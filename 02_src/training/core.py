"""CV学習のコアロジック

CVRunner: フレームワーク非依存のCVループ実行
CVResult: CV実行結果のデータクラス
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from evaluation.metrics import calculate_mape

if TYPE_CHECKING:
    from training.callbacks import Callback
    from training.fold_trainers import FoldTrainer


class PrunedException(Exception):
    """CV途中でpruneされた場合の例外"""

    pass


@dataclass
class CVResult:
    """CV実行結果"""

    oof_predictions: np.ndarray
    fold_scores: list[float]
    models: list[Any]
    feature_importance: np.ndarray | None = None
    best_iterations: list[int] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        """スコアの平均"""
        return float(np.mean(self.fold_scores))

    @property
    def std_score(self) -> float:
        """スコアの標準偏差"""
        return float(np.std(self.fold_scores))


class CVRunner:
    """CVループの実行（フレームワーク非依存）

    責務:
      - foldループの実行
      - OOF予測の集約
      - スコア計算
      - コールバック呼び出し

    責務外:
      - モデル学習（FoldTrainerに委譲）
      - CV分割（外部から受け取る）
      - 評価関数の実装（外部から受け取る）
    """

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_splits: list[tuple[np.ndarray, np.ndarray]],
        fold_trainer: FoldTrainer,
        scorer: Callable[[np.ndarray, np.ndarray], float] | None = None,
        callbacks: list[Callback] | None = None,
    ) -> CVResult:
        """CV実行

        Args:
            X: 特徴量（numpy array or pandas DataFrame）
            y: ターゲット（元スケール）
            cv_splits: [(train_idx, val_idx), ...]
            fold_trainer: 1 fold学習を担当するオブジェクト
            scorer: 評価関数（デフォルト: MAPE）
            callbacks: コールバックリスト

        Returns:
            CVResult: OOF予測、スコア、モデル等を含む結果

        Raises:
            PrunedException: コールバックがpruneを要求した場合
        """
        scorer = scorer or calculate_mape
        callbacks = callbacks or []

        # pandas/polars → numpy変換
        if hasattr(X, "values"):
            X_arr = X.values
        elif hasattr(X, "to_numpy"):
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)

        n_samples = len(y)
        oof = np.zeros(n_samples)
        scores: list[float] = []
        models: list[Any] = []
        feature_importances: list[np.ndarray] = []
        best_iterations: list[int] = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            # コールバック: fold開始
            for cb in callbacks:
                cb.on_fold_start(fold_idx)

            # FoldTrainerにtrain_idxを通知（sample_weight用）
            if hasattr(fold_trainer, "set_train_idx"):
                fold_trainer.set_train_idx(train_idx)

            # 1 fold学習
            result = fold_trainer.train_fold(
                X_arr[train_idx],
                y[train_idx],
                X_arr[val_idx],
                y[val_idx],
            )

            # 結果を集約
            oof[val_idx] = result.predictions
            score = scorer(y[val_idx], result.predictions)
            scores.append(score)
            models.append(result.model)

            if result.feature_importance is not None:
                feature_importances.append(result.feature_importance)

            if result.best_iteration is not None:
                best_iterations.append(result.best_iteration)

            # コールバック: fold終了
            for cb in callbacks:
                action = cb.on_fold_end(fold_idx, score)
                if action == "prune":
                    raise PrunedException(f"Pruned at fold {fold_idx}")

        # 特徴量重要度の平均
        fi = None
        if feature_importances:
            fi = np.mean(feature_importances, axis=0)

        return CVResult(
            oof_predictions=oof,
            fold_scores=scores,
            models=models,
            feature_importance=fi,
            best_iterations=best_iterations,
        )
