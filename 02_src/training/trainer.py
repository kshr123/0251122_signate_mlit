"""Trainer - 高レベルCV学習API（後方互換）

内部実装はCVRunnerとLightGBMFoldTrainerに委譲。
既存のAPIを維持しながら、新しいアーキテクチャを利用。

Examples:
    >>> from training import Trainer
    >>> from training.transforms import Log1pTransform
    >>>
    >>> trainer = Trainer(
    ...     target_transform=Log1pTransform(),
    ...     seed=42,
    ...     early_stopping_rounds=200,
    ... )
    >>> cv_result = trainer.cv_train(X, y, lgb_params, cv_splits)
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from evaluation.metrics import calculate_mape
from training.transforms import TargetTransform, Log1pTransform, IdentityTransform
from training.core import CVRunner
from training.fold_trainers import LightGBMFoldTrainer
from training.callbacks import VerboseCallback


class Trainer:
    """高レベルCV学習API（後方互換）

    内部実装:
      - CVループ: CVRunner
      - 1 fold学習: LightGBMFoldTrainer
      - ログ出力: VerboseCallback

    既存のAPIを維持しているため、train.pyの変更は不要。
    """

    def __init__(
        self,
        target_transform: Union[TargetTransform, str, None] = None,
        seed: int = 42,
        early_stopping_rounds: int = 200,
        verbose: bool = True,
        lgb_callbacks: list | None = None,
    ):
        """初期化

        Args:
            target_transform: ターゲット変換
                - TargetTransformインスタンス: そのまま使用
                - "log1p": Log1pTransform()
                - "none" or None: IdentityTransform()
            seed: 乱数シード
            early_stopping_rounds: 早期終了ラウンド数
            verbose: 詳細出力するか
            lgb_callbacks: LightGBM用追加コールバック（学習進捗ログ等）
        """
        # TargetTransformの解決
        if isinstance(target_transform, TargetTransform):
            self._transform = target_transform
        elif target_transform == "log1p":
            self._transform = Log1pTransform()
        else:
            self._transform = IdentityTransform()

        self._seed = seed
        self._early_stopping_rounds = early_stopping_rounds
        self._verbose = verbose
        self._lgb_callbacks = lgb_callbacks or []

    def prepare_target(self, y: np.ndarray) -> np.ndarray:
        """ターゲット変数を学習用に変換

        Args:
            y: 元スケールのターゲット

        Returns:
            変換後のターゲット
        """
        return self._transform.transform(y)

    def inverse_target(self, y: np.ndarray) -> np.ndarray:
        """予測値を元スケールに逆変換

        Args:
            y: 変換スケールの予測値

        Returns:
            元スケールの予測値
        """
        return self._transform.inverse_transform(y)

    def cv_train(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        model_params: dict,
        cv_splits: list[tuple],
        sample_weight: np.ndarray | None = None,
    ) -> dict:
        """CV学習を実行

        Args:
            X: 特徴量（numpy, pandas, polars対応）
            y: ターゲット（元スケール）
            model_params: LightGBMパラメータ
            cv_splits: CVのインデックス分割 [(train_idx, val_idx), ...]
            sample_weight: サンプル重み（オプション）

        Returns:
            {
                "cv_mapes": list[float],
                "cv_mape_mean": float,
                "cv_mape_std": float,
                "oof_predictions": np.ndarray,
                "best_iterations": list[int],
                "models": list[LGBMRegressor],
                "feature_importance": np.ndarray,
            }
        """
        # polars → pandas変換
        if hasattr(X, "to_pandas"):
            X_pd = X.to_pandas()
        elif isinstance(X, pd.DataFrame):
            X_pd = X
        else:
            X_pd = pd.DataFrame(X)

        n_splits = len(cv_splits)

        # FoldTrainer作成
        fold_trainer = LightGBMFoldTrainer(
            params=model_params,
            target_transform=self._transform,
            early_stopping_rounds=self._early_stopping_rounds,
            sample_weight=sample_weight,
            verbose=self._verbose,
            extra_callbacks=self._lgb_callbacks,
        )

        # コールバック
        callbacks = []
        if self._verbose:
            callbacks.append(VerboseCallback(n_splits=n_splits, metric_name="MAPE"))

        # CVRunner実行
        runner = CVRunner()
        cv_result = runner.run(
            X=X_pd,
            y=y,
            cv_splits=cv_splits,
            fold_trainer=fold_trainer,
            scorer=calculate_mape,
            callbacks=callbacks,
        )

        # 後方互換の形式に変換
        return {
            "cv_mapes": cv_result.fold_scores,
            "cv_mape_mean": cv_result.mean_score,
            "cv_mape_std": cv_result.std_score,
            "oof_predictions": cv_result.oof_predictions,
            "best_iterations": cv_result.best_iterations,
            "models": cv_result.models,
            "feature_importance": cv_result.feature_importance,
        }

    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MAPE計算（evaluation/metrics.pyに委譲）

        Args:
            y_true: 真値（元スケール）
            y_pred: 予測値（元スケール）

        Returns:
            MAPE (%)
        """
        return calculate_mape(y_true, y_pred)
