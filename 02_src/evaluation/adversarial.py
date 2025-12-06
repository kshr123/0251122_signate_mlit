"""Adversarial Validation モジュール.

Train/Test分布差を検出するための分類ベースの検証手法。
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


@dataclass
class AdversarialResult:
    """Adversarial Validation結果."""

    auc_mean: float
    auc_std: float
    fold_aucs: list[float]
    feature_importance: pl.DataFrame  # name, importance
    probabilities: np.ndarray  # 全サンプルの確率
    train_probs: np.ndarray  # Trainサンプルの確率
    test_probs: np.ndarray  # Testサンプルの確率
    n_train: int
    n_test: int


class AdversarialValidator:
    """Adversarial Validation実行クラス.

    Train/Testを見分ける分類モデルを学習し、AUCで分布差を測定する。

    Attributes:
        seed: 乱数シード
        n_splits: CV分割数
        model_params: LightGBMパラメータ
        early_stopping_rounds: Early stopping rounds

    Example:
        >>> validator = AdversarialValidator(config["adversarial"])
        >>> result = validator.run(X_train, X_test)
        >>> print(f"AUC: {result.auc_mean:.4f}")
    """

    def __init__(self, config: dict):
        """初期化.

        Args:
            config: adversarialセクションの設定辞書
                - seed: 乱数シード
                - n_splits: CV分割数
                - model.params: LightGBMパラメータ
                - model.early_stopping_rounds: Early stopping
        """
        self.seed = config["seed"]
        self.n_splits = config["n_splits"]
        self.model_params = config["model"]["params"]
        self.early_stopping_rounds = config["model"]["early_stopping_rounds"]

    def run(
        self,
        X_train: np.ndarray | pl.DataFrame,
        X_test: np.ndarray | pl.DataFrame,
        feature_names: list[str] | None = None,
    ) -> AdversarialResult:
        """Adversarial Validation実行.

        Args:
            X_train: Train特徴量
            X_test: Test特徴量
            feature_names: 特徴量名リスト（省略時は自動生成）

        Returns:
            AdversarialResult: AUC、特徴量重要度、確率
        """
        # polars DataFrameの場合は特徴量名を取得してnumpyに変換
        if isinstance(X_train, pl.DataFrame):
            if feature_names is None:
                feature_names = X_train.columns
            X_train = X_train.to_numpy()
        if isinstance(X_test, pl.DataFrame):
            X_test = X_test.to_numpy()

        n_train = len(X_train)
        n_test = len(X_test)

        # データセット作成
        X, y = self._create_dataset(X_train, X_test)

        # CV学習
        probs, fold_aucs, feature_importance = self._train_cv(X, y, feature_names)

        # 結果集計
        auc_mean = float(np.mean(fold_aucs))
        auc_std = float(np.std(fold_aucs))

        return AdversarialResult(
            auc_mean=auc_mean,
            auc_std=auc_std,
            fold_aucs=fold_aucs,
            feature_importance=feature_importance,
            probabilities=probs,
            train_probs=probs[:n_train],
            test_probs=probs[n_train:],
            n_train=n_train,
            n_test=n_test,
        )

    def _create_dataset(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train/Test結合・ラベル作成.

        Args:
            X_train: Train特徴量
            X_test: Test特徴量

        Returns:
            結合データX, ラベルy のタプル（Train=0, Test=1）
        """
        X = np.vstack([X_train, X_test])
        y = np.array([0] * len(X_train) + [1] * len(X_test))
        return X, y

    def _train_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None,
    ) -> tuple[np.ndarray, list[float], pl.DataFrame]:
        """CV学習・予測.

        Args:
            X: 特徴量
            y: ラベル（0=Train, 1=Test）
            feature_names: 特徴量名リスト

        Returns:
            確率, fold別AUC, 特徴量重要度 のタプル
        """
        cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed
        )

        probs = np.zeros(len(X))
        fold_aucs: list[float] = []
        importance_list: list[np.ndarray] = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # LightGBM Dataset
            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            # 学習
            model = lgb.train(
                self.model_params,
                train_data,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

            # 予測
            probs[val_idx] = model.predict(X_val)

            # AUC
            auc = roc_auc_score(y_val, probs[val_idx])
            fold_aucs.append(float(auc))

            # 特徴量重要度
            importance_list.append(model.feature_importance(importance_type="gain"))

            print(f"  Fold {fold + 1}/{self.n_splits}: AUC = {auc:.4f}")

        # 特徴量重要度の平均
        importance_mean = np.mean(importance_list, axis=0)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        feature_importance = pl.DataFrame({
            "name": feature_names,
            "importance": importance_mean,
        }).sort("importance", descending=True)

        return probs, fold_aucs, feature_importance
