"""TabNet用FoldTrainer

pytorch-tabnetのTabNetRegressorをラップし、
既存のFoldTrainerインターフェースに統合。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor

from training.fold_trainers import FoldResult, FoldTrainer
from training.transforms import IdentityTransform, TargetTransform


class TabNetFoldTrainer(FoldTrainer):
    """TabNet用FoldTrainer

    pytorch-tabnetのTabNetRegressorをラップ。
    CVRunnerから透過的に使用可能。

    Usage:
        from training.nn import TabNetFoldTrainer

        fold_trainer = TabNetFoldTrainer(
            tabnet_params={"n_d": 8, "n_a": 8, "n_steps": 3},
            target_transform=Log1pTransform(),
            max_epochs=100,
            patience=20,
        )

        result = fold_trainer.train_fold(X_train, y_train, X_val, y_val)
    """

    def __init__(
        self,
        tabnet_params: dict | None = None,
        target_transform: TargetTransform | None = None,
        # 学習パラメータ
        max_epochs: int = 100,
        patience: int = 20,
        batch_size: int = 1024,
        virtual_batch_size: int = 128,
        # デバイス
        device_name: str = "cpu",
        # ログ
        verbose: int = 1,
    ):
        """初期化

        Args:
            tabnet_params: TabNetのパラメータ（n_d, n_a, n_steps等）
            target_transform: ターゲット変換（None時はIdentityTransform）
            max_epochs: 最大エポック数
            patience: Early stoppingのpatience
            batch_size: バッチサイズ
            virtual_batch_size: Ghost BatchNormのバッチサイズ
            device_name: デバイス ("cpu", "cuda")
            verbose: ログレベル（0: なし, 1: あり）
        """
        self.tabnet_params = tabnet_params or {}
        self.transform = target_transform or IdentityTransform()
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.device_name = device_name
        self.verbose = verbose

    def train_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> FoldResult:
        """1 foldの学習を実行

        Args:
            X_train: 学習用特徴量 (n_train, n_features)
            y_train: 学習用ターゲット (n_train,)
            X_val: 検証用特徴量 (n_val, n_features)
            y_val: 検証用ターゲット (n_val,)

        Returns:
            FoldResult: 予測値、モデル、feature_importance、best_iteration
        """
        # 0. NaN処理（TabNetはNaN非対応）
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)

        # 1. ターゲット変換（TabNetは(n, 1)形式を期待）
        y_train_t = self.transform.transform(y_train).reshape(-1, 1)
        y_val_t = self.transform.transform(y_val).reshape(-1, 1)

        # 2. TabNetRegressor作成
        model = TabNetRegressor(
            device_name=self.device_name,
            verbose=self.verbose,
            **self.tabnet_params,
        )

        # 3. 学習（内部でearly stoppingあり）
        model.fit(
            X_train=X_train,
            y_train=y_train_t,
            eval_set=[(X_val, y_val_t)],
            eval_metric=["rmse"],
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
        )

        # 4. 予測（逆変換して元スケールに戻す）
        pred_t = model.predict(X_val).flatten()
        pred = self.transform.inverse_transform(pred_t)

        # 5. 特徴量重要度（TabNet固有）
        feature_importance = model.feature_importances_

        return FoldResult(
            predictions=pred,
            model=TabNetModelWrapper(model, self.transform),
            feature_importance=feature_importance,
            best_iteration=model.best_epoch if hasattr(model, "best_epoch") else 0,
        )


class TabNetModelWrapper:
    """TabNet推論用ラッパー

    GBDT系モデルと同じpredict()インターフェースを提供。
    """

    def __init__(
        self,
        model: TabNetRegressor,
        target_transform: TargetTransform,
    ):
        self.model = model
        self.target_transform = target_transform

    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測（変換後スケール）

        Note:
            CVRunnerとの互換性のため、変換後スケールで返す。
            呼び出し側でinverse_transformする。
        """
        # TabNetはNaN非対応
        X = np.nan_to_num(X, nan=0.0)
        return self.model.predict(X).flatten()

    def save_model(self, path: str) -> None:
        """モデル保存"""
        self.model.save_model(path)

    @classmethod
    def load_model(cls, path: str) -> "TabNetModelWrapper":
        """モデル読み込み"""
        model = TabNetRegressor()
        model.load_model(path)
        return cls(model, IdentityTransform())
