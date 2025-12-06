"""NN用FoldTrainer

PyTorch NNをCVRunner/FoldTrainerアーキテクチャに統合。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.fold_trainers import FoldResult, FoldTrainer
from training.transforms import IdentityTransform, TargetTransform

from .dataset import RegressionDataset
from .early_stopping import EarlyStopping
from .transforms import FeatureScaler


class NNFoldTrainer(FoldTrainer):
    """PyTorch NN用FoldTrainer

    既存のFoldTrainerと同じインターフェースを実装。
    CVRunnerから透過的に使用可能。

    Usage:
        from training.nn import NNFoldTrainer, MLP

        fold_trainer = NNFoldTrainer(
            model_class=MLP,
            model_params={"hidden_dims": [512, 256, 128], "dropout": 0.3},
            target_transform=Log1pTransform(),
            epochs=100,
            early_stopping_rounds=20,
        )

        result = fold_trainer.train_fold(X_train, y_train, X_val, y_val)
    """

    def __init__(
        self,
        model_class: type,
        model_params: dict,
        target_transform: TargetTransform | None = None,
        # 学習パラメータ
        epochs: int = 100,
        batch_size: int = 1024,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        optimizer: str = "adamw",
        # Early stopping
        early_stopping_rounds: int = 20,
        # 特徴量スケーリング
        scale_features: bool = True,
        # デバイス
        device: str = "auto",
        # ログ
        verbose: bool = True,
    ):
        """初期化

        Args:
            model_class: nn.Moduleのクラス（MLP等）
            model_params: モデルのパラメータ（hidden_dims, dropout等）
            target_transform: ターゲット変換（None時はIdentityTransform）
            epochs: 最大エポック数
            batch_size: バッチサイズ
            lr: 学習率
            weight_decay: L2正則化
            optimizer: 最適化アルゴリズム ("adam", "adamw", "sgd")
            early_stopping_rounds: Early stoppingのpatience
            scale_features: 特徴量を正規化するか
            device: デバイス ("auto", "cpu", "mps", "cuda")
            verbose: ログ出力
        """
        self.model_class = model_class
        self.model_params = model_params
        self.transform = target_transform or IdentityTransform()

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer

        self.early_stopping_rounds = early_stopping_rounds
        self.scale_features = scale_features
        self.device = self._get_device(device)
        self.verbose = verbose

        # fold学習時に初期化
        self.feature_scaler: FeatureScaler | None = None

    def _get_device(self, device: str) -> torch.device:
        """デバイス自動選択"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """オプティマイザ作成"""
        if self.optimizer_name == "adam":
            return torch.optim.Adam(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "adamw":
            return torch.optim.AdamW(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

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
            FoldResult: 予測値、モデル、best_iteration等
        """
        # 1. 特徴量スケーリング（NN専用）
        if self.scale_features:
            self.feature_scaler = FeatureScaler()
            X_train = self.feature_scaler.fit_transform(X_train)
            X_val = self.feature_scaler.transform(X_val)

        # 2. ターゲット変換
        y_train_t = self.transform.transform(y_train)
        y_val_t = self.transform.transform(y_val)

        # 3. DataLoader作成
        train_loader = DataLoader(
            RegressionDataset(X_train, y_train_t),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            RegressionDataset(X_val, y_val_t),
            batch_size=self.batch_size,
            shuffle=False,
        )

        # 4. モデル作成
        input_dim = X_train.shape[1]
        model = self.model_class(input_dim=input_dim, **self.model_params)
        model = model.to(self.device)

        # 5. オプティマイザ・損失関数
        optimizer = self._create_optimizer(model)
        criterion = nn.MSELoss()

        # 6. Early stopping
        early_stopping = EarlyStopping(patience=self.early_stopping_rounds)

        # 7. 学習ループ
        for epoch in range(self.epochs):
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
            val_loss = self._validate_epoch(model, val_loader, criterion)

            if self.verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

            if early_stopping(val_loss, model, epoch):
                if self.verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

        # 8. ベストモデル復元
        early_stopping.load_best_model(model)

        # 9. 予測（行列演算で一括）
        pred_t = self._predict_all(model, X_val)
        pred = self.transform.inverse_transform(pred_t)

        return FoldResult(
            predictions=pred,
            model=self._create_model_wrapper(model),
            feature_importance=None,
            best_iteration=early_stopping.best_epoch,
        )

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """1エポック学習（行列演算でバッチ処理）"""
        model.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad()
            pred = model(X_batch).squeeze()
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _validate_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> float:
        """1エポック検証（行列演算で一括）"""
        model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                pred = model(X_batch).squeeze()
                loss = criterion(pred, y_batch)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def _predict_all(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        """全データを一括予測（行列演算）"""
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pred = model(X_tensor).squeeze()

        return pred.cpu().numpy()

    def _create_model_wrapper(self, model: nn.Module) -> "NNModelWrapper":
        """推論用ラッパー作成（CVResultで使用）"""
        return NNModelWrapper(
            model=model,
            feature_scaler=self.feature_scaler,
            target_transform=self.transform,
            device=self.device,
        )


class NNModelWrapper:
    """NN推論用ラッパー

    GBDT系モデルと同じpredict()インターフェースを提供。
    スケーリング・逆変換を内包。
    """

    def __init__(
        self,
        model: nn.Module,
        feature_scaler: FeatureScaler | None,
        target_transform: TargetTransform,
        device: torch.device,
    ):
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_transform = target_transform
        self.device = device

    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測（変換後スケール）

        Note:
            CVRunnerとの互換性のため、変換後スケールで返す。
            呼び出し側でinverse_transformする。
        """
        # スケーリング
        if self.feature_scaler is not None:
            X = self.feature_scaler.transform(X)

        # 予測
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pred = self.model(X_tensor).squeeze()

        return pred.cpu().numpy()
