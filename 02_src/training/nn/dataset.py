"""PyTorch Dataset for regression tasks"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class RegressionDataset(Dataset):
    """回帰タスク用Dataset

    numpy配列をTensorに変換してDataLoaderで使用。

    Usage:
        # 学習時（ラベルあり）
        dataset = RegressionDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=1024, shuffle=True)

        # 推論時（ラベルなし）
        dataset = RegressionDataset(X_test)
        loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        """初期化

        Args:
            X: 特徴量 (n_samples, n_features)
            y: ターゲット (n_samples,) or None（推論時）
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """インデックスでアクセス

        Returns:
            学習時: (X[idx], y[idx])
            推論時: X[idx]
        """
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]
