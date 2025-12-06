"""NN Models for tabular data"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """シンプルなMLP

    構成: [Linear → BatchNorm → Activation → Dropout] × n_layers → Linear

    Usage:
        model = MLP(input_dim=100, hidden_dims=[512, 256, 128], dropout=0.3)
        output = model(x)  # (batch_size, 1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = None,
        dropout: float = 0.3,
        activation: str = "relu",
        use_batchnorm: bool = True,
    ):
        """初期化

        Args:
            input_dim: 入力次元（特徴量数）
            hidden_dims: 各隠れ層の次元リスト（デフォルト: [512, 256, 128]）
            dropout: ドロップアウト率
            activation: 活性化関数 ("relu", "leaky_relu", "gelu", "silu")
            use_batchnorm: BatchNormを使うか
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        # 活性化関数
        activation_fn = self._get_activation(activation)

        # レイヤー構築
        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(activation_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim

        # 出力層
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)

        # 重み初期化
        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """活性化関数を取得"""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
        return activations[name]

    def _init_weights(self):
        """重み初期化（Kaiming初期化）"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播

        Args:
            x: (batch_size, input_dim)

        Returns:
            (batch_size, 1)
        """
        return self.model(x)
