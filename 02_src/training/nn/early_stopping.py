"""PyTorch用Early Stopping"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EarlyStopping:
    """PyTorch用Early Stopping

    検証ロスが改善しない場合に学習を停止。
    ベストモデルの状態を保持。

    Usage:
        early_stopping = EarlyStopping(patience=20)

        for epoch in range(epochs):
            train_loss = train_one_epoch(...)
            val_loss = validate(...)

            if early_stopping(val_loss, model):
                print(f"Early stopping at epoch {epoch}")
                break

        # ベストモデルを復元
        early_stopping.load_best_model(model)
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """初期化

        Args:
            patience: 改善がない場合に待つエポック数
            min_delta: 改善とみなす最小変化量
            mode: "min"（ロス最小化）or "max"（スコア最大化）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_score: float | None = None
        self.best_epoch: int = 0
        self.best_state_dict: dict | None = None

        # mode に応じた比較関数
        if mode == "min":
            self._is_better = lambda score, best: score < best - min_delta
        elif mode == "max":
            self._is_better = lambda score, best: score > best + min_delta
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def __call__(self, score: float, model: nn.Module, epoch: int = 0) -> bool:
        """スコアをチェック

        Args:
            score: 検証スコア（ロス or メトリクス）
            model: PyTorchモデル
            epoch: 現在のエポック番号

        Returns:
            True: 学習停止すべき
            False: 継続
        """
        if self.best_score is None or self._is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            return True

        return False

    def load_best_model(self, model: nn.Module) -> None:
        """ベストモデルの重みを復元

        Args:
            model: 重みを復元するモデル
        """
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
