"""NN専用の特徴量変換

FeatureScaler: 特徴量の正規化（StandardScaler wrapper）
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """NN用特徴量スケーラー

    sklearn.preprocessing.StandardScalerのラッパー。
    fit/transform分離でデータリーク防止。

    Usage:
        scaler = FeatureScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)  # fit済みのパラメータで変換
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        """学習データでfit

        Args:
            X: 特徴量 (n_samples, n_features)

        Returns:
            self
        """
        self.scaler.fit(X)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """変換（fit後のみ使用可能）

        Args:
            X: 特徴量 (n_samples, n_features)

        Returns:
            スケーリング済み特徴量

        Raises:
            RuntimeError: fit()が呼ばれていない場合
        """
        if not self._fitted:
            raise RuntimeError("fit() must be called before transform()")
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """fit + transform

        Args:
            X: 特徴量 (n_samples, n_features)

        Returns:
            スケーリング済み特徴量
        """
        return self.fit(X).transform(X)

    @property
    def mean_(self) -> np.ndarray:
        """各特徴量の平均"""
        return self.scaler.mean_

    @property
    def scale_(self) -> np.ndarray:
        """各特徴量のスケール（標準偏差）"""
        return self.scaler.scale_
