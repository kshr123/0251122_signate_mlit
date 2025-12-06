"""ターゲット変換モジュール（Strategy Pattern）

価格予測などでターゲット変数を変換するためのクラス群。
Log1p変換が標準だが、必要に応じて他の変換も追加可能。

Examples:
    >>> transform = Log1pTransform()
    >>> y_transformed = transform.transform(y)
    >>> y_original = transform.inverse_transform(y_transformed)
"""

from abc import ABC, abstractmethod

import numpy as np


class TargetTransform(ABC):
    """ターゲット変換の抽象クラス（Strategy Pattern）"""

    @abstractmethod
    def transform(self, y: np.ndarray) -> np.ndarray:
        """元スケール → 学習用スケール

        Args:
            y: ターゲット配列（元スケール）

        Returns:
            変換後の配列
        """
        pass

    @abstractmethod
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """学習用スケール → 元スケール

        Args:
            y: 変換済み配列

        Returns:
            元スケールの配列
        """
        pass


class Log1pTransform(TargetTransform):
    """log1p変換（価格予測の標準）

    y_transformed = log(1 + y)
    y_original = exp(y_transformed) - 1

    Notes:
        - 負の予測値は0にクリップ
    """

    def transform(self, y: np.ndarray) -> np.ndarray:
        return np.log1p(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return np.maximum(np.expm1(y), 0)


class IdentityTransform(TargetTransform):
    """変換なし（そのまま使用）"""

    def transform(self, y: np.ndarray) -> np.ndarray:
        return y

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return y
