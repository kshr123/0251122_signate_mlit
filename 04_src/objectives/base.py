"""
Custom objective functions for LightGBM

汎用的なカスタム目的関数の基底クラスと実装。
BaseObjectiveを継承して独自の目的関数を作成可能。

使用方法:
    from objectives import MAPEObjective, FocalObjective

    # 基本的な使い方
    objective = MAPEObjective()
    model = LGBMRegressor(objective=objective, metric="none")
    model.fit(X, y, eval_metric=objective.eval_metric)

    # パラメータ付き
    objective = FocalObjective(gamma=2.0)

    # 継承してカスタマイズ
    class MyObjective(BaseObjective):
        name = "my_loss"
        def gradient(self, y_true, y_pred): ...
        def hessian(self, y_true, y_pred): ...
        def loss(self, y_true, y_pred): ...
"""

import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod


class BaseObjective(ABC):
    """カスタム目的関数の基底クラス

    LightGBM用のカスタム目的関数を定義するための抽象基底クラス。
    サブクラスで gradient, hessian, loss の3メソッドを実装する。

    Attributes:
        name: 目的関数の名前（eval_metricで使用）
    """

    name: str = "base"

    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """勾配（一次微分）を計算

        Args:
            y_true: 正解値
            y_pred: 予測値

        Returns:
            各サンプルの勾配
        """
        pass

    @abstractmethod
    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """ヘシアン（二次微分）を計算

        Args:
            y_true: 正解値
            y_pred: 予測値

        Returns:
            各サンプルのヘシアン
        """
        pass

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """損失値を計算（eval_metric用）

        Args:
            y_true: 正解値
            y_pred: 予測値

        Returns:
            損失値（スカラー）
        """
        pass

    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """LightGBMから呼び出される目的関数

        Args:
            y_true: 正解値
            y_pred: 予測値

        Returns:
            (gradient, hessian) のタプル
        """
        return self.gradient(y_true, y_pred), self.hessian(y_true, y_pred)

    def eval_metric(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[str, float, bool]:
        """early stopping用の評価関数

        LightGBM sklearn APIのeval_metricパラメータに渡す。

        Note:
            sklearn APIでは (y_true, y_pred) の順で渡される。
            ネイティブAPIの (y_pred, dataset) とは異なるので注意。

        Args:
            y_true: 正解値
            y_pred: 予測値

        Returns:
            (metric_name, metric_value, is_higher_better)
        """
        return self.name, self.loss(y_true, y_pred), False  # lower is better


class MAPEObjective(BaseObjective):
    """MAPE (Mean Absolute Percentage Error) 目的関数

    MAPE = mean(|y_true - y_pred| / |y_true|) * 100

    低価格帯の相対誤差を直接最小化したい場合に有効。
    注意: MAPEは二次微分が存在しないため、ヘシアンは定数で近似。

    Args:
        hessian_const: ヘシアン近似値（default: 1.0）
            - 大きいほど慎重な学習（更新幅が小さい）
            - 小さいほど積極的な学習（更新幅が大きい）
            - 推奨範囲: 0.1〜2.0
    """

    name: str = "mape"

    def __init__(self, hessian_const: float = 1.0):
        self.hessian_const = hessian_const

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """MAPEの勾配

        d/dy_pred (|y_true - y_pred| / |y_true|)
        = sign(y_pred - y_true) / |y_true|
        """
        y_true_safe = np.maximum(np.abs(y_true), 1e-8)
        return np.sign(y_pred - y_true) / y_true_safe

    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """MAPEのヘシアン（近似）

        MAPEは絶対値を含むため二次微分が存在しない。
        定数で近似して学習を安定させる。
        """
        return np.ones_like(y_true) * self.hessian_const

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MAPE損失値"""
        y_true_safe = np.maximum(np.abs(y_true), 1e-8)
        return float(np.mean(np.abs(y_true - y_pred) / y_true_safe) * 100)


class FocalObjective(BaseObjective):
    """Focal Loss 目的関数（回帰版）

    難しいサンプル（誤差が大きいサンプル）に重みを付ける。
    weight = |residual|^gamma
    Loss = mean(weight * residual^2)

    Args:
        gamma: フォーカシングパラメータ（default: 2.0）
               大きいほど難サンプルを重視

    分類タスクのFocal Lossを回帰に応用したもの。
    外れ値や予測困難なサンプルの学習を強化。
    """

    name: str = "focal"

    def __init__(self, gamma: float = 2.0):
        self.gamma = gamma

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Focal Lossの勾配

        L = weight * residual^2
        weight = |residual|^gamma
        簡略化した勾配: 2 * weight * residual
        """
        residual = y_pred - y_true
        weight = np.power(np.abs(residual) + 1e-8, self.gamma)
        return 2 * weight * residual

    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Focal Lossのヘシアン（近似）"""
        residual = y_pred - y_true
        weight = np.power(np.abs(residual) + 1e-8, self.gamma)
        return 2 * weight

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Focal Loss損失値"""
        residual = y_pred - y_true
        weight = np.power(np.abs(residual) + 1e-8, self.gamma)
        return float(np.mean(weight * residual**2))
