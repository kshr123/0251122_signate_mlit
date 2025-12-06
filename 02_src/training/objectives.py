"""カスタム目的関数・評価関数

LightGBM用のカスタム目的関数（objective）と評価関数（metric）を定義。
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def asymmetric_mse_objective(alpha: float = 0.6) -> Callable:
    """非対称MSE目的関数（LightGBM用）

    過大予測と過小予測で異なる重みを適用する非対称な損失関数。

    Args:
        alpha: 過大予測の重み（0.5より大きいと過大ペナルティ強化）
            - alpha=0.5: 通常のMSE
            - alpha=0.6: 過大:過小 = 6:4
            - alpha=0.7: 過大:過小 = 7:3

    Returns:
        LightGBM用目的関数 (y_true, y_pred) -> (grad, hess)

    Example:
        >>> params = {
        ...     "objective": asymmetric_mse_objective(alpha=0.6),
        ...     "metric": "None",  # カスタム目的関数使用時
        ...     ...
        ... }
    """

    def objective(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """勾配とヘシアンを計算"""
        residual = y_pred - y_true
        # 過大予測（residual > 0）と過小予測で重みを変える
        grad = np.where(residual > 0, 2 * alpha * residual, 2 * (1 - alpha) * residual)
        hess = np.where(residual > 0, 2 * alpha, 2 * (1 - alpha))
        return grad, hess

    return objective


def asymmetric_mse_metric(alpha: float = 0.6) -> Callable:
    """非対称MSE評価関数（LightGBM用）

    asymmetric_mse_objectiveと対になる評価関数。

    Args:
        alpha: 過大予測の重み

    Returns:
        LightGBM用評価関数 (y_true, y_pred) -> (name, value, is_higher_better)
    """

    def metric(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
        """非対称MSEを計算"""
        residual = y_pred - y_true
        weights = np.where(residual > 0, alpha, 1 - alpha)
        loss = np.mean(weights * residual**2)
        return "asymmetric_mse", loss, False

    return metric


def huber_objective(delta: float = 1.0) -> Callable:
    """Huber損失目的関数（LightGBM用）

    外れ値に対してロバストな損失関数。
    |residual| <= delta: MSE的な挙動
    |residual| > delta: MAE的な挙動

    Args:
        delta: MSEとMAEの切り替え閾値

    Returns:
        LightGBM用目的関数 (y_true, y_pred) -> (grad, hess)
    """

    def objective(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """勾配とヘシアンを計算"""
        residual = y_pred - y_true
        abs_residual = np.abs(residual)

        # |residual| <= delta: grad = residual, hess = 1
        # |residual| > delta: grad = delta * sign(residual), hess = 0
        grad = np.where(abs_residual <= delta, residual, delta * np.sign(residual))
        hess = np.where(abs_residual <= delta, 1.0, 0.0)

        return grad, hess

    return objective


def huber_metric(delta: float = 1.0) -> Callable:
    """Huber損失評価関数（LightGBM用）

    Args:
        delta: MSEとMAEの切り替え閾値

    Returns:
        LightGBM用評価関数 (y_true, y_pred) -> (name, value, is_higher_better)
    """

    def metric(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
        """Huber損失を計算"""
        residual = y_pred - y_true
        abs_residual = np.abs(residual)

        # |residual| <= delta: 0.5 * residual^2
        # |residual| > delta: delta * (|residual| - 0.5 * delta)
        loss = np.where(
            abs_residual <= delta,
            0.5 * residual**2,
            delta * (abs_residual - 0.5 * delta),
        )
        return "huber", np.mean(loss), False

    return metric
