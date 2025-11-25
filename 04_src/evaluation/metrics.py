"""
評価指標モジュール

MAPE等の評価指標を計算
"""

import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from typing import Union


def calculate_mape(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> float:
    """
    MAPE (Mean Absolute Percentage Error) を計算

    sklearn.metrics.mean_absolute_percentage_errorのラッパー関数。
    返り値をパーセンテージ（0-100）に変換する。

    Args:
        y_true: 真値
        y_pred: 予測値

    Returns:
        MAPE（パーセンテージ、0-100の範囲）

    Examples:
        >>> y_true = [100, 200, 300]
        >>> y_pred = [110, 190, 310]
        >>> mape = calculate_mape(y_true, y_pred)
        >>> print(f"MAPE: {mape:.2f}%")
        MAPE: 6.11%

    Notes:
        - sklearnのMAPEは0-1の範囲で返すため、100倍してパーセンテージに変換
        - y_true=0の場合、sklearnが自動的に処理
    """
    # sklearnのMAPEは0-1の範囲で返すため、100倍してパーセンテージに変換
    mape_ratio = mean_absolute_percentage_error(y_true, y_pred)
    return float(mape_ratio * 100)
