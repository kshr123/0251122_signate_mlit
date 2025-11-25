"""
評価指標のテスト
"""

import pytest
import numpy as np

from evaluation.metrics import calculate_mape


def test_calculate_mape_perfect_prediction():
    """完全予測の場合、MAPEは0になること"""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 200.0, 300.0])

    mape = calculate_mape(y_true, y_pred)

    assert mape == 0.0


def test_calculate_mape_known_values():
    """既知の値でMAPEが正しく計算されること"""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])

    # |100-110|/100 = 0.1 = 10%
    # |200-190|/200 = 0.05 = 5%
    # |300-310|/300 = 0.0333... = 3.33%
    # 平均 = (10 + 5 + 3.33) / 3 = 6.11%
    expected = (10.0 + 5.0 + 10.0/3) / 3

    mape = calculate_mape(y_true, y_pred)

    assert abs(mape - expected) < 0.01


def test_calculate_mape_with_small_true_values():
    """真値が小さい場合もMAPEが計算できること（epsilon対策）"""
    y_true = np.array([0.1, 0.2, 0.3])
    y_pred = np.array([0.11, 0.19, 0.31])

    mape = calculate_mape(y_true, y_pred)

    assert isinstance(mape, float)
    assert mape >= 0


def test_calculate_mape_percentage_format():
    """MAPEがパーセンテージ（0-100）で返されること"""
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 220.0])

    # |100-110|/100 = 0.1 = 10%
    # |200-220|/200 = 0.1 = 10%
    # 平均 = 10%
    mape = calculate_mape(y_true, y_pred)

    assert abs(mape - 10.0) < 0.01


def test_calculate_mape_handles_arrays():
    """numpy配列とリストの両方に対応すること"""
    y_true_array = np.array([100.0, 200.0, 300.0])
    y_pred_array = np.array([110.0, 190.0, 310.0])

    y_true_list = [100.0, 200.0, 300.0]
    y_pred_list = [110.0, 190.0, 310.0]

    mape_array = calculate_mape(y_true_array, y_pred_array)
    mape_list = calculate_mape(y_true_list, y_pred_list)

    assert abs(mape_array - mape_list) < 0.01
