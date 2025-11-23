"""相関分析のテスト"""

import pytest
import polars as pl
import numpy as np
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent / "04_src"
sys.path.insert(0, str(project_root))

from eda.correlation import calculate_correlations, find_high_correlation_pairs


def test_calculate_correlations_returns_sorted():
    """相関係数が絶対値の降順でソートされる"""
    df = pl.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'x2': [5, 4, 3, 2, 1],  # 負の相関
        'x3': [1, 1, 1, 1, 1],  # 相関=0
        'target': [1, 2, 3, 4, 5]
    })

    result = calculate_correlations(df, 'target', ['x1', 'x2', 'x3'])

    # x1が最も相関が高い（+1.0）
    assert result['feature'][0] == 'x1'
    # x2が次（-1.0）
    assert result['feature'][1] == 'x2'
    # x3が最も低い（0.0）
    assert result['feature'][2] == 'x3'


def test_calculate_correlations_with_nan():
    """NaN値を含むカラムでも計算できる"""
    df = pl.DataFrame({
        'x1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'target': [1.0, 2.0, 3.0, 4.0, 5.0]
    })

    result = calculate_correlations(df, 'target', ['x1'])

    # NaNを除外して計算される
    assert len(result) == 1
    assert not np.isnan(result['correlation'][0])


def test_calculate_correlations_all_same_value():
    """全て同じ値の場合は相関=0"""
    df = pl.DataFrame({
        'x1': [1, 1, 1, 1, 1],
        'target': [1, 2, 3, 4, 5]
    })

    result = calculate_correlations(df, 'target', ['x1'])

    # 分散が0なので相関は0
    assert result['correlation'][0] == 0.0


def test_calculate_correlations_perfect_correlation():
    """完全相関（+1.0）を検出"""
    df = pl.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'target': [1, 2, 3, 4, 5]
    })

    result = calculate_correlations(df, 'target', ['x1'])

    assert abs(result['correlation'][0] - 1.0) < 1e-10


def test_calculate_correlations_negative_correlation():
    """負の相関を検出"""
    df = pl.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'target': [5, 4, 3, 2, 1]
    })

    result = calculate_correlations(df, 'target', ['x1'])

    assert abs(result['correlation'][0] - (-1.0)) < 1e-10


def test_find_high_correlation_pairs_detects_pairs():
    """高相関ペアを検出"""
    df = pl.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'x2': [1, 2, 3, 4, 5],  # x1と完全相関
        'x3': [5, 4, 3, 2, 1]   # 負の相関
    })

    pairs = find_high_correlation_pairs(df, ['x1', 'x2', 'x3'], threshold=0.8)

    # x1-x2ペアが検出される
    assert len(pairs) >= 1
    assert ('x1', 'x2') in [(p[0], p[1]) for p in pairs]


def test_find_high_correlation_pairs_no_pairs():
    """高相関ペアがない場合は空リスト"""
    df = pl.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'x2': [1, 1, 1, 1, 1],
        'x3': [2, 3, 1, 5, 4]  # ランダムな値で相関を低く
    })

    pairs = find_high_correlation_pairs(df, ['x1', 'x2', 'x3'], threshold=0.8)

    assert len(pairs) == 0


def test_find_high_correlation_pairs_sorted_by_abs_corr():
    """相関係数の絶対値で降順ソートされる"""
    df = pl.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'x2': [1, 2, 3, 4, 5],      # +1.0
        'x3': [1.0, 2.1, 2.9, 4.0, 5.1],  # 約+0.99
        'x4': [5, 4, 3, 2, 1]       # -1.0
    })

    pairs = find_high_correlation_pairs(df, ['x1', 'x2', 'x3', 'x4'], threshold=0.8)

    # 最初のペアが最も相関が高い
    if len(pairs) >= 2:
        assert abs(pairs[0][2]) >= abs(pairs[1][2])
