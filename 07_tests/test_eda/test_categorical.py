"""カテゴリ変数分析のテスト"""

import pytest
import polars as pl
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent / "04_src"
sys.path.insert(0, str(project_root))

from eda.categorical import (
    classify_cardinality,
    calculate_target_encoding_potential,
    get_category_target_stats
)


def test_classify_cardinality_low():
    """低カーディナリティが正しく分類される"""
    df = pl.DataFrame({
        'cat1': ['A', 'B', 'A', 'B', 'C'],  # 3個
        'cat2': ['X', 'Y', 'X', 'Y', 'Z'],  # 3個
    })

    result = classify_cardinality(df, ['cat1', 'cat2'])

    assert len(result['low']) == 2
    assert len(result['medium']) == 0
    assert len(result['high']) == 0


def test_classify_cardinality_medium():
    """中カーディナリティが正しく分類される"""
    # 長さを揃えるため、繰り返しでデータを作成
    df = pl.DataFrame({
        'cat1': [i % 20 for i in range(100)],  # 20個のユニーク値
        'cat2': [i % 30 for i in range(100)],  # 30個のユニーク値
    })

    result = classify_cardinality(df, ['cat1', 'cat2'])

    assert len(result['low']) == 0
    assert len(result['medium']) == 2
    assert len(result['high']) == 0


def test_classify_cardinality_high():
    """高カーディナリティが正しく分類される"""
    # 長さを揃えるため、繰り返しでデータを作成
    df = pl.DataFrame({
        'cat1': [i % 100 for i in range(300)],  # 100個のユニーク値
        'cat2': [i % 200 for i in range(300)],  # 200個のユニーク値
    })

    result = classify_cardinality(df, ['cat1', 'cat2'])

    assert len(result['low']) == 0
    assert len(result['medium']) == 0
    assert len(result['high']) == 2


def test_classify_cardinality_mixed():
    """混在した場合も正しく分類される"""
    df = pl.DataFrame({
        'low': ['A', 'B'] * 50,      # 2個
        'medium': list(range(25)) * 4,  # 25個
        'high': list(range(100))     # 100個
    })

    result = classify_cardinality(df, ['low', 'medium', 'high'])

    assert len(result['low']) == 1
    assert result['low'][0][0] == 'low'
    assert len(result['medium']) == 1
    assert result['medium'][0][0] == 'medium'
    assert len(result['high']) == 1
    assert result['high'][0][0] == 'high'


def test_classify_cardinality_sorted():
    """各カテゴリ内でn_uniqueの降順にソートされる"""
    # 長さを揃えるため、繰り返しでデータを作成
    df = pl.DataFrame({
        'cat1': [i % 3 for i in range(10)],  # 3個のユニーク値
        'cat2': [i % 2 for i in range(10)],  # 2個のユニーク値
        'cat3': [i % 5 for i in range(10)]   # 5個のユニーク値
    })

    result = classify_cardinality(df, ['cat1', 'cat2', 'cat3'])

    # lowカテゴリ内でソートされている
    assert result['low'][0][1] > result['low'][1][1]  # 5 > 3
    assert result['low'][1][1] > result['low'][2][1]  # 3 > 2


def test_target_encoding_potential_high_variance():
    """カテゴリ間でターゲットの差が大きい場合、大きな値を返す"""
    df = pl.DataFrame({
        'category': ['A', 'A', 'B', 'B'],
        'target': [1.0, 2.0, 100.0, 200.0]  # A平均=1.5, B平均=150
    })

    potential = calculate_target_encoding_potential(df, 'category', 'target')

    # 標準偏差が大きい
    assert potential > 50


def test_target_encoding_potential_low_variance():
    """カテゴリ間でターゲットの差が小さい場合、小さな値を返す"""
    df = pl.DataFrame({
        'category': ['A', 'A', 'B', 'B'],
        'target': [10.0, 11.0, 10.5, 11.5]  # A平均=10.5, B平均=11
    })

    potential = calculate_target_encoding_potential(df, 'category', 'target')

    # 標準偏差が小さい
    assert potential < 1.0


def test_target_encoding_potential_single_category():
    """カテゴリが1つの場合は0を返す"""
    df = pl.DataFrame({
        'category': ['A', 'A', 'A', 'A'],
        'target': [1.0, 2.0, 3.0, 4.0]
    })

    potential = calculate_target_encoding_potential(df, 'category', 'target')

    assert potential == 0.0


def test_get_category_target_stats_basic():
    """カテゴリ別統計量が正しく計算される"""
    df = pl.DataFrame({
        'category': ['A', 'A', 'B', 'B', 'C'],
        'target': [1.0, 2.0, 10.0, 20.0, 100.0]
    })

    stats = get_category_target_stats(df, 'category', 'target')

    # 3カテゴリ存在
    assert stats.shape[0] == 3
    # 必要なカラムが存在
    assert 'count' in stats.columns
    assert 'mean_target' in stats.columns
    assert 'std_target' in stats.columns


def test_get_category_target_stats_sorted_by_count():
    """頻度順にソートされる"""
    df = pl.DataFrame({
        'category': ['A'] * 10 + ['B'] * 5 + ['C'] * 2,
        'target': [1.0] * 17
    })

    stats = get_category_target_stats(df, 'category', 'target')

    # 頻度降順
    assert stats['count'][0] == 10
    assert stats['count'][1] == 5
    assert stats['count'][2] == 2


def test_get_category_target_stats_top_n():
    """top_nで件数制限できる"""
    df = pl.DataFrame({
        'category': list('ABCDEFGHIJ'),
        'target': [1.0] * 10
    })

    stats = get_category_target_stats(df, 'category', 'target', top_n=3)

    # 上位3件のみ
    assert stats.shape[0] == 3
