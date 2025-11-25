"""
CategoryNuniqueBlockのテスト

カテゴリAでグループ化したとき、カテゴリBのユニーク数（nunique）を計算するBlock。

注意: N個のカテゴリカラムで N×(N-1) 個の特徴量が生成されるため、
カラム数が多い場合は計算量・メモリに注意。
"""

import pytest
import polars as pl
import numpy as np

from features.blocks.aggregation import CategoryNuniqueBlock


def test_category_nunique_basic():
    """CategoryNuniqueBlock: カテゴリごとの別カテゴリのnuniqueを計算すること"""
    df = pl.DataFrame({
        "cat_a": ["X", "X", "X", "Y", "Y"],
        "cat_b": ["p", "q", "p", "r", "r"],
    })
    # cat_a="X"のとき cat_b は {p, q} → nunique=2
    # cat_a="Y"のとき cat_b は {r} → nunique=1
    # cat_b="p"のとき cat_a は {X} → nunique=1
    # cat_b="q"のとき cat_a は {X} → nunique=1
    # cat_b="r"のとき cat_a は {Y} → nunique=1

    block = CategoryNuniqueBlock(columns=["cat_a", "cat_b"])
    result = block.fit(df)

    # 2カラム × (2-1) = 2つの特徴量
    assert result.shape[1] == 2
    assert "nunique_cat_a_groupby_cat_b" in result.columns
    assert "nunique_cat_b_groupby_cat_a" in result.columns

    # cat_aでgroupbyしたcat_bのnunique
    values_a = result["nunique_cat_a_groupby_cat_b"].to_list()
    assert values_a[0] == 2  # X → 2
    assert values_a[1] == 2  # X → 2
    assert values_a[2] == 2  # X → 2
    assert values_a[3] == 1  # Y → 1
    assert values_a[4] == 1  # Y → 1


def test_category_nunique_three_columns():
    """CategoryNuniqueBlock: 3カラムで6つの特徴量が生成されること"""
    df = pl.DataFrame({
        "cat_a": ["X", "X", "Y", "Y"],
        "cat_b": ["p", "q", "p", "q"],
        "cat_c": ["1", "1", "2", "2"],
    })

    block = CategoryNuniqueBlock(columns=["cat_a", "cat_b", "cat_c"])
    result = block.fit(df)

    # 3カラム × (3-1) = 6つの特徴量
    assert result.shape[1] == 6

    expected_cols = [
        "nunique_cat_a_groupby_cat_b",
        "nunique_cat_a_groupby_cat_c",
        "nunique_cat_b_groupby_cat_a",
        "nunique_cat_b_groupby_cat_c",
        "nunique_cat_c_groupby_cat_a",
        "nunique_cat_c_groupby_cat_b",
    ]
    for col in expected_cols:
        assert col in result.columns


def test_category_nunique_transform_consistency():
    """CategoryNuniqueBlock: train/testで一貫したマッピングが適用されること"""
    train_df = pl.DataFrame({
        "cat_a": ["X", "X", "Y", "Y"],
        "cat_b": ["p", "q", "r", "r"],
    })
    test_df = pl.DataFrame({
        "cat_a": ["X", "Y"],
        "cat_b": ["p", "r"],
    })

    block = CategoryNuniqueBlock(columns=["cat_a", "cat_b"])
    _ = block.fit(train_df)
    test_result = block.transform(test_df)

    # trainで計算したnuniqueがtestにも適用される
    # X → cat_bのnunique=2 (p, q)
    # Y → cat_bのnunique=1 (r)
    values = test_result["nunique_cat_a_groupby_cat_b"].to_list()
    assert values[0] == 2  # X
    assert values[1] == 1  # Y


def test_category_nunique_unknown_category():
    """CategoryNuniqueBlock: 未知カテゴリを0で埋めること"""
    train_df = pl.DataFrame({
        "cat_a": ["X", "X", "Y"],
        "cat_b": ["p", "q", "r"],
    })
    test_df = pl.DataFrame({
        "cat_a": ["X", "Z"],  # Zは未知
        "cat_b": ["p", "s"],  # sは未知
    })

    block = CategoryNuniqueBlock(columns=["cat_a", "cat_b"])
    _ = block.fit(train_df)
    test_result = block.transform(test_df)

    values_a = test_result["nunique_cat_a_groupby_cat_b"].to_list()
    values_b = test_result["nunique_cat_b_groupby_cat_a"].to_list()

    # 既知カテゴリ
    assert values_a[0] == 2  # X → 2
    assert values_b[0] == 1  # p → 1

    # 未知カテゴリは0
    assert values_a[1] == 0  # Z → 0
    assert values_b[1] == 0  # s → 0


def test_category_nunique_not_fitted_error():
    """CategoryNuniqueBlock: fit前のtransformでRuntimeError"""
    df = pl.DataFrame({
        "cat_a": ["X", "Y"],
        "cat_b": ["p", "q"],
    })

    block = CategoryNuniqueBlock(columns=["cat_a", "cat_b"])

    with pytest.raises(RuntimeError, match="fit\\(\\)を先に実行してください"):
        block.transform(df)


def test_category_nunique_immutability():
    """CategoryNuniqueBlock: 元のDataFrameを変更しないこと"""
    df = pl.DataFrame({
        "cat_a": ["X", "Y", "X"],
        "cat_b": ["p", "q", "r"],
    })

    original_a = df["cat_a"].to_list()
    original_b = df["cat_b"].to_list()

    block = CategoryNuniqueBlock(columns=["cat_a", "cat_b"])
    _ = block.fit(df)

    assert df["cat_a"].to_list() == original_a
    assert df["cat_b"].to_list() == original_b


def test_category_nunique_feature_count_warning():
    """CategoryNuniqueBlock: 生成される特徴量数を確認できること"""
    # 4カラム → 4×3 = 12特徴量
    df = pl.DataFrame({
        "a": ["X", "Y"],
        "b": ["p", "q"],
        "c": ["1", "2"],
        "d": ["i", "j"],
    })

    block = CategoryNuniqueBlock(columns=["a", "b", "c", "d"])
    result = block.fit(df)

    # N × (N-1) = 4 × 3 = 12
    assert result.shape[1] == 12


def test_category_nunique_single_column_error():
    """CategoryNuniqueBlock: 1カラムのみではValueError"""
    df = pl.DataFrame({
        "cat_a": ["X", "Y"],
    })

    with pytest.raises(ValueError, match="2つ以上"):
        block = CategoryNuniqueBlock(columns=["cat_a"])
        block.fit(df)
