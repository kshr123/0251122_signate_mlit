"""
GroupByAggBlockのテスト

カテゴリごとに数値カラムの統計量を計算し、特徴量として追加するBlock。
"""

import pytest
import polars as pl
import numpy as np

from features.blocks.aggregation import GroupByAggBlock


def test_groupby_agg_basic():
    """GroupByAggBlock: カテゴリごとの統計量を計算すること"""
    df = pl.DataFrame({
        "cat": ["A", "A", "B", "B"],
        "value": [100, 200, 300, 400],
    })

    block = GroupByAggBlock(
        cat_column="cat",
        num_columns=["value"],
        aggs=["mean"]
    )
    result = block.fit(df)

    # A: mean=150, B: mean=350
    assert "groupby_cat_value_mean" in result.columns

    values = result["groupby_cat_value_mean"].to_list()
    assert values[0] == 150  # A
    assert values[1] == 150  # A
    assert values[2] == 350  # B
    assert values[3] == 350  # B


def test_groupby_agg_multiple_aggs():
    """GroupByAggBlock: 複数の集計方法を適用できること"""
    df = pl.DataFrame({
        "cat": ["A", "A", "B", "B"],
        "value": [100, 200, 300, 400],
    })

    block = GroupByAggBlock(
        cat_column="cat",
        num_columns=["value"],
        aggs=["mean", "std", "min", "max"]
    )
    result = block.fit(df)

    # 4つの集計 × 1カラム = 4カラム（ratio/diffなし）
    assert "groupby_cat_value_mean" in result.columns
    assert "groupby_cat_value_std" in result.columns
    assert "groupby_cat_value_min" in result.columns
    assert "groupby_cat_value_max" in result.columns


def test_groupby_agg_multiple_num_columns():
    """GroupByAggBlock: 複数の数値カラムを集計できること"""
    df = pl.DataFrame({
        "cat": ["A", "A", "B", "B"],
        "value1": [100, 200, 300, 400],
        "value2": [10, 20, 30, 40],
    })

    block = GroupByAggBlock(
        cat_column="cat",
        num_columns=["value1", "value2"],
        aggs=["mean"]
    )
    result = block.fit(df)

    assert "groupby_cat_value1_mean" in result.columns
    assert "groupby_cat_value2_mean" in result.columns


def test_groupby_agg_ratio_diff():
    """GroupByAggBlock: ratio/diffを計算すること"""
    df = pl.DataFrame({
        "cat": ["A", "A", "B", "B"],
        "value": [100, 200, 300, 400],
    })

    block = GroupByAggBlock(
        cat_column="cat",
        num_columns=["value"],
        aggs=["mean"],
        add_ratio_diff=True
    )
    result = block.fit(df)

    # mean + ratio + diff = 3カラム
    assert "groupby_cat_value_mean" in result.columns
    assert "groupby_cat_value_mean_ratio" in result.columns
    assert "groupby_cat_value_mean_diff" in result.columns

    # A: mean=150, value=[100, 200]
    # ratio = value / mean = [100/150, 200/150]
    # diff = value - mean = [100-150, 200-150]
    ratios = result["groupby_cat_value_mean_ratio"].to_list()
    diffs = result["groupby_cat_value_mean_diff"].to_list()

    assert abs(ratios[0] - 100/150) < 0.01
    assert abs(ratios[1] - 200/150) < 0.01
    assert diffs[0] == -50  # 100 - 150
    assert diffs[1] == 50   # 200 - 150


def test_groupby_agg_transform_consistency():
    """GroupByAggBlock: train/testで一貫したマッピングが適用されること"""
    train_df = pl.DataFrame({
        "cat": ["A", "A", "B", "B"],
        "value": [100, 200, 300, 400],
    })
    test_df = pl.DataFrame({
        "cat": ["A", "B"],
        "value": [150, 350],
    })

    block = GroupByAggBlock(
        cat_column="cat",
        num_columns=["value"],
        aggs=["mean"]
    )
    _ = block.fit(train_df)
    test_result = block.transform(test_df)

    # trainで計算した平均がtestにも適用される
    values = test_result["groupby_cat_value_mean"].to_list()
    assert values[0] == 150  # A: mean=150
    assert values[1] == 350  # B: mean=350


def test_groupby_agg_unknown_category():
    """GroupByAggBlock: 未知カテゴリを全体平均で埋めること"""
    train_df = pl.DataFrame({
        "cat": ["A", "A", "B", "B"],
        "value": [100, 200, 300, 400],
    })
    test_df = pl.DataFrame({
        "cat": ["A", "C"],  # Cは未知
        "value": [150, 500],
    })

    block = GroupByAggBlock(
        cat_column="cat",
        num_columns=["value"],
        aggs=["mean"]
    )
    _ = block.fit(train_df)
    test_result = block.transform(test_df)

    # 全体平均 = (100+200+300+400)/4 = 250
    values = test_result["groupby_cat_value_mean"].to_list()
    assert values[0] == 150  # A: mean=150
    assert values[1] == 250  # C: 全体平均=250


def test_groupby_agg_not_fitted_error():
    """GroupByAggBlock: fit前のtransformでRuntimeError"""
    df = pl.DataFrame({
        "cat": ["A", "B"],
        "value": [100, 200],
    })

    block = GroupByAggBlock(
        cat_column="cat",
        num_columns=["value"],
        aggs=["mean"]
    )

    with pytest.raises(RuntimeError, match="fit\\(\\)を先に実行してください"):
        block.transform(df)


def test_groupby_agg_immutability():
    """GroupByAggBlock: 元のDataFrameを変更しないこと"""
    df = pl.DataFrame({
        "cat": ["A", "B", "A"],
        "value": [100, 200, 300],
    })

    original_cat = df["cat"].to_list()
    original_value = df["value"].to_list()

    block = GroupByAggBlock(
        cat_column="cat",
        num_columns=["value"],
        aggs=["mean"]
    )
    _ = block.fit(df)

    assert df["cat"].to_list() == original_cat
    assert df["value"].to_list() == original_value


def test_groupby_agg_prefix():
    """GroupByAggBlock: カラム名にプレフィックスがつくこと"""
    df = pl.DataFrame({
        "prefecture": ["東京", "東京", "大阪"],
        "price": [1000, 2000, 3000],
    })

    block = GroupByAggBlock(
        cat_column="prefecture",
        num_columns=["price"],
        aggs=["mean"]
    )
    result = block.fit(df)

    # groupby_{cat_column}_{num_column}_{agg}
    assert "groupby_prefecture_price_mean" in result.columns
