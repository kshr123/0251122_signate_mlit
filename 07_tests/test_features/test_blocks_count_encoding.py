"""
CountEncodingBlockのテスト（category_encoders版）
"""

import pytest
import polars as pl

from features.blocks.encoding import CountEncodingBlock


def test_count_encoding_basic():
    """CountEncodingBlock: カテゴリを出現頻度に変換すること"""
    df = pl.DataFrame({
        "cat1": ["A", "B", "A", "A", "B", "C"],  # A:3, B:2, C:1
        "cat2": ["X", "X", "Y", "X", "Y", "Y"],  # X:3, Y:3
    })

    block = CountEncodingBlock(columns=["cat1", "cat2"])
    result = block.fit(df)

    # 数値型に変換されている
    assert result["cat1"].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float64]
    assert result["cat2"].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float64]

    # カラム数が正しい
    assert result.columns == ["cat1", "cat2"]
    assert result.shape == (6, 2)

    # 出現頻度が正しい
    cat1_values = result["cat1"].to_list()
    assert cat1_values[0] == 3  # "A" -> 3回
    assert cat1_values[1] == 2  # "B" -> 2回
    assert cat1_values[5] == 1  # "C" -> 1回


def test_count_encoding_transform_consistency():
    """CountEncodingBlock: train/testで一貫したマッピングが適用されること"""
    train_df = pl.DataFrame({
        "cat_col": ["A", "B", "A", "A", "B", "C"]  # A:3, B:2, C:1
    })

    test_df = pl.DataFrame({
        "cat_col": ["A", "C", "B"]
    })

    block = CountEncodingBlock(columns=["cat_col"])

    # trainでfit
    train_result = block.fit(train_df)

    # testでtransform（trainで学習した頻度を使用）
    test_result = block.transform(test_df)

    # trainでの頻度を確認
    train_counts = train_result["cat_col"].to_list()
    assert train_counts[0] == 3  # "A"
    assert train_counts[1] == 2  # "B"
    assert train_counts[5] == 1  # "C"

    # testでも同じ頻度が使われている
    test_counts = test_result["cat_col"].to_list()
    assert test_counts[0] == 3  # "A"
    assert test_counts[1] == 1  # "C"
    assert test_counts[2] == 2  # "B"


def test_count_encoding_unknown_category():
    """CountEncodingBlock: 未知カテゴリを適切に処理すること"""
    train_df = pl.DataFrame({
        "cat_col": ["A", "B", "A", "C"]
    })

    test_df = pl.DataFrame({
        "cat_col": ["A", "D", "E"]  # "D", "E"は未知
    })

    block = CountEncodingBlock(columns=["cat_col"])
    _ = block.fit(train_df)
    test_result = block.transform(test_df)

    test_counts = test_result["cat_col"].to_list()

    # 未知カテゴリは0またはNaN（handle_unknownの設定による）
    # category_encodersのデフォルトでは0になる
    assert test_counts[1] == 0 or test_counts[1] is None  # "D"
    assert test_counts[2] == 0 or test_counts[2] is None  # "E"

    # 既知カテゴリは正の値
    assert test_counts[0] == 2  # "A"


def test_count_encoding_missing_value():
    """CountEncodingBlock: 欠損値を適切に処理すること"""
    train_df = pl.DataFrame({
        "cat_col": ["A", "B", None, "A", None]  # A:2, B:1, None:2
    })

    block = CountEncodingBlock(columns=["cat_col"])
    result = block.fit(train_df)

    counts = result["cat_col"].to_list()

    # NaNも1つのカテゴリとして頻度がカウントされる
    assert counts[0] == 2  # "A"
    assert counts[1] == 1  # "B"
    assert counts[2] == 2  # None


def test_count_encoding_not_fitted_error():
    """CountEncodingBlock: fit前のtransformでRuntimeError"""
    df = pl.DataFrame({
        "cat_col": ["A", "B"]
    })

    block = CountEncodingBlock(columns=["cat_col"])

    with pytest.raises(RuntimeError, match="fit\\(\\)を先に実行してください"):
        block.transform(df)


def test_count_encoding_immutability():
    """CountEncodingBlock: 元のDataFrameを変更しないこと"""
    df = pl.DataFrame({
        "cat_col": ["A", "B", "C"],
        "other_col": [1, 2, 3]
    })

    original_data = df["cat_col"].to_list()
    original_columns = df.columns.copy()

    block = CountEncodingBlock(columns=["cat_col"])
    _ = block.fit(df)

    # 元のDataFrameが変更されていない
    assert df.columns == original_columns
    assert df["cat_col"].to_list() == original_data


def test_count_encoding_normalize():
    """CountEncodingBlock: normalize=Trueで正規化された頻度を返すこと"""
    df = pl.DataFrame({
        "cat_col": ["A", "B", "A", "A", "B", "C"]  # A:3, B:2, C:1, total:6
    })

    block = CountEncodingBlock(columns=["cat_col"], normalize=True)
    result = block.fit(df)

    counts = result["cat_col"].to_list()

    # 正規化された値（頻度 / 総数）
    assert abs(counts[0] - 0.5) < 0.01  # A: 3/6 = 0.5
    assert abs(counts[1] - 1/3) < 0.01  # B: 2/6 = 0.333...
    assert abs(counts[5] - 1/6) < 0.01  # C: 1/6 = 0.166...
