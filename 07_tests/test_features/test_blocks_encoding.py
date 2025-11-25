"""
LabelEncodingBlockのテスト（category_encoders版）
"""

import pytest
import polars as pl

from features.blocks.encoding import LabelEncodingBlock


def test_label_encoding_basic():
    """LabelEncodingBlock: 基本的な文字列を数値に変換すること"""
    df = pl.DataFrame({
        "cat1": ["A", "B", "A", "C"],
        "cat2": ["X", "Y", "X", "Y"],
    })

    block = LabelEncodingBlock(columns=["cat1", "cat2"])
    result = block.fit(df)

    # 数値型に変換されている
    assert result["cat1"].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]
    assert result["cat2"].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]

    # カラム数が正しい
    assert result.columns == ["cat1", "cat2"]
    assert result.shape == (4, 2)

    # 同じ値は同じコードに変換されている
    cat1_values = result["cat1"].to_list()
    assert cat1_values[0] == cat1_values[2]  # 両方とも"A"


def test_label_encoding_transform_consistency():
    """LabelEncodingBlock: train/testで一貫したマッピングが適用されること"""
    train_df = pl.DataFrame({
        "cat_col": ["A", "B", "C", "A"]
    })

    test_df = pl.DataFrame({
        "cat_col": ["A", "C", "B"]
    })

    block = LabelEncodingBlock(columns=["cat_col"])

    # trainでfit
    train_result = block.fit(train_df)

    # testでtransform
    test_result = block.transform(test_df)

    # trainでの"A"のコードを取得
    train_a_code = train_result["cat_col"].to_list()[0]
    train_b_code = train_result["cat_col"].to_list()[1]
    train_c_code = train_result["cat_col"].to_list()[2]

    # testでも同じコードが使われている
    test_codes = test_result["cat_col"].to_list()
    assert test_codes[0] == train_a_code  # "A"
    assert test_codes[1] == train_c_code  # "C"
    assert test_codes[2] == train_b_code  # "B"


def test_label_encoding_unknown_category():
    """LabelEncodingBlock: 未知カテゴリを-1にエンコードすること"""
    train_df = pl.DataFrame({
        "cat_col": ["A", "B", "C"]
    })

    test_df = pl.DataFrame({
        "cat_col": ["A", "D", "E"]  # "D", "E"は未知
    })

    block = LabelEncodingBlock(columns=["cat_col"])
    _ = block.fit(train_df)
    test_result = block.transform(test_df)

    test_codes = test_result["cat_col"].to_list()

    # 未知カテゴリは-1
    assert test_codes[1] == -1  # "D"
    assert test_codes[2] == -1  # "E"

    # 既知カテゴリは正の値
    assert test_codes[0] >= 1  # "A"


def test_label_encoding_missing_value_in_train():
    """LabelEncodingBlock: trainに欠損値がある場合、1つのカテゴリとしてエンコードすること"""
    train_df = pl.DataFrame({
        "cat_col": ["A", "B", None, "C"]
    })

    block = LabelEncodingBlock(columns=["cat_col"])
    result = block.fit(train_df)

    codes = result["cat_col"].to_list()

    # trainに含まれるNaNは正の整数（1つのカテゴリとして扱われる）
    assert codes[2] >= 1

    # 非欠損値も正の値
    assert codes[0] >= 1
    assert codes[1] >= 1
    assert codes[3] >= 1


def test_label_encoding_missing_value_in_test():
    """LabelEncodingBlock: testで初めてNaNが出現した場合、-2にエンコードすること"""
    train_df = pl.DataFrame({
        "cat_col": ["A", "B", "C"]  # NaNなし
    })

    test_df = pl.DataFrame({
        "cat_col": ["A", None, "B"]  # NaNあり
    })

    block = LabelEncodingBlock(columns=["cat_col"])
    _ = block.fit(train_df)
    test_result = block.transform(test_df)

    test_codes = test_result["cat_col"].to_list()

    # testで初めて出現したNaNは-2
    assert test_codes[1] == -2

    # 既知カテゴリは正の値
    assert test_codes[0] >= 1
    assert test_codes[2] >= 1


def test_label_encoding_not_fitted_error():
    """LabelEncodingBlock: fit前のtransformでRuntimeError"""
    df = pl.DataFrame({
        "cat_col": ["A", "B"]
    })

    block = LabelEncodingBlock(columns=["cat_col"])

    # fit()を実行していない状態でtransform()を呼ぶ
    with pytest.raises(RuntimeError, match="fit\\(\\)を先に実行してください"):
        block.transform(df)


def test_label_encoding_immutability():
    """LabelEncodingBlock: 元のDataFrameを変更しないこと"""
    df = pl.DataFrame({
        "cat_col": ["A", "B", "C"],
        "other_col": [1, 2, 3]
    })

    original_data = df["cat_col"].to_list()
    original_columns = df.columns.copy()

    block = LabelEncodingBlock(columns=["cat_col"])
    _ = block.fit(df)

    # 元のDataFrameが変更されていない
    assert df.columns == original_columns
    assert df["cat_col"].to_list() == original_data


def test_label_encoding_numeric_passthrough():
    """LabelEncodingBlock: 数値型カラムもそのまま処理できること"""
    df = pl.DataFrame({
        "num_col": [1, 2, 3],
        "cat_col": ["A", "B", "C"],
    })

    block = LabelEncodingBlock(columns=["num_col", "cat_col"])
    result = block.fit(df)

    # 両方とも数値型
    assert result["num_col"].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]
    assert result["cat_col"].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]

    # カラム数が正しい
    assert result.shape == (3, 2)
