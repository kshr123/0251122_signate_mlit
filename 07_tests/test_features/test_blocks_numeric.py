"""
NumericBlockのテスト
"""

import pytest
import polars as pl

from features.blocks.numeric import NumericBlock


def test_numeric_block_normal():
    """NumericBlock: 正常系 - 指定したカラムをそのまま返すこと"""
    df = pl.DataFrame({
        "num1": [1, 2, 3],
        "num2": [1.5, 2.5, 3.5],
        "cat": ["A", "B", "C"]
    })

    block = NumericBlock(columns=["num1", "num2"])
    result = block.fit(df)

    # 指定したカラムのみ返される
    assert result.columns == ["num1", "num2"]
    assert result.shape == (3, 2)

    # 値が変更されていない
    assert result["num1"].to_list() == [1, 2, 3]
    assert result["num2"].to_list() == [1.5, 2.5, 3.5]


def test_numeric_block_transform():
    """NumericBlock: fit後のtransformが正常に動作すること"""
    train_df = pl.DataFrame({
        "num1": [1, 2, 3],
        "num2": [1.5, 2.5, 3.5],
        "cat": ["A", "B", "C"]
    })

    test_df = pl.DataFrame({
        "num1": [10, 20],
        "num2": [10.5, 20.5],
        "cat": ["X", "Y"]
    })

    block = NumericBlock(columns=["num1", "num2"])

    # trainでfit
    train_result = block.fit(train_df)

    # testでtransform
    test_result = block.transform(test_df)

    assert test_result.columns == ["num1", "num2"]
    assert test_result.shape == (2, 2)
    assert test_result["num1"].to_list() == [10, 20]


def test_numeric_block_not_fitted_error():
    """NumericBlock: fit前のtransformでRuntimeError"""
    df = pl.DataFrame({
        "num1": [1, 2, 3],
        "cat": ["A", "B", "C"]
    })

    block = NumericBlock(columns=["num1"])

    # fit()を実行していない状態でtransform()を呼ぶ
    with pytest.raises(RuntimeError, match="fit\\(\\)を先に実行してください"):
        block.transform(df)


def test_numeric_block_immutability():
    """NumericBlock: 元のDataFrameを変更しないこと"""
    df = pl.DataFrame({
        "num1": [1, 2, 3],
        "num2": [4, 5, 6],
        "cat": ["A", "B", "C"]
    })

    original_columns = df.columns.copy()
    original_shape = df.shape

    block = NumericBlock(columns=["num1"])
    _ = block.fit(df)

    # 元のDataFrameが変更されていない
    assert df.columns == original_columns
    assert df.shape == original_shape
    assert "cat" in df.columns
