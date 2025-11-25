"""
TargetYmBlockのテスト
"""

import pytest
import polars as pl

from features.blocks.temporal import TargetYmBlock


def test_target_ym_block_normal():
    """TargetYmBlock: 正常系 - YYYYMMを年・月に分解すること"""
    df = pl.DataFrame({
        "target_ym": [202301, 202312, 202404]
    })

    block = TargetYmBlock()
    result = block.fit(df)

    # 年・月に分解されている
    assert "target_year" in result.columns
    assert "target_month" in result.columns
    assert result.shape == (3, 2)

    # 値が正しい
    assert result["target_year"].to_list() == [2023, 2023, 2024]
    assert result["target_month"].to_list() == [1, 12, 4]


def test_target_ym_block_custom_column():
    """TargetYmBlock: カスタムカラム名を指定できること"""
    df = pl.DataFrame({
        "custom_ym": [202301, 202312]
    })

    block = TargetYmBlock(source_col="custom_ym")
    result = block.fit(df)

    assert result["target_year"].to_list() == [2023, 2023]
    assert result["target_month"].to_list() == [1, 12]


def test_target_ym_block_transform():
    """TargetYmBlock: fit後のtransformが正常に動作すること"""
    train_df = pl.DataFrame({
        "target_ym": [202301, 202312]
    })

    test_df = pl.DataFrame({
        "target_ym": [202401, 202502]
    })

    block = TargetYmBlock()

    # trainでfit
    train_result = block.fit(train_df)

    # testでtransform
    test_result = block.transform(test_df)

    assert test_result["target_year"].to_list() == [2024, 2025]
    assert test_result["target_month"].to_list() == [1, 2]


def test_target_ym_block_not_fitted_error():
    """TargetYmBlock: fit前のtransformでRuntimeError"""
    df = pl.DataFrame({
        "target_ym": [202301]
    })

    block = TargetYmBlock()

    # fit()を実行していない状態でtransform()を呼ぶ
    with pytest.raises(RuntimeError, match="fit\\(\\)を先に実行してください"):
        block.transform(df)


def test_target_ym_block_immutability():
    """TargetYmBlock: 元のDataFrameを変更しないこと"""
    df = pl.DataFrame({
        "target_ym": [202301, 202312],
        "other_col": [1, 2]
    })

    original_data = df["target_ym"].to_list()
    original_columns = df.columns.copy()

    block = TargetYmBlock()
    _ = block.fit(df)

    # 元のDataFrameが変更されていない
    assert df["target_ym"].to_list() == original_data
    assert df.columns == original_columns
