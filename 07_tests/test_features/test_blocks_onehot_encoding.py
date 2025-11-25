"""
OneHotEncodingBlockのテスト（category_encoders版）

注意: One-Hot Encodingは低カーディナリティのカテゴリ変数向けです。
高カーディナリティの場合は次元爆発に注意。
min_countで低頻度カテゴリを除外可能。
"""

import pytest
import polars as pl

from features.blocks.encoding import OneHotEncodingBlock


def test_onehot_encoding_basic():
    """OneHotEncodingBlock: カテゴリをダミー変数に変換すること"""
    df = pl.DataFrame({
        "cat1": ["A", "B", "A", "C"],
    })

    block = OneHotEncodingBlock(columns=["cat1"])
    result = block.fit(df)

    # 3カテゴリ → 3カラム（デフォルトでdrop_first=False）
    assert result.shape[0] == 4
    assert result.shape[1] == 3  # A, B, C

    # すべてのカラムが0/1の値を持つ
    for col in result.columns:
        unique_values = set(result[col].to_list())
        assert unique_values.issubset({0, 1, 0.0, 1.0})


def test_onehot_encoding_column_names():
    """OneHotEncodingBlock: カラム名が正しく生成されること"""
    df = pl.DataFrame({
        "color": ["red", "blue", "green"],
    })

    block = OneHotEncodingBlock(columns=["color"])
    result = block.fit(df)

    # カラム名に元のカラム名が含まれている
    for col in result.columns:
        assert "color" in col


def test_onehot_encoding_transform_consistency():
    """OneHotEncodingBlock: train/testで一貫したカラムが生成されること"""
    train_df = pl.DataFrame({
        "cat_col": ["A", "B", "C", "A"]
    })

    test_df = pl.DataFrame({
        "cat_col": ["A", "C"]
    })

    block = OneHotEncodingBlock(columns=["cat_col"])

    # trainでfit
    train_result = block.fit(train_df)

    # testでtransform
    test_result = block.transform(test_df)

    # 同じカラム数
    assert train_result.columns == test_result.columns
    assert train_result.shape[1] == test_result.shape[1]


def test_onehot_encoding_unknown_category():
    """OneHotEncodingBlock: 未知カテゴリを適切に処理すること"""
    train_df = pl.DataFrame({
        "cat_col": ["A", "B", "C"]
    })

    test_df = pl.DataFrame({
        "cat_col": ["A", "D"]  # "D"は未知
    })

    block = OneHotEncodingBlock(columns=["cat_col"])
    _ = block.fit(train_df)
    test_result = block.transform(test_df)

    # trainで学習したカラム数と同じ
    assert test_result.shape[1] == 3  # A, B, C

    # 未知カテゴリ"D"の行はすべて0
    row_d = test_result.row(1)
    assert sum(row_d) == 0  # すべて0


def test_onehot_encoding_missing_value():
    """OneHotEncodingBlock: 欠損値を適切に処理すること"""
    train_df = pl.DataFrame({
        "cat_col": ["A", "B", None, "C"]
    })

    block = OneHotEncodingBlock(columns=["cat_col"])
    result = block.fit(train_df)

    # NaNも1つのカテゴリとして扱われる可能性がある
    # または全て0になる
    assert result.shape[0] == 4


def test_onehot_encoding_not_fitted_error():
    """OneHotEncodingBlock: fit前のtransformでRuntimeError"""
    df = pl.DataFrame({
        "cat_col": ["A", "B"]
    })

    block = OneHotEncodingBlock(columns=["cat_col"])

    with pytest.raises(RuntimeError, match="fit\\(\\)を先に実行してください"):
        block.transform(df)


def test_onehot_encoding_immutability():
    """OneHotEncodingBlock: 元のDataFrameを変更しないこと"""
    df = pl.DataFrame({
        "cat_col": ["A", "B", "C"],
        "other_col": [1, 2, 3]
    })

    original_data = df["cat_col"].to_list()
    original_columns = df.columns.copy()

    block = OneHotEncodingBlock(columns=["cat_col"])
    _ = block.fit(df)

    # 元のDataFrameが変更されていない
    assert df.columns == original_columns
    assert df["cat_col"].to_list() == original_data


def test_onehot_encoding_multiple_columns():
    """OneHotEncodingBlock: 複数カラムを処理できること"""
    df = pl.DataFrame({
        "color": ["red", "blue", "red"],
        "size": ["S", "M", "L"],
    })

    block = OneHotEncodingBlock(columns=["color", "size"])
    result = block.fit(df)

    # color: 2種類, size: 3種類 → 5カラム
    assert result.shape[1] == 5
    assert result.shape[0] == 3


def test_onehot_encoding_use_cat_names():
    """OneHotEncodingBlock: use_cat_names=Trueでカテゴリ名をカラム名に使用"""
    df = pl.DataFrame({
        "color": ["red", "blue", "green"],
    })

    block = OneHotEncodingBlock(columns=["color"], use_cat_names=True)
    result = block.fit(df)

    # カラム名にカテゴリ値が含まれる
    col_names = result.columns
    assert any("red" in col or "blue" in col or "green" in col for col in col_names)


def test_onehot_encoding_min_count():
    """OneHotEncodingBlock: min_countで低頻度カテゴリを除外すること"""
    df = pl.DataFrame({
        # A: 5回, B: 3回, C: 1回
        "cat_col": ["A", "A", "A", "A", "A", "B", "B", "B", "C"]
    })

    # min_count=2: 2回以下のカテゴリ(C)を除外
    block = OneHotEncodingBlock(columns=["cat_col"], min_count=2)
    result = block.fit(df)

    # A, Bのみ → 2カラム
    assert result.shape[1] == 2

    # Cのカラムは存在しない
    col_names = [str(c) for c in result.columns]
    assert not any("C" in col for col in col_names)


def test_onehot_encoding_min_count_transform():
    """OneHotEncodingBlock: min_countで除外されたカテゴリはtransformでも無視されること"""
    train_df = pl.DataFrame({
        # A: 5回, B: 3回, C: 1回
        "cat_col": ["A", "A", "A", "A", "A", "B", "B", "B", "C"]
    })
    test_df = pl.DataFrame({
        "cat_col": ["A", "B", "C", "D"]  # C, Dは除外/未知
    })

    block = OneHotEncodingBlock(columns=["cat_col"], min_count=2)
    _ = block.fit(train_df)
    test_result = block.transform(test_df)

    # A, Bのみ → 2カラム
    assert test_result.shape[1] == 2
    assert test_result.shape[0] == 4

    # C, Dの行はすべて0
    row_c = test_result.row(2)
    row_d = test_result.row(3)
    assert sum(row_c) == 0
    assert sum(row_d) == 0


def test_onehot_encoding_min_count_zero():
    """OneHotEncodingBlock: min_count=0で全カテゴリを含むこと"""
    df = pl.DataFrame({
        "cat_col": ["A", "A", "B", "C"]  # A:2, B:1, C:1
    })

    block = OneHotEncodingBlock(columns=["cat_col"], min_count=0)
    result = block.fit(df)

    # 全3カテゴリ
    assert result.shape[1] == 3
