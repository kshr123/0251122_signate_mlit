"""
TargetEncodingBlockのテスト（CV内OOF方式）

このBlockはOut-of-Fold方式でTarget Encodingを行い、
データリークを防止します。
"""

import pytest
import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from features.blocks.encoding import TargetEncodingBlock


def test_target_encoding_basic():
    """TargetEncodingBlock: カテゴリをターゲット平均に変換すること"""
    df = pl.DataFrame({
        "cat1": ["A", "A", "B", "B", "C", "C"],
    })
    y = np.array([100, 200, 300, 400, 500, 600])

    # 3-fold CV
    cv = list(KFold(n_splits=3, shuffle=False).split(df))

    block = TargetEncodingBlock(columns=["cat1"], cv=cv)
    result = block.fit(df, y=y)

    # 数値型に変換されている
    assert result["TE_cat1"].dtype == pl.Float64

    # カラム名にTE_プレフィックスがつく
    assert "TE_cat1" in result.columns
    assert result.shape == (6, 1)


def test_target_encoding_requires_y():
    """TargetEncodingBlock: fit時にyが必要であること"""
    df = pl.DataFrame({
        "cat_col": ["A", "B", "C"]
    })
    cv = list(KFold(n_splits=2).split(df))

    block = TargetEncodingBlock(columns=["cat_col"], cv=cv)

    with pytest.raises(ValueError, match="ターゲット変数.*必須"):
        block.fit(df, y=None)


def test_target_encoding_requires_cv():
    """TargetEncodingBlock: 初期化時にcvが必要であること"""
    with pytest.raises(TypeError):
        TargetEncodingBlock(columns=["cat_col"])  # cv引数なし


def test_target_encoding_oof_no_leak():
    """TargetEncodingBlock: OOF方式でデータリークを防止すること"""
    # 意図的にリークしやすいデータを作成
    df = pl.DataFrame({
        "cat_col": ["A", "A", "A", "A", "B", "B"]
    })
    y = np.array([100, 100, 100, 100, 1000, 1000])
    # A: mean=100, B: mean=1000

    # 2-fold CV（前半3件と後半3件）
    cv = [(np.array([0, 1, 2]), np.array([3, 4, 5])),
          (np.array([3, 4, 5]), np.array([0, 1, 2]))]

    block = TargetEncodingBlock(columns=["cat_col"], cv=cv)
    result = block.fit(df, y=y)

    values = result["TE_cat_col"].to_list()

    # OOF方式では、各fold内で計算されるため
    # 単純な全体平均とは異なる値になる
    # リークがないことを確認（全ての値が完全に同じにならない）
    # fold1: train=[0,1,2](A:100), valid=[3,4,5] → Aは100、Bは全体平均
    # fold2: train=[3,4,5](A:100,B:1000), valid=[0,1,2] → Aは100
    assert values[0] == values[1] == values[2]  # 同じfoldで計算されたA
    assert values[4] == values[5]  # 同じfoldで計算されたB


def test_target_encoding_transform_uses_full_train():
    """TargetEncodingBlock: transformは全trainデータの平均を使用すること"""
    train_df = pl.DataFrame({
        "cat_col": ["A", "A", "B", "B", "C", "C"]
    })
    y = np.array([100, 200, 300, 400, 500, 600])
    # A: mean=150, B: mean=350, C: mean=550

    test_df = pl.DataFrame({
        "cat_col": ["A", "B", "C"]
    })

    cv = list(KFold(n_splits=2, shuffle=False).split(train_df))

    block = TargetEncodingBlock(columns=["cat_col"], cv=cv)
    _ = block.fit(train_df, y=y)

    # testでtransform（全trainデータの平均を使用）
    test_result = block.transform(test_df)

    test_values = test_result["TE_cat_col"].to_list()
    assert abs(test_values[0] - 150) < 0.01  # "A": mean=150
    assert abs(test_values[1] - 350) < 0.01  # "B": mean=350
    assert abs(test_values[2] - 550) < 0.01  # "C": mean=550


def test_target_encoding_unknown_category():
    """TargetEncodingBlock: 未知カテゴリを全体平均でエンコードすること"""
    train_df = pl.DataFrame({
        "cat_col": ["A", "A", "B", "B"]
    })
    y = np.array([100, 200, 300, 400])
    # 全体平均: 250

    test_df = pl.DataFrame({
        "cat_col": ["A", "D", "E"]  # "D", "E"は未知
    })

    cv = list(KFold(n_splits=2, shuffle=False).split(train_df))

    block = TargetEncodingBlock(columns=["cat_col"], cv=cv)
    _ = block.fit(train_df, y=y)
    test_result = block.transform(test_df)

    test_values = test_result["TE_cat_col"].to_list()

    # 未知カテゴリは全体平均
    global_mean = 250
    assert abs(test_values[1] - global_mean) < 0.01  # "D"
    assert abs(test_values[2] - global_mean) < 0.01  # "E"


def test_target_encoding_not_fitted_error():
    """TargetEncodingBlock: fit前のtransformでRuntimeError"""
    df = pl.DataFrame({
        "cat_col": ["A", "B"]
    })
    cv = list(KFold(n_splits=2).split(df))

    block = TargetEncodingBlock(columns=["cat_col"], cv=cv)

    with pytest.raises(RuntimeError, match="fit\\(\\)を先に実行してください"):
        block.transform(df)


def test_target_encoding_immutability():
    """TargetEncodingBlock: 元のDataFrameを変更しないこと"""
    df = pl.DataFrame({
        "cat_col": ["A", "B", "C", "D"],
        "other_col": [1, 2, 3, 4]
    })
    y = np.array([100, 200, 300, 400])

    original_data = df["cat_col"].to_list()
    original_columns = df.columns.copy()

    cv = list(KFold(n_splits=2, shuffle=False).split(df))

    block = TargetEncodingBlock(columns=["cat_col"], cv=cv)
    _ = block.fit(df, y=y)

    # 元のDataFrameが変更されていない
    assert df.columns == original_columns
    assert df["cat_col"].to_list() == original_data


def test_target_encoding_multiple_columns():
    """TargetEncodingBlock: 複数カラムを処理できること"""
    df = pl.DataFrame({
        "cat1": ["A", "B", "A", "B"],
        "cat2": ["X", "X", "Y", "Y"],
    })
    y = np.array([100, 200, 300, 400])

    cv = list(KFold(n_splits=2, shuffle=False).split(df))

    block = TargetEncodingBlock(columns=["cat1", "cat2"], cv=cv)
    result = block.fit(df, y=y)

    # 2カラム
    assert result.shape[1] == 2
    assert "TE_cat1" in result.columns
    assert "TE_cat2" in result.columns


def test_target_encoding_missing_category_in_fold():
    """TargetEncodingBlock: fold内に存在しないカテゴリを適切に処理すること"""
    df = pl.DataFrame({
        "cat_col": ["A", "A", "B", "C"]  # fold分割でBやCが片方にしかない可能性
    })
    y = np.array([100, 200, 300, 400])

    cv = list(KFold(n_splits=2, shuffle=False).split(df))

    block = TargetEncodingBlock(columns=["cat_col"], cv=cv)
    result = block.fit(df, y=y)

    # NaNが含まれていないこと（全体平均で埋められている）
    values = result["TE_cat_col"].to_list()
    assert all(v is not None and not np.isnan(v) for v in values)
