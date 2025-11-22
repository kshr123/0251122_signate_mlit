"""
DataLoaderのテスト（TDD - Red）
"""

import pytest
import polars as pl
from pathlib import Path
from src.data.loader import DataLoader
from src.utils.config import load_config


@pytest.fixture
def data_config():
    """データ設定のフィクスチャ"""
    return load_config("data", config_dir="03_configs")


def test_load_train_returns_dataframe(data_config):
    """訓練データを読み込んでDataFrameが返ること"""
    loader = DataLoader(data_config)
    train = loader.load_train()

    # Polarsデータフレームであること
    assert isinstance(train, pl.DataFrame)

    # 行数が正の値であること
    assert train.height > 0

    # 列数が149以上であること（データ定義書による）
    assert train.width >= 149

    # 目的変数が含まれていること
    assert "money_room" in train.columns


def test_load_test_returns_dataframe(data_config):
    """テストデータを読み込んでDataFrameが返ること"""
    loader = DataLoader(data_config)
    test = loader.load_test()

    # Polarsデータフレームであること
    assert isinstance(test, pl.DataFrame)

    # 行数が正の値であること
    assert test.height > 0

    # idカラムが含まれていること（テストデータのみ）
    assert "id" in test.columns

    # 目的変数が含まれていないこと
    assert "money_room" not in test.columns


def test_load_sample_submit_returns_dataframe(data_config):
    """サンプル提出ファイルを読み込んでDataFrameが返ること"""
    loader = DataLoader(data_config)
    sample = loader.load_sample_submit()

    # Polarsデータフレームであること
    assert isinstance(sample, pl.DataFrame)

    # 必要なカラムが含まれていること
    assert "id" in sample.columns
    assert "money_room" in sample.columns

    # カラム数が2つであること
    assert sample.width == 2


def test_load_train_file_not_found():
    """存在しないファイルの読み込みでエラーが発生すること"""
    config = {
        "data": {
            "train_path": "nonexistent/path/train.csv",
            "test_path": "data/raw/test.csv",
            "sample_submit_path": "data/raw/sample_submit.csv",
        }
    }

    loader = DataLoader(config)

    with pytest.raises(FileNotFoundError):
        loader.load_train()


def test_train_test_have_same_columns_except_target_and_id(data_config):
    """訓練データとテストデータが目的変数とID以外は同じカラム構成であること"""
    loader = DataLoader(data_config)
    train = loader.load_train()
    test = loader.load_test()

    # 訓練データから目的変数を除いたカラム
    train_cols = set(train.columns) - {"money_room"}

    # テストデータからIDを除いたカラム
    test_cols = set(test.columns) - {"id"}

    # 両者が一致すること
    assert train_cols == test_cols
