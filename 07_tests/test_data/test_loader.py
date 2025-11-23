"""
DataLoaderのテスト（TDD - Red）
"""

import pytest
import polars as pl
from pathlib import Path
from data.loader import DataLoader
from utils.config import load_config


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

    # 列数が149以上であること（データ定義書による + 住所カラム2列）
    assert train.width >= 151

    # 目的変数が含まれていること
    assert "money_room" in train.columns

    # 住所カラムが追加されていること（デフォルト動作）
    assert "prefecture_name" in train.columns
    assert "city_name" in train.columns


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


def test_loader_without_address_columns(data_config):
    """add_address_columns=Falseで住所カラムが追加されないこと"""
    loader = DataLoader(data_config, add_address_columns=False)
    train = loader.load_train()

    # 住所カラムが追加されていないこと
    assert "prefecture_name" not in train.columns
    assert "city_name" not in train.columns

    # 元のカラムは存在すること
    assert "addr1_1" in train.columns
    assert "full_address" in train.columns


def test_address_columns_valid_values(data_config):
    """追加された住所カラムの値が妥当であること"""
    loader = DataLoader(data_config)
    train = loader.load_train()

    # prefecture_nameが47都道府県のいずれかであること
    unique_prefs = train["prefecture_name"].unique().to_list()
    assert all(pref.endswith(('都', '道', '府', '県')) for pref in unique_prefs if pref)

    # city_nameが市区町村のいずれかであること
    unique_cities = train["city_name"].unique().to_list()
    assert all(
        any(city.endswith(suffix) for suffix in ['市', '区', '町', '村'])
        for city in unique_cities if city
    )

    # prefecture_nameとaddr1_1の対応が正しいこと（サンプル確認）
    sample = train.select(['addr1_1', 'prefecture_name']).head(10)
    for row in sample.iter_rows(named=True):
        if row['addr1_1'] == 13:
            assert row['prefecture_name'] == '東京都'
        elif row['addr1_1'] == 27:
            assert row['prefecture_name'] == '大阪府'
