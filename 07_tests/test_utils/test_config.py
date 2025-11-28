"""
設定ファイル読み込みのテスト
"""

import pytest
from pathlib import Path
from utils.config import Config, load_config, get_config_value, deep_merge


# =============================================================================
# deep_merge関数のテスト
# =============================================================================


class TestDeepMerge:
    """deep_merge関数のテスト"""

    def test_deep_merge_basic(self):
        """基本的なマージが動作する"""
        base = {"a": 1, "b": 2}
        override = {"c": 3}

        result = deep_merge(base, override)

        assert result == {"a": 1, "b": 2, "c": 3}
        # 元の辞書は変更されない
        assert base == {"a": 1, "b": 2}

    def test_deep_merge_override(self):
        """上書きが正しく動作する"""
        base = {"a": 1, "b": 2}
        override = {"b": 20}

        result = deep_merge(base, override)

        assert result == {"a": 1, "b": 20}

    def test_deep_merge_nested(self):
        """ネストした辞書のマージが動作する"""
        base = {
            "a": 1,
            "b": {
                "x": 10,
                "y": 20,
            },
        }
        override = {
            "b": {
                "y": 30,
                "z": 40,
            },
            "c": 3,
        }

        result = deep_merge(base, override)

        assert result == {
            "a": 1,
            "b": {
                "x": 10,  # baseから維持
                "y": 30,  # overrideで上書き
                "z": 40,  # overrideから追加
            },
            "c": 3,
        }

    def test_deep_merge_empty(self):
        """空の辞書のマージ"""
        assert deep_merge({}, {"a": 1}) == {"a": 1}
        assert deep_merge({"a": 1}, {}) == {"a": 1}


# =============================================================================
# Configクラスのテスト
# =============================================================================


def test_load_single_config():
    """単一の設定ファイルを読み込めること"""
    config = load_config("base", config_dir="03_configs")

    assert "project" in config
    assert config["project"]["name"] == "real_estate_price_prediction"
    assert config["project"]["task_type"] == "regression"


def test_load_all_configs():
    """すべての設定ファイルをマージして読み込めること"""
    config = load_config(config_dir="03_configs")

    # 各設定ファイルの内容が含まれていること
    assert "project" in config  # base.yaml
    assert "data" in config  # data.yaml
    assert "preprocessing" in config  # preprocessing.yaml
    assert "features" in config  # features.yaml
    assert "model" in config  # model.yaml
    assert "training" in config  # training.yaml


def test_config_get_method():
    """ドット区切りでのアクセスができること"""
    cfg = Config("03_configs")
    cfg.load_all()

    # ドット区切りでアクセス
    assert cfg.get("project.name") == "real_estate_price_prediction"
    assert cfg.get("data.target_column") == "money_room"
    assert cfg.get("model.type") == "lightgbm"

    # デフォルト値
    assert cfg.get("nonexistent.key", default="default_value") == "default_value"


def test_get_config_value():
    """get_config_value関数が正しく動作すること"""
    # 階層アクセス
    train_path = get_config_value("data.train_path", config_dir="03_configs")
    assert train_path == "data/raw/train.csv"

    # デフォルト値
    nonexistent = get_config_value("nonexistent.key", default="default", config_dir="03_configs")
    assert nonexistent == "default"


def test_config_not_found():
    """存在しない設定ファイルの読み込みでエラーが発生すること"""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent", config_dir="03_configs")


def test_config_dict_access():
    """辞書スタイルでのアクセスができること"""
    cfg = Config("03_configs")
    cfg.load_all()

    assert cfg["project"]["name"] == "real_estate_price_prediction"
    assert "data" in cfg
