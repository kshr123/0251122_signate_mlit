"""
設定ファイル読み込みユーティリティ

YAML形式の設定ファイルを読み込み、マージする機能を提供
"""

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    2つの辞書を再帰的にマージ（override側が優先）

    Args:
        base: ベースとなる辞書
        override: 上書きする辞書

    Returns:
        マージされた辞書

    Examples:
        >>> base = {"a": 1, "b": {"x": 10, "y": 20}}
        >>> override = {"b": {"y": 30, "z": 40}, "c": 3}
        >>> deep_merge(base, override)
        {"a": 1, "b": {"x": 10, "y": 30, "z": 40}, "c": 3}
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """設定管理クラス"""

    def __init__(self, config_dir: Union[str, Path] = "03_configs"):
        """
        Args:
            config_dir: 設定ファイルディレクトリのパス
        """
        self.config_dir = Path(config_dir)
        self._config: Dict[str, Any] = {}

    def load(self, config_name: str) -> Dict[str, Any]:
        """
        設定ファイルを読み込む

        Args:
            config_name: 設定ファイル名（拡張子なし）
                        例: "base", "data", "model"

        Returns:
            設定内容の辞書

        Raises:
            FileNotFoundError: 設定ファイルが見つからない場合
        """
        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config or {}

    def load_all(self) -> Dict[str, Any]:
        """
        すべての設定ファイルを読み込んでマージ

        Returns:
            マージされた設定の辞書
        """
        config_files = [
            "base",
            "data",
            "preprocessing",
            "features",
            "model",
            "training",
        ]

        merged_config = {}
        for config_name in config_files:
            try:
                config = self.load(config_name)
                merged_config.update(config)
            except FileNotFoundError:
                # 設定ファイルが存在しない場合はスキップ
                continue

        self._config = merged_config
        return merged_config

    def get(self, key: str, default: Any = None) -> Any:
        """
        設定値を取得

        Args:
            key: 設定キー（ドット区切りで階層アクセス可能）
                例: "data.train_path", "model.lightgbm.learning_rate"
            default: デフォルト値

        Returns:
            設定値
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """辞書スタイルでのアクセス"""
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """in演算子のサポート"""
        return key in self._config


def load_config(
    config_name: Union[str, None] = None, config_dir: Union[str, Path] = "03_configs"
) -> Dict[str, Any]:
    """
    設定ファイルを読み込む（関数版）

    Args:
        config_name: 設定ファイル名（拡張子なし）
                    Noneの場合はすべての設定をマージ
        config_dir: 設定ファイルディレクトリのパス

    Returns:
        設定内容の辞書

    Examples:
        >>> # 単一の設定ファイルを読み込み
        >>> data_config = load_config("data")
        >>> print(data_config["data"]["train_path"])

        >>> # すべての設定をマージして読み込み
        >>> config = load_config()
        >>> print(config["data"]["train_path"])
        >>> print(config["model"]["type"])
    """
    cfg = Config(config_dir)

    if config_name is None:
        return cfg.load_all()
    else:
        return cfg.load(config_name)


def get_config_value(key: str, default: Any = None, config_dir: Union[str, Path] = "03_configs") -> Any:
    """
    設定値を取得する（関数版）

    Args:
        key: 設定キー（ドット区切りで階層アクセス可能）
        default: デフォルト値
        config_dir: 設定ファイルディレクトリのパス

    Returns:
        設定値

    Examples:
        >>> # ドット区切りで階層アクセス
        >>> train_path = get_config_value("data.train_path")
        >>> learning_rate = get_config_value("model.lightgbm.learning_rate", default=0.05)
    """
    cfg = Config(config_dir)
    cfg.load_all()
    return cfg.get(key, default)
