"""
データ読み込みモジュール

Polarsを使用してCSVファイルを読み込む

デフォルトではマスター結合済みの enriched ファイルを読み込む:
- data/processed/train_enriched.csv（都道府県名・市区町村名付き）
- data/processed/test_enriched.csv

生データが必要な場合は data.yaml の raw_train_path / raw_test_path を参照
"""

from pathlib import Path
from typing import Any, Dict
import polars as pl


class DataLoader:
    """データ読み込みクラス

    Note:
        enriched ファイルには既に都道府県名・市区町村名が含まれているため、
        add_address_columns=False がデフォルト。
    """

    def __init__(self, config: Dict[str, Any], add_address_columns: bool = False):
        """
        Args:
            config: data.yaml の内容
            add_address_columns: 都道府県・市区町村カラムを追加するかどうか
                                 enriched使用時は False でOK（デフォルト: False）
        """
        self.config = config
        self.data_config = config.get("data", config)
        self.add_address_columns = add_address_columns

        # パスの取得
        self.train_path = Path(self.data_config["train_path"])
        self.test_path = Path(self.data_config["test_path"])
        self.sample_submit_path = Path(self.data_config["sample_submit_path"])

        # AddressParserの初期化（遅延インポート）
        self._address_parser = None
        self._address_mapping_built = False

    def _get_address_parser(self):
        """AddressParserを取得（遅延インポート・初期化）"""
        if self._address_parser is None:
            from preprocessing.address_parser import AddressParser
            self._address_parser = AddressParser()
        return self._address_parser

    def _build_address_mapping_if_needed(self, df: pl.DataFrame) -> None:
        """必要に応じて住所マッピングを構築"""
        if self.add_address_columns and not self._address_mapping_built:
            parser = self._get_address_parser()
            parser.build_city_mapping(df)
            self._address_mapping_built = True

    def _add_address_info(self, df: pl.DataFrame) -> pl.DataFrame:
        """住所情報を追加"""
        if not self.add_address_columns:
            return df

        parser = self._get_address_parser()
        return parser.add_address_columns(df)

    def load_train(self) -> pl.DataFrame:
        """
        訓練データを読み込む

        Returns:
            訓練データのDataFrame（add_address_columns=Trueの場合、prefecture_name, city_nameが追加される）

        Raises:
            FileNotFoundError: ファイルが存在しない場合
        """
        if not self.train_path.exists():
            raise FileNotFoundError(f"訓練データが見つかりません: {self.train_path}")

        # スキーマ推論をデータ全体から行う（混合型のカラムがあるため）
        df = pl.read_csv(self.train_path, infer_schema_length=None)

        # 住所マッピングを構築（初回のみ）
        self._build_address_mapping_if_needed(df)

        # 住所カラムを追加
        df = self._add_address_info(df)

        return df

    def load_test(self) -> pl.DataFrame:
        """
        テストデータを読み込む

        Returns:
            テストデータのDataFrame（add_address_columns=Trueの場合、prefecture_name, city_nameが追加される）

        Raises:
            FileNotFoundError: ファイルが存在しない場合
        """
        if not self.test_path.exists():
            raise FileNotFoundError(f"テストデータが見つかりません: {self.test_path}")

        # スキーマ推論をデータ全体から行う（混合型のカラムがあるため）
        df = pl.read_csv(self.test_path, infer_schema_length=None)

        # 住所マッピングを構築（必要に応じて）
        self._build_address_mapping_if_needed(df)

        # 住所カラムを追加
        df = self._add_address_info(df)

        return df

    def load_sample_submit(self) -> pl.DataFrame:
        """
        サンプル提出ファイルを読み込む

        Returns:
            サンプル提出データのDataFrame

        Raises:
            FileNotFoundError: ファイルが存在しない場合
        """
        if not self.sample_submit_path.exists():
            raise FileNotFoundError(
                f"サンプル提出ファイルが見つかりません: {self.sample_submit_path}"
            )

        # ヘッダーなしのCSV（id, money_room）
        df = pl.read_csv(self.sample_submit_path, has_header=False, new_columns=["id", "money_room"])
        return df

    def get_info(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        データフレームの基本情報を取得

        Args:
            df: データフレーム

        Returns:
            基本情報の辞書
        """
        return {
            "shape": (df.height, df.width),
            "columns": df.columns,
            "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
            "memory_usage_mb": df.estimated_size("mb"),
        }
