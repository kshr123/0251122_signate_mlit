"""住所情報のパース・抽出モジュール

このモジュールは以下の機能を提供します：
- full_addressから都道府県・市区町村名を抽出
- addr1_1（都道府県コード）から都道府県名を取得
- addr1_2（市区町村コード）から市区町村名を取得
"""

import re
import polars as pl
from typing import Dict, Tuple


# 都道府県マスタ（JISコード準拠）
PREFECTURE_MASTER = {
    1: '北海道', 2: '青森県', 3: '岩手県', 4: '宮城県', 5: '秋田県',
    6: '山形県', 7: '福島県', 8: '茨城県', 9: '栃木県', 10: '群馬県',
    11: '埼玉県', 12: '千葉県', 13: '東京都', 14: '神奈川県', 15: '新潟県',
    16: '富山県', 17: '石川県', 18: '福井県', 19: '山梨県', 20: '長野県',
    21: '岐阜県', 22: '静岡県', 23: '愛知県', 24: '三重県', 25: '滋賀県',
    26: '京都府', 27: '大阪府', 28: '兵庫県', 29: '奈良県', 30: '和歌山県',
    31: '鳥取県', 32: '島根県', 33: '岡山県', 34: '広島県', 35: '山口県',
    36: '徳島県', 37: '香川県', 38: '愛媛県', 39: '高知県', 40: '福岡県',
    41: '佐賀県', 42: '長崎県', 43: '熊本県', 44: '大分県', 45: '宮崎県',
    46: '鹿児島県', 47: '沖縄県'
}


class AddressParser:
    """住所情報パーサー

    full_addressから都道府県・市区町村を抽出し、
    コードとの対応付けを行う。
    """

    def __init__(self):
        """初期化"""
        self.pref_master = PREFECTURE_MASTER
        self._city_mapping: Dict[Tuple[int, int], str] = {}

    def build_city_mapping(self, df: pl.DataFrame) -> None:
        """市区町村マッピングを構築

        DataFrameから (pref_code, city_code) -> city_name のマッピングを作成。

        Args:
            df: 'addr1_1', 'addr1_2', 'full_address' カラムを持つDataFrame
        """
        if not all(col in df.columns for col in ['addr1_1', 'addr1_2', 'full_address']):
            raise ValueError("Required columns: addr1_1, addr1_2, full_address")

        # ユニークな組み合わせを取得
        unique_cities = df.select(['addr1_1', 'addr1_2', 'full_address']).unique(
            subset=['addr1_1', 'addr1_2']
        )

        # 市区町村名を抽出
        for row in unique_cities.iter_rows(named=True):
            pref_code = row['addr1_1']
            city_code = row['addr1_2']
            full_addr = row['full_address']

            if pref_code not in self.pref_master or not full_addr:
                continue

            pref_name = self.pref_master[pref_code]

            # 都道府県名以降を抽出
            if pref_name in full_addr:
                after_pref = full_addr.replace(pref_name, '', 1)
                # 市区町村名を抽出（市・区・町・村で終わるパターン）
                city_match = re.match(r'^([^0-9]+?[市区町村])', after_pref)
                if city_match:
                    city_name = city_match.group(1)
                    key = (pref_code, city_code)
                    self._city_mapping[key] = city_name

    def add_address_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """住所カラムを追加

        以下のカラムを追加：
        - prefecture_name: 都道府県名
        - city_name: 市区町村名

        Args:
            df: 元のDataFrame（addr1_1, addr1_2を含む）

        Returns:
            住所カラムが追加されたDataFrame
        """
        if 'addr1_1' not in df.columns:
            raise ValueError("Column 'addr1_1' is required")

        # 都道府県名を追加
        df = df.with_columns(
            pl.col('addr1_1').replace_strict(
                old=list(self.pref_master.keys()),
                new=list(self.pref_master.values()),
                default=None
            ).alias('prefecture_name')
        )

        # 市区町村名を追加（マッピングが構築されている場合）
        if self._city_mapping and 'addr1_2' in df.columns:
            # Polarsで効率的にマッピング適用
            # (pref_code, city_code) -> city_name
            city_map_df = pl.DataFrame([
                {'addr1_1': k[0], 'addr1_2': k[1], 'city_name': v}
                for k, v in self._city_mapping.items()
            ])

            df = df.join(
                city_map_df,
                on=['addr1_1', 'addr1_2'],
                how='left'
            )

        return df

    def get_city_name(self, pref_code: int, city_code: int) -> str | None:
        """市区町村名を取得

        Args:
            pref_code: 都道府県コード
            city_code: 市区町村コード

        Returns:
            市区町村名（見つからない場合はNone）
        """
        return self._city_mapping.get((pref_code, city_code))

    @property
    def city_mapping_size(self) -> int:
        """市区町村マッピングのサイズを取得"""
        return len(self._city_mapping)
