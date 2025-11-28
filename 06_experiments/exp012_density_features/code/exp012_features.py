"""
exp012固有の特徴量Block

密度特徴量（郵便番号内物件数のパーセンタイルビン分け）

設計方針（04_feature_engineering_rules.md準拠）:
- 04_srcの共通Blockを最大限活用
- 共通Blockで対応できない場合のみ実験固有Blockを作成
"""

from typing import List, Optional

import numpy as np
import polars as pl

from features.base import BaseBlock
from features.blocks.encoding import CountEncodingBlock


class DensityBinBlock(BaseBlock):
    """
    パーセンタイルベースの密度ビン分けBlock

    郵便番号内物件数をパーセンタイルで4カテゴリに分類：
    - very_low: 0-10%ile (MAPE 15.41%)
    - low: 10-30%ile (MAPE 12.84%)
    - medium: 30-70%ile (MAPE 11.91%)
    - high: 70-100%ile (MAPE 10.81%)

    重要: train時にパーセンタイル境界を学習し、testに適用
    """

    def __init__(
        self,
        column: str,
        percentile_boundaries: List[float] = None,
        output_suffix: str = "_density_bin",
    ):
        """
        Args:
            column: カウントが格納されたカラム名（例: 'post_full_count'）
            percentile_boundaries: パーセンタイル境界リスト
                デフォルト: [0, 10, 30, 70, 100]
            output_suffix: 出力カラムのサフィックス
        """
        super().__init__()
        self._column = column
        self._percentile_boundaries = percentile_boundaries or [0, 10, 30, 70, 100]
        self._output_suffix = output_suffix

        # fit時に計算される閾値
        self._thresholds: Optional[np.ndarray] = None

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        """trainデータからパーセンタイル閾値を計算し、ビン分けを実行"""
        counts = input_df[self._column].to_numpy()

        # パーセンタイル境界から実際の閾値を計算（中間値のみ）
        # 例: [0, 10, 30, 70, 100] → 10, 30, 70パーセンタイル値
        inner_percentiles = self._percentile_boundaries[1:-1]
        self._thresholds = np.percentile(counts, inner_percentiles)

        self._fitted = True
        return self._transform(input_df)

    def _transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """学習済み閾値でビン分けを実行"""
        counts = input_df[self._column].to_numpy()

        # np.digitize: 閾値より小さい→0, 最初の閾値以上→1, ...
        bins = np.digitize(counts, self._thresholds)

        output_col = f"{self._column}{self._output_suffix}"
        return pl.DataFrame({output_col: bins.astype(np.int32)})

    def get_thresholds(self) -> Optional[np.ndarray]:
        """計算された閾値を取得（デバッグ用）"""
        return self._thresholds


class AreaAgeCategoryBlock(BaseBlock):
    """
    面積×築年数カテゴリBlock（4カテゴリ版）

    カテゴリ判定順序: cat3 → cat2 → cat1 → cat0（上位条件を優先）

    | カテゴリ | 条件 | 期待MAPE |
    |----------|------|----------|
    | cat0 | それ以外 | 11.8% |
    | cat1 | 100㎡+ AND 35年+ | 15.4% |
    | cat2 | 150㎡+ AND 45年+ | 18.2% |
    | cat3 | 200㎡+ AND 45年+ | 20.5% |

    Note: statelessなBlockなのでfit()はtransform()を呼ぶだけ
    """

    def __init__(
        self,
        area_column: str = "house_area",
        age_column: str = "building_age",
        cat1_area: float = 100.0,
        cat1_age: int = 35,
        cat2_area: float = 150.0,
        cat2_age: int = 45,
        cat3_area: float = 200.0,
        cat3_age: int = 45,
        output_column: str = "area_age_category",
    ):
        """
        Args:
            area_column: 面積カラム名
            age_column: 築年数カラム名
            cat1_area, cat1_age: カテゴリ1の閾値
            cat2_area, cat2_age: カテゴリ2の閾値
            cat3_area, cat3_age: カテゴリ3の閾値
            output_column: 出力カラム名
        """
        super().__init__()
        self._area_column = area_column
        self._age_column = age_column
        self._cat1_area = cat1_area
        self._cat1_age = cat1_age
        self._cat2_area = cat2_area
        self._cat2_age = cat2_age
        self._cat3_area = cat3_area
        self._cat3_age = cat3_age
        self._output_column = output_column

    def _transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """カテゴリを計算（優先順: cat3 > cat2 > cat1 > cat0）"""
        area = input_df[self._area_column].to_numpy()
        age = input_df[self._age_column].to_numpy()

        # デフォルトはcat0
        category = np.zeros(len(input_df), dtype=np.int32)

        # cat1: 100㎡+ AND 35年+
        mask_cat1 = (area >= self._cat1_area) & (age >= self._cat1_age)
        category[mask_cat1] = 1

        # cat2: 150㎡+ AND 45年+（cat1を上書き）
        mask_cat2 = (area >= self._cat2_area) & (age >= self._cat2_age)
        category[mask_cat2] = 2

        # cat3: 200㎡+ AND 45年+（cat2を上書き）
        mask_cat3 = (area >= self._cat3_area) & (age >= self._cat3_age)
        category[mask_cat3] = 3

        return pl.DataFrame({self._output_column: category})


class PostalCountBlock(CountEncodingBlock):
    """
    郵便番号内物件数をカウントするBlock

    04_srcのCountEncodingBlockを継承し、出力カラム名をカスタマイズ。
    """

    def __init__(
        self,
        column: str = "post_full",
        output_column: str = "post_full_count",
    ):
        super().__init__(columns=[column])
        self._input_column = column
        self._output_column = output_column

    def _transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """カウント結果をカスタム名でリネーム"""
        result = super()._transform(input_df)
        return result.rename({self._input_column: self._output_column})
