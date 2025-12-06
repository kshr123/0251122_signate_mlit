"""
汎用ビニングBlock

数値カラムをビン（カテゴリ）に変換する汎用Block。
特定カラムに依存しない設計。
"""

from typing import List, Optional

import numpy as np
import polars as pl

from features.base import BaseBlock


class PercentileBinBlock(BaseBlock):
    """パーセンタイルベースのビニングBlock（stateful）

    任意の数値カラムをパーセンタイル閾値でビニングする汎用Block。
    fit時にtrainデータから閾値を学習し、transform時に同じ閾値を適用。

    Args:
        column: ビニング対象のカラム名
        percentiles: パーセンタイル境界リスト（例: [33.3, 66.7]）
        output_column: 出力カラム名（デフォルト: {column}_bin）

    Examples:
        >>> block = PercentileBinBlock(
        ...     column="house_area",
        ...     percentiles=[33.3, 66.7],
        ...     output_column="area_bin_3"
        ... )
        >>> # fit時: trainデータから33.3%, 66.7%タイル値を学習
        >>> # transform時: 学習した閾値でビン分け（0, 1, 2）
    """

    def __init__(
        self,
        column: str,
        percentiles: List[float],
        output_column: Optional[str] = None,
    ):
        super().__init__()
        self.column = column
        self.percentiles = percentiles
        self.output_column = output_column or f"{column}_bin"
        self._thresholds: Optional[np.ndarray] = None

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        """trainデータからパーセンタイル閾値を学習

        Args:
            input_df: 入力DataFrame（trainデータ）
            y: ターゲット変数（使用しない）

        Returns:
            ビニング後のDataFrame
        """
        values = input_df[self.column].to_numpy()
        # NaN を除外してパーセンタイルを計算
        valid_values = values[~np.isnan(values)]
        self._thresholds = np.percentile(valid_values, self.percentiles)
        self._fitted = True
        return self._transform(input_df)

    def _transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """学習した閾値でビン分け

        Args:
            input_df: 入力DataFrame

        Returns:
            ビニング後のDataFrame（0, 1, 2, ...）
        """
        values = input_df[self.column].to_numpy()
        bins = np.digitize(values, self._thresholds)
        return pl.DataFrame({self.output_column: bins.astype(np.int32)})

    def get_thresholds(self) -> Optional[np.ndarray]:
        """学習した閾値を取得（デバッグ用）"""
        return self._thresholds


class GroupPercentileBinBlock(BaseBlock):
    """グループ別パーセンタイルビニングBlock（stateful）

    グループ（エリアなど）ごとにパーセンタイル閾値を学習し、
    同じグループ内での相対位置でビニングする。

    Args:
        column: ビニング対象のカラム名
        group_column: グループ化に使うカラム名（例: "addr1_2"）
        percentiles: パーセンタイル境界リスト（例: [33.3, 66.7]）
        output_column: 出力カラム名（デフォルト: {column}_bin_in_{group_column}）
        fallback_to_global: グループにデータが少ない場合、全体パーセンタイルにフォールバック
        min_samples: フォールバック判定の最小サンプル数（デフォルト: 10）

    Examples:
        >>> block = GroupPercentileBinBlock(
        ...     column="house_area",
        ...     group_column="addr1_2",
        ...     percentiles=[33.3, 66.7],
        ...     output_column="area_bin_in_addr"
        ... )
        >>> # fit時: 各addr1_2ごとにhouse_areaの33.3%, 66.7%タイル値を学習
        >>> # transform時: 同じグループの閾値でビン分け（0, 1, 2）
    """

    def __init__(
        self,
        column: str,
        group_column: str,
        percentiles: List[float],
        output_column: Optional[str] = None,
        fallback_to_global: bool = True,
        min_samples: int = 10,
    ):
        super().__init__()
        self.column = column
        self.group_column = group_column
        self.percentiles = percentiles
        self.output_column = output_column or f"{column}_bin_in_{group_column}"
        self.fallback_to_global = fallback_to_global
        self.min_samples = min_samples
        self._group_thresholds: dict = {}  # {group_value: np.ndarray}
        self._global_thresholds: Optional[np.ndarray] = None

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        """trainデータからグループ別パーセンタイル閾値を学習

        Args:
            input_df: 入力DataFrame（trainデータ）
            y: ターゲット変数（使用しない）

        Returns:
            ビニング後のDataFrame
        """
        # 全体のパーセンタイルを計算（フォールバック用）
        all_values = input_df[self.column].to_numpy()
        valid_all = all_values[~np.isnan(all_values)]
        self._global_thresholds = np.percentile(valid_all, self.percentiles)

        # グループごとにパーセンタイルを計算
        self._group_thresholds = {}
        for group_val in input_df[self.group_column].unique().to_list():
            if group_val is None:
                continue
            mask = input_df[self.group_column] == group_val
            group_values = input_df.filter(mask)[self.column].to_numpy()
            valid_values = group_values[~np.isnan(group_values)]

            if len(valid_values) >= self.min_samples:
                self._group_thresholds[group_val] = np.percentile(
                    valid_values, self.percentiles
                )
            elif self.fallback_to_global:
                self._group_thresholds[group_val] = self._global_thresholds

        self._fitted = True
        return self._transform(input_df)

    def _transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """学習したグループ別閾値でビン分け

        Args:
            input_df: 入力DataFrame

        Returns:
            ビニング後のDataFrame（0, 1, 2, ...）
        """
        values = input_df[self.column].to_numpy()
        groups = input_df[self.group_column].to_list()
        bins = np.zeros(len(input_df), dtype=np.int32)

        for i, (val, grp) in enumerate(zip(values, groups)):
            if np.isnan(val):
                bins[i] = -1  # NaNは-1
                continue

            # グループの閾値を取得（なければグローバル）
            thresholds = self._group_thresholds.get(grp, self._global_thresholds)
            if thresholds is not None:
                bins[i] = np.digitize([val], thresholds)[0]
            else:
                bins[i] = 0  # フォールバックもない場合

        return pl.DataFrame({self.output_column: bins})

    def get_group_thresholds(self) -> dict:
        """学習したグループ別閾値を取得（デバッグ用）"""
        return self._group_thresholds


class FixedBinBlock(BaseBlock):
    """固定刻みビニングBlock（stateless）

    任意の数値カラムを固定幅でビニングする汎用Block。
    閾値はパラメータで決まるため学習不要。

    Args:
        column: ビニング対象のカラム名
        bin_width: ビン幅（例: 10 = 10年刻み）
        max_bin: 最大ビン値（例: 5 = 50年以上は5にクリップ）
        output_column: 出力カラム名（デフォルト: {column}_bin）

    Examples:
        >>> block = FixedBinBlock(
        ...     column="building_age",
        ...     bin_width=10,
        ...     max_bin=5,
        ...     output_column="age_bin_10y"
        ... )
        >>> # 0-9年 → 0, 10-19年 → 1, ..., 50年以上 → 5
    """

    def __init__(
        self,
        column: str,
        bin_width: int,
        max_bin: int,
        output_column: Optional[str] = None,
    ):
        super().__init__()
        self.column = column
        self.bin_width = bin_width
        self.max_bin = max_bin
        self.output_column = output_column or f"{column}_bin"

    def _transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """固定幅でビン分け

        Args:
            input_df: 入力DataFrame

        Returns:
            ビニング後のDataFrame（0, 1, 2, ..., max_bin）
        """
        values = input_df[self.column].to_numpy()
        # NaN は -1 にマップ（後続処理で適切に扱う）
        bins = np.where(
            np.isnan(values),
            -1,
            np.clip((values / self.bin_width).astype(int), 0, self.max_bin),
        )
        return pl.DataFrame({self.output_column: bins.astype(np.int32)})
