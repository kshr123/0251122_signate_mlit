"""
集計系Block

カテゴリごとに数値カラムの統計量を計算し、特徴量として追加する。
カテゴリごとに別カテゴリのnuniqueを計算する。
"""

import polars as pl
import numpy as np
import pandas as pd
from features.base import BaseBlock


class GroupByAggBlock(BaseBlock):
    """カテゴリごとに数値カラムの統計量を計算するBlock

    指定したカテゴリカラムでグループ化し、数値カラムの統計量（mean, std, min, max等）を
    計算して特徴量として追加します。

    特徴:
    - trainデータで計算した統計量をtestデータにも適用
    - 未知カテゴリ: 全体平均で埋める
    - オプションでratio（比率）とdiff（差分）も計算可能
    - カラム名: groupby_{cat_column}_{num_column}_{agg}

    Args:
        cat_column: グループ化に使用するカテゴリカラム名
        num_columns: 集計対象の数値カラムリスト
        aggs: 集計方法のリスト（"mean", "std", "min", "max", "median", "sum"等）
        add_ratio_diff: Trueの場合、mean/min/maxに対してratio/diffも計算

    Examples:
        >>> df = pl.DataFrame({
        ...     "prefecture": ["東京", "東京", "大阪", "大阪"],
        ...     "price": [1000, 2000, 1500, 2500],
        ... })
        >>> block = GroupByAggBlock(
        ...     cat_column="prefecture",
        ...     num_columns=["price"],
        ...     aggs=["mean", "std"]
        ... )
        >>> result = block.fit(df)
        >>> # result.columns = ["groupby_prefecture_price_mean", "groupby_prefecture_price_std"]
    """

    def __init__(
        self,
        cat_column: str,
        num_columns: list[str],
        aggs: list[str],
        add_ratio_diff: bool = False
    ):
        """初期化

        Args:
            cat_column: グループ化に使用するカテゴリカラム名
            num_columns: 集計対象の数値カラムリスト
            aggs: 集計方法のリスト
            add_ratio_diff: ratio/diffを追加するか
        """
        super().__init__()
        self.cat_column = cat_column
        self.num_columns = num_columns
        self.aggs = aggs
        self.add_ratio_diff = add_ratio_diff
        self.agg_dict_ = {}  # {(num_col, agg): {cat_value: stat_value}}
        self.global_stats_ = {}  # {(num_col, agg): global_value}

    def fit(self, input_df: pl.DataFrame, y=None) -> pl.DataFrame:
        """trainデータで統計量を計算

        Args:
            input_df: 入力DataFrame（trainデータ）
            y: ターゲット変数（使用しない）

        Returns:
            集計特徴量のDataFrame
        """
        # Polars → pandas変換
        pdf = input_df.to_pandas()

        # グループごとの統計量を計算
        group = pdf.groupby(self.cat_column)[self.num_columns]
        agg_result = group.agg(self.aggs)

        # MultiIndexのカラムを辞書に変換
        # agg_result.columns = MultiIndex([('value', 'mean'), ('value', 'std'), ...])
        self.agg_dict_ = agg_result.to_dict()

        # 全体の統計量を計算（未知カテゴリ用）
        for num_col in self.num_columns:
            for agg in self.aggs:
                if agg == "mean":
                    self.global_stats_[(num_col, agg)] = pdf[num_col].mean()
                elif agg == "std":
                    self.global_stats_[(num_col, agg)] = pdf[num_col].std()
                elif agg == "min":
                    self.global_stats_[(num_col, agg)] = pdf[num_col].min()
                elif agg == "max":
                    self.global_stats_[(num_col, agg)] = pdf[num_col].max()
                elif agg == "median":
                    self.global_stats_[(num_col, agg)] = pdf[num_col].median()
                elif agg == "sum":
                    self.global_stats_[(num_col, agg)] = pdf[num_col].sum()
                else:
                    # その他の集計方法
                    self.global_stats_[(num_col, agg)] = pdf[num_col].agg(agg)

        self._fitted = True
        return self.transform(input_df)

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """学習した統計量をマッピング

        Args:
            input_df: 入力DataFrame（train/testどちらでも可）

        Returns:
            集計特徴量のDataFrame

        Raises:
            RuntimeError: fit()を先に実行していない場合
        """
        if not self._fitted:
            raise RuntimeError("GroupByAggBlock: fit()を先に実行してください")

        # Polars → pandas変換
        pdf = input_df.to_pandas()

        out_df = pd.DataFrame()

        for key, mapping in self.agg_dict_.items():
            num_col, agg = key
            col_name = f"{num_col}_{agg}"

            # カテゴリ値でマッピング、未知カテゴリは全体平均で埋める
            global_val = self.global_stats_[(num_col, agg)]
            out_df[col_name] = pdf[self.cat_column].map(mapping).fillna(global_val)

            # ratio/diffを追加（mean, min, maxの場合のみ）
            if self.add_ratio_diff and agg in ["mean", "min", "max"]:
                # ratio = 元の値 / 統計値
                out_df[f"{col_name}_ratio"] = pdf[num_col] / out_df[col_name]
                # diff = 元の値 - 統計値
                out_df[f"{col_name}_diff"] = pdf[num_col] - out_df[col_name]

        # プレフィックスを追加
        out_df = out_df.add_prefix(f"groupby_{self.cat_column}_")

        # pandas → Polars変換して返す
        return pl.from_pandas(out_df)


class CategoryNuniqueBlock(BaseBlock):
    """カテゴリAでグループ化したときのカテゴリBのnuniqueを計算するBlock

    複数のカテゴリカラムを指定し、各カラムでグループ化したときの
    他カラムのユニーク数（nunique）を計算して特徴量として追加します。

    注意:
    - N個のカテゴリカラムで N×(N-1) 個の特徴量が生成される
    - カラム数が多い場合は計算量・メモリに注意

    特徴:
    - trainデータで計算したnuniqueをtestデータにも適用
    - 未知カテゴリ: 0で埋める
    - カラム名: nunique_{groupby_col}_groupby_{target_col}

    Args:
        columns: 対象のカテゴリカラムリスト（2つ以上必要）

    Examples:
        >>> df = pl.DataFrame({
        ...     "prefecture": ["東京", "東京", "大阪"],
        ...     "type": ["マンション", "戸建", "戸建"],
        ... })
        >>> block = CategoryNuniqueBlock(columns=["prefecture", "type"])
        >>> result = block.fit(df)
        >>> # result.columns = [
        >>> #     "nunique_prefecture_groupby_type",  # 都道府県ごとの種類のnunique
        >>> #     "nunique_type_groupby_prefecture",  # 種類ごとの都道府県のnunique
        >>> # ]
    """

    def __init__(self, columns: list[str]):
        """初期化

        Args:
            columns: 対象のカテゴリカラムリスト（2つ以上必要）
        """
        super().__init__()
        self.columns = columns
        self.mapping_df_ = {}  # {groupby_col: {target_col: {cat_value: nunique}}}

    def fit(self, input_df: pl.DataFrame, y=None) -> pl.DataFrame:
        """trainデータでnuniqueを計算

        Args:
            input_df: 入力DataFrame（trainデータ）
            y: ターゲット変数（使用しない）

        Returns:
            nunique特徴量のDataFrame

        Raises:
            ValueError: columnsが2つ未満の場合
        """
        if len(self.columns) < 2:
            raise ValueError("CategoryNuniqueBlock: columnsは2つ以上必要です")

        # Polars → pandas変換
        pdf = input_df.to_pandas()

        self.mapping_df_ = {}

        for groupby_col in self.columns:
            # groupby_col以外のカラムが対象
            target_cols = [c for c in self.columns if c != groupby_col]

            # グループごとのnuniqueを計算
            _df = pdf[target_cols].groupby(pdf[groupby_col]).nunique()

            # {target_col: {cat_value: nunique}} の形式で保存
            self.mapping_df_[groupby_col] = _df.to_dict()

        self._fitted = True
        return self.transform(input_df)

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """学習したnuniqueをマッピング

        Args:
            input_df: 入力DataFrame（train/testどちらでも可）

        Returns:
            nunique特徴量のDataFrame

        Raises:
            RuntimeError: fit()を先に実行していない場合
        """
        if not self._fitted:
            raise RuntimeError("CategoryNuniqueBlock: fit()を先に実行してください")

        # Polars → pandas変換
        pdf = input_df.to_pandas()

        out_df = pd.DataFrame()

        for groupby_col in self.mapping_df_:
            for target_col in self.mapping_df_[groupby_col]:
                col_name = f"nunique_{groupby_col}_groupby_{target_col}"
                mapping = self.mapping_df_[groupby_col][target_col]

                # カテゴリ値でマッピング、未知カテゴリは0で埋める
                out_df[col_name] = pdf[groupby_col].map(mapping).fillna(0).astype(int)

        # pandas → Polars変換して返す
        return pl.from_pandas(out_df)
