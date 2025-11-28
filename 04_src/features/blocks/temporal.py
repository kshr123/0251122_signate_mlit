"""
時系列特徴量Block

YYYYMMフォーマットの時系列データを処理
"""

import polars as pl
from features.base import BaseBlock


class TargetYmBlock(BaseBlock):
    """target_ymを年・月に分解するBlock

    YYYYMMフォーマットの列を年（YYYY）と月（MM）に分解します。

    Args:
        source_col: 分解する列名（デフォルト: "target_ym"）

    Examples:
        >>> df = pl.DataFrame({
        ...     "target_ym": [202301, 202312, 202401]
        ... })
        >>> block = TargetYmBlock()
        >>> result = block.fit(df)
        >>> result["target_year"].to_list()
        [2023, 2023, 2024]
        >>> result["target_month"].to_list()
        [1, 12, 4]
    """

    def __init__(self, source_col: str = "target_ym"):
        """初期化

        Args:
            source_col: 分解する列名（デフォルト: "target_ym"）
        """
        super().__init__()
        self.source_col = source_col

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        """統計量の学習（不要なのでそのままtransform）

        Args:
            input_df: 入力DataFrame
            y: ターゲット変数（使用しない）

        Returns:
            変換後のDataFrame
        """
        self._fitted = True
        return self._transform(input_df)

    def _transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """YYYYMMを年・月に分解

        処理内容:
        - year = target_ym // 100  (例: 202301 → 2023)
        - month = target_ym % 100  (例: 202301 → 1)

        Args:
            input_df: 入力DataFrame

        Returns:
            target_year, target_month を含むDataFrame

        Note:
            source_colは整数型（Int64等）である必要があります。
            YYYYMMフォーマット（例: 202301, 202412）を想定しています。
        """
        return input_df.select([
            (pl.col(self.source_col) // 100).alias("target_year"),
            (pl.col(self.source_col) % 100).alias("target_month"),
        ])
