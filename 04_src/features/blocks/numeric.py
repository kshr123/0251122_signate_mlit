"""
数値特徴量Block

数値特徴量をそのまま返す（前処理なし）
"""

import polars as pl
from features.base import BaseBlock


class NumericBlock(BaseBlock):
    """数値特徴量をそのまま返すBlock

    指定された数値カラムをそのまま返します。
    前処理は行いません。

    Args:
        columns: 対象の数値カラムリスト

    Examples:
        >>> df = pl.DataFrame({
        ...     "num1": [1, 2, 3],
        ...     "num2": [1.5, 2.5, 3.5],
        ...     "cat": ["A", "B", "C"]
        ... })
        >>> block = NumericBlock(columns=["num1", "num2"])
        >>> result = block.fit(df)
        >>> result.columns
        ['num1', 'num2']
    """

    def __init__(self, columns: list[str]):
        """初期化

        Args:
            columns: 対象の数値カラムリスト
        """
        super().__init__()
        self.columns = columns

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        """統計量の学習（不要なのでそのままtransform）

        Args:
            input_df: 入力DataFrame
            y: ターゲット変数（使用しない）

        Returns:
            変換後のDataFrame
        """
        self._fitted = True
        return self.transform(input_df)

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """指定されたカラムをそのまま返す

        Args:
            input_df: 入力DataFrame

        Returns:
            指定されたカラムのみを含むDataFrame

        Raises:
            RuntimeError: fit()を先に実行していない場合
        """
        if not self._fitted:
            raise RuntimeError("NumericBlock: fit()を先に実行してください")

        return input_df.select(self.columns)
