"""
リネームBlock

出力カラム名をリネームするラッパーBlock。
"""

from typing import Any, Dict

import polars as pl

from features.base import BaseBlock


class RenameBlock(BaseBlock):
    """出力カラム名をリネームするラッパーBlock

    他のBlockの出力カラム名を変更したい場合に使用。
    rename_mapで個別指定、またはsuffixで一括追加が可能。

    Args:
        block: ラップするBlock
        rename_map: {元カラム名: 新カラム名}の辞書
        suffix: 全カラムに付けるサフィックス

    Examples:
        >>> from features import TargetEncodingBlock
        >>> te_block = TargetEncodingBlock(columns=['category'])
        >>> # サフィックスを追加
        >>> renamed = RenameBlock(te_block, suffix='_v2')
        >>> train_result = renamed.fit(train_df, y)
        >>> test_result = renamed.transform(test_df)
        >>>
        >>> # 個別リネーム
        >>> renamed = RenameBlock(te_block, rename_map={'category_te': 'cat_encoded'})
    """

    def __init__(
        self,
        block: Any,
        rename_map: Dict[str, str] = None,
        suffix: str = None,
    ):
        super().__init__()
        self.block = block
        self.rename_map = rename_map
        self.suffix = suffix

    def _apply_rename(self, df: pl.DataFrame) -> pl.DataFrame:
        """リネームを適用"""
        if self.rename_map:
            return df.rename(self.rename_map)
        elif self.suffix:
            return df.rename({col: f"{col}{self.suffix}" for col in df.columns})
        return df

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        """ラップされたBlockをfitし、リネームを適用"""
        if hasattr(self.block, "fit_transform"):
            result = self.block.fit_transform(input_df, y)
        elif hasattr(self.block, "fit"):
            result = self.block.fit(input_df, y)
        else:
            raise ValueError("Block must have fit_transform or fit method")
        self._fitted = True
        return self._apply_rename(result)

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """ラップされたBlockをtransformし、リネームを適用"""
        if not self._fitted:
            raise RuntimeError(f"{self.__class__.__name__}: fit()を先に実行してください")
        result = self.block.transform(input_df)
        return self._apply_rename(result)
