"""
テキスト変換Block

TF-IDF変換を行う。
"""

from typing import List, Optional

import polars as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from features.base import BaseBlock


class TfidfBlock(BaseBlock):
    """TF-IDF変換Block

    複数のテキストカラムを結合し、TF-IDFベクトルに変換します。

    Args:
        columns: TF-IDF対象のカラムリスト
        max_features: 出力次元数（デフォルト: 20）
        separator: カラム結合時の区切り文字（デフォルト: ' '）
        prefix: 出力カラム名のプレフィックス（デフォルト: 'tfidf'）
        token_pattern: トークン抽出の正規表現（デフォルト: 日本語対応）

    Examples:
        >>> df = pl.DataFrame({
        ...     "text1": ["東京 渋谷", "大阪 梅田"],
        ...     "text2": ["駅前", "繁華街"],
        ... })
        >>> block = TfidfBlock(columns=["text1", "text2"], max_features=10)
        >>> result = block.fit(df)
        >>> # result.columns = ["tfidf_0", "tfidf_1", ..., "tfidf_9"]
    """

    def __init__(
        self,
        columns: List[str],
        max_features: int = 20,
        separator: str = ' ',
        prefix: str = 'tfidf',
        token_pattern: str = r'(?u)\b\w+\b',
    ):
        super().__init__()
        self.columns = columns
        self.max_features = max_features
        self.separator = separator
        self.prefix = prefix
        self.token_pattern = token_pattern

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            token_pattern=token_pattern,
        )
        self.feature_names: List[str] = []

    def _create_text(self, df: pl.DataFrame) -> List[str]:
        """複数カラムを結合してテキスト化"""
        text_cols = [
            pl.col(col).cast(pl.Utf8).fill_null('')
            for col in self.columns
        ]
        combined = df.select(
            pl.concat_str(text_cols, separator=self.separator).alias('text')
        )
        return combined['text'].to_list()

    def fit(self, input_df: pl.DataFrame, y=None) -> pl.DataFrame:
        """学習データでfit

        Args:
            input_df: 入力DataFrame（trainデータ）
            y: ターゲット変数（使用しない）

        Returns:
            TF-IDF変換後のDataFrame
        """
        texts = self._create_text(input_df)
        self.vectorizer.fit(texts)
        self.feature_names = [f'{self.prefix}_{i}' for i in range(self.max_features)]

        self._fitted = True
        return self._transform(input_df)

    def _transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """学習した語彙でTF-IDF変換

        Args:
            input_df: 入力DataFrame

        Returns:
            TF-IDF変換後のDataFrame
        """
        texts = self._create_text(input_df)
        tfidf_matrix = self.vectorizer.transform(texts)
        tfidf_array = tfidf_matrix.toarray()

        return pl.DataFrame({
            name: tfidf_array[:, i]
            for i, name in enumerate(self.feature_names)
        })

    def get_vocabulary(self) -> dict:
        """学習した語彙を取得"""
        return self.vectorizer.vocabulary_
