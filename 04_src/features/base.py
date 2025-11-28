"""
特徴量エンジニアリング基盤モジュール

再現性確保のためのSeedManager等を提供
"""

import random
import os
import numpy as np
import polars as pl
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """
    すべての乱数生成器のシードを固定

    Args:
        seed: シード値

    Examples:
        >>> set_seed(42)
        >>> # 以降、random, numpy等の乱数が再現可能
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    # PyTorch（使用する場合）
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass  # PyTorchがインストールされていない場合は無視


class BaseBlock:
    """特徴量Blockの基底クラス

    すべての特徴量処理Blockはこのクラスを継承します。
    fit/transformパターンでデータリークを防止します。

    設計原則:
        - fit(): 統計量を学習し、変換結果を返す
        - transform(): 学習済み統計量で変換
        - _transform(): 実際の変換ロジック（子クラスでオーバーライド）

    子クラスの実装パターン:
        1. Stateless（統計量不要）: _transform()のみオーバーライド
        2. Stateful（統計量必要）: fit()と_transform()をオーバーライド

    Examples:
        >>> class MyBlock(BaseBlock):
        ...     def _transform(self, input_df):
        ...         return input_df.select("feature_col")
        ...
        >>> block = MyBlock()
        >>> train_features = block.fit(train_df)
        >>> test_features = block.transform(test_df)
    """

    def __init__(self):
        """初期化"""
        self._fitted = False

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        """統計量を学習し、変換結果を返す

        Args:
            input_df: 入力DataFrame（trainデータ）
            y: ターゲット変数（Target Encodingなどで使用）

        Returns:
            変換後のDataFrame

        Note:
            このメソッドは**trainデータのみ**で実行してください。
            testデータでは必ずtransform()を使用してください。
        """
        self._fitted = True
        return self._transform(input_df)

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """学習した統計量で変換

        Args:
            input_df: 入力DataFrame（train/testどちらでも可）

        Returns:
            変換後のDataFrame

        Raises:
            RuntimeError: fit()を先に実行していない場合

        Note:
            このメソッドはfit()実行後に使用してください。
        """
        if not self._fitted:
            raise RuntimeError(f"{self.__class__.__name__}: fit()を先に実行してください")
        return self._transform(input_df)

    def _transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """実際の変換ロジック（子クラスでオーバーライド）

        Args:
            input_df: 入力DataFrame

        Returns:
            変換後のDataFrame

        Note:
            子クラスはこのメソッドをオーバーライドして変換ロジックを実装します。
            fit()とtransform()の両方からこのメソッドが呼ばれます。
        """
        raise NotImplementedError()

    def fit_transform(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        """fit()とtransform()を連続実行（sklearn互換）

        Args:
            input_df: 入力DataFrame（trainデータ）
            y: ターゲット変数（Target Encodingなどで使用）

        Returns:
            変換後のDataFrame

        Note:
            このメソッドは内部でfit()を呼ぶため、trainデータにのみ使用してください。
        """
        return self.fit(input_df, y)
