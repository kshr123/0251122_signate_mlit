"""
BaseBlockとSeedManagerのテスト
"""

import pytest
import random
import os
import numpy as np
import polars as pl

from features.base import set_seed, BaseBlock


def test_set_seed_fixes_random():
    """set_seedがPython標準ライブラリのrandomを固定すること"""
    set_seed(42)
    values1 = [random.random() for _ in range(10)]

    set_seed(42)
    values2 = [random.random() for _ in range(10)]

    assert values1 == values2


def test_set_seed_fixes_numpy():
    """set_seedがNumPyの乱数を固定すること"""
    set_seed(42)
    values1 = np.random.rand(10).tolist()

    set_seed(42)
    values2 = np.random.rand(10).tolist()

    assert values1 == values2


def test_set_seed_sets_pythonhashseed():
    """set_seedがPYTHONHASHSEEDを設定すること"""
    set_seed(42)
    assert os.environ.get('PYTHONHASHSEED') == '42'


def test_different_seeds_produce_different_values():
    """異なるシードで異なる乱数列が生成されること"""
    set_seed(42)
    values1 = [random.random() for _ in range(10)]

    set_seed(100)
    values2 = [random.random() for _ in range(10)]

    assert values1 != values2


# ===== BaseBlock のテスト =====

class DummyBlock(BaseBlock):
    """テスト用のダミーBlock"""

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        if not self._fitted:
            raise RuntimeError("DummyBlock: fit()を先に実行してください")
        return input_df.select("col1")


def test_base_block_fit_transform():
    """BaseBlock: fit()実行後にtransform()が動作すること"""
    df = pl.DataFrame({
        "col1": [1, 2, 3],
        "col2": [4, 5, 6],
    })

    block = DummyBlock()
    result = block.fit(df)

    # fit()が_fittedフラグを立てること
    assert block._fitted is True

    # fit()が変換結果を返すこと
    assert result.columns == ["col1"]
    assert result.shape == (3, 1)


def test_base_block_not_fitted_error():
    """BaseBlock: fit()前にtransform()を呼ぶとRuntimeErrorが発生すること"""
    df = pl.DataFrame({
        "col1": [1, 2, 3],
        "col2": [4, 5, 6],
    })

    block = DummyBlock()

    # fit()を実行していない状態でtransform()を呼ぶ
    with pytest.raises(RuntimeError, match="fit\\(\\)を先に実行してください"):
        block.transform(df)
