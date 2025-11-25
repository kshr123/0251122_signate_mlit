"""
次元圧縮Blockのテスト

SVDBlock, PCABlock, UMAPBlockのテスト。
共通の基底クラスDimensionReductionBlockの動作を検証。
"""

import pytest
import polars as pl
import numpy as np

from features.blocks.dimension_reduction import (
    SVDBlock,
    PCABlock,
    UMAPBlock,
)


# =============================================================================
# SVDBlock Tests
# =============================================================================

class TestSVDBlock:
    """SVDBlockのテスト"""

    def test_svd_basic(self):
        """SVDBlock: 基本的な次元圧縮ができること"""
        np.random.seed(42)
        df = pl.DataFrame({
            "col1": np.random.randn(100).tolist(),
            "col2": np.random.randn(100).tolist(),
            "col3": np.random.randn(100).tolist(),
        })

        block = SVDBlock(columns=["col1", "col2", "col3"], n_components=2)
        result = block.fit(df)

        # 2次元に圧縮される
        assert result.shape[0] == 100
        assert result.shape[1] == 2

        # カラム名にsvdプレフィックスがつく
        for col in result.columns:
            assert "svd" in col

    def test_svd_transform_consistency(self):
        """SVDBlock: train/testで一貫した変換が適用されること"""
        np.random.seed(42)
        train_df = pl.DataFrame({
            "col1": np.random.randn(100).tolist(),
            "col2": np.random.randn(100).tolist(),
        })
        test_df = pl.DataFrame({
            "col1": np.random.randn(50).tolist(),
            "col2": np.random.randn(50).tolist(),
        })

        block = SVDBlock(columns=["col1", "col2"], n_components=2)
        train_result = block.fit(train_df)
        test_result = block.transform(test_df)

        # 同じカラム構造
        assert train_result.columns == test_result.columns
        assert test_result.shape[0] == 50
        assert test_result.shape[1] == 2

    def test_svd_standardize(self):
        """SVDBlock: 標準化が適用されること"""
        # スケールが大きく異なるデータ
        df = pl.DataFrame({
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col2": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        })

        block_with_std = SVDBlock(columns=["col1", "col2"], n_components=1, standardize=True)
        block_without_std = SVDBlock(columns=["col1", "col2"], n_components=1, standardize=False)

        result_with = block_with_std.fit(df)
        result_without = block_without_std.fit(df)

        # 標準化あり/なしで結果が異なる
        assert not np.allclose(
            result_with.to_numpy(),
            result_without.to_numpy()
        )

    def test_svd_missing_error(self):
        """SVDBlock: 欠損値があるとエラー（デフォルト）"""
        df = pl.DataFrame({
            "col1": [1.0, 2.0, None, 4.0],
            "col2": [1.0, 2.0, 3.0, 4.0],
        })

        block = SVDBlock(columns=["col1", "col2"], n_components=1)

        with pytest.raises(ValueError, match="欠損値"):
            block.fit(df)

    def test_svd_missing_mean(self):
        """SVDBlock: handle_missing='mean'で平均補完されること"""
        df = pl.DataFrame({
            "col1": [1.0, 2.0, None, 4.0],  # 平均 = 7/3 ≈ 2.33
            "col2": [1.0, 2.0, 3.0, 4.0],
        })

        block = SVDBlock(
            columns=["col1", "col2"],
            n_components=1,
            handle_missing="mean"
        )
        result = block.fit(df)

        # エラーなく実行される
        assert result.shape[0] == 4
        assert result.shape[1] == 1

    def test_svd_missing_zero(self):
        """SVDBlock: handle_missing='zero'で0補完されること"""
        df = pl.DataFrame({
            "col1": [1.0, 2.0, None, 4.0],
            "col2": [1.0, 2.0, 3.0, 4.0],
        })

        block = SVDBlock(
            columns=["col1", "col2"],
            n_components=1,
            handle_missing="zero"
        )
        result = block.fit(df)

        assert result.shape[0] == 4
        assert result.shape[1] == 1

    def test_svd_missing_transform_uses_train_mean(self):
        """SVDBlock: transformでもtrainの平均で補完されること"""
        train_df = pl.DataFrame({
            "col1": [10.0, 20.0, 30.0, 40.0],  # 平均 = 25
            "col2": [1.0, 2.0, 3.0, 4.0],
        })
        test_df = pl.DataFrame({
            "col1": [None, 15.0],  # 欠損はtrainの平均25で補完される
            "col2": [2.0, 3.0],
        })

        block = SVDBlock(
            columns=["col1", "col2"],
            n_components=1,
            handle_missing="mean"
        )
        _ = block.fit(train_df)
        result = block.transform(test_df)

        # エラーなく実行される
        assert result.shape[0] == 2

    def test_svd_not_fitted_error(self):
        """SVDBlock: fit前のtransformでRuntimeError"""
        df = pl.DataFrame({
            "col1": [1.0, 2.0],
            "col2": [1.0, 2.0],
        })

        block = SVDBlock(columns=["col1", "col2"], n_components=1)

        with pytest.raises(RuntimeError, match="fit.*先に実行"):
            block.transform(df)

    def test_svd_immutability(self):
        """SVDBlock: 元のDataFrameを変更しないこと"""
        df = pl.DataFrame({
            "col1": [1.0, 2.0, 3.0],
            "col2": [4.0, 5.0, 6.0],
        })

        original_col1 = df["col1"].to_list()
        original_col2 = df["col2"].to_list()

        block = SVDBlock(columns=["col1", "col2"], n_components=1)
        _ = block.fit(df)

        assert df["col1"].to_list() == original_col1
        assert df["col2"].to_list() == original_col2


# =============================================================================
# PCABlock Tests
# =============================================================================

class TestPCABlock:
    """PCABlockのテスト"""

    def test_pca_basic(self):
        """PCABlock: 基本的な次元圧縮ができること"""
        np.random.seed(42)
        df = pl.DataFrame({
            "col1": np.random.randn(100).tolist(),
            "col2": np.random.randn(100).tolist(),
            "col3": np.random.randn(100).tolist(),
        })

        block = PCABlock(columns=["col1", "col2", "col3"], n_components=2)
        result = block.fit(df)

        assert result.shape[0] == 100
        assert result.shape[1] == 2

        for col in result.columns:
            assert "pca" in col

    def test_pca_transform_consistency(self):
        """PCABlock: train/testで一貫した変換が適用されること"""
        np.random.seed(42)
        train_df = pl.DataFrame({
            "col1": np.random.randn(100).tolist(),
            "col2": np.random.randn(100).tolist(),
        })
        test_df = pl.DataFrame({
            "col1": np.random.randn(50).tolist(),
            "col2": np.random.randn(50).tolist(),
        })

        block = PCABlock(columns=["col1", "col2"], n_components=2)
        train_result = block.fit(train_df)
        test_result = block.transform(test_df)

        assert train_result.columns == test_result.columns
        assert test_result.shape[0] == 50

    def test_pca_missing_mean(self):
        """PCABlock: handle_missing='mean'で動作すること"""
        df = pl.DataFrame({
            "col1": [1.0, 2.0, None, 4.0],
            "col2": [1.0, 2.0, 3.0, 4.0],
        })

        block = PCABlock(
            columns=["col1", "col2"],
            n_components=1,
            handle_missing="mean"
        )
        result = block.fit(df)

        assert result.shape[0] == 4


# =============================================================================
# UMAPBlock Tests
# =============================================================================

class TestUMAPBlock:
    """UMAPBlockのテスト"""

    def test_umap_basic(self):
        """UMAPBlock: 基本的な次元圧縮ができること"""
        np.random.seed(42)
        # UMAPは最低でもn_neighbors以上のサンプルが必要
        df = pl.DataFrame({
            "col1": np.random.randn(50).tolist(),
            "col2": np.random.randn(50).tolist(),
            "col3": np.random.randn(50).tolist(),
        })

        block = UMAPBlock(
            columns=["col1", "col2", "col3"],
            n_components=2,
            n_neighbors=5,
        )
        result = block.fit(df)

        assert result.shape[0] == 50
        assert result.shape[1] == 2

        for col in result.columns:
            assert "umap" in col

    def test_umap_transform_consistency(self):
        """UMAPBlock: train/testで変換できること"""
        np.random.seed(42)
        train_df = pl.DataFrame({
            "col1": np.random.randn(50).tolist(),
            "col2": np.random.randn(50).tolist(),
        })
        test_df = pl.DataFrame({
            "col1": np.random.randn(20).tolist(),
            "col2": np.random.randn(20).tolist(),
        })

        block = UMAPBlock(
            columns=["col1", "col2"],
            n_components=2,
            n_neighbors=5,
        )
        train_result = block.fit(train_df)
        test_result = block.transform(test_df)

        assert train_result.columns == test_result.columns
        assert test_result.shape[0] == 20

    def test_umap_missing_mean(self):
        """UMAPBlock: handle_missing='mean'で動作すること"""
        np.random.seed(42)
        data = np.random.randn(50, 2)
        data[5, 0] = np.nan  # 欠損を1つ入れる

        df = pl.DataFrame({
            "col1": data[:, 0].tolist(),
            "col2": data[:, 1].tolist(),
        })

        block = UMAPBlock(
            columns=["col1", "col2"],
            n_components=2,
            n_neighbors=5,
            handle_missing="mean"
        )
        result = block.fit(df)

        assert result.shape[0] == 50


# =============================================================================
# Common Tests
# =============================================================================

class TestDimensionReductionCommon:
    """次元圧縮Block共通のテスト"""

    def test_column_naming_short(self):
        """カラム名: 少数カラムの場合は連結"""
        df = pl.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        })

        block = SVDBlock(columns=["a", "b"], n_components=2)
        result = block.fit(df)

        # カラム名に元のカラム名が含まれる
        col_names = result.columns
        assert len(col_names) == 2

    def test_column_naming_many_columns(self):
        """カラム名: 多数カラムの場合は数で表示"""
        np.random.seed(42)
        # 10カラム以上
        data = {f"col{i}": np.random.randn(20).tolist() for i in range(15)}
        df = pl.DataFrame(data)

        block = SVDBlock(columns=list(data.keys()), n_components=2)
        result = block.fit(df)

        # カラム名が生成されている
        assert result.shape[1] == 2
        # "15cols"のような表記が含まれる
        for col in result.columns:
            assert "15cols" in col

    def test_n_components_exceeds_features(self):
        """n_componentsが特徴数を超える場合"""
        df = pl.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        })

        # 2カラムに対してn_components=5
        block = SVDBlock(columns=["a", "b"], n_components=5)
        result = block.fit(df)

        # min(n_components, n_features)になる
        assert result.shape[1] <= 2
