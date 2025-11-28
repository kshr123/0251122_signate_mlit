"""FeaturePipelineのテスト"""

import pytest
import polars as pl

from features.pipeline import FeaturePipeline, BlockInfo
from features.base import BaseBlock


class DummyBlock(BaseBlock):
    """テスト用ダミーBlock"""

    def __init__(self, column: str, output_suffix: str = "_out"):
        super().__init__()
        self._column = column
        self._suffix = output_suffix

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        self._fitted = True
        return input_df.select(
            pl.col(self._column).alias(f"{self._column}{self._suffix}")
        )

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        if not self._fitted:
            raise RuntimeError("Not fitted")
        return input_df.select(
            pl.col(self._column).alias(f"{self._column}{self._suffix}")
        )


class TestBlockInfo:
    """BlockInfoのテスト"""

    def test_block_info_creation(self):
        """BlockInfoが正しく作成される"""
        block = DummyBlock("col1")
        info = BlockInfo(
            name="test",
            block=block,
            input_columns=["col1"],
            description="Test block",
        )
        assert info.name == "test"
        assert info.input_columns == ["col1"]
        assert info.description == "Test block"
        assert info.output_columns == []


class TestFeaturePipeline:
    """FeaturePipelineのテスト"""

    @pytest.fixture
    def sample_df(self):
        """テスト用DataFrame"""
        return pl.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [10, 20, 30, 40, 50],
            }
        )

    @pytest.fixture
    def sample_y(self):
        """テスト用ターゲット"""
        return pl.Series("y", [100, 200, 300, 400, 500])

    def test_feature_pipeline_init(self):
        """初期化が正しく行われる"""
        pipeline = FeaturePipeline()
        assert pipeline.blocks == []
        assert pipeline._feature_names == []
        assert pipeline._fitted is False

    def test_add_block(self):
        """Blockが正しく追加される"""
        pipeline = FeaturePipeline()
        block = DummyBlock("a")

        result = pipeline.add_block(
            name="block1", block=block, input_columns=["a"], description="Test"
        )

        assert result is pipeline  # メソッドチェーン
        assert len(pipeline.blocks) == 1
        assert pipeline.blocks[0].name == "block1"

    def test_fit_transform(self, sample_df, sample_y):
        """fit_transformが正しく動作する"""
        pipeline = FeaturePipeline()
        pipeline.add_block("block1", DummyBlock("a", "_v1"), ["a"], "Block 1")
        pipeline.add_block("block2", DummyBlock("b", "_v2"), ["b"], "Block 2")

        result = pipeline.fit_transform(sample_df, sample_y)

        assert pipeline._fitted is True
        assert result.shape == (5, 2)
        assert "a_v1" in result.columns
        assert "b_v2" in result.columns
        assert pipeline.get_feature_names() == ["a_v1", "b_v2"]

    def test_transform_before_fit_raises(self, sample_df):
        """fit前にtransformするとエラー"""
        pipeline = FeaturePipeline()
        pipeline.add_block("block1", DummyBlock("a"), ["a"], "Block 1")

        with pytest.raises(RuntimeError, match="fit_transform"):
            pipeline.transform(sample_df)

    def test_transform_after_fit(self, sample_df, sample_y):
        """fit後にtransformが正しく動作する"""
        pipeline = FeaturePipeline()
        pipeline.add_block("block1", DummyBlock("a", "_out"), ["a"], "Block 1")

        # fit_transform
        pipeline.fit_transform(sample_df, sample_y)

        # transform（別データ）
        test_df = pl.DataFrame({"a": [10, 20], "b": [100, 200]})
        result = pipeline.transform(test_df)

        assert result.shape == (2, 1)
        assert "a_out" in result.columns

    def test_describe(self, sample_df, sample_y):
        """describeが正しい情報を返す"""
        pipeline = FeaturePipeline()
        pipeline.add_block("block1", DummyBlock("a", "_out"), ["a"], "Test block")
        pipeline.fit_transform(sample_df, sample_y)

        desc = pipeline.describe()

        assert "block1" in desc
        assert desc["block1"]["input_columns"] == ["a"]
        assert desc["block1"]["output_columns"] == ["a_out"]
        assert desc["block1"]["description"] == "Test block"

    def test_summary(self, sample_df, sample_y):
        """summaryが文字列を返す"""
        pipeline = FeaturePipeline()
        pipeline.add_block("block1", DummyBlock("a"), ["a"], "Block 1")
        pipeline.fit_transform(sample_df, sample_y)

        summary = pipeline.summary()

        assert isinstance(summary, str)
        assert "block1" in summary
        assert "Block 1" in summary
        assert "合計特徴量数: 1" in summary
