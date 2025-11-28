"""
特徴量パイプライン

複数の特徴量Blockを順番に適用し、変換結果を結合するパイプラインクラス。
どの特徴量にどの変換が適用されたかを追跡可能。

使用例:
    from features.pipeline import FeaturePipeline
    from features.blocks.encoding import TargetEncodingBlock

    pipeline = FeaturePipeline()
    pipeline.add_block("te", TargetEncodingBlock(...), ["city"], "市区町村TE")
    X_train = pipeline.fit_transform(train_df, y_train)
    X_test = pipeline.transform(test_df)

    print(pipeline.summary())
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import polars as pl


@dataclass
class BlockInfo:
    """Block情報を保持"""

    name: str
    block: Any
    input_columns: List[str]
    output_columns: List[str] = field(default_factory=list)
    description: str = ""


class FeaturePipeline:
    """
    特徴量変換パイプライン

    複数のBlockを順番に適用し、変換結果を結合。
    どの特徴量にどの変換が適用されたかを追跡可能。

    Attributes:
        blocks: BlockInfoのリスト
        _feature_names: 変換後の特徴量名リスト
        _fitted: fit済みかどうか
    """

    def __init__(self):
        self.blocks: List[BlockInfo] = []
        self._feature_names: List[str] = []
        self._fitted: bool = False

    def add_block(
        self,
        name: str,
        block: Any,
        input_columns: List[str],
        description: str = "",
    ) -> "FeaturePipeline":
        """
        Blockを追加

        Args:
            name: Block名（識別用）
            block: Blockインスタンス
            input_columns: 入力カラム名リスト
            description: 変換の説明

        Returns:
            self（メソッドチェーン用）
        """
        self.blocks.append(
            BlockInfo(
                name=name,
                block=block,
                input_columns=input_columns,
                description=description,
            )
        )
        return self

    def fit_transform(self, df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        """
        全Blockをfit & transform

        Args:
            df: 入力DataFrame（trainデータ）
            y: ターゲット変数

        Returns:
            変換後のDataFrame
        """
        results = []
        self._feature_names = []

        for block_info in self.blocks:
            # fit_transformを実行
            if hasattr(block_info.block, "fit_transform"):
                feat = block_info.block.fit_transform(df, y)
            elif hasattr(block_info.block, "fit"):
                feat = block_info.block.fit(df, y)
            else:
                raise ValueError(
                    f"{block_info.name}: fit_transform or fit method required"
                )

            # 出力カラム名を記録
            block_info.output_columns = list(feat.columns)
            self._feature_names.extend(feat.columns)
            results.append(feat)

        self._fitted = True
        return pl.concat(results, how="horizontal")

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        全Blockをtransform（fit済みのパラメータを使用）

        Args:
            df: 入力DataFrame（testデータ）

        Returns:
            変換後のDataFrame
        """
        if not self._fitted:
            raise RuntimeError("FeaturePipeline: fit_transform()を先に実行してください")

        results = []
        for block_info in self.blocks:
            feat = block_info.block.transform(df)
            results.append(feat)

        return pl.concat(results, how="horizontal")

    def get_feature_names(self) -> List[str]:
        """変換後の特徴量名リストを取得"""
        return self._feature_names

    def describe(self) -> Dict[str, Dict[str, Any]]:
        """
        パイプラインの変換内容を一覧表示

        Returns:
            {block名: {input_columns, output_columns, description}}
        """
        return {
            block_info.name: {
                "input_columns": block_info.input_columns,
                "output_columns": block_info.output_columns,
                "n_input": len(block_info.input_columns),
                "n_output": len(block_info.output_columns),
                "description": block_info.description,
            }
            for block_info in self.blocks
        }

    def summary(self) -> str:
        """パイプラインのサマリーを文字列で取得"""
        lines = ["=" * 60, "Feature Pipeline Summary", "=" * 60]

        for block_info in self.blocks:
            n_in = len(block_info.input_columns)
            n_out = len(block_info.output_columns)
            lines.append(f"\n[{block_info.name}]")
            lines.append(f"  入力: {n_in}カラム → 出力: {n_out}カラム")
            if block_info.description:
                lines.append(f"  説明: {block_info.description}")
            if n_out <= 5:
                lines.append(f"  出力カラム: {block_info.output_columns}")
            else:
                lines.append(
                    f"  出力カラム: {block_info.output_columns[:3]} ... (+{n_out - 3})"
                )

        lines.append("\n" + "=" * 60)
        lines.append(f"合計特徴量数: {len(self._feature_names)}")
        lines.append("=" * 60)

        return "\n".join(lines)
