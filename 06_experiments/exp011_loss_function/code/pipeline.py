"""
exp010 特徴量パイプライン

特徴量変換を一元管理するパイプラインクラス。
どの特徴量にどの変換が適用されたかを追跡可能。

使用例:
    pipeline = create_pipeline(cv_splits)
    X_train = pipeline.fit_transform(train, y_train)
    X_test = pipeline.transform(test)

    # 変換内容を確認
    print(pipeline.describe())
    print(pipeline.get_feature_names())
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "04_src"))
# exp010のcodeディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl
import numpy as np

# 04_srcの共通Block
from features.base import BaseBlock
from features.blocks.numeric import NumericBlock
from features.blocks.encoding import (
    CountEncodingBlock,
    TargetEncodingBlock,
    LabelEncodingBlock,
)
from features.blocks.rename import RenameBlock
from features.blocks.text import TfidfBlock
from features.blocks.dimension_reduction import PCABlock
from features.blocks.multi_hot import (
    MultiHotSVDBlock,
    MultiColumnMultiHotSVDBlock,
    MultiColumnOneHotSVDBlock,
)
from features.blocks.aggregation import GroupByAggBlock

# exp010の特徴量
from exp011_features import (
    # Blockクラス
    PostalCodeTEBlock,
    AreaRegionalRatioBlock,
    AreaAgeCategoryTEBlock,
    DistanceRegionalRatioBlock,
    # exp010固有関数
    add_lp_area_value,
    add_area_age_category,
    # 定数
    CURRENT_USE_TOP_CATEGORIES,
    LP_RATIO_COLUMNS,
)

# 04_srcの共通Block（上位N+その他LE）
from features import TopNCategoryLEBlock

# カラム定義（constants.pyから）
from constants import (
    NUMERIC_FEATURES,
    TARGET_ENCODING_COLUMNS,
    COUNT_ENCODING_COLUMNS,
    ROUTE_LE_COLUMNS,
    TFIDF_TEXT_COLUMNS,
    GEO_COLUMNS,
)

# デフォルトパラメータ（experiment.yamlから上書き可能）
DEFAULT_FEATURE_PARAMS = {
    "tfidf_max_features": 20,
    "geo_pca_components": 2,
    "building_tag_svd_dim": 15,
    "unit_tag_svd_dim": 30,
    "statuses_svd_dim": 30,
    "reform_svd_dim": 7,
    "post_full_min_count": 30,
    "random_seed": 42,
    "landprice_road_svd_dim": 13,
}


# =============================================================================
# パイプラインクラス
# =============================================================================

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
        description: str = ""
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
        self.blocks.append(BlockInfo(
            name=name,
            block=block,
            input_columns=input_columns,
            description=description,
        ))
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
            if hasattr(block_info.block, 'fit_transform'):
                feat = block_info.block.fit_transform(df, y)
            elif hasattr(block_info.block, 'fit'):
                feat = block_info.block.fit(df, y)
            else:
                raise ValueError(f"{block_info.name}: fit_transform or fit method required")

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
                lines.append(f"  出力カラム: {block_info.output_columns[:3]} ... (+{n_out - 3})")

        lines.append("\n" + "=" * 60)
        lines.append(f"合計特徴量数: {len(self._feature_names)}")
        lines.append("=" * 60)

        return "\n".join(lines)


# =============================================================================
# 地価公示特徴量用の特殊Block
# =============================================================================

class LandpriceFeatureBlock(BaseBlock):
    """
    地価公示データの特徴量生成Block

    preprocessing.pyで地価公示データは既に結合済み。
    このBlockは結合済みデータから特徴量を生成する：
    - 土地形状PCA（04_src/PCABlockで代替）
    - 道路状況SVD（MultiColumnOneHotSVDBlock）
    - 利用現況LE（TopNCategoryLEBlock）
    - 価格時系列比率（Polars直書き）
    - カテゴリ別地価比率（GroupByAggBlock）
    """

    def __init__(
        self,
        road_svd_dim: int = DEFAULT_FEATURE_PARAMS["landprice_road_svd_dim"],
        random_seed: int = DEFAULT_FEATURE_PARAMS["random_seed"],
    ):
        super().__init__()
        self._road_svd_dim = road_svd_dim
        self._random_seed = random_seed

        # 出力特徴量カラム（21次元 + 18次元 = 39次元 + 新規4次元 = 43次元）
        self.LP_BASE_COLUMNS = [
            # 土地形状PCA（1次元）
            "lp_shape_pca",
            # 道路状況SVD（N次元）+ 道路幅員（1次元）
            *[f"lp_road_svd_{i}" for i in range(road_svd_dim)],
            "lp_road_width",
            # 利用現況LE（1次元）
            "lp_current_use_le",
            # 価格関連（3次元）
            "lp_price",
            "lp_change_rate",
            "lp_nearest_dist",
            # 価格時系列（2次元）
            "lp_ratio_1to3",
            "lp_ratio_3to5",
            # 新規追加（4次元）: 地積・駅距離・防火地域・駅距離比率
            "lp_land_area_log",
            "lp_station_dist",
            "lp_fire_zone_le",
            "station_dist_ratio",
        ]

        # 防火地域LE
        self._fire_zone_le = LabelEncodingBlock(columns=["lp_fire_zone"])

        # 各変換器
        self._shape_pca = PCABlock(
            columns=["lp_frontage_ratio", "lp_depth_ratio"],
            n_components=1,
            standardize=False,
            handle_missing="mean",
            random_state=random_seed,
        )
        # 道路状況SVD
        self._road_svd = MultiColumnOneHotSVDBlock(
            columns=["lp_road_type", "lp_road_direction", "lp_side_road"],
            n_components=road_svd_dim,
            output_prefix="lp_road",
            null_value="_",
            random_state=random_seed,
        )
        # 利用現況LE
        self._current_use_le = TopNCategoryLEBlock(
            column="lp_current_use",
            top_categories=CURRENT_USE_TOP_CATEGORIES,
            other_label="その他",
            output_column="lp_current_use_le",
        )
        # カテゴリ別地価比率
        self._lp_ratio_blocks = {
            col: GroupByAggBlock(
                cat_column=col,
                num_columns=["lp_price"],
                aggs=["mean"],
                derived=["ratio"],
            )
            for col in LP_RATIO_COLUMNS
        }
        self._output_columns = []

    def _apply_transformations(self, df: pl.DataFrame, is_train: bool) -> pl.DataFrame:
        """各種変換を適用"""
        # 1. 土地形状PCA
        if is_train:
            shape_pca_result = self._shape_pca.fit(df)
        else:
            shape_pca_result = self._shape_pca.transform(df)
        original_col = shape_pca_result.columns[0]
        df = df.with_columns(shape_pca_result[original_col].alias("lp_shape_pca"))

        # 2. 道路状況SVD
        if is_train:
            road_svd_result = self._road_svd.fit(df)
        else:
            road_svd_result = self._road_svd.transform(df)
        df = df.with_columns(road_svd_result.get_columns())

        # 3. 利用現況LE
        if is_train:
            current_use_result = self._current_use_le.fit(df)
        else:
            current_use_result = self._current_use_le.transform(df)
        df = df.with_columns(current_use_result["lp_current_use_le"])

        # 4. 価格時系列比率
        df = df.with_columns([
            pl.when(pl.col("lp_price_3y_ago") > 0)
            .then(pl.col("lp_price_1y_ago") / pl.col("lp_price_3y_ago"))
            .otherwise(None).alias("lp_ratio_1to3"),
            pl.when(pl.col("lp_price_5y_ago") > 0)
            .then(pl.col("lp_price_3y_ago") / pl.col("lp_price_5y_ago"))
            .otherwise(None).alias("lp_ratio_3to5"),
        ])

        # 5. 地積（log変換）
        df = df.with_columns(
            pl.when(pl.col("lp_land_area") > 0)
            .then(pl.col("lp_land_area").log())
            .otherwise(None).alias("lp_land_area_log")
        )

        # 6. 防火地域LE
        if is_train:
            fire_zone_result = self._fire_zone_le.fit(df)
        else:
            fire_zone_result = self._fire_zone_le.transform(df)
        df = df.with_columns(fire_zone_result["lp_fire_zone"].alias("lp_fire_zone_le"))

        # 7. 駅距離比率 (walk_distance1 / lp_station_dist)
        df = df.with_columns(
            pl.when(pl.col("lp_station_dist") > 0)
            .then(pl.col("walk_distance1") / pl.col("lp_station_dist"))
            .otherwise(None).alias("station_dist_ratio")
        )

        return df

    def _apply_ratio_blocks(self, df: pl.DataFrame, is_train: bool) -> pl.DataFrame:
        """カテゴリ別地価比率を計算"""
        ratio_results = []
        for col, block in self._lp_ratio_blocks.items():
            if is_train:
                result = block.fit(df)
            else:
                result = block.transform(df)
            ratio_results.append(result)
        return pl.concat(ratio_results, how="horizontal")

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        """trainデータで地価公示特徴量を生成（データは既に結合済み）"""
        df_with_features = self._apply_transformations(input_df, is_train=True)
        result_base = df_with_features.select(self.LP_BASE_COLUMNS)
        result_ratio = self._apply_ratio_blocks(df_with_features, is_train=True)
        result = pl.concat([result_base, result_ratio], how="horizontal")
        self._output_columns = list(result.columns)
        self._fitted = True
        return result

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """testデータで地価公示特徴量を生成（データは既に結合済み）"""
        if not self._fitted:
            raise RuntimeError("LandpriceFeatureBlock: fit()を先に実行してください")
        df_with_features = self._apply_transformations(input_df, is_train=False)
        result_base = df_with_features.select(self.LP_BASE_COLUMNS)
        result_ratio = self._apply_ratio_blocks(df_with_features, is_train=False)
        return pl.concat([result_base, result_ratio], how="horizontal")


class Exp010FeatureBlock(BaseBlock):
    """
    exp010固有特徴量を生成するBlock

    - lp_area_value: 土地価値目安
    - area_age_category: 面積×築年数カテゴリ
    - area_age_cat_te_*: カテゴリ×属性TE
    """

    def __init__(self, cv: list = None):
        super().__init__()
        self.cv = cv
        self._cat_te_block = None
        self._output_columns = []

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        """trainデータでexp010固有特徴量を生成（lp_priceは既に結合済み）"""
        # lp_area_value（土地価値目安）
        df_with_features = add_lp_area_value(input_df)

        # area_age_category（面積×築年数カテゴリ）
        df_with_features = add_area_age_category(df_with_features)

        # 基本特徴量
        result_basic = df_with_features.select(['lp_area_value', 'area_age_category'])

        # カテゴリ×属性TE
        self._cat_te_block = AreaAgeCategoryTEBlock(
            attr_columns=['addr1_1', 'land_youto'],
            cv=self.cv
        )
        result_te = self._cat_te_block.fit(df_with_features, y=y)

        result = pl.concat([result_basic, result_te], how="horizontal")
        self._output_columns = list(result.columns)
        self._fitted = True
        return result

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """testデータでexp010固有特徴量を生成（lp_priceは既に結合済み）"""
        if not self._fitted:
            raise RuntimeError("Exp010FeatureBlock: fit()を先に実行してください")

        df_with_features = add_lp_area_value(input_df)
        df_with_features = add_area_age_category(df_with_features)
        result_basic = df_with_features.select(['lp_area_value', 'area_age_category'])
        result_te = self._cat_te_block.transform(df_with_features)
        return pl.concat([result_basic, result_te], how="horizontal")


# =============================================================================
# パイプライン構築関数
# =============================================================================

def create_pipeline(
    cv_splits: list = None,
    feature_params: dict = None,
) -> FeaturePipeline:
    """
    exp010用の特徴量パイプラインを構築

    Args:
        cv_splits: CVのfold情報（TargetEncoding用）
        feature_params: 特徴量パラメータ（experiment.yamlのfeaturesセクション）
                       省略時はDEFAULT_FEATURE_PARAMSを使用

    Returns:
        構成済みのFeaturePipeline
    """
    # パラメータ設定
    params = DEFAULT_FEATURE_PARAMS.copy()
    if feature_params:
        # experiment.yamlの構造からフラットな辞書に変換
        if "tfidf" in feature_params:
            params["tfidf_max_features"] = feature_params["tfidf"].get("max_features", params["tfidf_max_features"])
        if "geo_pca" in feature_params:
            params["geo_pca_components"] = feature_params["geo_pca"].get("n_components", params["geo_pca_components"])
        if "building_tag_svd" in feature_params:
            params["building_tag_svd_dim"] = feature_params["building_tag_svd"].get("n_components", params["building_tag_svd_dim"])
        if "unit_tag_svd" in feature_params:
            params["unit_tag_svd_dim"] = feature_params["unit_tag_svd"].get("n_components", params["unit_tag_svd_dim"])
        if "statuses_svd" in feature_params:
            params["statuses_svd_dim"] = feature_params["statuses_svd"].get("n_components", params["statuses_svd_dim"])
        if "reform_svd" in feature_params:
            params["reform_svd_dim"] = feature_params["reform_svd"].get("n_components", params["reform_svd_dim"])
        if "postal_te" in feature_params:
            params["post_full_min_count"] = feature_params["postal_te"].get("post_full_min_count", params["post_full_min_count"])
        if "landprice" in feature_params:
            params["landprice_road_svd_dim"] = feature_params["landprice"].get("road_svd_dim", params["landprice_road_svd_dim"])
        # training.seedがあればそれを使用
        if "random_seed" in feature_params:
            params["random_seed"] = feature_params["random_seed"]

    pipeline = FeaturePipeline()

    # -------------------------------------------------------------------------
    # 1. 数値特徴量
    # -------------------------------------------------------------------------
    pipeline.add_block(
        name="NumericBlock",
        block=NumericBlock(columns=NUMERIC_FEATURES),
        input_columns=NUMERIC_FEATURES,
        description="基本数値特徴量をそのまま使用"
    )

    # -------------------------------------------------------------------------
    # 2. ターゲットエンコーディング
    # -------------------------------------------------------------------------
    te_rename_map = {f'TE_{col}': f'{col}_te' for col in TARGET_ENCODING_COLUMNS}
    pipeline.add_block(
        name="TargetEncodingBlock",
        block=RenameBlock(
            TargetEncodingBlock(columns=TARGET_ENCODING_COLUMNS, cv=cv_splits),
            rename_map=te_rename_map
        ),
        input_columns=TARGET_ENCODING_COLUMNS,
        description="OOFターゲットエンコーディング"
    )

    # -------------------------------------------------------------------------
    # 3. 郵便番号TE
    # -------------------------------------------------------------------------
    pipeline.add_block(
        name="PostalCodeTEBlock",
        block=PostalCodeTEBlock(min_count=params["post_full_min_count"], cv=cv_splits),
        input_columns=['post1', 'post_full'],
        description="郵便番号TE（3桁 + 7桁）"
    )

    # -------------------------------------------------------------------------
    # 4. カウントエンコーディング
    # -------------------------------------------------------------------------
    count_rename_map = {col: f'{col}_count' for col in COUNT_ENCODING_COLUMNS}
    pipeline.add_block(
        name="CountEncodingBlock",
        block=RenameBlock(
            CountEncodingBlock(columns=COUNT_ENCODING_COLUMNS),
            rename_map=count_rename_map
        ),
        input_columns=COUNT_ENCODING_COLUMNS,
        description="出現回数エンコーディング"
    )

    # -------------------------------------------------------------------------
    # 5. ラベルエンコーディング（路線・駅名）
    # -------------------------------------------------------------------------
    pipeline.add_block(
        name="LabelEncodingBlock",
        block=LabelEncodingBlock(columns=ROUTE_LE_COLUMNS),
        input_columns=ROUTE_LE_COLUMNS,
        description="路線・駅名のラベルエンコーディング"
    )

    # -------------------------------------------------------------------------
    # 6. TF-IDF
    # -------------------------------------------------------------------------
    pipeline.add_block(
        name="TfidfBlock",
        block=TfidfBlock(
            columns=TFIDF_TEXT_COLUMNS,
            max_features=params["tfidf_max_features"],
            separator=' ',
            prefix='tfidf',
        ),
        input_columns=TFIDF_TEXT_COLUMNS,
        description="交通アクセステキストのTF-IDF"
    )

    # -------------------------------------------------------------------------
    # 7. Geo PCA
    # -------------------------------------------------------------------------
    pipeline.add_block(
        name="GeoPCABlock",
        block=PCABlock(
            columns=GEO_COLUMNS,
            n_components=params["geo_pca_components"],
            standardize=True,
            handle_missing='mean',
            random_state=params["random_seed"],
            prefix='geo_pca',
        ),
        input_columns=GEO_COLUMNS,
        description="緯度経度4次元→2次元PCA"
    )

    # -------------------------------------------------------------------------
    # 8. タグSVD（building）
    # -------------------------------------------------------------------------
    pipeline.add_block(
        name="BuildingTagSVD",
        block=MultiHotSVDBlock(
            column='building_tag_id',
            n_components=params["building_tag_svd_dim"],
            separator='/',
            prefix='building_tag',
            random_state=params["random_seed"],
        ),
        input_columns=['building_tag_id'],
        description="建物タグのSVD圧縮"
    )

    # -------------------------------------------------------------------------
    # 9. タグSVD（unit）
    # -------------------------------------------------------------------------
    pipeline.add_block(
        name="UnitTagSVD",
        block=MultiHotSVDBlock(
            column='unit_tag_id',
            n_components=params["unit_tag_svd_dim"],
            separator='/',
            prefix='unit_tag',
            random_state=params["random_seed"],
        ),
        input_columns=['unit_tag_id'],
        description="部屋タグのSVD圧縮"
    )

    # -------------------------------------------------------------------------
    # 10. ステータスSVD
    # -------------------------------------------------------------------------
    pipeline.add_block(
        name="StatusesSVD",
        block=MultiHotSVDBlock(
            column='statuses',
            n_components=params["statuses_svd_dim"],
            separator='/',
            prefix='statuses',
            random_state=params["random_seed"],
        ),
        input_columns=['statuses'],
        description="ステータスコードのSVD圧縮（143コード→30次元）"
    )

    # -------------------------------------------------------------------------
    # 11. リフォームSVD
    # -------------------------------------------------------------------------
    pipeline.add_block(
        name="ReformSVD",
        block=MultiColumnMultiHotSVDBlock(
            columns=['reform_wet_area', 'reform_interior'],
            prefixes=['wet', 'int'],
            n_components=params["reform_svd_dim"],
            separator='/',
            output_prefix='reform',
            random_state=params["random_seed"],
        ),
        input_columns=['reform_wet_area', 'reform_interior'],
        description="リフォーム情報のSVD圧縮"
    )

    # -------------------------------------------------------------------------
    # 12. 面積地域平均比率
    # -------------------------------------------------------------------------
    pipeline.add_block(
        name="AreaRegionalRatioBlock",
        block=AreaRegionalRatioBlock(cv=cv_splits),
        input_columns=['house_area', 'snapshot_land_area', 'unit_area', 'addr1_1', 'addr1_2'],
        description="面積の地域平均比率"
    )

    # -------------------------------------------------------------------------
    # 13. 距離系地域平均比率
    # -------------------------------------------------------------------------
    pipeline.add_block(
        name="DistanceRegionalRatioBlock",
        block=DistanceRegionalRatioBlock(cv=cv_splits),
        input_columns=['walk_time1', 'total_access_time1', 'addr1_1', 'addr1_2'],
        description="距離・時間の地域平均比率（4次元）"
    )

    # -------------------------------------------------------------------------
    # 14. 地価公示特徴量（preprocessing.pyで既に結合済み）
    # -------------------------------------------------------------------------
    pipeline.add_block(
        name="LandpriceFeatureBlock",
        block=LandpriceFeatureBlock(
            road_svd_dim=params["landprice_road_svd_dim"],
            random_seed=params["random_seed"],
        ),
        input_columns=['lp_price', 'lp_land_area', 'lp_station_dist', 'lp_fire_zone', 'walk_distance1',
                       'post1', 'post_full', 'addr1_1', 'addr1_2',
                       'bukken_type', 'land_youto', 'land_toshi', 'building_age_bin', 'rosen_name1'],
        description="地価公示特徴量（43次元）"
    )

    # -------------------------------------------------------------------------
    # 15. exp010固有特徴量
    # -------------------------------------------------------------------------
    pipeline.add_block(
        name="Exp010FeatureBlock",
        block=Exp010FeatureBlock(cv=cv_splits),
        input_columns=['house_area', 'building_age', 'lp_price', 'addr1_1', 'land_youto'],
        description="exp010固有（lp_area_value, area_age_category, カテゴリTE）"
    )

    return pipeline
