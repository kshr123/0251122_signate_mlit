"""
exp012 特徴量パイプライン

exp011のパイプラインをベースに、密度特徴量を追加。

使用例:
    pipeline = create_pipeline(cv_splits, config)
    X_train = pipeline.fit_transform(train, y_train)
    X_test = pipeline.transform(test)

    # 変換内容を確認
    print(pipeline.summary())
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "04_src"))
# exp012のcodeディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl

# 04_srcの共通コンポーネント
from features.pipeline import FeaturePipeline
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
from features import TopNCategoryLEBlock

# exp012の特徴量
from exp012_features import (
    DensityBinBlock,
    AreaAgeCategoryBlock,
    PostalCountBlock,
)

# カラム定義
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
    # exp012追加
    "density_percentile_boundaries": [0, 10, 30, 70, 100],
}

# 地価公示関連定数（exp011から継承）
CURRENT_USE_TOP_CATEGORIES = [
    '住宅', '共同住宅', '店舗', '事務所', '駐車場', '店舗兼住宅',
    '作業場', '倉庫', '事務所兼住宅', '工場'
]

LP_RATIO_COLUMNS = [
    'post1', 'post_full', 'addr1_1', 'addr1_2',
    'bukken_type', 'land_youto', 'land_toshi', 'building_age_bin', 'rosen_name1'
]


# =============================================================================
# 地価公示特徴量用の特殊Block（exp011から継承）
# =============================================================================

class LandpriceFeatureBlock(BaseBlock):
    """
    地価公示データの特徴量生成Block

    preprocessing.pyで地価公示データは既に結合済み。
    このBlockは結合済みデータから特徴量を生成する。
    """

    def __init__(
        self,
        road_svd_dim: int = DEFAULT_FEATURE_PARAMS["landprice_road_svd_dim"],
        random_seed: int = DEFAULT_FEATURE_PARAMS["random_seed"],
    ):
        super().__init__()
        self._road_svd_dim = road_svd_dim
        self._random_seed = random_seed

        # 出力特徴量カラム
        self.LP_BASE_COLUMNS = [
            "lp_shape_pca",
            *[f"lp_road_svd_{i}" for i in range(road_svd_dim)],
            "lp_road_width",
            "lp_current_use_le",
            "lp_price",
            "lp_change_rate",
            "lp_nearest_dist",
            "lp_ratio_1to3",
            "lp_ratio_3to5",
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
        self._road_svd = MultiColumnOneHotSVDBlock(
            columns=["lp_road_type", "lp_road_direction", "lp_side_road"],
            n_components=road_svd_dim,
            output_prefix="lp_road",
            null_value="_",
            random_state=random_seed,
        )
        self._current_use_le = TopNCategoryLEBlock(
            column="lp_current_use",
            top_categories=CURRENT_USE_TOP_CATEGORIES,
            other_label="その他",
            output_column="lp_current_use_le",
        )
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

        # 7. 駅距離比率
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
        df_with_features = self._apply_transformations(input_df, is_train=True)
        result_base = df_with_features.select(self.LP_BASE_COLUMNS)
        result_ratio = self._apply_ratio_blocks(df_with_features, is_train=True)
        result = pl.concat([result_base, result_ratio], how="horizontal")
        self._output_columns = list(result.columns)
        self._fitted = True
        return result

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        if not self._fitted:
            raise RuntimeError("LandpriceFeatureBlock: fit()を先に実行してください")
        df_with_features = self._apply_transformations(input_df, is_train=False)
        result_base = df_with_features.select(self.LP_BASE_COLUMNS)
        result_ratio = self._apply_ratio_blocks(df_with_features, is_train=False)
        return pl.concat([result_base, result_ratio], how="horizontal")


# =============================================================================
# exp010/011継承の特徴量Block
# =============================================================================

class AreaRegionalRatioBlock(BaseBlock):
    """面積の地域平均比率Block"""

    def __init__(self, cv: list = None):
        super().__init__()
        self.cv = cv
        self._ratio_blocks = {}
        self._output_columns = []

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        results = []
        for num_col in ['house_area', 'snapshot_land_area', 'unit_area']:
            for cat_col, suffix in [('addr1_1', 'pref'), ('addr1_2', 'city')]:
                block = GroupByAggBlock(
                    cat_column=cat_col,
                    num_columns=[num_col],
                    aggs=["mean"],
                    derived=["ratio"],
                )
                result = block.fit(input_df)
                # リネーム
                result = result.rename({result.columns[0]: f"{num_col}_{suffix}_ratio"})
                results.append(result)
                self._ratio_blocks[(num_col, cat_col)] = block
        result = pl.concat(results, how="horizontal")
        self._output_columns = list(result.columns)
        self._fitted = True
        return result

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        if not self._fitted:
            raise RuntimeError("fit()を先に実行してください")
        results = []
        for (num_col, cat_col), block in self._ratio_blocks.items():
            suffix = 'pref' if cat_col == 'addr1_1' else 'city'
            result = block.transform(input_df)
            result = result.rename({result.columns[0]: f"{num_col}_{suffix}_ratio"})
            results.append(result)
        return pl.concat(results, how="horizontal")


class DistanceRegionalRatioBlock(BaseBlock):
    """距離・時間の地域平均比率Block"""

    def __init__(self, cv: list = None):
        super().__init__()
        self.cv = cv
        self._ratio_blocks = {}
        self._output_columns = []

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        results = []
        for num_col in ['walk_time1', 'total_access_time1']:
            for cat_col, suffix in [('addr1_1', 'pref'), ('addr1_2', 'city')]:
                block = GroupByAggBlock(
                    cat_column=cat_col,
                    num_columns=[num_col],
                    aggs=["mean"],
                    derived=["ratio"],
                )
                result = block.fit(input_df)
                result = result.rename({result.columns[0]: f"{num_col}_{suffix}_ratio"})
                results.append(result)
                self._ratio_blocks[(num_col, cat_col)] = block
        result = pl.concat(results, how="horizontal")
        self._output_columns = list(result.columns)
        self._fitted = True
        return result

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        if not self._fitted:
            raise RuntimeError("fit()を先に実行してください")
        results = []
        for (num_col, cat_col), block in self._ratio_blocks.items():
            suffix = 'pref' if cat_col == 'addr1_1' else 'city'
            result = block.transform(input_df)
            result = result.rename({result.columns[0]: f"{num_col}_{suffix}_ratio"})
            results.append(result)
        return pl.concat(results, how="horizontal")


class PostalCodeTEBlock(BaseBlock):
    """郵便番号TE（3桁 + 7桁）"""

    def __init__(self, min_count: int = 30, cv: list = None):
        super().__init__()
        self._min_count = min_count
        self.cv = cv
        self._te_post1 = TargetEncodingBlock(columns=['post1'], cv=cv)
        self._te_post_full = TargetEncodingBlock(columns=['post_full'], cv=cv)
        self._output_columns = []

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        result1 = self._te_post1.fit(input_df, y)
        result1 = result1.rename({'TE_post1': 'post1_te'})

        result_full = self._te_post_full.fit(input_df, y)
        result_full = result_full.rename({'TE_post_full': 'post_full_te'})

        result = pl.concat([result1, result_full], how="horizontal")
        self._output_columns = list(result.columns)
        self._fitted = True
        return result

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        if not self._fitted:
            raise RuntimeError("fit()を先に実行してください")
        result1 = self._te_post1.transform(input_df)
        result1 = result1.rename({'TE_post1': 'post1_te'})
        result_full = self._te_post_full.transform(input_df)
        result_full = result_full.rename({'TE_post_full': 'post_full_te'})
        return pl.concat([result1, result_full], how="horizontal")


class AreaAgeCategoryTEBlock(BaseBlock):
    """面積×築年数カテゴリ × 属性のTE"""

    def __init__(self, attr_columns: list = None, cv: list = None):
        super().__init__()
        self.attr_columns = attr_columns or ['addr1_1', 'land_youto']
        self.cv = cv
        self._te_blocks = {}
        self._output_columns = []

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        results = []
        for attr_col in self.attr_columns:
            # area_age_category × attr_col の組み合わせでTE
            combined_col = f"area_age_cat_{attr_col}"
            df_with_combined = input_df.with_columns(
                (pl.col('area_age_category').cast(pl.Utf8) + '_' + pl.col(attr_col).cast(pl.Utf8))
                .alias(combined_col)
            )
            te_block = TargetEncodingBlock(columns=[combined_col], cv=self.cv)
            result = te_block.fit(df_with_combined, y)
            result = result.rename({f'TE_{combined_col}': f'area_age_cat_te_{attr_col.replace("addr1_", "")}'})
            results.append(result)
            self._te_blocks[attr_col] = te_block
        result = pl.concat(results, how="horizontal")
        self._output_columns = list(result.columns)
        self._fitted = True
        return result

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        if not self._fitted:
            raise RuntimeError("fit()を先に実行してください")
        results = []
        for attr_col, te_block in self._te_blocks.items():
            combined_col = f"area_age_cat_{attr_col}"
            df_with_combined = input_df.with_columns(
                (pl.col('area_age_category').cast(pl.Utf8) + '_' + pl.col(attr_col).cast(pl.Utf8))
                .alias(combined_col)
            )
            result = te_block.transform(df_with_combined)
            result = result.rename({f'TE_{combined_col}': f'area_age_cat_te_{attr_col.replace("addr1_", "")}'})
            results.append(result)
        return pl.concat(results, how="horizontal")


def add_lp_area_value(df: pl.DataFrame) -> pl.DataFrame:
    """土地価値目安を追加"""
    return df.with_columns(
        (pl.col('lp_price') * pl.col('house_area')).alias('lp_area_value')
    )


# =============================================================================
# exp012固有の密度特徴量Block
# =============================================================================

class DensityFeatureBlock(BaseBlock):
    """
    密度特徴量を生成するBlock

    1. 郵便番号内物件数をカウント (post_full_count)
    2. パーセンタイルベースでビン分け (post_full_density_bin)
    3. ビン×ターゲットエンコーディング (post_full_density_bin_te)
    """

    def __init__(
        self,
        percentile_boundaries: list = None,
        cv: list = None,
    ):
        super().__init__()
        self._percentile_boundaries = percentile_boundaries or [0, 10, 30, 70, 100]
        self.cv = cv

        self._count_block = PostalCountBlock(
            column='post_full',
            output_column='post_full_count',
        )
        self._bin_block = DensityBinBlock(
            column='post_full_count',
            percentile_boundaries=self._percentile_boundaries,
            output_suffix='_density_bin',
        )
        self._te_block = None  # fit時に作成
        self._output_columns = []

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        # 1. カウント
        count_result = self._count_block.fit(input_df, y)

        # カウント結果をdfに追加
        df_with_count = input_df.with_columns(count_result.get_columns())

        # 2. ビン分け
        bin_result = self._bin_block.fit(df_with_count, y)

        # ビン結果をdfに追加
        df_with_bin = df_with_count.with_columns(bin_result.get_columns())

        # 3. ビン×TE
        bin_col = 'post_full_count_density_bin'
        self._te_block = TargetEncodingBlock(columns=[bin_col], cv=self.cv)
        te_result = self._te_block.fit(df_with_bin, y)
        te_result = te_result.rename({f'TE_{bin_col}': 'post_full_density_bin_te'})

        # 結果を結合
        result = pl.concat([count_result, bin_result, te_result], how="horizontal")
        self._output_columns = list(result.columns)
        self._fitted = True
        return result

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        if not self._fitted:
            raise RuntimeError("fit()を先に実行してください")

        # 1. カウント
        count_result = self._count_block.transform(input_df)
        df_with_count = input_df.with_columns(count_result.get_columns())

        # 2. ビン分け
        bin_result = self._bin_block.transform(df_with_count)
        df_with_bin = df_with_count.with_columns(bin_result.get_columns())

        # 3. ビン×TE
        bin_col = 'post_full_count_density_bin'
        te_result = self._te_block.transform(df_with_bin)
        te_result = te_result.rename({f'TE_{bin_col}': 'post_full_density_bin_te'})

        return pl.concat([count_result, bin_result, te_result], how="horizontal")


class Exp012FeatureBlock(BaseBlock):
    """
    exp012固有特徴量を生成するBlock

    - lp_area_value: 土地価値目安
    - area_age_category: 面積×築年数カテゴリ（4カテゴリ版）
    - area_age_cat_te_*: カテゴリ×属性TE
    """

    def __init__(self, config: dict = None, cv: list = None):
        super().__init__()
        self.config = config or {}
        self.cv = cv

        # 設定から閾値を取得
        area_age_cfg = self.config.get('exp010', {}).get('area_age_category', {})
        self._area_age_block = AreaAgeCategoryBlock(
            cat1_area=area_age_cfg.get('cat1_area_threshold', 100.0),
            cat1_age=area_age_cfg.get('cat1_age_threshold', 35),
            cat2_area=area_age_cfg.get('cat2_area_threshold', 150.0),
            cat2_age=area_age_cfg.get('cat2_age_threshold', 45),
            cat3_area=area_age_cfg.get('cat3_area_threshold', 200.0),
            cat3_age=area_age_cfg.get('cat3_age_threshold', 45),
        )
        self._cat_te_block = None
        self._output_columns = []

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        # lp_area_value
        df_with_features = add_lp_area_value(input_df)

        # area_age_category（4カテゴリ版）
        cat_result = self._area_age_block.fit(df_with_features, y)
        df_with_features = df_with_features.with_columns(cat_result.get_columns())

        # 基本特徴量
        result_basic = pl.concat([
            df_with_features.select(['lp_area_value']),
            cat_result
        ], how="horizontal")

        # カテゴリ×属性TE
        self._cat_te_block = AreaAgeCategoryTEBlock(
            attr_columns=['addr1_1', 'land_youto'],
            cv=self.cv
        )
        result_te = self._cat_te_block.fit(df_with_features, y)

        result = pl.concat([result_basic, result_te], how="horizontal")
        self._output_columns = list(result.columns)
        self._fitted = True
        return result

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        if not self._fitted:
            raise RuntimeError("fit()を先に実行してください")

        df_with_features = add_lp_area_value(input_df)
        cat_result = self._area_age_block.transform(df_with_features)
        df_with_features = df_with_features.with_columns(cat_result.get_columns())

        result_basic = pl.concat([
            df_with_features.select(['lp_area_value']),
            cat_result
        ], how="horizontal")
        result_te = self._cat_te_block.transform(df_with_features)
        return pl.concat([result_basic, result_te], how="horizontal")


# =============================================================================
# パイプライン構築関数
# =============================================================================

def create_pipeline(
    cv_splits: list = None,
    config: dict = None,
) -> FeaturePipeline:
    """
    exp012用の特徴量パイプラインを構築

    Args:
        cv_splits: CVのfold情報（TargetEncoding用）
        config: 実験設定（experiment.yaml全体）

    Returns:
        構成済みのFeaturePipeline
    """
    # パラメータ設定
    params = DEFAULT_FEATURE_PARAMS.copy()
    feature_params = config.get('features', {}) if config else {}

    if feature_params:
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

    # exp012設定
    exp012_cfg = config.get('exp012', {}) if config else {}
    density_cfg = exp012_cfg.get('postal_density', {})
    params["density_percentile_boundaries"] = density_cfg.get(
        'percentile_boundaries', params["density_percentile_boundaries"]
    )

    # training.seedがあればそれを使用
    training_cfg = config.get('training', {}) if config else {}
    params["random_seed"] = training_cfg.get('seed', params["random_seed"])

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
    # 14. 地価公示特徴量
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
    # 15. exp012固有特徴量（面積×築年数カテゴリ 4版）
    # -------------------------------------------------------------------------
    pipeline.add_block(
        name="Exp012FeatureBlock",
        block=Exp012FeatureBlock(config=config, cv=cv_splits),
        input_columns=['house_area', 'building_age', 'lp_price', 'addr1_1', 'land_youto'],
        description="exp012固有（lp_area_value, area_age_category 4版, カテゴリTE）"
    )

    # -------------------------------------------------------------------------
    # 16. 密度特徴量（exp012新規）
    # -------------------------------------------------------------------------
    pipeline.add_block(
        name="DensityFeatureBlock",
        block=DensityFeatureBlock(
            percentile_boundaries=params["density_percentile_boundaries"],
            cv=cv_splits,
        ),
        input_columns=['post_full'],
        description="郵便番号密度特徴量（count, bin, bin_te）"
    )

    return pipeline
