"""
exp010 特徴量モジュール

実験固有の特徴量変換器・関数を提供。
基本変換器（TF-IDF, LabelEncoding, PCA, SVD等）は04_src/featuresを直接使用。
地価公示データ結合はpreprocessing.pyで実行済み。

クラス:
- PostalCodeTEBlock: 郵便番号TE（階層フォールバック付き）
- AreaRegionalRatioBlock: 面積地域平均比率
- AreaAgeCategoryTEBlock: 面積×築年数カテゴリTE

関数:
- add_lp_area_value: 土地価値目安（lp_price × house_area）
- add_area_age_category: 面積×築年数カテゴリ（0,1,2の3段階）
"""

from pathlib import Path
from typing import List, Set

import numpy as np
import polars as pl
from sklearn.model_selection import KFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "04_src"))

from features.base import BaseBlock
from features.blocks.encoding import TargetEncodingBlock
from features.blocks.aggregation import GroupByAggBlock
from features import MultiKeyTEBlock

# デフォルト値（experiment.yamlから上書き可能）
DEFAULT_POST_FULL_MIN_COUNT = 30
DEFAULT_AREA_AGE_CAT2_AREA_THRESHOLD = 150
DEFAULT_AREA_AGE_CAT2_AGE_THRESHOLD = 45
DEFAULT_AREA_AGE_CAT1_AREA_THRESHOLD = 100
DEFAULT_AREA_AGE_CAT1_AGE_THRESHOLD = 35
DEFAULT_RANDOM_SEED = 42
DEFAULT_N_SPLITS = 3


# =============================================================================
# 定数定義
# =============================================================================

# 道路関連のカテゴリカラム
ROAD_CATEGORY_COLUMNS = ["lp_road_type", "lp_road_direction", "lp_side_road"]

# 利用現況の上位カテゴリ（trainから集計済み）
CURRENT_USE_TOP_CATEGORIES = [
    "住宅", "店舗兼住宅", "店舗", "店舗兼共同住宅", "店舗兼事務所",
    "事務所", "共同住宅", "工場", "倉庫", "事務所兼住宅",
]

# 地価公示価格の平均・比率を計算するカテゴリカラム
LP_RATIO_COLUMNS = [
    'post1', 'post_full', 'addr1_1', 'addr1_2', 'bukken_type',
    'land_youto', 'land_toshi', 'building_age_bin', 'rosen_name1',
]


# =============================================================================
# exp010固有特徴量関数
# =============================================================================

def add_lp_area_value(df: pl.DataFrame) -> pl.DataFrame:
    """
    土地価値目安を計算して追加

    地価公示価格（円/㎡）と物件面積（㎡）を掛け合わせて、
    土地の概算価値を算出する。これにより価格の下限目安を特徴量として持てる。

    計算式: lp_area_value = lp_price × house_area

    Args:
        df: 入力DataFrame（lp_price, house_areaカラムを含む）

    Returns:
        pl.DataFrame: lp_area_valueカラムが追加されたDataFrame

    Note:
        - lp_priceまたはhouse_areaがnullの場合、lp_area_valueもnullになる
        - lp_priceは地価公示データからの結合で取得済みを想定
    """
    return df.with_columns(
        (pl.col('lp_price') * pl.col('house_area')).alias('lp_area_value')
    )


def add_area_age_category(
    df: pl.DataFrame,
    cat2_area: int = DEFAULT_AREA_AGE_CAT2_AREA_THRESHOLD,
    cat2_age: int = DEFAULT_AREA_AGE_CAT2_AGE_THRESHOLD,
    cat1_area: int = DEFAULT_AREA_AGE_CAT1_AREA_THRESHOLD,
    cat1_age: int = DEFAULT_AREA_AGE_CAT1_AGE_THRESHOLD,
) -> pl.DataFrame:
    """
    面積×築年数カテゴリを追加

    予測困難度に基づいて3段階のカテゴリを付与:
    - カテゴリ2（最も予測困難）: 面積150㎡以上 かつ 築45年以上
      → MAPE 20.24%相当、低価格帯の外れ値が多い
    - カテゴリ1（中程度）: 面積100㎡以上 かつ 築35年以上（カテゴリ2除く）
      → MAPE 16.16%相当
    - カテゴリ0（標準）: その他
      → MAPE 11.82%相当

    Args:
        df: 入力DataFrame（house_area, building_ageカラムを含む）
        cat2_area: カテゴリ2の面積閾値（デフォルト: 150）
        cat2_age: カテゴリ2の築年数閾値（デフォルト: 45）
        cat1_area: カテゴリ1の面積閾値（デフォルト: 100）
        cat1_age: カテゴリ1の築年数閾値（デフォルト: 35）

    Returns:
        pl.DataFrame: area_age_categoryカラム（Int64、0/1/2）が追加されたDataFrame

    Note:
        - building_ageは築年数（年）として事前に計算済みを想定
        - 閾値はexperiment.yamlで設定可能
    """
    return df.with_columns(
        pl.when(
            (pl.col('house_area') >= cat2_area) &
            (pl.col('building_age') >= cat2_age)
        ).then(pl.lit(2))
        .when(
            (pl.col('house_area') >= cat1_area) &
            (pl.col('building_age') >= cat1_age)
        ).then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias('area_age_category')
    )


# =============================================================================
# 特徴量Blockクラス
# =============================================================================

class PostalCodeTEBlock(TargetEncodingBlock):
    """
    郵便番号ターゲットエンコーディング（OOF方式 + 階層フォールバック）

    TargetEncodingBlockを継承し、郵便番号専用の階層フォールバック機能を追加。

    2種類のTEを計算:
    - post1_te: 郵便番号上3桁のTE
    - post_full_te: 郵便番号7桁のTE

    post_full_teは出現回数が少ない場合（デフォルト30件未満）、
    より粗いpost1_teで補完することで安定した推定を実現。

    Attributes:
        min_count: post_fullのTE計算に必要な最低出現回数（デフォルト: 30）
        valid_post_fulls: min_count以上出現した郵便番号7桁のセット
    """

    def __init__(self, min_count: int = DEFAULT_POST_FULL_MIN_COUNT, cv: list = None):
        super().__init__(columns=['post1', 'post_full'], cv=cv)
        self.min_count = min_count
        self.valid_post_fulls: Set[str] = set()

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        if y is None:
            raise ValueError("PostalCodeTEBlock: ターゲット変数(y)は必須です")

        # post_fullの出現回数を計算して有効な郵便番号を特定
        post_full_counts = input_df.group_by('post_full').agg(pl.count().alias('count'))
        valid_post_fulls_df = post_full_counts.filter(pl.col('count') >= self.min_count)
        self.valid_post_fulls = set(valid_post_fulls_df['post_full'].to_list())

        # 親クラスのfitでOOF TE計算（post1, post_full両方）
        te_result = super().fit(input_df, y)

        # カラム名を変換して階層フォールバック適用
        post1_te = te_result['TE_post1'].to_numpy()
        post_full_te_raw = te_result['TE_post_full'].to_numpy()
        post_full_values = input_df['post_full'].to_list()

        # 階層フォールバック: 出現回数が少ないpost_fullはpost1_teで補完
        post_full_te = np.where(
            [pf in self.valid_post_fulls for pf in post_full_values],
            post_full_te_raw,
            post1_te
        )

        return pl.DataFrame({
            'post1_te': post1_te,
            'post_full_te': post_full_te,
        })

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"{self.__class__.__name__}: fit()を先に実行してください")

        te_result = super().transform(input_df)

        post1_te = te_result['TE_post1'].to_numpy()
        post_full_te_raw = te_result['TE_post_full'].to_numpy()
        post_full_values = input_df['post_full'].to_list()

        post_full_te = np.where(
            [pf in self.valid_post_fulls for pf in post_full_values],
            post_full_te_raw,
            post1_te
        )

        return pl.DataFrame({
            'post1_te': post1_te,
            'post_full_te': post_full_te,
        })

    def get_stats(self) -> dict:
        return {
            'post1_unique': len(self.mapping_df_.get('post1', {})) if self.mapping_df_ else 0,
            'post_full_unique': len(self.mapping_df_.get('post_full', {})) if self.mapping_df_ else 0,
            'valid_post_fulls': len(self.valid_post_fulls),
            'global_mean': self.y_mean_,
        }


class AreaRegionalRatioBlock(BaseBlock):
    """
    面積の地域平均比率特徴量

    3種類の面積カラム（house_area, snapshot_land_area, unit_area）について
    都道府県・市区町村単位での平均値との比率を計算。

    比率 = 物件の面積 / 地域平均面積
    - 比率 > 1: 地域平均より広い（相対的に広い物件）
    - 比率 < 1: 地域平均より狭い（相対的に狭い物件）

    内部で04_srcのGroupByAggBlockを使用。
    """

    AREA_COLUMNS = ['house_area', 'snapshot_land_area', 'unit_area']
    REGION_COLUMNS = {
        'addr1_1': 'pref',   # 都道府県
        'addr1_2': 'city',   # 市区町村
    }

    def __init__(self, cv: list = None):
        super().__init__()
        # 内部Block（6個: 3面積 × 2地域）
        self._blocks = {}
        for area_col in self.AREA_COLUMNS:
            for region_col, suffix in self.REGION_COLUMNS.items():
                key = (area_col, suffix)
                self._blocks[key] = GroupByAggBlock(
                    cat_column=region_col,
                    num_columns=[area_col],
                    aggs=["mean"],
                    derived=["ratio"],
                )

        # 出力特徴量名
        self.feature_names = []
        for col in self.AREA_COLUMNS:
            self.feature_names.append(f'{col}_pref_ratio')
            self.feature_names.append(f'{col}_city_ratio')

    def fit(self, input_df: pl.DataFrame, y=None) -> pl.DataFrame:
        results = {}

        for (area_col, suffix), block in self._blocks.items():
            result = block.fit(input_df)
            ratio_col = [c for c in result.columns if c.endswith("_ratio")][0]
            new_name = f"{area_col}_{suffix}_ratio"
            results[new_name] = result[ratio_col]

        self._fitted = True
        return pl.DataFrame(results).select(self.feature_names)

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"{self.__class__.__name__}: fit()を先に実行してください")

        results = {}

        for (area_col, suffix), block in self._blocks.items():
            result = block.transform(input_df)
            ratio_col = [c for c in result.columns if c.endswith("_ratio")][0]
            new_name = f"{area_col}_{suffix}_ratio"
            results[new_name] = result[ratio_col]

        return pl.DataFrame(results).select(self.feature_names)


class AreaAgeCategoryTEBlock(MultiKeyTEBlock):
    """
    面積×築年数カテゴリ × 属性のターゲットエンコーディング

    MultiKeyTEBlockを継承し、primary_key='area_age_category'に固定。
    area_age_categoryと他の属性（都道府県、用途地域など）を組み合わせた
    複合キーでターゲットエンコーディングを行う。

    これにより、カテゴリ毎の地域差・用途地域差を捉える特徴量を生成。
    例: 「カテゴリ2（広面積×築古）の東京都」の平均価格を特徴量化
    """

    PRIMARY_KEY = 'area_age_category'
    OUTPUT_PREFIX = 'area_age_cat_te'

    def __init__(
        self,
        attr_columns: List[str],
        cv: list = None,
        n_splits: int = DEFAULT_N_SPLITS,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ):
        cv_list = cv if cv is not None else []
        super().__init__(
            primary_key=self.PRIMARY_KEY,
            attr_columns=attr_columns,
            cv=cv_list,
            output_prefix=self.OUTPUT_PREFIX,
        )
        self._cv_auto_generate = cv is None
        self._n_splits = n_splits
        self._random_seed = random_seed

    def fit(self, input_df: pl.DataFrame, y=None) -> pl.DataFrame:
        if self._cv_auto_generate and len(self.cv) == 0:
            self.cv = list(KFold(
                n_splits=self._n_splits,
                shuffle=True,
                random_state=self._random_seed
            ).split(input_df))
        return super().fit(input_df, y)

    @property
    def category_means(self) -> dict:
        """後方互換性のためのエイリアス"""
        return self.primary_key_means

    def get_stats(self) -> dict:
        stats = super().get_stats()
        return {
            'n_combinations': stats['n_combinations'],
            'category_means': stats['primary_key_means'],
            'global_mean': stats['global_mean'],
        }


class DistanceRegionalRatioBlock(BaseBlock):
    """
    距離・時間の地域平均比率特徴量

    距離・時間系カラム（walk_time1, total_access_time1）について
    都道府県・市区町村単位での平均値との比率を計算。

    比率 = 物件の値 / 地域平均値
    - 比率 > 1: 地域平均より遠い/時間がかかる
    - 比率 < 1: 地域平均より近い/時間が短い

    内部で04_srcのGroupByAggBlockを使用。
    """

    DISTANCE_COLUMNS = ['walk_time1', 'total_access_time1']
    REGION_COLUMNS = {
        'addr1_1': 'pref',   # 都道府県
        'addr1_2': 'city',   # 市区町村
    }

    def __init__(self, cv: list = None):
        super().__init__()
        # 内部Block（4個: 2距離 × 2地域）
        self._blocks = {}
        for dist_col in self.DISTANCE_COLUMNS:
            for region_col, suffix in self.REGION_COLUMNS.items():
                key = (dist_col, suffix)
                self._blocks[key] = GroupByAggBlock(
                    cat_column=region_col,
                    num_columns=[dist_col],
                    aggs=["mean"],
                    derived=["ratio"],
                )

        # 出力特徴量名
        self.feature_names = []
        for col in self.DISTANCE_COLUMNS:
            self.feature_names.append(f'{col}_pref_ratio')
            self.feature_names.append(f'{col}_city_ratio')

    def fit(self, input_df: pl.DataFrame, y=None) -> pl.DataFrame:
        results = {}

        for (dist_col, suffix), block in self._blocks.items():
            result = block.fit(input_df)
            ratio_col = [c for c in result.columns if c.endswith("_ratio")][0]
            new_name = f"{dist_col}_{suffix}_ratio"
            results[new_name] = result[ratio_col]

        self._fitted = True
        return pl.DataFrame(results).select(self.feature_names)

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"{self.__class__.__name__}: fit()を先に実行してください")

        results = {}

        for (dist_col, suffix), block in self._blocks.items():
            result = block.transform(input_df)
            ratio_col = [c for c in result.columns if c.endswith("_ratio")][0]
            new_name = f"{dist_col}_{suffix}_ratio"
            results[new_name] = result[ratio_col]

        return pl.DataFrame(results).select(self.feature_names)


# =============================================================================
# エクスポート用
# =============================================================================

__all__ = [
    # exp010固有関数
    "add_lp_area_value",
    "add_area_age_category",
    # Blockクラス
    "PostalCodeTEBlock",
    "AreaRegionalRatioBlock",
    "AreaAgeCategoryTEBlock",
    "DistanceRegionalRatioBlock",
    # 定数
    "CURRENT_USE_TOP_CATEGORIES",
    "ROAD_CATEGORY_COLUMNS",
    "LP_RATIO_COLUMNS",
]
