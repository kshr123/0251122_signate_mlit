"""地価公示データから特徴量を生成するモジュール"""

from typing import Optional, List, Dict, Tuple
import numpy as np
import polars as pl
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.sparse import csr_matrix


# =============================================
# 定数定義
# =============================================

# 道路関連のカテゴリカラム
ROAD_CATEGORY_COLUMNS = ["lp_road_type", "lp_road_direction", "lp_side_road"]

# SVD次元数（累積分散説明率91%）
ROAD_SVD_DIM = 13

# 利用現況の上位カテゴリ数
CURRENT_USE_TOP_N = 10

# 利用現況の上位カテゴリ（trainから集計済み）
CURRENT_USE_TOP_CATEGORIES = [
    "住宅",
    "店舗兼住宅",
    "店舗",
    "店舗兼共同住宅",
    "店舗兼事務所",
    "事務所",
    "共同住宅",
    "工場",
    "倉庫",
    "事務所兼住宅",
]


class LandpriceShapePCATransformer:
    """
    間口比率・奥行比率をPCAで1次元に圧縮

    処理内容:
    - 間口比率、奥行比率の2次元を1次元に圧縮
    - 土地形状の特徴を捉える
    """

    def __init__(self, n_components: int = 1):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=42)
        self.is_fitted = False

    def fit(self, df: pl.DataFrame) -> "LandpriceShapePCATransformer":
        """trainデータでPCAをfit"""
        X = df.select(["lp_frontage_ratio", "lp_depth_ratio"]).to_numpy()
        # 欠損値を平均で埋める
        X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
        self.pca.fit(X)
        self.is_fitted = True
        print(f"ShapePCA fitted: explained_variance_ratio={self.pca.explained_variance_ratio_}")
        return self

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """fitしてtransform（sklearn互換）"""
        return self.fit(df).transform(df)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """PCAで変換"""
        if not self.is_fitted:
            raise ValueError("Transformer is not fitted yet. Call fit() first.")

        X = df.select(["lp_frontage_ratio", "lp_depth_ratio"]).to_numpy()
        # 欠損値を平均で埋める
        X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
        X_pca = self.pca.transform(X)

        # DataFrameに追加
        result = df.with_columns([
            pl.Series(f"lp_shape_pca", X_pca[:, 0])
        ])

        return result


class LandpriceRoadSVDTransformer:
    """
    道路関連カテゴリをOne-hot + SVDで圧縮

    処理内容:
    - 前面道路区分、前面道路方位、側道状況をOne-hot encoding
    - SVDで13次元に圧縮
    """

    def __init__(self, n_components: int = ROAD_SVD_DIM):
        self.n_components = n_components
        self.encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.is_fitted = False

    def fit(self, df: pl.DataFrame) -> "LandpriceRoadSVDTransformer":
        """trainデータでEncoder + SVDをfit"""
        X = df.select(ROAD_CATEGORY_COLUMNS).to_numpy()
        # 欠損値を"_"で埋める
        X = np.where(X == None, "_", X)
        X = np.where(X == "", "_", X)

        # One-hot encoding
        X_onehot = self.encoder.fit_transform(X)
        print(f"RoadSVD One-hot shape: {X_onehot.shape}")

        # SVD
        self.svd.fit(X_onehot)
        cumsum = np.cumsum(self.svd.explained_variance_ratio_)
        print(f"RoadSVD fitted: cumulative_variance_ratio at {self.n_components}={cumsum[-1]:.3f}")
        self.is_fitted = True
        return self

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """fitしてtransform（sklearn互換）"""
        return self.fit(df).transform(df)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """SVDで変換"""
        if not self.is_fitted:
            raise ValueError("Transformer is not fitted yet. Call fit() first.")

        X = df.select(ROAD_CATEGORY_COLUMNS).to_numpy()
        # 欠損値を"_"で埋める
        X = np.where(X == None, "_", X)
        X = np.where(X == "", "_", X)

        # One-hot encoding
        X_onehot = self.encoder.transform(X)

        # SVD
        X_svd = self.svd.transform(X_onehot)

        # DataFrameに追加
        svd_columns = [pl.Series(f"lp_road_svd_{i}", X_svd[:, i]) for i in range(self.n_components)]
        result = df.with_columns(svd_columns)

        return result


class LandpriceCurrentUseLETransformer:
    """
    利用現況をLabel Encoding（上位N + その他）

    処理内容:
    - 上位10カテゴリはそのままエンコード
    - それ以外は「その他」にまとめる
    """

    def __init__(self, top_categories: Optional[List[str]] = None):
        self.top_categories = top_categories or CURRENT_USE_TOP_CATEGORIES
        self.le = LabelEncoder()
        self.is_fitted = False

    def fit(self, df: pl.DataFrame) -> "LandpriceCurrentUseLETransformer":
        """trainデータでLabelEncoderをfit"""
        # 上位カテゴリ + その他
        categories = self.top_categories + ["その他"]
        self.le.fit(categories)
        self.is_fitted = True
        print(f"CurrentUseLE fitted: categories={list(self.le.classes_)}")
        return self

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """fitしてtransform（sklearn互換）"""
        return self.fit(df).transform(df)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Label Encodingで変換"""
        if not self.is_fitted:
            raise ValueError("Transformer is not fitted yet. Call fit() first.")

        # 上位カテゴリ以外は「その他」に置換
        current_use = df["lp_current_use"].to_numpy()
        current_use_mapped = np.where(
            np.isin(current_use, self.top_categories),
            current_use,
            "その他"
        )

        # Label Encoding
        X_le = self.le.transform(current_use_mapped)

        # DataFrameに追加
        result = df.with_columns([
            pl.Series("lp_current_use_le", X_le)
        ])

        return result


# 地価公示価格の平均・比率を計算するカテゴリカラム
LP_RATIO_COLUMNS = [
    'post1',            # 郵便番号上3桁
    'post_full',        # 郵便番号7桁
    'addr1_1',          # 都道府県コード
    'addr1_2',          # 市区町村コード
    'bukken_type',      # 物件タイプ
    'land_youto',       # 用途地域
    'land_toshi',       # 都市計画
    'building_age_bin', # 築年数カテゴリ（前処理で作成）
    'rosen_name1',      # 路線名1
]


class LandpriceCategoryRatioTransformer:
    """
    カテゴリ別の地価公示価格（lp_price）平均と比率を計算

    処理内容:
    - 各カテゴリカラムに対してlp_priceの平均を計算
    - train/testそれぞれ独立に計算（地価公示は外部データのためリーケージなし）
    - 出力カラム名: {col}_lp_mean, {col}_lp_ratio
    - 比率 = lp_price / カテゴリ別平均
    """

    def __init__(self, columns: Optional[List[str]] = None):
        """
        Parameters
        ----------
        columns : List[str], optional
            対象カラム。Noneの場合はLP_RATIO_COLUMNS
        """
        self.columns = columns or LP_RATIO_COLUMNS

    def fit(self, df: pl.DataFrame = None) -> "LandpriceCategoryRatioTransformer":
        """
        fitメソッド（インターフェース統一のため）

        このTransformerはtrain/testそれぞれ独立に計算するため、
        fitでパラメータを保存しない。ただしfit/transformパターンに従う。
        """
        return self

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """fitしてtransform（sklearn互換）"""
        return self.fit(df).transform(df)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        カテゴリ別平均と比率を計算

        Parameters
        ----------
        df : pl.DataFrame
            地価公示データが結合済みのDataFrame（lp_priceカラムが必要）

        Returns
        -------
        pl.DataFrame
            平均と比率のDataFrame（{col}_lp_mean, {col}_lp_ratio カラム）
        """

        # lp_priceがNullでない行のみで平均計算
        valid_df = df.filter(pl.col("lp_price").is_not_null())
        n_valid = len(valid_df)
        print(f"LandpriceCategoryRatio: 有効データ数 = {n_valid:,} / {len(df):,}")

        # グローバル平均
        global_mean = float(valid_df["lp_price"].mean())
        print(f"  global_mean = {global_mean:,.0f}")

        # lp_priceを取得
        lp_price = df["lp_price"].to_numpy()

        results = {}
        for col in self.columns:
            if col not in df.columns:
                print(f"  Warning: {col} not in DataFrame, skipping")
                continue

            # カテゴリ別平均を計算
            mean_df = valid_df.group_by(col).agg(
                pl.col("lp_price").mean().alias("mean")
            )
            mean_map = dict(zip(
                mean_df[col].to_list(),
                mean_df["mean"].to_list()
            ))

            # 各行に平均を適用
            col_values = df[col].to_numpy()
            mean_values = np.array([
                mean_map.get(v, global_mean) for v in col_values
            ])

            # 比率を計算（0除算防止）
            ratio_values = np.where(
                mean_values > 0,
                lp_price / mean_values,
                np.nan
            )

            results[f"{col}_lp_mean"] = mean_values
            results[f"{col}_lp_ratio"] = ratio_values

        return pl.DataFrame(results)

    @staticmethod
    def get_feature_columns(columns: Optional[List[str]] = None) -> List[str]:
        """出力される特徴量カラム名のリストを返す"""
        cols = columns or LP_RATIO_COLUMNS
        feature_cols = []
        for col in cols:
            feature_cols.append(f"{col}_lp_mean")
            feature_cols.append(f"{col}_lp_ratio")
        return feature_cols


class LandpricePriceRatioTransformer:
    """
    価格時系列から比率特徴量を生成（スライド方式）

    処理内容:
    - lp_ratio_1to3 = 1年前価格 / 3年前価格
    - lp_ratio_3to5 = 3年前価格 / 5年前価格
    """

    def fit(self, df: pl.DataFrame = None) -> "LandpricePriceRatioTransformer":
        """
        fitメソッド（インターフェース統一のため）

        このTransformerはtrain/testそれぞれ独立に計算するため、
        fitでパラメータを保存しない。ただしfit/transformパターンに従う。
        """
        return self

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """fitしてtransform（sklearn互換）"""
        return self.fit(df).transform(df)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """比率計算"""
        result = df.with_columns([
            # 1年前 / 3年前
            pl.when(pl.col("lp_price_3y_ago") > 0)
            .then(pl.col("lp_price_1y_ago") / pl.col("lp_price_3y_ago"))
            .otherwise(None)
            .alias("lp_ratio_1to3"),

            # 3年前 / 5年前
            pl.when(pl.col("lp_price_5y_ago") > 0)
            .then(pl.col("lp_price_3y_ago") / pl.col("lp_price_5y_ago"))
            .otherwise(None)
            .alias("lp_ratio_3to5"),
        ])

        return result


def create_landprice_features(
    df: pl.DataFrame,
    shape_pca: Optional[LandpriceShapePCATransformer] = None,
    road_svd: Optional[LandpriceRoadSVDTransformer] = None,
    current_use_le: Optional[LandpriceCurrentUseLETransformer] = None,
    is_train: bool = True,
) -> tuple[pl.DataFrame, dict]:
    """
    地価公示データから全特徴量を生成

    Parameters
    ----------
    df : pl.DataFrame
        地価公示データが結合されたDataFrame
    shape_pca : LandpriceShapePCATransformer, optional
        学習済みPCA変換器（test時に指定）
    road_svd : LandpriceRoadSVDTransformer, optional
        学習済みSVD変換器（test時に指定）
    current_use_le : LandpriceCurrentUseLETransformer, optional
        学習済みLE変換器（test時に指定）
    is_train : bool
        trainデータの場合True

    Returns
    -------
    tuple[pl.DataFrame, dict]
        特徴量追加後のDataFrameと、学習済み変換器の辞書
    """
    transformers = {}

    # 1. 土地属性（PCA）
    if is_train:
        shape_pca = LandpriceShapePCATransformer()
        shape_pca.fit(df)
    df = shape_pca.transform(df)
    transformers["shape_pca"] = shape_pca

    # 2. 道路状況（SVD）
    if is_train:
        road_svd = LandpriceRoadSVDTransformer()
        road_svd.fit(df)
    df = road_svd.transform(df)
    transformers["road_svd"] = road_svd

    # 3. 周辺環境（Label Encoding）
    if is_train:
        current_use_le = LandpriceCurrentUseLETransformer()
        current_use_le.fit(df)
    df = current_use_le.transform(df)
    transformers["current_use_le"] = current_use_le

    # 4. 価格時系列（比率計算）
    price_ratio = LandpricePriceRatioTransformer()
    price_ratio.fit()
    df = price_ratio.transform(df)

    # 5. 数値特徴量はそのまま（lp_price, lp_change_rate, lp_road_width, lp_nearest_dist）
    # 既に結合済み

    return df, transformers


def get_landprice_feature_columns(include_ratio: bool = True) -> List[str]:
    """地価公示から生成する特徴量のカラム名リストを返す

    Parameters
    ----------
    include_ratio : bool
        カテゴリ別地価平均・比率を含めるか（デフォルトTrue）

    Returns
    -------
    List[str]
        特徴量カラム名のリスト
        - include_ratio=False: 21次元
        - include_ratio=True: 21 + 18 = 39次元
    """
    columns = [
        # 土地属性（1次元）
        "lp_shape_pca",

        # 道路状況（14次元）
        *[f"lp_road_svd_{i}" for i in range(ROAD_SVD_DIM)],
        "lp_road_width",

        # 周辺環境（1次元）
        "lp_current_use_le",

        # 価格（3次元）
        "lp_price",
        "lp_change_rate",
        "lp_nearest_dist",

        # 価格時系列（2次元）
        "lp_ratio_1to3",
        "lp_ratio_3to5",
    ]

    # カテゴリ別地価平均・比率（18次元: 9カテゴリ × 2）
    if include_ratio:
        columns.extend(LandpriceCategoryRatioTransformer.get_feature_columns())

    return columns


if __name__ == "__main__":
    from pathlib import Path
    import sys

    # プロジェクトルート
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root / "06_experiments/exp009_landprice/code"))

    from join_landprice import join_landprice_by_year

    # trainデータを読み込み・結合
    train = pl.read_csv(project_root / "data/raw/train.csv", infer_schema_length=50000)
    print(f"Train shape: {train.shape}")

    # サンプルで検証（高速化のため）
    train_sample = train.head(10000)

    # 地価公示データと結合
    train_joined = join_landprice_by_year(
        train_sample,
        base_path=project_root / "data/external/landprice"
    )
    print(f"Joined shape: {train_joined.shape}")

    # 特徴量生成
    train_features, transformers = create_landprice_features(train_joined, is_train=True)
    print(f"Features shape: {train_features.shape}")

    # 生成された特徴量カラム
    lp_columns = get_landprice_feature_columns()
    print(f"\n=== 生成された特徴量 ({len(lp_columns)}個) ===")
    for col in lp_columns:
        if col in train_features.columns:
            series = train_features[col]
            print(f"  {col}: dtype={series.dtype}, null={series.null_count()}")
