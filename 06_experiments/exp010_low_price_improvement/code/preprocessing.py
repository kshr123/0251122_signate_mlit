"""
exp010_low_price_improvement 前処理モジュール

低価格帯（特に広面積×築古）の予測精度改善を目的とした前処理。

前処理内容:
1. 基本前処理（年変換、築年数、アクセス時間等）
2. 地価公示データ結合（年度別に最近傍探索）
3. 特徴量パイプライン実行

使用例:
    X_train, X_test, y_train, pipeline = preprocess_for_training(train, test, cv_splits)
    print(pipeline.summary())  # 変換内容を確認
"""

import sys
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "04_src"))
# exp010のcodeディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl
from sklearn.model_selection import KFold
from sklearn.neighbors import BallTree

# パイプライン
from pipeline import FeaturePipeline, create_pipeline

# パス定義
from constants import LANDPRICE_BASE_PATH

# デフォルト値（experiment.yamlから上書き可能）
DEFAULT_RANDOM_SEED = 42
DEFAULT_N_SPLITS = 3
DEFAULT_AGE_THRESHOLD = 35
DEFAULT_AREA_THRESHOLD = 80
DEFAULT_MAJOR_CITIES = [13, 14, 23, 27]


# =============================================================================
# 地価公示データ読み込み・結合
# =============================================================================

# 年度ごとの設定（価格カラム名、共通カラムマッピング）
YEAR_CONFIG = {
    2019: {
        "filename": "L01-31P-48-01.0a/L01-31P-2K.csv",
        "price_current": "Ｈ３１価格",
        "price_1y_ago": "Ｈ３０価格",
        "price_3y_ago": "Ｈ２８価格",
        "price_5y_ago": "Ｈ２６価格",
        "columns": {
            "経度": "経度", "緯度": "緯度",
            "間口（比率）": "間口比率", "奥行（比率）": "奥行比率",
            "前面道路区分": "前面道路区分", "前面道路の方位区分": "前面道路方位",
            "前面道路の幅員": "前面道路幅員", "側道区分": "側道状況", "利用の現況": "利用現況",
            "地積": "地積", "駅距離": "駅距離", "防火区分": "防火地域",
        },
        "schema_overrides": {"選定年次ビット": pl.Utf8},
    },
    2020: {
        "filename": "L01-2020P-48-01.0a/L01-2020P-2K.csv",
        "price_current": "Ｒ２価格", "price_1y_ago": "Ｈ３１価格",
        "price_3y_ago": "Ｈ２９価格", "price_5y_ago": "Ｈ２７価格",
        "columns": {
            "経度": "経度", "緯度": "緯度",
            "間口（比率）": "間口比率", "奥行（比率）": "奥行比率",
            "前面道路区分": "前面道路区分", "前面道路の方位区分": "前面道路方位",
            "前面道路の幅員": "前面道路幅員", "側道区分": "側道状況", "利用の現況": "利用現況",
            "地積": "地積", "駅距離": "駅距離", "防火区分": "防火地域",
        },
        "schema_overrides": {"選定年次ビット": pl.Utf8},
    },
    2021: {
        "filename": "L01-2021P-48-01.0a/L01-2021P-2K.csv",
        "price_current": "Ｒ３価格", "price_1y_ago": "Ｒ２価格",
        "price_3y_ago": "Ｈ３０価格", "price_5y_ago": "Ｈ２８価格",
        "columns": {
            "経度": "経度", "緯度": "緯度",
            "間口（比率）": "間口比率", "奥行（比率）": "奥行比率",
            "前面道路区分": "前面道路区分", "前面道路の方位区分": "前面道路方位",
            "前面道路の幅員": "前面道路幅員", "側道区分": "側道状況", "利用の現況": "利用現況",
            "地積": "地積", "駅距離": "駅距離", "防火区分": "防火地域",
        },
        "schema_overrides": {"選定年次ビット": pl.Utf8},
    },
    2022: {
        "filename": "L01-2022P-48-01.0a/L01-2022P-2K.csv",
        "price_current": "Ｒ４価格", "price_1y_ago": "Ｒ３価格",
        "price_3y_ago": "Ｈ３１価格", "price_5y_ago": "Ｈ２９価格",
        "columns": {
            "経度": "経度", "緯度": "緯度",
            "間口（比率）": "間口比率", "奥行（比率）": "奥行比率",
            "前面道路区分": "前面道路区分", "前面道路の方位区分": "前面道路方位",
            "前面道路の幅員": "前面道路幅員", "側道区分": "側道状況", "利用の現況": "利用現況",
            "地積": "地積", "駅距離": "駅距離", "防火区分": "防火地域",
        },
        "schema_overrides": {"選定年次ビット": pl.Utf8},
    },
    2023: {
        "filename": "L01-2023P-48-01.0a/L01-2023P-2K.csv",
        "price_current": "Ｒ５価格", "price_1y_ago": "Ｒ４価格",
        "price_3y_ago": "Ｒ２価格", "price_5y_ago": "Ｈ３０価格",
        "columns": {
            "経度": "経度", "緯度": "緯度",
            "間口（比率）": "間口比率", "奥行（比率）": "奥行比率",
            "前面道路区分": "前面道路区分", "前面道路の方位区分": "前面道路方位",
            "前面道路の幅員": "前面道路幅員", "側道区分": "側道状況", "利用の現況": "利用現況",
            "地積": "地積", "駅距離": "駅距離", "防火区分": "防火地域",
        },
        "schema_overrides": {"選定年次ビット": pl.Utf8},
    },
    2024: {
        "filename": "L01-2024P-48-01.0a/L01-2024P-2K.csv",
        "price_current": "価格R06", "price_1y_ago": "価格R05",
        "price_3y_ago": "価格R03", "price_5y_ago": "価格R01",
        "columns": {
            "経度": "経度", "緯度": "緯度",
            "間口比率": "間口比率", "奥行比率": "奥行比率",
            "前面道路区分": "前面道路区分", "前面道路方位": "前面道路方位",
            "前面道路幅員": "前面道路幅員", "側道状況": "側道状況", "利用現況": "利用現況",
            "対前年変動率": "対前年変動率",
            "地積": "地積", "駅距離": "駅距離", "防火地域": "防火地域",
        },
        "schema_overrides": {"選定年次フラグ": pl.Utf8},
    },
}

# target_ymの年度と使用する地価公示データの年度のマッピング
TARGET_YEAR_TO_LANDPRICE_YEAR = {2019: 2019, 2020: 2020, 2021: 2021, 2022: 2022, 2023: 2023}


def _load_landprice(year: int, base_path: Path) -> pl.DataFrame:
    """指定年度の地価公示データを読み込み、座標変換を行う"""
    config = YEAR_CONFIG[year]
    file_path = base_path / config["filename"]

    columns_to_read = list(config["columns"].keys()) + [
        config["price_current"], config["price_1y_ago"],
        config["price_3y_ago"], config["price_5y_ago"],
    ]

    df = pl.read_csv(
        file_path, encoding="shift_jis", columns=columns_to_read,
        infer_schema_length=50000, schema_overrides=config.get("schema_overrides", {}),
    )

    rename_map = {k: v for k, v in config["columns"].items() if k != v}
    if rename_map:
        df = df.rename(rename_map)

    # 座標変換: 秒 → 度
    df = df.with_columns([
        (pl.col("経度") / 3600).alias("lon_wgs"),
        (pl.col("緯度") / 3600).alias("lat_wgs"),
    ])

    df = df.rename({
        config["price_current"]: "lp_price",
        config["price_1y_ago"]: "lp_price_1y_ago",
        config["price_3y_ago"]: "lp_price_3y_ago",
        config["price_5y_ago"]: "lp_price_5y_ago",
    })

    # 対前年変動率
    if "対前年変動率" not in df.columns:
        df = df.with_columns(
            pl.when(pl.col("lp_price_1y_ago") > 0)
            .then((pl.col("lp_price") / pl.col("lp_price_1y_ago") - 1) * 100)
            .otherwise(None).alias("lp_change_rate")
        )
    else:
        df = df.rename({"対前年変動率": "lp_change_rate"})

    # 価格0をnullに
    price_columns = ["lp_price", "lp_price_1y_ago", "lp_price_3y_ago", "lp_price_5y_ago"]
    df = df.with_columns([
        pl.when(pl.col(col) == 0).then(None).otherwise(pl.col(col)).alias(col)
        for col in price_columns
    ])

    df = df.drop(["経度", "緯度"])
    output_columns = [
        "lon_wgs", "lat_wgs", "lp_price", "lp_price_1y_ago", "lp_price_3y_ago", "lp_price_5y_ago",
        "lp_change_rate", "間口比率", "奥行比率", "前面道路区分", "前面道路方位", "前面道路幅員", "側道状況", "利用現況",
        "地積", "駅距離", "防火地域",
    ]
    return df.select([c for c in output_columns if c in df.columns])


def _join_landprice_by_nearest(
    df: pl.DataFrame, landprice_df: pl.DataFrame, max_distance_km: float = 5.0
) -> pl.DataFrame:
    """BallTree最近傍探索で地価公示データを結合"""
    df_coords = np.radians(df.select(["lat", "lon"]).to_numpy())
    lp_coords = np.radians(landprice_df.select(["lat_wgs", "lon_wgs"]).to_numpy())

    tree = BallTree(lp_coords, metric="haversine")
    distances, indices = tree.query(df_coords, k=1)
    distances_km = distances.flatten() * 6371
    nearest_indices = indices.flatten()

    lp_column_map = {
        "lp_price": ("lp_price", pl.Float64),
        "lp_price_1y_ago": ("lp_price_1y_ago", pl.Float64),
        "lp_price_3y_ago": ("lp_price_3y_ago", pl.Float64),
        "lp_price_5y_ago": ("lp_price_5y_ago", pl.Float64),
        "lp_change_rate": ("lp_change_rate", pl.Float64),
        "間口比率": ("lp_frontage_ratio", pl.Float64),
        "奥行比率": ("lp_depth_ratio", pl.Float64),
        "前面道路区分": ("lp_road_type", pl.Utf8),
        "前面道路方位": ("lp_road_direction", pl.Utf8),
        "前面道路幅員": ("lp_road_width", pl.Float64),
        "側道状況": ("lp_side_road", pl.Utf8),
        "利用現況": ("lp_current_use", pl.Utf8),
        "地積": ("lp_land_area", pl.Float64),
        "駅距離": ("lp_station_dist", pl.Float64),
        "防火地域": ("lp_fire_zone", pl.Utf8),
    }

    joined_data = {}
    for src_col, (dst_col, dtype) in lp_column_map.items():
        col_data = landprice_df[src_col].to_numpy()[nearest_indices]
        joined_data[dst_col] = np.array(col_data, dtype=np.float64) if dtype == pl.Float64 else col_data

    joined_df = pl.DataFrame(joined_data)
    lp_columns = [c[0] for c in lp_column_map.values()]
    joined_df = joined_df.with_columns(pl.Series("lp_nearest_dist", distances_km))

    if max_distance_km is not None:
        for col in lp_columns + ["lp_nearest_dist"]:
            joined_df = joined_df.with_columns(
                pl.when(pl.col("lp_nearest_dist") > max_distance_km)
                .then(None).otherwise(pl.col(col)).alias(col)
            )

    return pl.concat([df, joined_df], how="horizontal")


def _impute_landprice_missing(df: pl.DataFrame) -> pl.DataFrame:
    """地価公示データの欠損値を階層的に補完"""
    columns = [
        "lp_price", "lp_price_1y_ago", "lp_price_3y_ago", "lp_price_5y_ago",
        "lp_change_rate", "lp_frontage_ratio", "lp_depth_ratio", "lp_road_width", "lp_nearest_dist",
    ]

    before_nulls = {col: df[col].null_count() for col in columns if col in df.columns}
    if sum(before_nulls.values()) == 0:
        return df

    df = df.with_columns([
        (pl.col("target_ym") // 100).alias("_target_year"),
        (pl.col("post1").cast(pl.Utf8) + pl.col("post2").cast(pl.Utf8)).alias("_full_postal")
    ])

    postal_means = df.filter(pl.col("lp_price").is_not_null()).group_by(["_target_year", "_full_postal"]).agg([
        pl.col(col).mean().alias(f"{col}_postal_mean") for col in columns if col in df.columns
    ])
    city_means = df.filter(pl.col("lp_price").is_not_null()).group_by(["_target_year", "addr1_2"]).agg([
        pl.col(col).mean().alias(f"{col}_city_mean") for col in columns if col in df.columns
    ])

    df = df.join(postal_means, on=["_target_year", "_full_postal"], how="left")
    df = df.join(city_means, on=["_target_year", "addr1_2"], how="left")

    for col in columns:
        if col not in df.columns:
            continue
        df = df.with_columns(
            pl.when(pl.col(col).is_null())
            .then(pl.when(pl.col(f"{col}_postal_mean").is_not_null())
                  .then(pl.col(f"{col}_postal_mean")).otherwise(pl.col(f"{col}_city_mean")))
            .otherwise(pl.col(col)).alias(col)
        )

    temp_cols = [c for c in df.columns if c.endswith("_postal_mean") or c.endswith("_city_mean")]
    temp_cols.extend(["_full_postal", "_target_year"])
    return df.drop(temp_cols)


def join_landprice_by_year(
    df: pl.DataFrame, base_path: Path = LANDPRICE_BASE_PATH, max_distance_km: float = 5.0
) -> pl.DataFrame:
    """target_ymの年度に応じた地価公示データを結合"""
    df = df.with_columns((pl.col("target_ym") // 100).alias("_target_year"))

    results = []
    for target_year, lp_year in TARGET_YEAR_TO_LANDPRICE_YEAR.items():
        year_df = df.filter(pl.col("_target_year") == target_year)
        if len(year_df) == 0:
            continue
        landprice_df = _load_landprice(lp_year, base_path)
        joined_df = _join_landprice_by_nearest(year_df, landprice_df, max_distance_km)
        results.append(joined_df)

    result = pl.concat(results)
    result = result.drop("_target_year")
    return _impute_landprice_missing(result)


# =============================================================================
# 基本前処理関数
# =============================================================================

def transform_year_built(df: pl.DataFrame) -> pl.DataFrame:
    """year_built を YYYYMM → YYYY に変換"""
    return df.with_columns([
        (pl.col('year_built').cast(pl.Utf8).str.slice(0, 4).cast(pl.Int64)).alias('year_built')
    ])


def aggregate_money_sonota(df: pl.DataFrame) -> pl.DataFrame:
    """money_sonota1, money_sonota2, money_sonota3 を合計"""
    return df.with_columns([
        (
            pl.col('money_sonota1').fill_null(0) +
            pl.col('money_sonota2').fill_null(0) +
            pl.col('money_sonota3').fill_null(0)
        ).alias('money_sonota_sum')
    ])


def extract_reference_year(df: pl.DataFrame) -> pl.DataFrame:
    """snapshot_create_dateから基準年を抽出"""
    return df.with_columns([
        pl.col('snapshot_create_date').str.slice(0, 4).cast(pl.Int64).alias('reference_year')
    ])


def add_post_full(df: pl.DataFrame) -> pl.DataFrame:
    """post_full（郵便番号7桁）カラムを追加"""
    return df.with_columns([
        (
            pl.col('post1').cast(pl.Utf8).fill_null('') +
            pl.col('post2').cast(pl.Utf8).fill_null('')
        ).alias('post_full')
    ])


def add_age_features(df: pl.DataFrame) -> pl.DataFrame:
    """築年数関連特徴量を追加"""
    raw_building_age = pl.col("reference_year") - pl.col("year_built")

    df = df.with_columns([
        # 未来物件フラグ
        (raw_building_age < 0).cast(pl.Int64).alias("is_future_building"),
        # 築年数（マイナスの場合は0にクリップ）
        raw_building_age.clip(0, None).alias("building_age"),
        # 築年数カテゴリ（5年単位、0-10にクリップ）
        (raw_building_age.clip(0, None) // 5).clip(0, 10).alias("building_age_bin"),
        # 地方フラグ
        (~pl.col("addr1_1").is_in(DEFAULT_MAJOR_CITIES)).cast(pl.Int64).alias("rural_flag"),
    ])

    df = df.with_columns([
        # 古くて広いフラグ
        (
            (pl.col("building_age") >= DEFAULT_AGE_THRESHOLD) &
            (pl.col("house_area") >= DEFAULT_AREA_THRESHOLD)
        ).cast(pl.Int64).alias("old_and_large_flag"),
        # 古くて地方フラグ
        (
            (pl.col("building_age") >= DEFAULT_AGE_THRESHOLD) &
            (pl.col("rural_flag") == 1)
        ).cast(pl.Int64).alias("old_and_rural_flag"),
    ])

    return df


def add_access_time_features(df: pl.DataFrame) -> pl.DataFrame:
    """交通アクセス時間特徴量を追加"""
    df = df.with_columns([
        (pl.col('walk_distance1') / 80).alias('walk_time1'),
        (pl.col('walk_distance2') / 80).alias('walk_time2'),
    ])

    df = df.with_columns([
        (pl.col('walk_time1') + pl.col('bus_time1').fill_null(0)).alias('total_access_time1'),
        (pl.col('walk_time2') + pl.col('bus_time2').fill_null(0)).alias('total_access_time2'),
    ])

    return df


def add_reform_year_features(df: pl.DataFrame) -> pl.DataFrame:
    """リフォーム経過年数特徴量を追加"""
    df = df.with_columns([
        (pl.col('reform_wet_area_date') / 100).floor().cast(pl.Int64).alias('wet_reform_year'),
        (pl.col('reform_interior_date') / 100).floor().cast(pl.Int64).alias('interior_reform_year'),
    ])

    df = df.with_columns([
        (pl.col('reference_year') - pl.col('wet_reform_year')).alias('years_since_wet_reform'),
        (pl.col('reference_year') - pl.col('interior_reform_year')).alias('years_since_interior_reform'),
    ])

    return df


def preprocess_base(df: pl.DataFrame) -> pl.DataFrame:
    """基本的な前処理"""
    df = transform_year_built(df)
    df = aggregate_money_sonota(df)
    df = extract_reference_year(df)
    df = add_post_full(df)
    df = add_age_features(df)
    df = add_access_time_features(df)
    df = add_reform_year_features(df)
    return df


# =============================================================================
# メイン前処理関数
# =============================================================================

def preprocess_for_training(
    train: pl.DataFrame,
    test: pl.DataFrame,
    cv_splits: list = None,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.Series, FeaturePipeline]:
    """
    学習用の前処理

    Args:
        train: 訓練データ（money_roomカラムを含む）
        test: テストデータ
        cv_splits: CVのfold情報（TargetEncoding用）

    Returns:
        (X_train, X_test, y_train, pipeline)のタプル
        pipelineを使って変換内容を確認可能
    """
    print("=" * 60)
    print("前処理開始（exp010: low_price_improvement）")
    print("=" * 60)

    # ターゲット変数を分離
    y_train = train["money_room"]
    print(f"\n✓ ターゲット変数分離: {len(y_train)}件")

    # CVインデックスの設定
    if cv_splits is None:
        cv_splits = list(KFold(
            n_splits=DEFAULT_N_SPLITS,
            shuffle=True,
            random_state=DEFAULT_RANDOM_SEED
        ).split(train))

    # =========================================================================
    # 基本前処理
    # =========================================================================
    print("\n[Step 1] 基本前処理")
    train = preprocess_base(train)
    test = preprocess_base(test)
    print("  → year_built, money_sonota_sum, reference_year, post_full")
    print("  → 築年数関連特徴量（4個）、アクセス時間（4個）、リフォーム経過年数（2個）")

    # =========================================================================
    # 地価公示データ結合
    # =========================================================================
    print("\n[Step 2] 地価公示データ結合")
    train = join_landprice_by_year(train)
    test = join_landprice_by_year(test)
    print("  → 年度別に最近傍地価公示データを結合（12カラム追加）")
    print("  → 欠損値を階層的に補完（郵便番号→市区町村）")

    # =========================================================================
    # 特徴量パイプライン
    # =========================================================================
    print("\n[Step 3] 特徴量パイプライン実行")
    pipeline = create_pipeline(cv_splits)

    # trainデータをfit & transform
    print("  → trainデータをfit_transform中...")
    X_train = pipeline.fit_transform(train, y_train)

    # testデータをtransform
    print("  → testデータをtransform中...")
    X_test = pipeline.transform(test)

    # =========================================================================
    # 結果確認
    # =========================================================================
    print(f"\n✓ 訓練データ: {X_train.shape}")
    print(f"✓ テストデータ: {X_test.shape}")
    print(f"✓ 特徴量数: {len(pipeline.get_feature_names())}個")

    print("\n" + "=" * 60)
    print("前処理完了")
    print("=" * 60)

    return X_train, X_test, y_train, pipeline


