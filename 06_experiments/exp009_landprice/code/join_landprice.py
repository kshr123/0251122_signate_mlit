"""train/testと地価公示データの結合処理モジュール（BallTree最近傍探索）"""

from pathlib import Path
from typing import Optional, List

import numpy as np
import polars as pl
from sklearn.neighbors import BallTree

from load_landprice import load_landprice, YEAR_CONFIG


# 補完対象の数値カラム（カテゴリカルは補完しない）
LP_NUMERIC_COLUMNS = [
    "lp_price",
    "lp_price_1y_ago",
    "lp_price_3y_ago",
    "lp_price_5y_ago",
    "lp_change_rate",
    "lp_frontage_ratio",
    "lp_depth_ratio",
    "lp_road_width",
    "lp_nearest_dist",
]

# target_ymの年度と使用する地価公示データの年度のマッピング
# train: 2019-2022年 → 同年の地価公示
# test: 2023年 → 2023年の地価公示（trainと同様に同年を使用）
TARGET_YEAR_TO_LANDPRICE_YEAR = {
    2019: 2019,
    2020: 2020,
    2021: 2021,
    2022: 2022,
    2023: 2023,  # testも同年の地価公示を使用
}


def join_landprice_by_nearest(
    df: pl.DataFrame,
    landprice_df: pl.DataFrame,
    max_distance_km: float = 5.0,
) -> pl.DataFrame:
    """
    BallTree最近傍探索で地価公示データを結合

    Parameters
    ----------
    df : pl.DataFrame
        train/testデータ（lon, lat カラムが必要）
    landprice_df : pl.DataFrame
        地価公示データ（lon_wgs, lat_wgs カラムが必要）
    max_distance_km : float
        最大結合距離（km）。これを超える場合はNaN

    Returns
    -------
    pl.DataFrame
        地価公示データが結合されたDataFrame
        - lp_nearest_dist: 最近傍地点までの距離（km）
        - その他地価公示カラム
    """
    # 座標をラジアンに変換（BallTree用）
    df_coords = np.radians(df.select(["lat", "lon"]).to_numpy())
    lp_coords = np.radians(landprice_df.select(["lat_wgs", "lon_wgs"]).to_numpy())

    # BallTree構築（Haversine距離）
    tree = BallTree(lp_coords, metric="haversine")

    # 最近傍探索
    distances, indices = tree.query(df_coords, k=1)

    # 距離をkmに変換（地球半径 6371km）
    distances_km = distances.flatten() * 6371

    # 最近傍のインデックス
    nearest_indices = indices.flatten()

    # 地価公示データのカラム定義（元カラム名と新カラム名）
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
    }

    # 各カラムを個別に取得して型を保持
    joined_data = {}
    for src_col, (dst_col, dtype) in lp_column_map.items():
        col_data = landprice_df[src_col].to_numpy()[nearest_indices]
        if dtype == pl.Float64:
            # 数値型の場合はfloat64に変換
            joined_data[dst_col] = np.array(col_data, dtype=np.float64)
        else:
            # 文字列型の場合はそのまま
            joined_data[dst_col] = col_data

    # 結合DataFrameを作成
    joined_df = pl.DataFrame(joined_data)

    # カラム名リスト（後続処理用）
    lp_columns = list(lp_column_map.values())
    lp_columns = [c[0] for c in lp_columns]

    # 距離カラムを追加
    joined_df = joined_df.with_columns(
        pl.Series("lp_nearest_dist", distances_km)
    )

    # 最大距離を超える場合はNaNに
    if max_distance_km is not None:
        for col in lp_columns + ["lp_nearest_dist"]:
            joined_df = joined_df.with_columns(
                pl.when(pl.col("lp_nearest_dist") > max_distance_km)
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )

    # 元のDataFrameと結合
    result = pl.concat([df, joined_df], how="horizontal")

    return result


def impute_landprice_missing(
    df: pl.DataFrame,
    columns: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    地価公示データの欠損値を補完（年度×フル郵便番号 → 年度×市区町村の階層的補完）

    Parameters
    ----------
    df : pl.DataFrame
        地価公示データが結合済みのDataFrame（target_ym, post1, post2, addr1_2 カラムが必要）
    columns : List[str], optional
        補完対象のカラム。Noneの場合はLP_NUMERIC_COLUMNS

    Returns
    -------
    pl.DataFrame
        欠損値補完後のDataFrame
    """
    if columns is None:
        columns = LP_NUMERIC_COLUMNS

    # 補完前の欠損数を記録
    before_nulls = {col: df[col].null_count() for col in columns if col in df.columns}
    total_before = sum(before_nulls.values())

    if total_before == 0:
        print("No missing values to impute")
        return df

    print(f"\n=== 欠損値補完（年度別） ===")
    print(f"補完前の欠損: {before_nulls}")

    # 年度カラムを作成（target_ymから年を抽出）
    df = df.with_columns(
        (pl.col("target_ym") // 100).alias("_target_year")
    )

    # フル郵便番号を作成（post1 + post2）
    df = df.with_columns(
        (pl.col("post1").cast(pl.Utf8) + pl.col("post2").cast(pl.Utf8)).alias("_full_postal")
    )

    # Step 1: 年度×フル郵便番号別の平均を計算
    postal_means = df.filter(pl.col("lp_price").is_not_null()).group_by(["_target_year", "_full_postal"]).agg([
        pl.col(col).mean().alias(f"{col}_postal_mean")
        for col in columns if col in df.columns
    ])
    print(f"年度×フル郵便番号グループ数: {len(postal_means)}")

    # Step 2: 年度×市区町村別の平均を計算
    city_means = df.filter(pl.col("lp_price").is_not_null()).group_by(["_target_year", "addr1_2"]).agg([
        pl.col(col).mean().alias(f"{col}_city_mean")
        for col in columns if col in df.columns
    ])
    print(f"年度×市区町村グループ数: {len(city_means)}")

    # Step 3: 平均値を結合
    df = df.join(postal_means, on=["_target_year", "_full_postal"], how="left")
    df = df.join(city_means, on=["_target_year", "addr1_2"], how="left")

    # Step 4: 階層的に補完（年度×フル郵便番号 → 年度×市区町村）
    impute_stats = {"postal": 0, "city": 0}

    for col in columns:
        if col not in df.columns:
            continue

        postal_mean_col = f"{col}_postal_mean"
        city_mean_col = f"{col}_city_mean"

        # 欠損箇所をカウント
        null_mask = df[col].is_null()
        null_count = null_mask.sum()

        if null_count == 0:
            continue

        # フル郵便番号平均で補完できる数
        can_impute_postal = (null_mask & df[postal_mean_col].is_not_null()).sum()
        # 市区町村平均で補完できる数（郵便番号で補完できない分）
        can_impute_city = (
            null_mask &
            df[postal_mean_col].is_null() &
            df[city_mean_col].is_not_null()
        ).sum()

        impute_stats["postal"] += can_impute_postal
        impute_stats["city"] += can_impute_city

        # 補完実行: 年度×フル郵便番号平均 → 年度×市区町村平均 → 元の値
        df = df.with_columns(
            pl.when(pl.col(col).is_null())
            .then(
                pl.when(pl.col(postal_mean_col).is_not_null())
                .then(pl.col(postal_mean_col))
                .otherwise(pl.col(city_mean_col))
            )
            .otherwise(pl.col(col))
            .alias(col)
        )

    # 一時カラムを削除
    temp_cols = [c for c in df.columns if c.endswith("_postal_mean") or c.endswith("_city_mean")]
    temp_cols.extend(["_full_postal", "_target_year"])
    df = df.drop(temp_cols)

    # 補完後の欠損数を確認
    after_nulls = {col: df[col].null_count() for col in columns if col in df.columns}
    total_after = sum(after_nulls.values())

    print(f"補完結果: 年度×フル郵便番号={impute_stats['postal']}, 年度×市区町村={impute_stats['city']}")
    print(f"補完後の欠損: {after_nulls}")
    print(f"欠損削減: {total_before} → {total_after} ({total_before - total_after}件補完)")

    return df


def join_landprice_by_year(
    df: pl.DataFrame,
    base_path: Optional[Path] = None,
    max_distance_km: float = 5.0,
    impute_missing: bool = True,
) -> pl.DataFrame:
    """
    target_ymの年度に応じた地価公示データを結合

    Parameters
    ----------
    df : pl.DataFrame
        train/testデータ（target_ym, lon, lat カラムが必要）
    base_path : Path, optional
        地価公示データのベースパス
    max_distance_km : float
        最大結合距離（km）
    impute_missing : bool
        欠損値を郵便番号・市区町村平均で補完するか

    Returns
    -------
    pl.DataFrame
        年度ごとの地価公示データが結合されたDataFrame
    """
    if base_path is None:
        base_path = Path("data/external/landprice")

    # target_ymから年を抽出
    df = df.with_columns(
        (pl.col("target_ym") // 100).alias("_target_year")
    )

    # 年度ごとに処理
    results = []
    for target_year, lp_year in TARGET_YEAR_TO_LANDPRICE_YEAR.items():
        # 該当年度のデータを抽出
        year_df = df.filter(pl.col("_target_year") == target_year)
        if len(year_df) == 0:
            continue

        print(f"Processing target_year={target_year} -> landprice_year={lp_year}, n={len(year_df):,}")

        # 地価公示データを読み込み
        landprice_df = load_landprice(lp_year, base_path)

        # 最近傍結合
        joined_df = join_landprice_by_nearest(
            year_df, landprice_df, max_distance_km
        )

        results.append(joined_df)

    # 結合
    result = pl.concat(results)

    # 一時カラムを削除
    result = result.drop("_target_year")

    # 欠損値補完
    if impute_missing:
        result = impute_landprice_missing(result)

    return result


def verify_join_results(df: pl.DataFrame) -> dict:
    """
    結合結果の検証

    Parameters
    ----------
    df : pl.DataFrame
        結合後のDataFrame

    Returns
    -------
    dict
        検証結果の統計情報
    """
    stats = {}

    # 距離の統計
    dist = df["lp_nearest_dist"]
    dist_mean = dist.mean()
    dist_median = dist.median()
    dist_std = dist.std()
    dist_min = dist.min()
    dist_max = dist.max()

    stats["distance"] = {
        "mean": float(dist_mean) if dist_mean is not None else None,
        "median": float(dist_median) if dist_median is not None else None,
        "std": float(dist_std) if dist_std is not None else None,
        "min": float(dist_min) if dist_min is not None else None,
        "max": float(dist_max) if dist_max is not None else None,
        "null_count": int(dist.null_count()),
        "null_rate": float(dist.null_count() / len(df)),
    }

    # 閾値別結合率
    for threshold in [0.5, 1.0, 2.0, 3.0, 5.0]:
        rate = float((dist <= threshold).sum() / len(df))
        stats[f"join_rate_{threshold}km"] = rate

    # 価格の統計（Object型の場合はスキップ）
    price = df["lp_price"]
    if price.dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
        price_mean = price.mean()
        price_median = price.median()
        stats["price"] = {
            "mean": float(price_mean) if price_mean is not None else None,
            "median": float(price_median) if price_median is not None else None,
            "null_count": int(price.null_count()),
            "null_rate": float(price.null_count() / len(df)),
        }
    else:
        # Object型の場合は数値に変換を試みる
        price_numeric = price.cast(pl.Float64, strict=False)
        price_mean = price_numeric.mean()
        price_median = price_numeric.median()
        stats["price"] = {
            "mean": float(price_mean) if price_mean is not None else None,
            "median": float(price_median) if price_median is not None else None,
            "null_count": int(price_numeric.null_count()),
            "null_rate": float(price_numeric.null_count() / len(df)),
        }

    return stats


if __name__ == "__main__":
    import sys
    # プロジェクトルートに移動
    project_root = Path(__file__).resolve().parents[3]

    # 動作確認
    print("=== 結合処理の動作確認 ===\n")

    # trainデータを読み込み
    train = pl.read_csv(project_root / "data/raw/train.csv", infer_schema_length=50000)
    print(f"Train shape: {train.shape}")

    # 結合実行（欠損値補完あり）
    train_joined = join_landprice_by_year(
        train,
        base_path=project_root / "data/external/landprice",
        impute_missing=True
    )
    print(f"\nJoined shape: {train_joined.shape}")
    print(f"New columns: {[c for c in train_joined.columns if c.startswith('lp_')]}")

    # 検証
    stats = verify_join_results(train_joined)
    print("\n=== 検証結果 ===")
    print(f"距離統計: mean={stats['distance']['mean']:.3f}km, "
          f"median={stats['distance']['median']:.3f}km")
    print(f"結合率: 1km={stats['join_rate_1.0km']:.1%}, "
          f"3km={stats['join_rate_3.0km']:.1%}, "
          f"5km={stats['join_rate_5.0km']:.1%}")
    print(f"価格: mean={stats['price']['mean']:,.0f}, "
          f"median={stats['price']['median']:,.0f}")
    print(f"価格欠損率: {stats['price']['null_rate']:.2%}")
