"""地価公示データの読み込み・座標変換モジュール（Polars版）"""

from pathlib import Path
from typing import Optional

import polars as pl


# 年度ごとの設定（価格カラム名、共通カラムマッピング）
YEAR_CONFIG = {
    2019: {
        "filename": "L01-31P-48-01.0a/L01-31P-2K.csv",
        "price_current": "Ｈ３１価格",
        "price_1y_ago": "Ｈ３０価格",
        "price_3y_ago": "Ｈ２８価格",
        "price_5y_ago": "Ｈ２６価格",
        "columns": {
            "経度": "経度",
            "緯度": "緯度",
            "間口（比率）": "間口比率",
            "奥行（比率）": "奥行比率",
            "前面道路区分": "前面道路区分",
            "前面道路の方位区分": "前面道路方位",
            "前面道路の幅員": "前面道路幅員",
            "側道区分": "側道状況",
            "利用の現況": "利用現況",
        },
        "schema_overrides": {"選定年次ビット": pl.Utf8},
    },
    2020: {
        "filename": "L01-2020P-48-01.0a/L01-2020P-2K.csv",
        "price_current": "Ｒ２価格",
        "price_1y_ago": "Ｈ３１価格",
        "price_3y_ago": "Ｈ２９価格",
        "price_5y_ago": "Ｈ２７価格",
        "columns": {
            "経度": "経度",
            "緯度": "緯度",
            "間口（比率）": "間口比率",
            "奥行（比率）": "奥行比率",
            "前面道路区分": "前面道路区分",
            "前面道路の方位区分": "前面道路方位",
            "前面道路の幅員": "前面道路幅員",
            "側道区分": "側道状況",
            "利用の現況": "利用現況",
        },
        "schema_overrides": {"選定年次ビット": pl.Utf8},
    },
    2021: {
        "filename": "L01-2021P-48-01.0a/L01-2021P-2K.csv",
        "price_current": "Ｒ３価格",
        "price_1y_ago": "Ｒ２価格",
        "price_3y_ago": "Ｈ３０価格",
        "price_5y_ago": "Ｈ２８価格",
        "columns": {
            "経度": "経度",
            "緯度": "緯度",
            "間口（比率）": "間口比率",
            "奥行（比率）": "奥行比率",
            "前面道路区分": "前面道路区分",
            "前面道路の方位区分": "前面道路方位",
            "前面道路の幅員": "前面道路幅員",
            "側道区分": "側道状況",
            "利用の現況": "利用現況",
        },
        "schema_overrides": {"選定年次ビット": pl.Utf8},
    },
    2022: {
        "filename": "L01-2022P-48-01.0a/L01-2022P-2K.csv",
        "price_current": "Ｒ４価格",
        "price_1y_ago": "Ｒ３価格",
        "price_3y_ago": "Ｈ３１価格",
        "price_5y_ago": "Ｈ２９価格",
        "columns": {
            "経度": "経度",
            "緯度": "緯度",
            "間口（比率）": "間口比率",
            "奥行（比率）": "奥行比率",
            "前面道路区分": "前面道路区分",
            "前面道路の方位区分": "前面道路方位",
            "前面道路の幅員": "前面道路幅員",
            "側道区分": "側道状況",
            "利用の現況": "利用現況",
        },
        "schema_overrides": {"選定年次ビット": pl.Utf8},
    },
    2023: {
        "filename": "L01-2023P-48-01.0a/L01-2023P-2K.csv",
        "price_current": "Ｒ５価格",
        "price_1y_ago": "Ｒ４価格",
        "price_3y_ago": "Ｒ２価格",
        "price_5y_ago": "Ｈ３０価格",
        "columns": {
            "経度": "経度",
            "緯度": "緯度",
            "間口（比率）": "間口比率",
            "奥行（比率）": "奥行比率",
            "前面道路区分": "前面道路区分",
            "前面道路の方位区分": "前面道路方位",
            "前面道路の幅員": "前面道路幅員",
            "側道区分": "側道状況",
            "利用の現況": "利用現況",
        },
        "schema_overrides": {"選定年次ビット": pl.Utf8},
    },
    2024: {
        "filename": "L01-2024P-48-01.0a/L01-2024P-2K.csv",
        "price_current": "価格R06",
        "price_1y_ago": "価格R05",
        "price_3y_ago": "価格R03",
        "price_5y_ago": "価格R01",
        "columns": {
            "経度": "経度",
            "緯度": "緯度",
            "間口比率": "間口比率",
            "奥行比率": "奥行比率",
            "前面道路区分": "前面道路区分",
            "前面道路方位": "前面道路方位",
            "前面道路幅員": "前面道路幅員",
            "側道状況": "側道状況",
            "利用現況": "利用現況",
            "対前年変動率": "対前年変動率",
        },
        "schema_overrides": {"選定年次フラグ": pl.Utf8},
    },
}


def load_landprice(
    year: int,
    base_path: Optional[Path] = None,
) -> pl.DataFrame:
    """
    指定年度の地価公示データを読み込み、座標変換を行う

    Parameters
    ----------
    year : int
        地価公示の年度（2019, 2020, 2021, 2022, 2024）
    base_path : Path, optional
        地価公示データのベースパス。デフォルトはdata/external/landprice

    Returns
    -------
    pl.DataFrame
        座標変換済みの地価公示データ
        - lon_wgs, lat_wgs: WGS84座標（度）
        - lp_price: 当年価格
        - lp_price_1y_ago: 1年前価格
        - lp_price_3y_ago: 3年前価格
        - lp_price_5y_ago: 5年前価格
        - lp_change_rate: 対前年変動率
        - 間口比率, 奥行比率, 前面道路区分, 前面道路方位, 前面道路幅員, 側道状況, 利用現況
    """
    if year not in YEAR_CONFIG:
        raise ValueError(f"Unsupported year: {year}. Supported: {list(YEAR_CONFIG.keys())}")

    if base_path is None:
        base_path = Path("data/external/landprice")

    config = YEAR_CONFIG[year]
    file_path = base_path / config["filename"]
    if not file_path.exists():
        raise FileNotFoundError(f"Land price data not found: {file_path}")

    # 読み込むカラムリスト
    columns_to_read = list(config["columns"].keys()) + [
        config["price_current"],
        config["price_1y_ago"],
        config["price_3y_ago"],
        config["price_5y_ago"],
    ]

    # データ読み込み（Shift-JIS）
    df = pl.read_csv(
        file_path,
        encoding="shift_jis",
        columns=columns_to_read,
        infer_schema_length=50000,
        schema_overrides=config.get("schema_overrides", {}),
    )

    # カラム名を統一（年度ごとに異なるカラム名を統一）
    rename_map = {k: v for k, v in config["columns"].items() if k != v}
    if rename_map:
        df = df.rename(rename_map)

    # 座標変換: 秒 → 度
    df = df.with_columns([
        (pl.col("経度") / 3600).alias("lon_wgs"),
        (pl.col("緯度") / 3600).alias("lat_wgs"),
    ])

    # 価格カラムをリネーム
    df = df.rename({
        config["price_current"]: "lp_price",
        config["price_1y_ago"]: "lp_price_1y_ago",
        config["price_3y_ago"]: "lp_price_3y_ago",
        config["price_5y_ago"]: "lp_price_5y_ago",
    })

    # 対前年変動率（2019-2022は存在しないので計算で代用）
    if "対前年変動率" not in df.columns:
        # (当年価格 / 前年価格 - 1) * 100
        df = df.with_columns(
            pl.when(pl.col("lp_price_1y_ago") > 0)
            .then((pl.col("lp_price") / pl.col("lp_price_1y_ago") - 1) * 100)
            .otherwise(None)
            .alias("lp_change_rate")
        )
    else:
        df = df.rename({"対前年変動率": "lp_change_rate"})

    # 価格が0のものはnullに変換（データなしを表す）
    price_columns = ["lp_price", "lp_price_1y_ago", "lp_price_3y_ago", "lp_price_5y_ago"]
    df = df.with_columns([
        pl.when(pl.col(col) == 0).then(None).otherwise(pl.col(col)).alias(col)
        for col in price_columns
    ])

    # 元の秒単位座標を削除
    df = df.drop(["経度", "緯度"])

    # 出力カラム順序を統一
    output_columns = [
        "lon_wgs", "lat_wgs",
        "lp_price", "lp_price_1y_ago", "lp_price_3y_ago", "lp_price_5y_ago",
        "lp_change_rate",
        "間口比率", "奥行比率",
        "前面道路区分", "前面道路方位", "前面道路幅員", "側道状況",
        "利用現況",
    ]
    df = df.select([c for c in output_columns if c in df.columns])

    return df


def load_all_landprice(base_path: Optional[Path] = None) -> dict[int, pl.DataFrame]:
    """
    全年度の地価公示データを読み込む

    Parameters
    ----------
    base_path : Path, optional
        地価公示データのベースパス

    Returns
    -------
    dict[int, pl.DataFrame]
        年度をキーとした地価公示データの辞書
    """
    result = {}
    for year in YEAR_CONFIG.keys():
        result[year] = load_landprice(year, base_path)
    return result


if __name__ == "__main__":
    # 動作確認
    for year in [2019, 2020, 2021, 2022, 2024]:
        df = load_landprice(year, Path("data/external/landprice"))
        print(f"\n=== {year}年 ===")
        print(f"レコード数: {len(df):,}")
        print(f"カラム: {df.columns}")
        print(f"座標範囲: lon={df['lon_wgs'].min():.2f}~{df['lon_wgs'].max():.2f}, "
              f"lat={df['lat_wgs'].min():.2f}~{df['lat_wgs'].max():.2f}")
        print(f"当年価格: mean={df['lp_price'].mean():,.0f}, "
              f"median={df['lp_price'].median():,.0f}")
