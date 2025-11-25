"""
マスター結合済みデータファイル作成スクリプト

data/raw/train.csv, test.csv に以下を結合して data/processed/ に保存:
- エリアマスター（都道府県名、市区町村名）
- 日付カラムの年月変換（building_create_date, building_modify_date）

Usage:
    python 08_scripts/create_enriched_data.py
"""

import polars as pl
from pathlib import Path
import os


def create_enriched_data():
    """マスター結合済みデータを作成"""

    print("=" * 60)
    print("train/test データにマスター情報を結合")
    print("=" * 60)

    # パス設定
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    master_dir = project_root / "data" / "master"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # マスターデータ読み込み
    print("\n📂 マスターデータ読み込み...")
    area_master = pl.read_csv(master_dir / "area_master.csv")
    print(f"  - area_master: {area_master.shape}")

    # addr1_1, addr1_2の型を統一
    area_master = area_master.with_columns([
        pl.col("addr1_1").cast(pl.Int64),
        pl.col("addr1_2").cast(pl.Int64),
    ])

    # train読み込み
    print("\n📂 train.csv 読み込み...")
    train = pl.read_csv(raw_dir / "train.csv", infer_schema_length=100000)
    print(f"  - 元サイズ: {train.shape}")

    # test読み込み
    print("\n📂 test.csv 読み込み...")
    test = pl.read_csv(raw_dir / "test.csv", infer_schema_length=100000)
    print(f"  - 元サイズ: {test.shape}")

    # 型を揃える
    train = train.with_columns([
        pl.col("addr1_1").cast(pl.Int64),
        pl.col("addr1_2").cast(pl.Int64),
    ])
    test = test.with_columns([
        pl.col("addr1_1").cast(pl.Int64),
        pl.col("addr1_2").cast(pl.Int64),
    ])

    # エリア情報結合（都道府県名、市区町村名を追加）
    print("\n🔧 エリア情報を結合...")
    train_enriched = train.join(
        area_master.select(["addr1_1", "addr1_2", "都道府県名", "市区町村名"]),
        on=["addr1_1", "addr1_2"],
        how="left"
    )
    test_enriched = test.join(
        area_master.select(["addr1_1", "addr1_2", "都道府県名", "市区町村名"]),
        on=["addr1_1", "addr1_2"],
        how="left"
    )

    print(f"  - train: {train.shape} → {train_enriched.shape}")
    print(f"  - test: {test.shape} → {test_enriched.shape}")

    # 日付カラムを年月形式に変換（target_ymと同じ YYYYMM 整数形式）
    print("\n🔧 日付カラムを年月形式に変換...")
    date_cols = ["building_create_date", "building_modify_date"]

    for col in date_cols:
        if col in train_enriched.columns:
            # "YYYY-MM-DD HH:MM:SS" → YYYYMM（整数）
            # 例: "2014-06-27 21:09:41" → 201406
            train_enriched = train_enriched.with_columns(
                pl.col(col).cast(pl.Utf8).str.slice(0, 7).str.replace("-", "").cast(pl.Int64).alias(col)
            )
            test_enriched = test_enriched.with_columns(
                pl.col(col).cast(pl.Utf8).str.slice(0, 7).str.replace("-", "").cast(pl.Int64).alias(col)
            )
            print(f"  - {col}: 'YYYY-MM-DD HH:MM:SS' → YYYYMM (Int64)")

    # 結合結果の確認
    train_null_pref = train_enriched["都道府県名"].null_count()
    test_null_pref = test_enriched["都道府県名"].null_count()
    print(f"\n📊 結合結果:")
    print(f"  - train 都道府県名 NULL数: {train_null_pref} / {len(train_enriched)}")
    print(f"  - test 都道府県名 NULL数: {test_null_pref} / {len(test_enriched)}")

    # サンプル表示
    print("\n📋 サンプルデータ（train先頭5行）:")
    print(train_enriched.select(["addr1_1", "addr1_2", "都道府県名", "市区町村名", "money_room"]).head(5))

    # 保存
    print("\n💾 保存中...")
    train_path = processed_dir / "train_enriched.csv"
    test_path = processed_dir / "test_enriched.csv"

    train_enriched.write_csv(train_path)
    test_enriched.write_csv(test_path)

    print(f"  ✓ {train_path} ({train_enriched.shape[0]:,} rows × {train_enriched.shape[1]} cols)")
    print(f"  ✓ {test_path} ({test_enriched.shape[0]:,} rows × {test_enriched.shape[1]} cols)")

    # ファイルサイズ確認
    train_size = os.path.getsize(train_path) / (1024 * 1024)
    test_size = os.path.getsize(test_path) / (1024 * 1024)
    print(f"\n📦 ファイルサイズ:")
    print(f"  - train_enriched.csv: {train_size:.1f} MB")
    print(f"  - test_enriched.csv: {test_size:.1f} MB")

    print("\n✅ 完了!")


if __name__ == "__main__":
    create_enriched_data()
