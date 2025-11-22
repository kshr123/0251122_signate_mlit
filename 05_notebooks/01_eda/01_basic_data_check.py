"""
基本データ確認スクリプト

このスクリプトでは以下を確認します：
1. データの形状（行数、列数）
2. カラム名とデータ型
3. 欠損値の状況
4. 基本統計量
5. 目的変数の分布
"""

import polars as pl
from pathlib import Path

# データパス
DATA_DIR = Path("../../data/raw")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMIT_PATH = DATA_DIR / "sample_submit.csv"

print("=" * 80)
print("不動産価格予測コンペ - 基本データ確認")
print("=" * 80)

# ========================================
# 1. データ読み込み
# ========================================
print("\n[1] データ読み込み中...")

try:
    train = pl.read_csv(TRAIN_PATH)
    test = pl.read_csv(TEST_PATH)
    sample_submit = pl.read_csv(SAMPLE_SUBMIT_PATH)
    print("✅ データ読み込み完了")
except Exception as e:
    print(f"❌ エラー: {e}")
    exit(1)

# ========================================
# 2. データの基本情報
# ========================================
print("\n[2] データの基本情報")
print("-" * 80)

print(f"\n📊 訓練データ (train.csv)")
print(f"  - 行数: {train.height:,}")
print(f"  - 列数: {train.width:,}")
print(f"  - サイズ: {train.estimated_size('mb'):.2f} MB")

print(f"\n📊 テストデータ (test.csv)")
print(f"  - 行数: {test.height:,}")
print(f"  - 列数: {test.width:,}")
print(f"  - サイズ: {test.estimated_size('mb'):.2f} MB")

print(f"\n📊 提出サンプル (sample_submit.csv)")
print(f"  - 行数: {sample_submit.height:,}")
print(f"  - 列数: {sample_submit.width:,}")

# ========================================
# 3. カラム確認
# ========================================
print("\n[3] カラム情報")
print("-" * 80)

print(f"\n訓練データのカラム数: {len(train.columns)}")
print(f"テストデータのカラム数: {len(test.columns)}")

# 訓練データのみに存在するカラム（目的変数）
train_only = set(train.columns) - set(test.columns)
print(f"\n訓練データのみに存在: {train_only}")

# テストデータのみに存在するカラム（id）
test_only = set(test.columns) - set(train.columns)
print(f"テストデータのみに存在: {test_only}")

# ========================================
# 4. データ型確認
# ========================================
print("\n[4] データ型")
print("-" * 80)

# データ型の集計
dtype_counts = {}
for col, dtype in zip(train.columns, train.dtypes):
    dtype_str = str(dtype)
    dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1

print("\n訓練データのデータ型分布:")
for dtype, count in sorted(dtype_counts.items(), key=lambda x: -x[1]):
    print(f"  {dtype}: {count}列")

# ========================================
# 5. 欠損値確認
# ========================================
print("\n[5] 欠損値")
print("-" * 80)

# 訓練データの欠損値
null_counts = train.null_count()
null_summary = pl.DataFrame({
    "column": train.columns,
    "null_count": null_counts.row(0),
    "null_ratio": [count / train.height for count in null_counts.row(0)]
}).sort("null_count", descending=True)

# 欠損値が多い上位10列
print("\n訓練データ - 欠損値が多いカラム（上位10件）:")
print(null_summary.head(10))

# 欠損値なしのカラム数
no_null = (null_summary.filter(pl.col("null_count") == 0)).height
print(f"\n欠損値なしのカラム: {no_null}列 / {train.width}列")

# ========================================
# 6. 目的変数の基本統計量
# ========================================
print("\n[6] 目的変数 (money_room) の基本統計量")
print("-" * 80)

if "money_room" in train.columns:
    target_stats = train.select([
        pl.col("money_room").count().alias("count"),
        pl.col("money_room").null_count().alias("null_count"),
        pl.col("money_room").min().alias("min"),
        pl.col("money_room").quantile(0.25).alias("q25"),
        pl.col("money_room").median().alias("median"),
        pl.col("money_room").quantile(0.75).alias("q75"),
        pl.col("money_room").max().alias("max"),
        pl.col("money_room").mean().alias("mean"),
        pl.col("money_room").std().alias("std"),
    ])

    print(target_stats.transpose(include_header=True))
else:
    print("❌ money_roomカラムが見つかりません")

# ========================================
# 7. 時系列情報確認
# ========================================
print("\n[7] 時系列情報 (target_ym)")
print("-" * 80)

if "target_ym" in train.columns:
    # 訓練データの期間分布
    train_ym_dist = (
        train
        .group_by("target_ym")
        .agg(pl.count().alias("count"))
        .sort("target_ym")
    )
    print("\n訓練データの年月分布:")
    print(train_ym_dist)

    # テストデータの期間分布
    if "target_ym" in test.columns:
        test_ym_dist = (
            test
            .group_by("target_ym")
            .agg(pl.count().alias("count"))
            .sort("target_ym")
        )
        print("\nテストデータの年月分布:")
        print(test_ym_dist)
else:
    print("❌ target_ymカラムが見つかりません")

# ========================================
# 8. サマリー保存
# ========================================
print("\n[8] サマリーデータを保存")
print("-" * 80)

# processed ディレクトリ作成
processed_dir = Path("../../data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

# カラム情報をCSVで保存
column_info = pl.DataFrame({
    "column_name": train.columns,
    "dtype": [str(dtype) for dtype in train.dtypes],
    "null_count": null_counts.row(0),
    "null_ratio": [count / train.height for count in null_counts.row(0)],
})
column_info.write_csv(processed_dir / "column_info.csv")
print(f"✅ カラム情報を保存: {processed_dir / 'column_info.csv'}")

print("\n" + "=" * 80)
print("基本データ確認完了")
print("=" * 80)
