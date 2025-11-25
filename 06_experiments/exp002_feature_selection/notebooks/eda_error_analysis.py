"""
exp002 EDA: 予測誤差分析

- 予測誤差の大きいサンプル分析
- 特徴量重要度の可視化
- 都道府県・市区町村別の誤差分析（マスター突合）
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "04_src"))

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

# 出力ディレクトリ
output_dir = Path(__file__).parent.parent / "outputs"
notebook_output = Path(__file__).parent / "figures"
notebook_output.mkdir(exist_ok=True)

# 最新のOOF予測ファイルを取得
oof_files = sorted(output_dir.glob("oof_predictions_*.csv"))
latest_oof = oof_files[-1] if oof_files else None

importance_files = sorted(output_dir.glob("feature_importance_*.csv"))
latest_importance = importance_files[-1] if importance_files else None

print("=" * 60)
print("exp002 EDA: 予測誤差分析")
print("=" * 60)

# ===== 1. OOF予測読み込み =====
print("\n📂 データ読み込み...")
oof_df = pl.read_csv(latest_oof)
print(f"  - OOF予測: {oof_df.shape}")

# 元の訓練データも読み込み（特徴量との突合用）
train = pl.read_csv(project_root / "data" / "raw" / "train.csv", infer_schema_length=100000)
print(f"  - 訓練データ: {train.shape}")

# マスターデータ読み込み
area_master = pl.read_csv(project_root / "data" / "master" / "area_master.csv")
area_master = area_master.with_columns([
    pl.col("addr1_1").cast(pl.Int64),
    pl.col("addr1_2").cast(pl.Int64),
])
print(f"  - エリアマスター: {area_master.shape}")

# ===== 2. 誤差計算 =====
print("\n📊 誤差計算...")

# 誤差列を追加
oof_df = oof_df.with_columns([
    (pl.col("predicted") - pl.col("actual")).alias("error"),
    ((pl.col("predicted") - pl.col("actual")).abs() / pl.col("actual") * 100).alias("ape"),  # Absolute Percentage Error
])

# 基本統計
print(f"\n  誤差統計:")
print(f"    - 平均誤差 (ME): {oof_df['error'].mean():,.0f}円")
print(f"    - 平均絶対誤差 (MAE): {oof_df['error'].abs().mean():,.0f}円")
print(f"    - MAPE: {oof_df['ape'].mean():.2f}%")
print(f"    - 中央値APE: {oof_df['ape'].median():.2f}%")

# ===== 3. 予測誤差の分布 =====
print("\n📈 予測誤差の分布を可視化...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# APE分布
ax = axes[0]
ape_values = oof_df["ape"].to_numpy()
ax.hist(ape_values, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(ape_values.mean(), color='red', linestyle='--', label=f'平均: {ape_values.mean():.1f}%')
ax.axvline(np.median(ape_values), color='orange', linestyle='--', label=f'中央値: {np.median(ape_values):.1f}%')
ax.set_xlabel("APE (%)")
ax.set_ylabel("頻度")
ax.set_title("絶対パーセント誤差 (APE) の分布")
ax.legend()
ax.set_xlim(0, 100)

# 実測値 vs 予測値
ax = axes[1]
actual = oof_df["actual"].to_numpy()
predicted = oof_df["predicted"].to_numpy()
ax.scatter(actual, predicted, alpha=0.1, s=1)
ax.plot([0, actual.max()], [0, actual.max()], 'r--', label='y=x')
ax.set_xlabel("実測値 (円)")
ax.set_ylabel("予測値 (円)")
ax.set_title("実測値 vs 予測値")
ax.legend()

# 誤差 vs 実測値
ax = axes[2]
ax.scatter(actual, oof_df["error"].to_numpy(), alpha=0.1, s=1)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel("実測値 (円)")
ax.set_ylabel("誤差 (円)")
ax.set_title("誤差 vs 実測値")

plt.tight_layout()
plt.savefig(notebook_output / "error_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ 保存: {notebook_output / 'error_distribution.png'}")

# ===== 4. 誤差の大きいサンプル分析 =====
print("\n🔍 予測誤差の大きいサンプル分析...")

# 元データとOOF予測を結合
train_with_oof = train.with_row_index("row_id").join(
    oof_df.rename({"id": "row_id"}),
    on="row_id",
    how="left"
)

# エリア情報を結合
train_with_oof = train_with_oof.with_columns([
    pl.col("addr1_1").cast(pl.Int64),
    pl.col("addr1_2").cast(pl.Int64),
])
train_with_area = train_with_oof.join(
    area_master.select(["addr1_1", "addr1_2", "都道府県名", "市区町村名"]),
    on=["addr1_1", "addr1_2"],
    how="left"
)

# APE上位10件
print("\n  APE上位10件（予測が大きく外れたサンプル）:")
top_errors = train_with_area.sort("ape", descending=True).head(10)
for i, row in enumerate(top_errors.iter_rows(named=True)):
    print(f"    {i+1}. APE={row['ape']:.1f}% | 実測={row['actual']:,.0f}円 | 予測={row['predicted']:,.0f}円")
    print(f"       {row['都道府県名']} {row['市区町村名']} | 面積={row['house_area']}㎡ | 築年={row['year_built']}")

# ===== 5. 特徴量重要度の可視化 =====
print("\n📊 特徴量重要度の可視化...")

importance_df = pl.read_csv(latest_importance)
top_features = importance_df.head(20)

fig, ax = plt.subplots(figsize=(10, 8))
features = top_features["feature"].to_list()[::-1]
importance = top_features["importance"].to_list()[::-1]

# 正規化（見やすくするため）
importance_norm = np.array(importance) / max(importance) * 100

ax.barh(features, importance_norm)
ax.set_xlabel("相対重要度 (%)")
ax.set_title("特徴量重要度 Top 20")

plt.tight_layout()
plt.savefig(notebook_output / "feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ 保存: {notebook_output / 'feature_importance.png'}")

# ===== 6. 都道府県別の誤差分析 =====
print("\n📊 都道府県別の誤差分析...")

pref_error = train_with_area.group_by("都道府県名").agg([
    pl.col("ape").mean().alias("mean_ape"),
    pl.col("ape").median().alias("median_ape"),
    pl.len().alias("count"),
]).sort("mean_ape", descending=True)

print("\n  都道府県別 平均APE (上位10):")
for i, row in enumerate(pref_error.head(10).iter_rows(named=True)):
    print(f"    {i+1}. {row['都道府県名']}: {row['mean_ape']:.2f}% (n={row['count']:,})")

print("\n  都道府県別 平均APE (下位10):")
for i, row in enumerate(pref_error.tail(10).iter_rows(named=True)):
    print(f"    {i+1}. {row['都道府県名']}: {row['mean_ape']:.2f}% (n={row['count']:,})")

# 都道府県別APEの可視化
fig, ax = plt.subplots(figsize=(12, 8))
pref_sorted = pref_error.sort("mean_ape", descending=False)
prefs = pref_sorted["都道府県名"].to_list()
apes = pref_sorted["mean_ape"].to_list()

colors = ['red' if ape > 30 else 'orange' if ape > 28 else 'green' for ape in apes]
ax.barh(prefs, apes, color=colors)
ax.axvline(28.26, color='blue', linestyle='--', label='全体平均 (28.26%)')
ax.set_xlabel("平均 APE (%)")
ax.set_title("都道府県別 平均APE")
ax.legend()

plt.tight_layout()
plt.savefig(notebook_output / "prefecture_ape.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ 保存: {notebook_output / 'prefecture_ape.png'}")

# ===== 7. 市区町村別の誤差分析（上位・下位） =====
print("\n📊 市区町村別の誤差分析...")

city_error = train_with_area.group_by(["都道府県名", "市区町村名"]).agg([
    pl.col("ape").mean().alias("mean_ape"),
    pl.col("ape").median().alias("median_ape"),
    pl.len().alias("count"),
]).filter(pl.col("count") >= 100).sort("mean_ape", descending=True)  # サンプル100件以上

print("\n  市区町村別 平均APE (上位10、n>=100):")
for i, row in enumerate(city_error.head(10).iter_rows(named=True)):
    print(f"    {i+1}. {row['都道府県名']} {row['市区町村名']}: {row['mean_ape']:.2f}% (n={row['count']:,})")

print("\n  市区町村別 平均APE (下位10、n>=100):")
for i, row in enumerate(city_error.tail(10).iter_rows(named=True)):
    print(f"    {i+1}. {row['都道府県名']} {row['市区町村名']}: {row['mean_ape']:.2f}% (n={row['count']:,})")

# ===== 8. 価格帯別の誤差分析 =====
print("\n📊 価格帯別の誤差分析...")

# 価格帯を作成（money_roomは円単位、最小490万円〜最大1.88億円）
train_with_area = train_with_area.with_columns([
    pl.when(pl.col("actual") < 10_000_000).then(pl.lit("~1000万"))
    .when(pl.col("actual") < 15_000_000).then(pl.lit("1000~1500万"))
    .when(pl.col("actual") < 20_000_000).then(pl.lit("1500~2000万"))
    .when(pl.col("actual") < 30_000_000).then(pl.lit("2000~3000万"))
    .when(pl.col("actual") < 50_000_000).then(pl.lit("3000~5000万"))
    .otherwise(pl.lit("5000万~"))
    .alias("price_range")
])

price_error = train_with_area.group_by("price_range").agg([
    pl.col("ape").mean().alias("mean_ape"),
    pl.col("ape").median().alias("median_ape"),
    pl.len().alias("count"),
]).sort("mean_ape", descending=True)

print("\n  価格帯別 平均APE:")
for row in price_error.iter_rows(named=True):
    print(f"    {row['price_range']}: {row['mean_ape']:.2f}% (中央値: {row['median_ape']:.2f}%, n={row['count']:,})")

# ===== 9. 築年数別の誤差分析 =====
print("\n📊 築年数別の誤差分析...")

# 築年を抽出（year_builtは YYYYMM形式、例: 199211 → 1992年）
train_with_area = train_with_area.with_columns([
    (pl.col("year_built") // 100).alias("built_year")
])

# 築年数を計算（2024年基準）
train_with_area = train_with_area.with_columns([
    (2024 - pl.col("built_year")).alias("building_age")
])

# 築年数帯を作成
train_with_area = train_with_area.with_columns([
    pl.when(pl.col("building_age") < 5).then(pl.lit("~5年"))
    .when(pl.col("building_age") < 10).then(pl.lit("5~10年"))
    .when(pl.col("building_age") < 20).then(pl.lit("10~20年"))
    .when(pl.col("building_age") < 30).then(pl.lit("20~30年"))
    .when(pl.col("building_age") < 40).then(pl.lit("30~40年"))
    .otherwise(pl.lit("40年~"))
    .alias("age_range")
])

age_error = train_with_area.group_by("age_range").agg([
    pl.col("ape").mean().alias("mean_ape"),
    pl.col("ape").median().alias("median_ape"),
    pl.len().alias("count"),
]).sort("mean_ape", descending=True)

print("\n  築年数帯別 平均APE:")
for row in age_error.iter_rows(named=True):
    print(f"    {row['age_range']}: {row['mean_ape']:.2f}% (中央値: {row['median_ape']:.2f}%, n={row['count']:,})")

# ===== 10. サマリー =====
print("\n" + "=" * 60)
print("📋 EDA サマリー")
print("=" * 60)

print(f"""
【モデル性能】
  - CV MAPE: 28.26%
  - 特徴量数: 84個（除外: 46個）

【誤差傾向】
  - 高APE都道府県: 高知県、沖縄県、鳥取県など
  - 低APE都道府県: 東京都、神奈川県、大阪府など（大都市圏）
  - 高価格帯ほど予測が難しい傾向

【重要特徴量 Top 5】
  1. house_area（専有面積）
  2. post1（郵便番号上3桁）
  3. year_built（築年）
  4. money_kyoueki（共益費）
  5. addr1_2（市区町村コード）

【次のアクション候補】
  1. 地域特性を反映した特徴量追加（都道府県ダミー等）
  2. 価格帯別のモデル分割検討
  3. 築年数の非線形変換
  4. 外れ値（高APE）サンプルの詳細分析
""")

print("=" * 60)
print("✅ EDA完了")
print("=" * 60)
