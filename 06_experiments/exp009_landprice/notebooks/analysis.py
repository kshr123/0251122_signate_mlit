# %% [markdown]
# # exp009 地価公示特徴量 - 分析レポート
#
# ## 実験概要
# - **実験ID**: exp009_landprice
# - **ベース**: exp008_reform_features (CV MAPE: 13.44%)
# - **追加特徴量**: 地価公示データ (39次元)
# - **結果**: CV MAPE 12.53% (約0.9pt改善)

# %% [markdown]
# ## 1. セットアップ

# %%
import sys
from pathlib import Path

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

# パス設定
exp_dir = Path(__file__).resolve().parent.parent
output_dir = exp_dir / "outputs"

# 日本語フォント設定
plt.rcParams['font.family'] = 'IPAexGothic'
plt.rcParams['axes.unicode_minus'] = False

# %% [markdown]
# ## 2. 特徴量重要度分析

# %%
# 特徴量重要度の読み込み
importance_df = pl.read_csv(output_dir / "feature_importance_20251127_091113.csv")
print(f"特徴量数: {len(importance_df)}")
importance_df.head(20)

# %%
# 上位30特徴量の可視化
top_n = 30
top_features = importance_df.head(top_n)

fig, ax = plt.subplots(figsize=(10, 10))
y_pos = np.arange(top_n)

ax.barh(y_pos, top_features["importance"].to_numpy()[::-1], color='steelblue')
ax.set_yticks(y_pos)
ax.set_yticklabels(top_features["feature"].to_list()[::-1])
ax.set_xlabel("重要度")
ax.set_title(f"特徴量重要度 Top {top_n}")
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "feature_importance_top30.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
# 地価公示特徴量の重要度
lp_features = importance_df.filter(
    pl.col("feature").str.starts_with("lp_") | pl.col("feature").str.contains("_lp_")
).sort("importance", descending=True)

print(f"\n地価公示特徴量数: {len(lp_features)}")
print(f"地価公示特徴量の重要度合計: {lp_features['importance'].sum():.0f}")
print(f"全体に占める割合: {lp_features['importance'].sum() / importance_df['importance'].sum() * 100:.1f}%")
print("\n地価公示特徴量の重要度:")
lp_features

# %%
# 地価公示特徴量の可視化
fig, ax = plt.subplots(figsize=(10, 8))
y_pos = np.arange(len(lp_features))

ax.barh(y_pos, lp_features["importance"].to_numpy()[::-1], color='coral')
ax.set_yticks(y_pos)
ax.set_yticklabels(lp_features["feature"].to_list()[::-1])
ax.set_xlabel("重要度")
ax.set_title("地価公示特徴量の重要度")
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "landprice_feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 3. エラー分析

# %%
# OOF予測の読み込み
oof_df = pl.read_csv(output_dir / "oof_predictions_20251127_091113.csv")
print(f"OOF予測数: {len(oof_df)}")
oof_df.head()

# %%
# エラー計算
oof_df = oof_df.with_columns([
    (pl.col("predicted") - pl.col("actual")).alias("error"),
    ((pl.col("predicted") - pl.col("actual")).abs() / pl.col("actual") * 100).alias("ape"),
])

print(f"MAPE: {oof_df['ape'].mean():.4f}%")
print(f"MAE: {oof_df['error'].abs().mean():,.0f}円")
print(f"RMSE: {np.sqrt((oof_df['error'] ** 2).mean()):,.0f}円")

# %%
# 予測値 vs 実測値の散布図
fig, ax = plt.subplots(figsize=(8, 8))

actual = oof_df["actual"].to_numpy()
predicted = oof_df["predicted"].to_numpy()

ax.scatter(actual, predicted, alpha=0.1, s=1)
ax.plot([0, actual.max()], [0, actual.max()], 'r--', label='y=x')
ax.set_xlabel("実測値（円）")
ax.set_ylabel("予測値（円）")
ax.set_title("予測値 vs 実測値")
ax.legend()
ax.grid(alpha=0.3)

ax.set_xscale('log')
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(output_dir / "prediction_vs_actual.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
# APEの分布
fig, ax = plt.subplots(figsize=(10, 5))

ape = oof_df["ape"].to_numpy()
ape_clipped = np.clip(ape, 0, 100)

ax.hist(ape_clipped, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(ape.mean(), color='red', linestyle='--', label=f'平均APE: {ape.mean():.2f}%')
ax.axvline(np.median(ape), color='orange', linestyle='--', label=f'中央値APE: {np.median(ape):.2f}%')
ax.set_xlabel("絶対パーセント誤差 (%)")
ax.set_ylabel("頻度")
ax.set_title("予測誤差の分布")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "ape_distribution.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
# 価格帯別のエラー分析
price_bins = [10_000_000, 20_000_000, 30_000_000, 50_000_000, 100_000_000]
price_labels = ['~1000万', '1000-2000万', '2000-3000万', '3000-5000万', '5000万-1億', '1億~']

oof_df = oof_df.with_columns([
    pl.col("actual").cut(price_bins, labels=price_labels).alias("price_bin")
])

price_bin_stats = oof_df.group_by("price_bin").agg([
    pl.col("ape").mean().alias("mape"),
    pl.col("ape").median().alias("median_ape"),
    pl.len().alias("count")
]).sort("price_bin")

print("価格帯別エラー分析:")
price_bin_stats

# %%
# 価格帯別MAPEの可視化
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(price_bin_stats))
width = 0.35

bars1 = ax.bar(x - width/2, price_bin_stats["mape"].to_numpy(), width, label='平均APE', color='steelblue')
bars2 = ax.bar(x + width/2, price_bin_stats["median_ape"].to_numpy(), width, label='中央値APE', color='coral')

ax.set_xlabel("価格帯")
ax.set_ylabel("APE (%)")
ax.set_title("価格帯別の予測誤差")
ax.set_xticks(x)
ax.set_xticklabels(price_bin_stats["price_bin"].to_list())
ax.legend()
ax.grid(axis='y', alpha=0.3)

for i, (bar, count) in enumerate(zip(bars1, price_bin_stats["count"].to_list())):
    ax.text(bar.get_x() + bar.get_width(), bar.get_height() + 0.5,
            f'n={count:,}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / "mape_by_price_bin.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
# 大きなエラーのサンプル分析
large_error_threshold = 50
large_errors = oof_df.filter(pl.col("ape") > large_error_threshold).sort("ape", descending=True)

print(f"\nAPE > {large_error_threshold}% のサンプル数: {len(large_errors)} ({len(large_errors)/len(oof_df)*100:.2f}%)")
print("\n最大エラーTop10:")
large_errors.head(10)

# %% [markdown]
# ## 4. 実験結果サマリー

# %%
print("=" * 60)
print("exp009 実験結果サマリー")
print("=" * 60)
print(f"\n【スコア】")
print(f"  CV MAPE: 12.53%")
print(f"  ベースライン (exp008): 13.44%")
print(f"  改善: 0.91pt")
print(f"\n【特徴量】")
print(f"  総特徴量数: 220")
print(f"  地価公示特徴量: 39次元")
print(f"  地価公示特徴量の重要度割合: {lp_features['importance'].sum() / importance_df['importance'].sum() * 100:.1f}%")
print(f"\n【モデル】")
print(f"  LightGBM (max_depth=10, num_leaves=127, min_child_samples=15)")
print(f"  learning_rate=0.05, n_estimators=50000")
print(f"  平均best_iteration: ~15,000")
print(f"\n【エラー分析】")
print(f"  MAPE: {oof_df['ape'].mean():.2f}%")
print(f"  中央値APE: {np.median(oof_df['ape'].to_numpy()):.2f}%")
print(f"  大エラー(>50%)率: {len(large_errors)/len(oof_df)*100:.2f}%")
print("=" * 60)

# %% [markdown]
# ## 5. 次回改善方針

# %%
print("""
============================================================
次回改善方針
============================================================

【高優先度】
1. 高価格帯(1億円以上)のエラー改善
   - 高価格物件向けの特徴量追加
   - 価格帯別モデルの検討

2. 大エラーサンプルの分析
   - 外れ値/異常値の特定と処理
   - 特殊物件の識別特徴量

【中優先度】
3. 地価公示特徴量の拡張
   - 時系列特徴量の追加（過去変動率）
   - 周辺地価の統計量（最小/最大/標準偏差）

4. ハイパーパラメータチューニング
   - Optunaによる自動チューニング
   - 正則化パラメータの調整

【低優先度】
5. アンサンブル
   - CatBoost, XGBoostとのブレンド
   - スタッキング

6. 新規外部データ
   - 人口統計データ
   - 駅乗降客数データ
============================================================
""")
