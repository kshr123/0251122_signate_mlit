# exp001_baseline - ベースラインモデル

**実験日**: 2025-11-24
**実験者**: System
**ステータス**: ✅ 完了

---

## 📝 実験概要

シンプルで再現性のあるベースラインモデルを構築し、初回提出を行う。

**目的**:
- 最小限の特徴量で動作するベースラインを確立
- 3-Fold CVでの性能評価
- 実験管理フロー（MLflow）の確立

---

## 🎯 実験結果

### クロスバリデーション結果

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| **MAPE** | **28.3432%** | **0.0883%** | 28.2762% | 28.4680% |

### Fold別スコア

| Fold | MAPE (%) | Best Iteration |
|------|----------|----------------|
| 1    | 28.4680  | 100            |
| 2    | 28.2762  | 100            |
| 3    | 28.2854  | 100            |

---

## 🔧 実験設定

### モデル

- **アルゴリズム**: LightGBM (GBDT)
- **目的関数**: Regression
- **評価指標**: MAPE

### ハイパーパラメータ

```yaml
objective: regression
metric: mape
boosting: gbdt
learning_rate: 0.05
num_leaves: 31
max_depth: -1
min_child_samples: 20
subsample: 0.8
subsample_freq: 1
colsample_bytree: 0.8
reg_alpha: 0.0
reg_lambda: 0.0
random_state: 42
num_boost_round: 100
early_stopping_rounds: 100
```

### 訓練設定

- **CV手法**: 3-Fold KFold (shuffle=True)
- **シード**: 42
- **Early Stopping**: 100 rounds

---

## 📊 データ

### データセット

- **Train**: 363,924 samples × 149 features
- **Test**: 112,437 samples × 149 features

### 前処理

**前処理クラス**: `SimplePreprocessor`

1. `target_ym` の分解 → `target_year`, `target_month`
2. 低カーディナリティカラムの抽出（閾値: 50）
3. 欠損値は未補完（LightGBMの自動処理に任せる）

### 特徴量

- **特徴量数**: 106
- **数値特徴量**: 96
- **カテゴリカル特徴量**: 8 (低カーディナリティのみ)

**カテゴリカル特徴量リスト**:
- `building_name_ruby`
- `reform_exterior`
- `name_ruby`
- `school_ele_code`
- `school_jun_code`
- `money_hoshou_company`
- `free_rent_duration`
- `free_rent_gen_timing`

**注意**: `target_year`, `target_month` も含む

---

## 🐛 発生した問題と解決策

### 問題1: train/testでデータ型が異なるカラム

**症状**:
```
ValueError: pandas dtypes must be int, float or bool.
Fields with bad pandas dtypes: traffic_car: object
```

**原因**:
- Train: `traffic_car` が `Int64` 型
- Test: `traffic_car` が `String` 型

元データの型が異なるため、`SimplePreprocessor` で異なる扱いを受け、testデータのみ文字列として残る。

**解決策**:
trainとtestの両方で文字列型カラムを検出し、すべてCategorical → ordinalに変換:

```python
# trainとtestで型が異なる可能性があるため、両方で文字列型を検出
string_cols_train = [col for col in X_train.columns if X_train[col].dtype == pl.Utf8]
string_cols_test = [col for col in X_test.columns if X_test[col].dtype == pl.Utf8]
string_cols = list(set(string_cols_train + string_cols_test))

# すべての文字列型カラムを数値に変換
for col in string_cols:
    if col in X_train.columns and X_train[col].dtype == pl.Utf8:
        X_train = X_train.with_columns(
            pl.col(col).cast(pl.Categorical).to_physical().alias(col)
        )
    if col in X_test.columns and X_test[col].dtype == pl.Utf8:
        X_test = X_test.with_columns(
            pl.col(col).cast(pl.Categorical).to_physical().alias(col)
        )
```

---

## 📂 生成ファイル

- **提出ファイル**: `submission_20251124_122920.csv`
- **MLflow Run ID**: `b1541b503505448d8567f82d22166a1d`

---

## 🔄 次のステップ

1. **特徴量追加**:
   - 住所情報（都道府県・市区町村名）の追加
   - 高カーディナリティカラムのエンコーディング

2. **モデル改善**:
   - ハイパーパラメータチューニング（Optuna）
   - アンサンブル

3. **リファクタリング**:
   - `DataLoader` でのデータ型統一
   - 型チェック機能の追加

---

## 📝 メモ

- すべてのFoldで `best_iteration=100` → Early Stopping未発動
  - `num_boost_round` を増やす余地あり
- CV標準偏差が小さい（0.0883%）→ モデルが安定している
- ベースライン完成により、以降の実験との比較が可能に

---

**実験担当**: Claude Code
**最終更新**: 2025-11-24 12:30
