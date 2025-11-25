# exp005: Log Transform + LR Tuning

## 実験概要

| 項目 | 内容 |
|------|------|
| 実験ID | exp005 |
| 実験名 | log_transform |
| ベース実験 | exp004_age_features |
| 目的 | 目的変数の対数変換 + Learning Rate調整 + 不要特徴量削除 |
| 目標 | CV MAPE 17.0%以下（exp004: 18.60%から1.5pt以上改善）|

---

## 背景・仮説

### 問題点（exp004の分析結果）

#### 1. 低価格帯の予測精度が極端に悪い
| 価格帯 | MAPE | 高誤差(>50%)率 | バイアス |
|--------|------|----------------|----------|
| Q1 (低価格) | **40.89%** | 29.8% | +240万円（過大評価）|
| Q2 | 25.02% | 12.1% | +182万円（過大評価）|
| Q7-Q9 | 12-13% | 1%未満 | 過小評価傾向 |

- 高誤差サンプル（MAPE>50%）の**51%がQ1に集中**
- 低価格帯は過大評価、高価格帯は過小評価の傾向

#### 2. 収束の問題
- `n_estimators=10,000`でearly_stopping**未到達**
- 全foldで`best_iteration=10,000`（モデルはまだ改善中）
- Learning Rate 0.01が小さすぎる

#### 3. 不要な特徴量
- `old_building_flag`（88位/89）：ほぼ無効、`building_age`と冗長

### 仮説

| 施策 | 仮説 | 期待効果 |
|------|------|----------|
| 目的変数の対数変換 | 価格スケールが均一化され、低価格帯の相対誤差が改善 | 1-2pt改善 |
| Learning Rate増加 | 適切な収束点に到達 | 0.5-1pt改善 |
| 不要特徴量削除 | ノイズ削減 | 微改善 |

---

## 変更点

### exp004からの変更一覧

| カテゴリ | 項目 | exp004 | exp005 | 理由 |
|----------|------|--------|--------|------|
| **前処理** | 目的変数 | そのまま | **log1p変換** | 低価格帯の精度改善 |
| ハイパラ | learning_rate | 0.01 | 0.03 | 収束速度向上 |
| ハイパラ | n_estimators | 10,000 | 15,000 | 収束余地確保 |
| 特徴量 | old_building_flag | あり | **削除** | 重要度88位、冗長 |

### 目的変数の対数変換（メイン施策）

```python
# 訓練時
y_train_log = np.log1p(y_train)  # log(1 + y)

# 予測時
pred_log = model.predict(X)
pred = np.expm1(pred_log)  # exp(pred) - 1 で元のスケールに戻す
```

**なぜ対数変換が効くか**：
- 賃料は正の値で右に歪んだ分布（低価格に集中、高価格は少数）
- 対数変換で分布が正規分布に近づく
- モデルが低価格帯と高価格帯を均等に学習できる
- MAPEは相対誤差なので、スケール均一化で全価格帯の精度が向上

### 削除する特徴量
- `old_building_flag`（88位/89）：`building_age`と冗長、情報量なし

### 維持する特徴量（築年数関連）
| 特徴量 | 順位 | 判断 |
|--------|------|------|
| building_age | 8位 | ⭐ 維持（高効果）|
| building_age_bin | 16位 | ○ 維持 |
| building_age_bin_te | 22位 | ○ 維持（TE効果あり）|
| old_and_large_flag | 49位 | △ 維持（様子見）|
| old_and_rural_flag | 65位 | △ 維持（様子見）|

---

## 設計

### ファイル構成
```
exp005_log_transform/
├── README.md                 # この仕様書
├── configs/
│   └── params.yaml          # ハイパーパラメータ設定
├── code/
│   ├── train.py             # 訓練スクリプト（YAML読み込み対応）
│   └── preprocessing.py     # 前処理（exp004ベース、特徴量削除）
├── outputs/
│   ├── oof_predictions_*.csv
│   ├── feature_importance_*.csv
│   └── submission_*.csv
└── notebooks/
    └── (エラー分析用)
```

### 設定ファイル構成（configs/params.yaml）
```yaml
experiment:
  id: "exp005"
  name: "log_transform"

training:
  seed: 42
  n_splits: 3
  early_stopping_rounds: 1000
  target_transform: "log1p"  # ここで対数変換を指定

model:
  params:
    learning_rate: 0.03
    n_estimators: 15000
    # ... その他パラメータ

preprocessing:
  drop_features:
    - "old_building_flag"
```

### 評価指標
- **訓練時メトリック**: RMSE（対数空間）
- **最終評価**: MAPE（元のスケールに戻してから計算）

---

## 成功基準

| 指標 | 目標 | exp004 |
|------|------|--------|
| CV MAPE | < 17.0% | 18.60% |
| Q1（低価格帯）MAPE | < 35% | 40.89% |
| early_stopping | 発動 | 未発動 |

---

## 失敗時の対応

| 問題 | 対応 |
|------|------|
| MAPE悪化 | 対数変換なし（lr=0.03のみ）で再実験 |
| Q1改善なし | 低価格帯専用モデル or 2段階モデル検討 |
| early_stopping未到達 | n_estimators=20,000に増加 |

---

## 実行方法

```bash
source .venv/bin/activate
PYTHONPATH=04_src python 06_experiments/exp005_log_transform/code/train.py
```

---

## 参考情報

### exp004の結果サマリー
- CV MAPE: 18.60% (±0.09%)
- Q1 MAPE: 40.89%（ボトルネック）
- Best iteration: 10,000（全fold、early_stopping未到達）
- 特徴量数: 89個 → exp005では88個（1個削除）
