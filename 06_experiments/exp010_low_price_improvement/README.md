# exp010_low_price_improvement

低価格帯（特に広面積×築古）の予測精度改善を目的とした実験。

---

## 概要

| 項目 | 内容 |
|------|------|
| 実験ID | exp010 |
| ベース | exp009_landprice (CV MAPE: 12.48%) |
| 目的 | 低価格帯（広面積×築古）の予測精度改善 |
| 追加特徴量 | 34次元（exp010固有4 + statuses SVD 30） |
| 合計特徴量 | 254次元 |

---

## クイックスタート

```bash
# プロジェクトルートで実行
cd /path/to/project
source .venv/bin/activate
export PYTHONPATH=04_src

# 本番実行（caffeinate -i でスリープ防止）
caffeinate -i python 06_experiments/exp010_low_price_improvement/code/train.py

# テスト実行（軽量: n_estimators=10）
caffeinate -i python 06_experiments/exp010_low_price_improvement/code/train.py --test
```

### MLflow UI

```bash
cd 06_experiments/exp010_low_price_improvement
mlflow ui
# http://localhost:5000
```

---

## ディレクトリ構成

```
exp010_low_price_improvement/
├── README.md               # このファイル
├── SPEC.md                 # 実験仕様書（背景・追加特徴量の詳細）
├── configs/
│   └── experiment.yaml     # 全ハイパーパラメータ（学習設定・モデル・特徴量）
├── code/                   # 実験コード
│   ├── README.md           # コード構成の詳細
│   ├── train.py            # エントリーポイント
│   ├── preprocessing.py    # 前処理
│   ├── pipeline.py         # 特徴量パイプライン
│   ├── constants.py        # パス定義・カラムリスト
│   └── exp010_features.py  # exp010固有Block・関数
├── notebooks/              # 分析ノートブック
│   └── exp010_analysis.ipynb  # 重要度・エラー分析
├── ideas/                  # 次回実験に向けたメモ
│   ├── loss_function.md    # 損失関数改善案
│   └── target_transform.md # 目的変数変換検討
├── outputs/                # 実験出力（Git管理外）
│   └── run_YYYYMMDD_HHMMSS/
│       ├── submission.csv
│       ├── oof_predictions.csv
│       └── feature_importance.json
└── mlruns/                 # MLflow記録（Git管理外）
```

---

## 設定ファイル構成

| ファイル | 役割 | 内容 |
|----------|------|------|
| `configs/experiment.yaml` | 全ハイパーパラメータ | 学習設定、モデルパラメータ、特徴量パラメータ、exp010固有閾値 |
| `code/constants.py` | 定数定義 | パス、カラムリスト（ハイパーパラメータは含まない） |

### experiment.yaml の構成

```yaml
experiment:     # 実験メタ情報（id, name, description）
training:       # 学習設定（seed, n_splits, early_stopping_rounds）
model:          # LightGBMパラメータ（params, params_test）
features:       # 特徴量パラメータ（tfidf, pca, svd等の次元数）
exp010:         # exp010固有（閾値設定）
```

---

## 追加特徴量（exp010固有）

### exp010固有特徴量（4次元）

| 特徴量名 | 計算式 | 意図 |
|----------|--------|------|
| `lp_area_value` | `lp_price × house_area` | 土地価値の目安 |
| `area_age_category` | 面積×築年数で3分類 | 予測困難度を明示 |
| `area_age_cat_te_addr1_1` | カテゴリ×都道府県TE | 地域別価格傾向 |
| `area_age_cat_te_land_youto` | カテゴリ×用途地域TE | 用途別価格傾向 |

### statuses SVD（30次元）

| 項目 | 内容 |
|------|------|
| 元データ | `statuses` カラム（MultiHot形式: `210201/210101/210301...`） |
| ユニークコード数 | 143 |
| 圧縮次元数 | 30 |
| 累積寄与率 | 78.8% |
| 出力特徴量 | `statuses_0` 〜 `statuses_29` |

詳細は [SPEC.md](./SPEC.md) を参照。

---

## 実験結果

### 本番実行

| 指標 | 値 |
|------|-----|
| CV MAPE | **12.20%** |
| ベース比較 | 12.48% → 12.20%（**-0.28pt改善**） |

### exp010追加特徴量の重要度

| 特徴量 | 重要度 | 全体順位 | 備考 |
|--------|--------|----------|------|
| `statuses_svd_0` | 21,709 | #5 | 最重要SVD成分 |
| `area_age_category` | 15,801 | #15 | 面積×築年数カテゴリ |
| `area_age_cat_te_addr1_1` | 13,363 | #25 | カテゴリ×都道府県TE |
| `lp_area_value` | 7,633 | #58 | 土地価値の目安 |
| `statuses_svd_1` | 13,156 | #28 | 2番目のSVD成分 |

**statuses SVD全体**: 30次元中、上位10次元が平均13,000超の重要度で貢献

### 詳細分析

詳細は [notebooks/exp010_analysis.ipynb](./notebooks/exp010_analysis.ipynb) を参照

---

## 関連ドキュメント

- [SPEC.md](./SPEC.md) - 実験仕様書
- [code/README.md](./code/README.md) - コード構成
- [exp009](../exp009_landprice/) - ベース実験
