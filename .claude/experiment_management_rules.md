# 実験管理ルール

> **このドキュメントについて**: 実験の完全な再現性を保証するためのディレクトリ構造とファイル管理ルール

---

## 📁 実験ディレクトリ構造

各実験は `06_experiments/expXXX_<実験名>/` 配下に以下の構造で管理する：

```
06_experiments/
└── exp001_baseline/
    ├── README.md                    # 実験サマリー（必須）
    ├── requirements.txt             # 依存関係（バージョン固定）
    │
    ├── configs/                     # パラメータ設定
    │   ├── model_params.yaml       # モデルハイパーパラメータ
    │   ├── preprocessing_config.yaml # 前処理設定
    │   ├── training_config.yaml    # 訓練設定（CV, seed等）
    │   └── feature_config.yaml     # 特徴量設定
    │
    ├── code/                        # 実験固有コード
    │   ├── train.py                # 訓練スクリプト
    │   ├── predict.py              # 推論スクリプト
    │   └── preprocessing.py        # 前処理詳細
    │
    ├── features/                    # 特徴量情報
    │   ├── feature_list.txt        # 使用特徴量リスト
    │   ├── feature_engineering.md  # 特徴量エンジニアリング詳細
    │   ├── feature_importance.csv  # 特徴量重要度
    │   └── categorical_features.txt # カテゴリカル特徴量リスト
    │
    ├── outputs/                     # 予測結果・メトリクス
    │   ├── submission_*.csv        # 提出ファイル
    │   ├── oof_predictions.csv     # Out-of-Fold予測値
    │   ├── cv_scores.json          # Fold別スコア詳細
    │   └── metrics.json            # 評価指標
    │
    ├── models/                      # 学習済みモデル
    │   ├── fold_1.txt              # Fold 1モデル
    │   ├── fold_2.txt              # Fold 2モデル
    │   ├── fold_3.txt              # Fold 3モデル
    │   └── final_model.txt         # 全データ再訓練モデル
    │
    ├── visualizations/              # 可視化
    │   ├── feature_importance.png  # 特徴量重要度
    │   ├── cv_scores.png           # CVスコア分布
    │   ├── prediction_vs_actual.png # 予測vs実測
    │   ├── residual_plot.png       # 残差プロット
    │   └── learning_curve.png      # 学習曲線
    │
    ├── analysis/                    # 分析結果
    │   ├── error_analysis.md       # エラー分析レポート
    │   ├── segment_analysis.csv    # セグメント別統計
    │   └── outlier_analysis.csv    # 外れ値分析
    │
    └── logs/                        # 実行ログ
        ├── training.log            # 訓練時の標準出力
        └── mlflow_run_id.txt       # MLflow Run ID
```

---

## 📝 各ディレクトリの詳細

### 1. `configs/` - パラメータ設定

**目的**: 実験のすべての設定を再現可能な形で記録

#### `model_params.yaml`
モデルのハイパーパラメータを記録：

```yaml
model_type: LightGBM
params:
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
  verbose: -1
  force_row_wise: true

training:
  num_boost_round: 100
  early_stopping_rounds: 100
```

#### `preprocessing_config.yaml`
前処理の設定：

```yaml
preprocessor: SimplePreprocessor
params:
  cardinality_threshold: 50
  fill_missing: false
  numeric_fill_value: -999
  categorical_fill_value: "missing"

target_transform:
  target_ym_split: true  # year/month分解

exclude_columns:
  - id
  - money_room
  - target_ym
```

#### `training_config.yaml`
訓練全般の設定：

```yaml
seed: 42
cv:
  method: KFold
  n_splits: 3
  shuffle: true
  random_state: 42

validation:
  metric: mape

data:
  train_path: data/raw/train.csv
  test_path: data/raw/test.csv
  add_address_columns: false
```

#### `feature_config.yaml`
特徴量の設定：

```yaml
# 使用した特徴量セット
feature_set: baseline_v1

# 数値特徴量
numeric_features:
  count: 96
  source: raw  # そのまま使用

# カテゴリカル特徴量
categorical_features:
  method: label_encoding
  cardinality_threshold: 50
  features:
    - building_name_ruby
    - reform_exterior
    - name_ruby
    - school_ele_code
    - school_jun_code
    - money_hoshou_company
    - free_rent_duration
    - free_rent_gen_timing

# 除外した特徴量
excluded_features:
  high_cardinality: []  # 高カーディナリティは除外

# 生成した特徴量
generated_features:
  - target_year   # target_ym // 100
  - target_month  # target_ym % 100

total_features: 106
```

---

### 2. `code/` - 実験固有コード

**目的**: この実験を実行するための完全なコード

#### `train.py`
- `04_src/training/train_baseline.py` のコピー + 実験固有の調整
- このファイルを実行すれば実験が再現できる
- **重要**: 04_src/は汎用コンポーネント、code/は実験固有

#### `predict.py`
- 学習済みモデルでの推論スクリプト
- テストデータから提出ファイル生成

#### `preprocessing.py`（オプション）
- 実験固有の前処理詳細
- SimplePreprocessorで表現しきれない処理がある場合

---

### 3. `features/` - 特徴量情報

**目的**: どんな特徴量を使ったか完全に記録

#### `feature_list.txt`
使用した全特徴量のリスト（1行1特徴量）：

```
floor_max
floor_min
age
area_room
...
target_year
target_month
```

#### `feature_engineering.md`
特徴量エンジニアリングの詳細：

```markdown
# 特徴量エンジニアリング詳細

## 1. 数値特徴量（96個）
- そのまま使用
- 欠損値は未補完（LightGBM自動処理）

## 2. カテゴリカル特徴量（8個）
- 低カーディナリティ（<50）のみ使用
- ラベルエンコーディング（Polars Categorical → ordinal）

## 3. 生成特徴量（2個）
- target_year = target_ym // 100
- target_month = target_ym % 100

## 4. 除外した特徴量
- 高カーディナリティ（>=50）
- id, money_room, target_ym
```

#### `feature_importance.csv`
特徴量重要度の記録：

```csv
feature,gain,split,permutation
floor_max,0.123,0.089,0.145
age,0.098,0.067,0.112
...
```

#### `categorical_features.txt`
カテゴリカル特徴量リスト：

```
building_name_ruby
reform_exterior
name_ruby
school_ele_code
school_jun_code
money_hoshou_company
free_rent_duration
free_rent_gen_timing
```

---

### 4. `outputs/` - 予測結果・メトリクス

**目的**: 実験の出力を記録

#### `submission_YYYYMMDD_HHMMSS.csv`
提出ファイル（ヘッダーなし）：

```
0,21223864.82
1,24102895.71
...
```

#### `oof_predictions.csv`
Out-of-Fold予測値（分析用）：

```csv
index,y_true,y_pred,fold
0,25000000,24500000,1
1,18000000,19200000,2
...
```

#### `cv_scores.json`
Fold別スコア詳細：

```json
{
  "metric": "mape",
  "scores": [28.468, 28.276, 28.285],
  "mean": 28.343,
  "std": 0.088,
  "min": 28.276,
  "max": 28.468
}
```

#### `metrics.json`
評価指標の完全記録：

```json
{
  "cv": {
    "mape": 28.343,
    "rmse": 12345678.9,
    "mae": 9876543.2
  },
  "fold_details": [
    {
      "fold": 1,
      "mape": 28.468,
      "best_iteration": 100
    },
    ...
  ]
}
```

---

### 5. `models/` - 学習済みモデル

**目的**: 予測の再現性を保証

#### ファイル
- `fold_1.txt` - Fold 1で訓練したモデル
- `fold_2.txt` - Fold 2で訓練したモデル
- `fold_3.txt` - Fold 3で訓練したモデル
- `final_model.txt` - 全データで再訓練したモデル（提出用）

**フォーマット**: LightGBMテキストフォーマット

**注意**: モデルファイルは大きくなる可能性があるため、.gitignoreに追加推奨

---

### 6. `visualizations/` - 可視化

**目的**: 実験結果の可視化による理解促進

#### `feature_importance.png`
特徴量重要度のプロット（gain/split/permutation）

#### `cv_scores.png`
CVスコアの分布（Fold別）

#### `prediction_vs_actual.png`
予測vs実測の散布図（validation）

#### `residual_plot.png`
残差のプロット

#### `learning_curve.png`
学習曲線（訓練・検証スコアの推移）

---

### 7. `analysis/` - 分析結果

**目的**: エラー分析・改善のヒント

#### `error_analysis.md`
エラー分析レポート：

```markdown
# エラー分析レポート

## 1. 全体統計
- MAPE: 28.34%
- RMSE: 12,345,678
- MAE: 9,876,543

## 2. エラー分布
- 残差の平均: -123.4
- 残差の標準偏差: 5,678,901

## 3. 外れ値
- 3σ外: 123件（0.03%）

## 4. セグメント別エラー
...
```

#### `segment_analysis.csv`
セグメント別の誤差統計：

```csv
segment,count,mape,rmse,mae
低価格,30000,25.2,8000000,6000000
中価格,50000,28.5,12000000,9000000
高価格,20000,32.1,18000000,14000000
```

#### `outlier_analysis.csv`
外れ値の詳細：

```csv
index,y_true,y_pred,residual,abs_residual,pct_error
12345,50000000,80000000,30000000,30000000,60.0
...
```

---

### 8. `logs/` - 実行ログ

**目的**: 実行時の情報を記録

#### `training.log`
訓練時の標準出力をすべて記録

#### `mlflow_run_id.txt`
MLflow Run IDを記録：

```
b1541b503505448d8567f82d22166a1d
```

---

### 9. ルートファイル

#### `README.md`
実験の概要・結果サマリー（必須）

- 実験目的
- 結果サマリー
- ハイパーパラメータ
- 観察事項
- 次のステップ

#### `requirements.txt`
この実験の依存関係（バージョン固定）：

```
polars==1.18.0
lightgbm==4.6.0
mlflow==3.6.0
numpy==1.26.4
...
```

---

## 🔄 実験実行フロー

### 1. 新規実験開始

```bash
# 実験ディレクトリ作成
mkdir -p 06_experiments/exp002_feature_add/{configs,code,features,outputs,models,visualizations,analysis,logs}

# テンプレートREADME作成
cp 06_experiments/exp001_baseline/README.md 06_experiments/exp002_feature_add/README.md
```

### 2. 訓練実行

```bash
# 実験ディレクトリに移動
cd 06_experiments/exp002_feature_add

# 訓練実行（ログ記録）
python code/train.py 2>&1 | tee logs/training.log
```

### 3. 実験終了後

- [ ] README.md更新（結果・観察事項）
- [ ] configs/ にすべての設定を保存
- [ ] features/ に特徴量情報を保存
- [ ] outputs/ に予測結果を保存
- [ ] models/ にモデルを保存
- [ ] visualizations/ に可視化を保存
- [ ] analysis/ に分析結果を保存
- [ ] Gitコミット

---

## 📦 Gitリポジトリ管理

### コミット対象

**必ず含める**:
- README.md
- configs/
- code/
- features/
- outputs/submission_*.csv（提出ファイルのみ）
- visualizations/
- analysis/
- logs/mlflow_run_id.txt
- requirements.txt

**含めない（.gitignore）**:
- models/ （大きいため）
- outputs/oof_predictions.csv （大きいため）
- logs/training.log （大きいため）

**オプション**:
- outputs/cv_scores.json
- outputs/metrics.json

---

## 🎯 実験再現手順

別の環境で実験を再現する場合：

```bash
# 1. リポジトリクローン
git clone <repo_url>
cd 20251122_signamte_mlit

# 2. 依存関係インストール
cd 06_experiments/exp001_baseline
pip install -r requirements.txt

# 3. データ準備
# data/raw/ にtrain.csv, test.csvを配置

# 4. 訓練実行
python code/train.py

# 5. 結果確認
cat logs/training.log
cat outputs/cv_scores.json
```

---

## 🔍 実験比較

複数実験を比較する際：

```bash
# CV結果比較
cat 06_experiments/exp001_baseline/outputs/cv_scores.json
cat 06_experiments/exp002_feature_add/outputs/cv_scores.json

# 特徴量重要度比較
diff 06_experiments/exp001_baseline/features/feature_importance.csv \
     06_experiments/exp002_feature_add/features/feature_importance.csv

# MLflow UI起動
mlflow ui
```

---

## ✅ チェックリスト

実験完了時に以下を確認：

- [ ] README.mdが更新されている
- [ ] configs/にすべての設定ファイルがある
- [ ] code/train.pyが実行可能
- [ ] features/に特徴量情報が記録されている
- [ ] outputs/に提出ファイルがある
- [ ] models/にモデルが保存されている
- [ ] visualizations/に可視化がある
- [ ] analysis/にエラー分析がある
- [ ] logs/にMLflow Run IDが記録されている
- [ ] requirements.txtがある
- [ ] Gitコミット済み

---

**最終更新**: 2025-11-24
**作成者**: Claude Code
