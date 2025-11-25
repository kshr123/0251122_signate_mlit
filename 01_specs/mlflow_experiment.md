# MLflow 実験記録 仕様書

Version: 1.0.0
Last Updated: 2025-11-24
Author: Claude Code

---

## 目次

1. [要件定義](#1-要件定義)
2. [記録方針](#2-記録方針)
3. [実装仕様](#3-実装仕様)
4. [成功基準](#4-成功基準)

---

## 1. 要件定義

### 1.1 背景と目的

**背景**
- データ分析コンペでは、複数の特徴量・モデル・パラメータを試行錯誤する
- 実験結果を記録しないと、何が効果的だったか分からなくなる
- 再現性を担保するため、実験条件を厳密に記録する必要がある

**目的**
- MLflowを使って実験を自動記録する仕組みを構築
- ベースラインモデルで記録の流れを確認
- 再現性を確保（シード、特徴量、パラメータ、データセット情報）

### 1.2 スコープ

#### 必須要件
- [ ] **実験の自動記録**（パラメータ、メトリクス、アーティファクト）
- [ ] **再現性の確保**（シード固定、環境情報記録）
- [ ] **提出ファイルの記録**（Run IDと紐付け）
- [ ] **特徴量情報の記録**（使用したカラム、生成方法）
- [ ] **CV結果の記録**（Fold別スコア、統計量）

#### スコープ外（個人開発のため）
- ❌ REST API / リモートトラッキングサーバー
- ❌ データベース（PostgreSQL, MySQL等）
- ❌ 複雑なモデルレジストリ
- ❌ 複数プロジェクト管理

**Note**: チーム開発時はリモートトラッキングサーバーを検討

---

## 2. 記録方針

### 2.1 記録するもの

#### 必須項目

##### 1. パラメータ（Parameters）
- **シード値**: `seed`
- **モデル種類**: `model_type` (例: "LightGBM")
- **モデルパラメータ**: `learning_rate`, `num_leaves`, `num_boost_round` 等
- **CV設定**: `cv_strategy`, `n_splits`
- **特徴量数**: `n_features`

##### 2. メトリクス（Metrics）
- **CVスコア統計量**: `cv_rmse_mean`, `cv_rmse_std`, `cv_rmse_min`, `cv_rmse_max`
- **Fold別スコア**: `cv_rmse_fold_0`, `cv_rmse_fold_1`, ...
- **データセット情報**: `train_size`, `test_size`

##### 3. アーティファクト（Artifacts）
- **提出ファイル**: `submissions/submission_{timestamp}.csv`
- **モデルファイル**: MLflow Model形式で保存
- **特徴量リスト**: 使用した特徴量のリスト（テキストまたはJSON）

##### 4. タグ（Tags）
- **実験種類**: `experiment_type` (baseline, feature_engineering, tuning, ensemble)
- **ステータス**: `status` (running, completed, failed)
- **フェーズ**: `phase` (initial, optimization, final)
- **メモ**: `note` (自由記述)

### 2.2 記録しないもの

- 大容量データファイル（train.csv, test.csv）
- 中間ファイル（前処理途中のデータ）
- ログファイル全体（標準出力のみ）

### 2.3 実験命名規則

```
{model_type}_{timestamp}
例: baseline_20251124_143022
```

---

## 3. 実装仕様

### 3.1 ディレクトリ構成

```
.
├── mlruns/                     # MLflowローカルファイルストレージ（.gitignore）
│                               # SQLite等のDBは不要、ファイルベースで十分
├── 04_src/
│   └── training/
│       ├── train_baseline.py   # ベースライン訓練スクリプト
│       └── utils/
│           └── mlflow_helper.py  # MLflow記録ヘルパー関数（オプション）
└── 03_configs/
    └── experiment.yaml         # 実験設定（オプション）
```

### 3.2 実装パターン

#### パターン1: ベースライン訓練スクリプト

```python
# 04_src/training/train_baseline.py

import mlflow
import polars as pl
import lightgbm as lgb
from pathlib import Path
from datetime import datetime

from data.loader import DataLoader
from utils.config import load_config
from features.base import SeedManager

SEED = 42


def main():
    """ベースラインモデルの訓練"""

    # シード固定
    SeedManager.set_seed(SEED)

    # MLflow実験設定
    mlflow.set_experiment("signate_mlit_rental_price")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"baseline_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        # === タグ設定 ===
        mlflow.set_tag("experiment_type", "baseline")
        mlflow.set_tag("model_family", "gbdt")
        mlflow.set_tag("status", "running")

        # === パラメータ記録 ===
        mlflow.log_param("seed", SEED)
        mlflow.log_param("model_type", "LightGBM")

        # === データ読み込み ===
        config = load_config("data", config_dir="03_configs")
        loader = DataLoader(config, add_address_columns=False)
        train = loader.load_train()
        test = loader.load_test()

        mlflow.log_metric("train_size", train.height)
        mlflow.log_metric("test_size", test.height)

        # === 特徴量選択（シンプルに） ===
        numeric_cols = ["area_sqm", "distance_station", ...]
        categorical_cols = ["structure_type", "direction", ...]

        # target_ym分解
        train = train.with_columns([
            (pl.col("target_ym") // 100).alias("target_year"),
            (pl.col("target_ym") % 100).alias("target_month"),
        ])
        test = test.with_columns([
            (pl.col("target_ym") // 100).alias("target_year"),
            (pl.col("target_ym") % 100).alias("target_month"),
        ])

        feature_cols = numeric_cols + categorical_cols + ["target_year", "target_month"]
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("feature_cols", feature_cols)

        # === モデル訓練 ===
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "seed": SEED,
            "verbose": -1,
        }

        mlflow.log_params(params)
        mlflow.log_param("num_boost_round", 100)

        target = train["money_room"]
        X_train = train.select(feature_cols).to_pandas()

        dtrain = lgb.Dataset(X_train, label=target.to_numpy())
        model = lgb.train(params, dtrain, num_boost_round=100)

        # === CV評価 ===
        from sklearn.model_selection import cross_val_score

        cv_scores = cross_val_score(
            model, X_train, target.to_numpy(),
            cv=5, scoring="neg_root_mean_squared_error"
        )

        # CV統計量
        mlflow.log_metric("cv_rmse_mean", -cv_scores.mean())
        mlflow.log_metric("cv_rmse_std", cv_scores.std())
        mlflow.log_metric("cv_rmse_min", -cv_scores.min())
        mlflow.log_metric("cv_rmse_max", -cv_scores.max())

        # Fold別スコア
        for i, score in enumerate(cv_scores):
            mlflow.log_metric(f"cv_rmse_fold_{i}", -score)

        # === 予測と提出 ===
        X_test = test.select(feature_cols).to_pandas()
        preds = model.predict(X_test)

        submission = test.select("id").with_columns(
            pl.Series("money_room", preds)
        )

        submission_dir = Path("06_submissions")
        submission_dir.mkdir(exist_ok=True)

        submission_path = submission_dir / f"submission_{timestamp}.csv"
        submission.write_csv(submission_path, has_header=False)

        # === アーティファクト保存 ===
        mlflow.log_artifact(submission_path)
        mlflow.lightgbm.log_model(model, "model")

        # === 完了 ===
        mlflow.set_tag("status", "completed")

        print(f"✅ Run completed: {mlflow.active_run().info.run_id}")
        print(f"📊 CV RMSE: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"📁 Submission: {submission_path}")


if __name__ == "__main__":
    main()
```

#### パターン2: ヘルパー関数

```python
# 04_src/training/utils/mlflow_helper.py

import mlflow
import polars as pl
import numpy as np
from typing import List


def log_dataset_info(df: pl.DataFrame, prefix: str = "train"):
    """データセット基本情報を記録"""
    mlflow.log_metric(f"{prefix}.n_rows", df.height)
    mlflow.log_metric(f"{prefix}.n_cols", df.width)

    # 欠損値情報
    null_counts = df.null_count()
    total_nulls = sum(null_counts.row(0))
    null_ratio = total_nulls / (df.height * df.width)

    mlflow.log_metric(f"{prefix}.total_nulls", total_nulls)
    mlflow.log_metric(f"{prefix}.null_ratio", null_ratio)


def log_cv_results(cv_scores: np.ndarray, metric_name: str = "rmse"):
    """CV詳細結果を記録"""
    mlflow.log_metric(f"cv_{metric_name}_mean", cv_scores.mean())
    mlflow.log_metric(f"cv_{metric_name}_std", cv_scores.std())
    mlflow.log_metric(f"cv_{metric_name}_min", cv_scores.min())
    mlflow.log_metric(f"cv_{metric_name}_max", cv_scores.max())

    for i, score in enumerate(cv_scores):
        mlflow.log_metric(f"cv_{metric_name}_fold_{i}", score)


def log_feature_list(feature_cols: List[str], filename: str = "features.txt"):
    """使用特徴量リストをアーティファクトとして保存"""
    from pathlib import Path

    temp_file = Path(filename)
    temp_file.write_text("\n".join(feature_cols))

    mlflow.log_artifact(temp_file)
    temp_file.unlink()  # 一時ファイル削除
```

### 3.3 MLflow UI の使用

#### 起動方法

```bash
mlflow ui --port 5000
```

ブラウザで http://localhost:5000 にアクセス

#### 実験比較

1. Experiments タブで実験選択
2. 比較したいRunにチェック
3. Compare ボタンをクリック
4. メトリクス・パラメータを並べて確認

#### 検索方法

```python
import mlflow

# タグで検索
runs = mlflow.search_runs(
    filter_string="tags.experiment_type = 'baseline'"
)

# メトリクスで検索
runs = mlflow.search_runs(
    filter_string="metrics.cv_rmse_mean < 10000"
)

# 最新のRun取得
runs = mlflow.search_runs(
    order_by=["start_time DESC"],
    max_results=1
)
```

---

## 4. 成功基準

### 4.1 機能面

- [ ] ベースライン訓練時に自動でMLflow記録される
- [ ] パラメータ（シード、モデル設定）が記録される
- [ ] メトリクス（CVスコア）が記録される
- [ ] アーティファクト（提出ファイル、モデル）が保存される
- [ ] MLflow UIで実験結果を確認できる
- [ ] Run IDから実験内容を再現できる

### 4.2 再現性

- [ ] 同じRun IDで訓練を再実行すると、同じスコアになる
- [ ] シード値が適切に記録・適用されている
- [ ] 使用した特徴量リストが記録されている

### 4.3 運用面

- [ ] 実行時間 < 1分（記録オーバーヘッド）
- [ ] mlrunsディレクトリが.gitignoreに含まれている
- [ ] README.mdにMLflow起動方法が記載されている

---

## 変更履歴

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-24 | Claude Code | 初版作成（ベースライン用） |

---

## 参考資料

- [MLflow公式ドキュメント](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- 参考実装: `/Users/kotaro/Desktop/dev/ML_designpattern/03_my_implementations/chapter2_training/01_model_db/`
