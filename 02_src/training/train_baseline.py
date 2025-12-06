"""
ベースラインモデル訓練スクリプト

シンプル・高速・再現性確保を優先したベースライン
"""

import sys
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlflow
import polars as pl
import lightgbm as lgb
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold

# プロジェクトモジュール
from data.loader import DataLoader
from preprocessing.simple import SimplePreprocessor
from features.base import set_seed
from evaluation.metrics import calculate_mape
from training.utils.mlflow_helper import (
    log_dataset_info,
    log_cv_results,
    log_feature_list,
    log_model_params,
)
from utils.config import load_config


# ===== 設定 =====
SEED = 42
N_SPLITS = 3
NUM_BOOST_ROUND = 100


def train_baseline():
    """ベースラインモデルの訓練"""

    # シード固定
    set_seed(SEED)

    # MLflow実験設定
    mlflow.set_experiment("signate_mlit_rental_price")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"baseline_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        print(f"🚀 訓練開始: {run_name}")

        # ===== タグ設定 =====
        mlflow.set_tag("experiment_type", "baseline")
        mlflow.set_tag("model_family", "gbdt")
        mlflow.set_tag("status", "running")

        # ===== パラメータ記録 =====
        mlflow.log_param("seed", SEED)
        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_param("n_splits", N_SPLITS)
        mlflow.log_param("num_boost_round", NUM_BOOST_ROUND)

        # ===== データ読み込み =====
        print("📂 データ読み込み中...")
        config = load_config("data", config_dir="03_configs")
        loader = DataLoader(config, add_address_columns=False)

        train = loader.load_train()
        test = loader.load_test()

        print(f"  - Train: {train.shape}")
        print(f"  - Test: {test.shape}")

        # データセット情報を記録
        log_dataset_info(train, prefix="train")
        log_dataset_info(test, prefix="test")

        # ===== 前処理 =====
        print("🔧 前処理中...")
        preprocessor = SimplePreprocessor(
            cardinality_threshold=50,
            fill_missing=False,  # LightGBMが欠損値を自動処理
        )

        # fitはtrainのみで実行
        X_train = preprocessor.fit_transform(train)
        y_train = train["money_room"].to_numpy()

        X_test = preprocessor.transform(test)

        # カテゴリカル特徴量の名前リスト（後でラベルエンコーディングに使用）
        categorical_features = [
            col for col in preprocessor.low_cardinality_cols
            if col in preprocessor.feature_cols
        ]

        print(f"  - 特徴量数: {len(preprocessor.feature_cols)}")
        print(f"  - 数値特徴量: {len(preprocessor.numeric_cols)}")
        print(f"  - カテゴリカル特徴量: {len(preprocessor.low_cardinality_cols)}")

        # 特徴量情報を記録
        mlflow.log_param("n_features", len(preprocessor.feature_cols))
        mlflow.log_param("n_numeric_features", len(preprocessor.numeric_cols))
        mlflow.log_param("n_categorical_features", len(preprocessor.low_cardinality_cols))

        # 特徴量リストを保存
        log_feature_list(preprocessor.feature_cols, artifact_path="features.txt")

        # ===== モデル訓練（3-Fold CV） =====
        print("🌲 モデル訓練中（3-Fold CV）...")

        # LightGBMパラメータ
        params = {
            "objective": "regression",
            "metric": "mape",
            "boosting": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "random_state": SEED,
            "verbose": -1,
            "force_row_wise": True,
        }

        log_model_params(params)
        mlflow.log_param("early_stopping_rounds", 100)

        # クロスバリデーション
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        cv_scores = []
        oof_predictions = np.zeros(len(y_train))
        models = []

        # Polars → pandas変換（LightGBM用）
        # trainとtestで型が異なる可能性があるため、両方で文字列型を検出して変換
        string_cols_train = [col for col in X_train.columns if X_train[col].dtype == pl.Utf8]
        string_cols_test = [col for col in X_test.columns if X_test[col].dtype == pl.Utf8]
        string_cols = list(set(string_cols_train + string_cols_test))

        print(f"  - Train文字列型: {string_cols_train}")
        print(f"  - Test文字列型: {string_cols_test}")

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

        X_train_pd = X_train.to_pandas()
        X_test_pd = X_test.to_pandas()

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_pd)):
            print(f"  - Fold {fold_idx + 1}/{N_SPLITS}")

            # データ分割
            X_tr = X_train_pd.iloc[train_idx]
            y_tr = y_train[train_idx]
            X_val = X_train_pd.iloc[val_idx]
            y_val = y_train[val_idx]

            # LightGBM Dataset作成
            train_data = lgb.Dataset(
                X_tr,
                label=y_tr,
                categorical_feature=categorical_features,
            )
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                categorical_feature=categorical_features,
                reference=train_data,
            )

            # 訓練（Early Stopping付き）
            callbacks = [
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),  # ログ出力なし
            ]

            model = lgb.train(
                params,
                train_data,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[train_data, val_data],
                valid_names=["train", "valid"],
                callbacks=callbacks,
            )

            models.append(model)

            # Validation予測
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            oof_predictions[val_idx] = y_pred

            # MAPE計算
            fold_mape = calculate_mape(y_val, y_pred)
            cv_scores.append(fold_mape)

            print(f"    MAPE: {fold_mape:.4f}% (best_iteration: {model.best_iteration})")

        cv_scores = np.array(cv_scores)

        # CV結果を記録
        log_cv_results(cv_scores, metric_name="mape")

        print(f"\n📊 CV結果:")
        print(f"  - MAPE: {cv_scores.mean():.4f}% ± {cv_scores.std():.4f}%")
        print(f"  - Min: {cv_scores.min():.4f}%")
        print(f"  - Max: {cv_scores.max():.4f}%")

        # ===== 全データで再訓練 =====
        print("\n🔄 全データで再訓練中...")
        full_train_data = lgb.Dataset(
            X_train_pd,
            label=y_train,
            categorical_feature=categorical_features,
        )

        final_model = lgb.train(
            params,
            full_train_data,
            num_boost_round=NUM_BOOST_ROUND,
        )

        # ===== 提出ファイル生成 =====
        print("📝 提出ファイル生成中...")
        test_predictions = final_model.predict(X_test_pd, num_iteration=final_model.best_iteration)

        submission = test.select("id").with_columns(
            pl.Series("money_room", test_predictions)
        )

        # 06_experiments/exp001_baseline/配下に保存
        exp_dir = Path("06_experiments/exp001_baseline")
        exp_dir.mkdir(parents=True, exist_ok=True)

        submission_path = exp_dir / f"submission_{timestamp}.csv"
        submission.write_csv(submission_path, include_header=False)

        print(f"  - 保存先: {submission_path}")

        # 提出ファイルをアーティファクトとして保存
        mlflow.log_artifact(str(submission_path))

        # ===== モデル保存（オプション） =====
        mlflow.lightgbm.log_model(final_model, "model")

        # ===== 完了 =====
        mlflow.set_tag("status", "completed")

        print(f"\n✅ 訓練完了!")
        print(f"  - Run ID: {mlflow.active_run().info.run_id}")
        print(f"  - CV MAPE: {cv_scores.mean():.4f}% ± {cv_scores.std():.4f}%")
        print(f"  - 提出ファイル: {submission_path}")


if __name__ == "__main__":
    train_baseline()
