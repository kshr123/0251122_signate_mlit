"""
exp003_baseline_v2 訓練スクリプト

特徴量を厳選し、エンコーディング手法を適用した新ベースラインモデル。
- ターゲットエンコーディング: addr1_1, addr1_2, bukken_type, land_youto, land_toshi
- ラベル + カウントエンコーディング: 11カラム
- year_built変換、money_sonota集約
"""

import sys
from pathlib import Path

# カレントディレクトリを優先
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(1, str(project_root / "04_src"))

import json
import mlflow
import polars as pl
import lightgbm as lgb
import numpy as np
import yaml
from datetime import datetime
from sklearn.model_selection import KFold

# 共通コンポーネント（04_src/）
from data.loader import DataLoader
from features.base import set_seed
from evaluation.metrics import calculate_mape
from training.utils.mlflow_helper import (
    log_dataset_info,
    log_cv_results,
    log_feature_list,
    log_model_params,
)

# 実験固有の前処理（このディレクトリ内）
from preprocessing import (
    preprocess_for_training,
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_ENCODING_COLUMNS,
    COUNT_ENCODING_COLUMNS,
    NUMERIC_FEATURES,
)


# ===== 設定 =====
SEED = 42
N_SPLITS = 3
NUM_BOOST_ROUND = 100


def train_baseline_v2():
    """exp003 baseline_v2 の訓練"""

    # シード固定
    set_seed(SEED)

    # MLflow実験設定
    mlflow.set_experiment("signate_mlit_rental_price")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"baseline_v2_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        print(f"🚀 訓練開始: {run_name}")

        # ===== タグ設定 =====
        mlflow.set_tag("experiment_type", "baseline_v2")
        mlflow.set_tag("experiment_id", "exp003")
        mlflow.set_tag("model_family", "gbdt")
        mlflow.set_tag("status", "running")

        # ===== パラメータ記録 =====
        mlflow.log_param("seed", SEED)
        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_param("n_splits", N_SPLITS)
        mlflow.log_param("num_boost_round", NUM_BOOST_ROUND)
        mlflow.log_param("n_target_encoding_columns", len(TARGET_ENCODING_COLUMNS))
        mlflow.log_param("n_count_encoding_columns", len(COUNT_ENCODING_COLUMNS))
        mlflow.log_param("n_numeric_features", len(NUMERIC_FEATURES))

        # ===== データ読み込み =====
        print("\n📂 データ読み込み中...")
        config_path = project_root / "03_configs" / "data.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            data_config = yaml.safe_load(f)

        # パスを絶対パスに変換
        data_config["data"]["train_path"] = str(project_root / data_config["data"]["train_path"])
        data_config["data"]["test_path"] = str(project_root / data_config["data"]["test_path"])
        data_config["data"]["sample_submit_path"] = str(project_root / data_config["data"]["sample_submit_path"])

        loader = DataLoader(config=data_config, add_address_columns=False)

        train = loader.load_train()
        test = loader.load_test()

        # ID列を先に保存（前処理後に参照するため）
        train_ids = np.arange(len(train))
        test_ids = test["id"].to_numpy()

        print(f"  - Train: {train.shape}")
        print(f"  - Test: {test.shape}")

        # データセット情報を記録
        log_dataset_info(train, prefix="train")
        log_dataset_info(test, prefix="test")

        # ===== CV分割を先に作成（TargetEncoding用） =====
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        cv_splits = list(cv.split(train))

        # ===== 前処理 =====
        print("\n🔧 前処理中...")
        X_train, X_test, y_train = preprocess_for_training(train, test, cv_splits=cv_splits)

        print(f"\n📊 特徴量情報:")
        print(f"  - 特徴量数: {len(ALL_FEATURES)}")
        print(f"  - カテゴリカル特徴量: {len(CATEGORICAL_FEATURES)}")

        # 特徴量情報を記録
        mlflow.log_param("n_features", len(ALL_FEATURES))
        mlflow.log_param("n_categorical_features", len(CATEGORICAL_FEATURES))

        # 特徴量リストを保存
        log_feature_list(ALL_FEATURES, artifact_path="features.txt")

        # ===== LightGBMパラメータ設定 =====
        lgb_params = {
            "objective": "regression",
            "metric": "mape",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "seed": SEED,
            "verbose": -1,
            "force_row_wise": True,
        }

        # パラメータ記録
        log_model_params(lgb_params, prefix="lgb")

        # ===== モデル訓練（3-Fold CV） =====
        print("\n🤖 3-Fold クロスバリデーション開始...")

        cv_scores = []
        oof_predictions = np.zeros(len(X_train))
        test_predictions = np.zeros(len(X_test))
        feature_importance = np.zeros(len(ALL_FEATURES))

        # Polars → NumPy変換
        X_train_np = X_train.to_numpy()
        X_test_np = X_test.to_numpy()
        y_train_np = y_train.to_numpy()

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"\n--- Fold {fold_idx + 1}/{N_SPLITS} ---")

            # データ分割
            X_tr, X_val = X_train_np[train_idx], X_train_np[val_idx]
            y_tr, y_val = y_train_np[train_idx], y_train_np[val_idx]

            # LightGBM Dataset作成
            cat_features = CATEGORICAL_FEATURES if CATEGORICAL_FEATURES else "auto"
            train_data = lgb.Dataset(
                X_tr,
                label=y_tr,
                feature_name=ALL_FEATURES,
                categorical_feature=cat_features,
            )

            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                feature_name=ALL_FEATURES,
                categorical_feature=cat_features,
                reference=train_data,
            )

            # 訓練
            model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[train_data, val_data],
                valid_names=["train", "valid"],
            )

            # 予測
            val_pred = model.predict(X_val)
            oof_predictions[val_idx] = val_pred

            # テスト予測（アンサンブル用）
            test_pred = model.predict(X_test_np)
            test_predictions += test_pred / N_SPLITS

            # MAPE計算
            mape_score = calculate_mape(y_val, val_pred)
            cv_scores.append(mape_score)

            # 特徴量重要度を蓄積（gain）
            feature_importance += model.feature_importance(importance_type="gain") / N_SPLITS

            print(f"  Validation MAPE: {mape_score:.4f}%")

        # ===== CV結果まとめ =====
        print("\n" + "=" * 60)
        print("📈 クロスバリデーション結果")
        print("=" * 60)
        print(f"  平均 MAPE: {np.mean(cv_scores):.4f}%")
        print(f"  標準偏差:   {np.std(cv_scores):.4f}%")
        print(f"  最小値:     {np.min(cv_scores):.4f}%")
        print(f"  最大値:     {np.max(cv_scores):.4f}%")
        print("=" * 60)

        # CV結果をMLflowに記録
        log_cv_results(np.array(cv_scores), metric_name="mape")

        # OOF MAPE
        oof_mape = calculate_mape(y_train_np, oof_predictions)
        mlflow.log_metric("oof_mape", oof_mape)
        print(f"\n  OOF MAPE: {oof_mape:.4f}%")

        # 出力ディレクトリ作成
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        # ===== OOF予測保存（エラー分析用） =====
        print("\n📊 OOF予測保存中...")
        oof_df = pl.DataFrame({
            "id": train_ids,
            "actual": y_train_np,
            "predicted": oof_predictions,
        })
        oof_path = output_dir / f"oof_predictions_{timestamp}.csv"
        oof_df.write_csv(oof_path)
        print(f"  ✓ 保存完了: {oof_path}")
        mlflow.log_artifact(oof_path, artifact_path="predictions")

        # ===== 特徴量重要度保存 =====
        print("\n📊 特徴量重要度保存中...")
        importance_dict = {
            "feature": ALL_FEATURES,
            "importance": feature_importance.tolist(),
        }
        # ソートして上位を表示
        sorted_indices = np.argsort(feature_importance)[::-1]
        print("  Top 10 Features:")
        for i, idx in enumerate(sorted_indices[:10]):
            print(f"    {i+1}. {ALL_FEATURES[idx]}: {feature_importance[idx]:.4f}")

        # JSON形式で保存
        importance_path = output_dir / f"feature_importance_{timestamp}.json"
        with open(importance_path, "w", encoding="utf-8") as f:
            json.dump(importance_dict, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 保存完了: {importance_path}")
        mlflow.log_artifact(importance_path, artifact_path="feature_importance")

        # CSV形式でも保存（可視化しやすい）
        importance_df = pl.DataFrame({
            "feature": ALL_FEATURES,
            "importance": feature_importance,
        }).sort("importance", descending=True)
        importance_csv_path = output_dir / f"feature_importance_{timestamp}.csv"
        importance_df.write_csv(importance_csv_path)
        mlflow.log_artifact(importance_csv_path, artifact_path="feature_importance")

        # ===== 提出ファイル生成 =====
        print("\n📤 提出ファイル生成中...")

        submission = pl.DataFrame({
            "id": test_ids,
            "money_room": test_predictions,
        })

        submission_path = output_dir / f"submission_{timestamp}.csv"
        submission.write_csv(submission_path)

        print(f"  ✓ 保存完了: {submission_path}")

        # MLflowに記録
        mlflow.log_artifact(submission_path, artifact_path="submissions")

        # ===== 実験完了 =====
        mlflow.set_tag("status", "completed")
        print("\n✅ 訓練完了！")


if __name__ == "__main__":
    train_baseline_v2()
