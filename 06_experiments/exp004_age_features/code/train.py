"""
exp004_age_features 訓練スクリプト

exp003をベースに、築年数関連特徴量を追加:
1. building_age: 築年数（2024 - year_built）
2. building_age_bin: 築年数5年単位カテゴリ（0-10）
3. old_building_flag: 築35年以上フラグ
4. old_and_large_flag: 築35年以上 & 80㎡以上フラグ
5. old_and_rural_flag: 築35年以上 & 地方フラグ

目標: CV MAPE 27.0%以下（exp003: 27.47%から0.5pt以上改善）

変更点:
- LGBMRegressor（sklearn API）に変更
- パラメータ調整（正則化、bagging、colsample等）
- early_stopping追加
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
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
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
    AGE_NUMERIC_FEATURES,
    AGE_THRESHOLD,
    AREA_THRESHOLD,
    MAJOR_CITIES,
)


# ===== 設定 =====
SEED = 42
N_SPLITS = 3
EARLY_STOPPING_ROUNDS = 1000


def train_exp004():
    """exp004 age_features の訓練"""

    # シード固定
    set_seed(SEED)

    # MLflow実験設定
    mlflow.set_experiment("signate_mlit_rental_price")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"age_features_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        print(f"🚀 訓練開始: {run_name}")

        # ===== タグ設定 =====
        mlflow.set_tag("experiment_type", "age_features")
        mlflow.set_tag("experiment_id", "exp004")
        mlflow.set_tag("model_family", "gbdt")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("base_experiment", "exp003_baseline_v2")

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
        print(f"  - 新規特徴量（築年数関連）: {len(AGE_NUMERIC_FEATURES)}")

        # 特徴量情報を記録
        mlflow.log_param("n_features", len(ALL_FEATURES))
        mlflow.log_param("n_categorical_features", len(CATEGORICAL_FEATURES))
        mlflow.log_param("seed", SEED)
        mlflow.log_param("n_splits", N_SPLITS)
        mlflow.log_param("early_stopping_rounds", EARLY_STOPPING_ROUNDS)

        # exp004固有パラメータ
        mlflow.log_param("n_age_features", len(AGE_NUMERIC_FEATURES))
        mlflow.log_param("age_threshold", AGE_THRESHOLD)
        mlflow.log_param("area_threshold", AREA_THRESHOLD)
        mlflow.log_param("major_cities", str(MAJOR_CITIES))

        # 特徴量リストを保存
        log_feature_list(ALL_FEATURES, artifact_path="features.txt")

        # ===== LightGBMパラメータ設定（sklearn API） =====
        lgb_params = {
            # 基本設定
            "objective": "regression",
            "metric": "mape",
            "boosting_type": "gbdt",

            # 学習率（小さめに設定、early_stoppingで制御）
            "learning_rate": 0.01,

            # 木の構造
            "max_depth": 7,
            "num_leaves": 63,  # 2^max_depth - 1 程度

            # 正則化
            "reg_lambda": 1.0,  # L2
            "reg_alpha": 0.1,   # L1

            # サンプリング（過学習防止）
            "colsample_bytree": 0.7,  # 特徴量の70%を使用
            "subsample": 0.9,         # データの90%を使用
            "subsample_freq": 3,      # 3回に1回bagging

            # 分割条件
            "min_child_samples": 20,

            # イテレーション（early_stoppingで制御）
            "n_estimators": 10000,

            # 再現性
            "random_state": SEED,
            "deterministic": True,

            # その他
            "importance_type": "gain",
            "verbose": -1,
            "force_row_wise": True,
        }

        # パラメータ記録
        log_model_params(lgb_params, prefix="lgb")

        print("\n📋 LightGBMパラメータ:")
        print(f"  - learning_rate: {lgb_params['learning_rate']}")
        print(f"  - max_depth: {lgb_params['max_depth']}")
        print(f"  - num_leaves: {lgb_params['num_leaves']}")
        print(f"  - reg_lambda (L2): {lgb_params['reg_lambda']}")
        print(f"  - reg_alpha (L1): {lgb_params['reg_alpha']}")
        print(f"  - colsample_bytree: {lgb_params['colsample_bytree']}")
        print(f"  - subsample: {lgb_params['subsample']}")
        print(f"  - n_estimators: {lgb_params['n_estimators']} (with early_stopping)")

        # ===== モデル訓練（3-Fold CV） =====
        print("\n🤖 3-Fold クロスバリデーション開始...")

        cv_scores = []
        oof_predictions = np.zeros(len(X_train))
        test_predictions = np.zeros(len(X_test))
        feature_importance = np.zeros(len(ALL_FEATURES))
        best_iterations = []

        # Polars → pandas変換（特徴量名を保持するため）
        X_train_pd = X_train.to_pandas()
        X_test_pd = X_test.to_pandas()
        y_train_np = y_train.to_numpy()

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"\n--- Fold {fold_idx + 1}/{N_SPLITS} ---")

            # データ分割（pandas DataFrameで特徴量名を保持）
            X_tr = X_train_pd.iloc[train_idx]
            X_val = X_train_pd.iloc[val_idx]
            y_tr, y_val = y_train_np[train_idx], y_train_np[val_idx]

            # モデル作成
            model = LGBMRegressor(**lgb_params)

            # 訓練（sklearn API）
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="mape",
                callbacks=[
                    early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
                    log_evaluation(period=500),
                ],
            )

            # best_iteration取得
            best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') else model.n_estimators
            best_iterations.append(best_iter)

            # 予測
            val_pred = model.predict(X_val)
            oof_predictions[val_idx] = val_pred

            # テスト予測（アンサンブル用）
            test_pred = model.predict(X_test_pd)
            test_predictions += test_pred / N_SPLITS

            # MAPE計算
            mape_score = calculate_mape(y_val, val_pred)
            cv_scores.append(mape_score)

            # 特徴量重要度を蓄積（gain）
            feature_importance += model.feature_importances_ / N_SPLITS

            print(f"  Validation MAPE: {mape_score:.4f}%")
            print(f"  Best iteration: {best_iter}")

        # ===== CV結果まとめ =====
        print("\n" + "=" * 60)
        print("📈 クロスバリデーション結果")
        print("=" * 60)
        print(f"  平均 MAPE: {np.mean(cv_scores):.4f}%")
        print(f"  標準偏差:   {np.std(cv_scores):.4f}%")
        print(f"  最小値:     {np.min(cv_scores):.4f}%")
        print(f"  最大値:     {np.max(cv_scores):.4f}%")
        print(f"  平均イテレーション: {np.mean(best_iterations):.0f}")
        print("=" * 60)

        # CV結果をMLflowに記録
        log_cv_results(np.array(cv_scores), metric_name="mape")
        mlflow.log_metric("avg_best_iteration", np.mean(best_iterations))

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
        print("  Top 15 Features:")
        for i, idx in enumerate(sorted_indices[:15]):
            feat_name = ALL_FEATURES[idx]
            # 新規特徴量にマークを付ける
            marker = "⭐" if feat_name in AGE_NUMERIC_FEATURES else ""
            print(f"    {i+1}. {feat_name}: {feature_importance[idx]:.4f} {marker}")

        # 新規特徴量の重要度を別途表示
        print("\n  新規特徴量（築年数関連）の重要度:")
        for feat in AGE_NUMERIC_FEATURES:
            if feat in ALL_FEATURES:
                idx = ALL_FEATURES.index(feat)
                rank = list(sorted_indices).index(idx) + 1
                print(f"    {feat}: {feature_importance[idx]:.4f} (順位: {rank}/{len(ALL_FEATURES)})")

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

        # 結果サマリー
        print("\n" + "=" * 60)
        print("📋 実験サマリー（exp004: age_features）")
        print("=" * 60)
        print(f"  ベースライン（exp003）: 27.47%")
        print(f"  今回の結果:              {np.mean(cv_scores):.2f}%")
        improvement = 27.47 - np.mean(cv_scores)
        if improvement > 0:
            print(f"  改善:                   {improvement:.2f}pt ✅")
        else:
            print(f"  悪化:                   {-improvement:.2f}pt ❌")
        print("=" * 60)
        print("\n✅ 訓練完了！")


if __name__ == "__main__":
    train_exp004()
