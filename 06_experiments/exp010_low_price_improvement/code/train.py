"""
exp010_low_price_improvement training script

低価格帯（特に広面積×築古）の予測精度改善を目的とした実験。

exp009からの継承:
- 地価公示データ結合・特徴量（39次元）
- リフォーム特徴量（7次元）
- 郵便番号TE（2次元）
- 面積地域平均比率（6次元）

exp010追加特徴量:
- lp_area_value: 土地価値目安（地価×面積）
- area_age_category: 面積×築年数カテゴリ（0,1,2）
- area_age_cat_te_*: カテゴリ×属性TE（2次元）

Target: CV MAPE < exp009 (13.62%)
"""

import sys
import argparse
from pathlib import Path

# Add project root to path (for 04_src)
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "04_src"))
# Add exp010 code directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import mlflow
import polars as pl
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
import numpy as np
import yaml
from datetime import datetime
from sklearn.model_selection import KFold

# Common components (04_src/)
from data.loader import DataLoader
from features.base import set_seed
from evaluation.metrics import calculate_mape
from training.utils.mlflow_helper import (
    log_dataset_info,
    log_cv_results,
    log_feature_list,
    log_model_params,
)

# exp010 preprocessing
from preprocessing import preprocess_for_training
from pipeline import FeaturePipeline
from constants import EXP010_FEATURE_COLUMNS


def load_config(test_mode: bool = False) -> dict:
    """Load experiment.yaml configuration file

    Args:
        test_mode: If True, use params_test instead of params
    """
    config_path = Path(__file__).parent.parent / "configs" / "experiment.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # テストモードの場合、params_testを使用
    if test_mode and "params_test" in config["model"]:
        config["model"]["params"] = config["model"]["params_test"]
        print("*** TEST MODE: Using params_test ***")

    return config


def train_exp010(test_mode: bool = False):
    """Train exp010 low_price_improvement

    Args:
        test_mode: If True, use test parameters for fast execution
    """

    # Load config
    config = load_config(test_mode=test_mode)
    exp_config = config["experiment"]
    training_config = config["training"]
    model_config = config["model"]

    # Set seed
    seed = training_config["seed"]
    set_seed(seed)

    # MLflow experiment setup
    mlflow.set_experiment("signate_mlit_rental_price")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{exp_config['name']}_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        print(f"Training started: {run_name}")

        # ===== Set tags =====
        mlflow.set_tag("experiment_type", exp_config["name"])
        mlflow.set_tag("experiment_id", exp_config["id"])
        mlflow.set_tag("model_family", "gbdt")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("base_experiment", exp_config["base"])
        mlflow.set_tag("target_transform", training_config.get("target_transform", "none"))

        # ===== Load data =====
        print("\nLoading data...")
        data_config_path = project_root / "03_configs" / "data.yaml"
        with open(data_config_path, "r", encoding="utf-8") as f:
            data_config = yaml.safe_load(f)

        # Convert to absolute paths
        data_config["data"]["train_path"] = str(project_root / data_config["data"]["train_path"])
        data_config["data"]["test_path"] = str(project_root / data_config["data"]["test_path"])
        data_config["data"]["sample_submit_path"] = str(project_root / data_config["data"]["sample_submit_path"])

        loader = DataLoader(config=data_config, add_address_columns=False)

        train = loader.load_train()
        test = loader.load_test()

        # Save IDs before preprocessing
        train_ids = np.arange(len(train))
        test_ids = test["id"].to_numpy()

        print(f"  - Train: {train.shape}")
        print(f"  - Test: {test.shape}")

        # Log dataset info
        log_dataset_info(train, prefix="train")
        log_dataset_info(test, prefix="test")

        # ===== Create CV splits for TargetEncoding =====
        n_splits = training_config["n_splits"]
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        cv_splits = list(cv.split(train))

        # ===== exp010 Preprocessing =====
        print("\n" + "=" * 60)
        print("exp010 Preprocessing (exp009 base + low_price features)")
        print("=" * 60)
        X_train, X_test, y_train, pipeline = preprocess_for_training(train, test, cv_splits=cv_splits)

        # Get actual feature list from pipeline
        ALL_FEATURES = pipeline.get_feature_names()

        # Display pipeline summary
        print("\n" + pipeline.summary())

        # Count feature categories
        n_exp010_features = len([c for c in ALL_FEATURES if c in EXP010_FEATURE_COLUMNS or 'area_age_cat' in c])
        n_landprice_features = len([c for c in ALL_FEATURES if c.startswith('lp_') or '_lp_' in c])
        n_base_features = len(ALL_FEATURES) - n_landprice_features - n_exp010_features

        print(f"\nTotal features: {len(ALL_FEATURES)}")
        print(f"  - Base features: {n_base_features}")
        print(f"  - Landprice features: {n_landprice_features}")
        print(f"  - exp010 features: {n_exp010_features}")

        # ===== Target variable log transform =====
        target_transform = training_config.get("target_transform", "none")
        y_train_np = y_train.to_numpy()

        if target_transform == "log1p":
            print("\nApplying log1p transform to target variable")
            y_train_transformed = np.log1p(y_train_np)
            print(f"  - Before: mean={y_train_np.mean():.2f}, std={y_train_np.std():.2f}")
            print(f"  - After: mean={y_train_transformed.mean():.4f}, std={y_train_transformed.std():.4f}")
        else:
            y_train_transformed = y_train_np
            print("\nTarget transform: none")

        # Log feature info
        mlflow.log_param("n_features", len(ALL_FEATURES))
        mlflow.log_param("n_base_features", n_base_features)
        mlflow.log_param("n_landprice_features", n_landprice_features)
        mlflow.log_param("n_exp010_features", n_exp010_features)
        mlflow.log_param("seed", seed)
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("early_stopping_rounds", training_config["early_stopping_rounds"])
        mlflow.log_param("target_transform", target_transform)

        # Save feature list
        log_feature_list(ALL_FEATURES, artifact_path="features.txt")

        # ===== LightGBM parameters =====
        lgb_params = model_config["params"].copy()
        lgb_params["random_state"] = seed
        lgb_params["deterministic"] = True

        # Log parameters
        log_model_params(lgb_params, prefix="lgb")

        print("\nLightGBM parameters:")
        print(f"  - learning_rate: {lgb_params['learning_rate']}")
        print(f"  - max_depth: {lgb_params['max_depth']}")
        print(f"  - num_leaves: {lgb_params['num_leaves']}")
        print(f"  - n_estimators: {lgb_params['n_estimators']} (with early_stopping)")

        # ===== Model training (3-Fold CV) =====
        print(f"\n{n_splits}-Fold Cross Validation starting...")

        cv_scores = []
        oof_predictions = np.zeros(len(X_train))
        test_predictions = np.zeros(len(X_test))
        feature_importance = np.zeros(len(ALL_FEATURES))
        best_iterations = []
        trained_models = []

        # Polars -> pandas (to preserve feature names)
        X_train_pd = X_train.to_pandas()
        X_test_pd = X_test.to_pandas()

        early_stopping_rounds = training_config["early_stopping_rounds"]

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")

            # Split data
            X_tr = X_train_pd.iloc[train_idx]
            X_val = X_train_pd.iloc[val_idx]
            y_tr = y_train_transformed[train_idx]
            y_val_transformed = y_train_transformed[val_idx]
            y_val_original = y_train_np[val_idx]

            # Create model
            model = LGBMRegressor(**lgb_params)

            # Train
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val_transformed)],
                callbacks=[
                    early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
                    log_evaluation(period=early_stopping_rounds),
                ],
            )

            # Get best_iteration
            best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') else model.n_estimators
            best_iterations.append(best_iter)
            trained_models.append(model)

            # Predict (in log space)
            val_pred_transformed = model.predict(X_val)

            # Inverse transform to original scale
            if target_transform == "log1p":
                val_pred = np.expm1(val_pred_transformed)
                val_pred = np.maximum(val_pred, 0)
            else:
                val_pred = val_pred_transformed

            oof_predictions[val_idx] = val_pred

            # Test prediction
            test_pred_transformed = model.predict(X_test_pd)
            if target_transform == "log1p":
                test_pred = np.expm1(test_pred_transformed)
                test_pred = np.maximum(test_pred, 0)
            else:
                test_pred = test_pred_transformed
            test_predictions += test_pred / n_splits

            # Calculate MAPE (on original scale)
            mape_score = calculate_mape(y_val_original, val_pred)
            cv_scores.append(mape_score)

            # Accumulate feature importance
            feature_importance += model.feature_importances_ / n_splits

            print(f"  Validation MAPE: {mape_score:.4f}%")
            print(f"  Best iteration: {best_iter}")

        # ===== CV Results Summary =====
        print("\n" + "=" * 60)
        print("Cross Validation Results")
        print("=" * 60)
        print(f"  Mean MAPE: {np.mean(cv_scores):.4f}%")
        print(f"  Std:       {np.std(cv_scores):.4f}%")
        print(f"  Min:       {np.min(cv_scores):.4f}%")
        print(f"  Max:       {np.max(cv_scores):.4f}%")
        print(f"  Avg iterations: {np.mean(best_iterations):.0f}")
        print("=" * 60)

        # Log CV results to MLflow
        log_cv_results(np.array(cv_scores), metric_name="mape")
        mlflow.log_metric("avg_best_iteration", np.mean(best_iterations))

        # OOF MAPE
        oof_mape = calculate_mape(y_train_np, oof_predictions)
        mlflow.log_metric("oof_mape", oof_mape)
        print(f"\n  OOF MAPE: {oof_mape:.4f}%")

        # Create output directory for this run (outputs/run_YYYYMMDD_HHMMSS/)
        base_output_dir = Path(__file__).parent.parent / "outputs"
        output_dir = base_output_dir / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Output directory: {output_dir}")

        # Create models directory
        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # ===== Save trained models =====
        print("\nSaving trained models...")
        for fold_idx, model in enumerate(trained_models):
            model_path = models_dir / f"model_fold{fold_idx}.txt"
            model.booster_.save_model(str(model_path))
            print(f"  Saved: {model_path}")
            mlflow.log_artifact(model_path, artifact_path="models")

        # Save feature list for inference
        features_path = models_dir / "features.json"
        with open(features_path, "w") as f:
            json.dump({"features": ALL_FEATURES}, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {features_path}")

        # ===== Save preprocessed data (for Permutation Importance) =====
        print("\nSaving preprocessed data...")
        X_train.write_parquet(output_dir / "X_train.parquet")
        pl.DataFrame({"target": y_train_np}).write_parquet(output_dir / "y_train.parquet")
        print(f"  Saved: X_train.parquet, y_train.parquet")

        # ===== Save OOF predictions =====
        print("\nSaving OOF predictions...")
        oof_df = pl.DataFrame({
            "id": train_ids,
            "actual": y_train_np,
            "predicted": oof_predictions,
        })
        oof_path = output_dir / "oof_predictions.csv"
        oof_df.write_csv(oof_path)
        print(f"  Saved: {oof_path}")
        mlflow.log_artifact(oof_path, artifact_path="predictions")

        # ===== Save feature importance =====
        print("\nSaving feature importance...")
        importance_dict = {
            "feature": ALL_FEATURES,
            "importance": feature_importance.tolist(),
        }

        # Sort and display top features
        sorted_indices = np.argsort(feature_importance)[::-1]
        print("  Top 30 Features:")
        for i, idx in enumerate(sorted_indices[:30]):
            feat_name = ALL_FEATURES[idx]
            # Mark new features
            if feat_name in EXP010_FEATURE_COLUMNS or 'area_age_cat' in feat_name:
                marker = "★"  # exp010 new features
            elif feat_name.startswith('lp_') or '_lp_' in feat_name:
                marker = "●"  # landprice features
            elif feat_name.startswith('reform_svd_') or feat_name in ['years_since_wet_reform', 'years_since_interior_reform']:
                marker = "◆"  # reform features
            elif feat_name in ['post1_te', 'post_full_te']:
                marker = "◇"  # postal TE
            else:
                marker = ""
            print(f"    {i+1}. {feat_name}: {feature_importance[idx]:.4f} {marker}")

        # Display exp010 feature importance
        print("\n  exp010 feature importance (NEW):")
        exp010_features = [f for f in ALL_FEATURES if f in EXP010_FEATURE_COLUMNS or 'area_age_cat' in f]
        for feat in sorted(exp010_features, key=lambda x: feature_importance[ALL_FEATURES.index(x)], reverse=True):
            idx = ALL_FEATURES.index(feat)
            rank = list(sorted_indices).index(idx) + 1
            print(f"    {feat}: {feature_importance[idx]:.4f} (rank: {rank}/{len(ALL_FEATURES)})")

        # Save as JSON
        importance_path = output_dir / "feature_importance.json"
        with open(importance_path, "w", encoding="utf-8") as f:
            json.dump(importance_dict, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {importance_path}")
        mlflow.log_artifact(importance_path, artifact_path="feature_importance")

        # Save as CSV
        importance_df = pl.DataFrame({
            "feature": ALL_FEATURES,
            "importance": feature_importance,
        }).sort("importance", descending=True)
        importance_csv_path = output_dir / "feature_importance.csv"
        importance_df.write_csv(importance_csv_path)
        mlflow.log_artifact(importance_csv_path, artifact_path="feature_importance")

        # ===== Create submission file =====
        print("\nCreating submission file...")
        # IDを6桁ゼロ埋め文字列に変換
        test_ids_str = [f"{int(id_):06d}" for id_ in test_ids]
        submission_df = pl.DataFrame({
            "id": test_ids_str,
            "price": test_predictions.astype(int),  # 整数に変換
        })
        submission_path = output_dir / "submission.csv"
        submission_df.write_csv(submission_path, include_header=False)  # ヘッダーなし
        print(f"  Saved: {submission_path}")
        mlflow.log_artifact(submission_path, artifact_path="submission")

        # Also save to project submissions directory
        project_submissions_dir = project_root / "09_submissions"
        project_submissions_dir.mkdir(parents=True, exist_ok=True)
        project_submission_path = project_submissions_dir / f"submission_{exp_config['id']}_{timestamp}.csv"
        submission_df.write_csv(project_submission_path, include_header=False)  # ヘッダーなし
        print(f"  Saved: {project_submission_path}")

        # ===== Experiment complete =====
        mlflow.set_tag("status", "completed")

        # Results summary
        print("\n" + "=" * 60)
        print(f"Experiment Summary ({exp_config['id']}: {exp_config['name']})")
        print("=" * 60)
        print(f"  Baseline (exp009): 13.62%")
        print(f"  This result:       {np.mean(cv_scores):.2f}%")
        improvement = 13.62 - np.mean(cv_scores)
        if improvement > 0:
            print(f"  Improvement:       {improvement:.2f}pt")
        else:
            print(f"  Degradation:       {-improvement:.2f}pt")
        print(f"  Features: {len(ALL_FEATURES)} (base + landprice + exp010)")
        print("=" * 60)
        print("\nTraining completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="exp010 low_price_improvement training")
    parser.add_argument("--test", action="store_true", help="Use test parameters for fast execution")
    args = parser.parse_args()

    train_exp010(test_mode=args.test)
