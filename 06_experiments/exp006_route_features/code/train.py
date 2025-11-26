"""
exp006_route_features training script

Based on exp005 with the following changes:
1. Add route/station features (TE, TF-IDF, LE, CE)
2. Add access time features (walk_time, total_access_time)
3. Add geo PCA features (4D -> 2D)

Target: CV MAPE < 14.0% (exp005: 14.74%, improvement > 0.7pt)
"""

import sys
from pathlib import Path

# Add current directory first
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

# Add project root to path
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

# Experiment-specific preprocessing (this directory)
from preprocessing import (
    preprocess_for_training,
    get_feature_names,
    CATEGORICAL_FEATURES,
    TARGET_ENCODING_COLUMNS,
    COUNT_ENCODING_COLUMNS,
    NUMERIC_FEATURES,
    AGE_NUMERIC_FEATURES,
    ACCESS_NUMERIC_FEATURES,
    AGE_THRESHOLD,
    AREA_THRESHOLD,
    MAJOR_CITIES,
    TFIDF_MAX_FEATURES,
)


def load_config() -> dict:
    """Load configuration file"""
    config_path = Path(__file__).parent.parent / "configs" / "params.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_exp006():
    """Train exp006 route_features"""

    # Load config
    config = load_config()
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

        # ===== Preprocessing =====
        print("\nPreprocessing...")
        X_train, X_test, y_train = preprocess_for_training(train, test, cv_splits=cv_splits)

        # Get actual feature list
        ALL_FEATURES = list(X_train.columns)

        print(f"\nFeature info:")
        print(f"  - Total features: {len(ALL_FEATURES)}")
        print(f"  - Categorical features: {len(CATEGORICAL_FEATURES)}")
        print(f"  - Age-related features: {len(AGE_NUMERIC_FEATURES)}")
        print(f"  - Access time features: {len(ACCESS_NUMERIC_FEATURES)}")
        print(f"  - TF-IDF features: {TFIDF_MAX_FEATURES}")
        print(f"  - Geo PCA features: 2")

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
        mlflow.log_param("n_categorical_features", len(CATEGORICAL_FEATURES))
        mlflow.log_param("seed", seed)
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("early_stopping_rounds", training_config["early_stopping_rounds"])
        mlflow.log_param("target_transform", target_transform)

        # exp006-specific parameters
        mlflow.log_param("n_age_features", len(AGE_NUMERIC_FEATURES))
        mlflow.log_param("n_access_features", len(ACCESS_NUMERIC_FEATURES))
        mlflow.log_param("tfidf_max_features", TFIDF_MAX_FEATURES)
        mlflow.log_param("geo_pca_components", 2)
        mlflow.log_param("age_threshold", AGE_THRESHOLD)
        mlflow.log_param("area_threshold", AREA_THRESHOLD)
        mlflow.log_param("major_cities", str(MAJOR_CITIES))

        # Save feature list
        log_feature_list(ALL_FEATURES, artifact_path="features.txt")

        # ===== LightGBM parameters (from YAML) =====
        lgb_params = model_config["params"].copy()
        lgb_params["random_state"] = seed
        lgb_params["deterministic"] = True

        # Log parameters
        log_model_params(lgb_params, prefix="lgb")

        print("\nLightGBM parameters:")
        print(f"  - learning_rate: {lgb_params['learning_rate']}")
        print(f"  - max_depth: {lgb_params['max_depth']}")
        print(f"  - num_leaves: {lgb_params['num_leaves']}")
        print(f"  - reg_lambda (L2): {lgb_params['reg_lambda']}")
        print(f"  - reg_alpha (L1): {lgb_params['reg_alpha']}")
        print(f"  - colsample_bytree: {lgb_params['colsample_bytree']}")
        print(f"  - subsample: {lgb_params['subsample']}")
        print(f"  - n_estimators: {lgb_params['n_estimators']} (with early_stopping)")

        # ===== Model training (3-Fold CV) =====
        print(f"\n{n_splits}-Fold Cross Validation starting...")

        cv_scores = []
        oof_predictions = np.zeros(len(X_train))
        test_predictions = np.zeros(len(X_test))
        feature_importance = np.zeros(len(ALL_FEATURES))
        best_iterations = []

        # Polars -> pandas (to preserve feature names)
        X_train_pd = X_train.to_pandas()
        X_test_pd = X_test.to_pandas()

        early_stopping_rounds = training_config["early_stopping_rounds"]

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")

            # Split data (pandas DataFrame to preserve feature names)
            X_tr = X_train_pd.iloc[train_idx]
            X_val = X_train_pd.iloc[val_idx]
            y_tr = y_train_transformed[train_idx]
            y_val_transformed = y_train_transformed[val_idx]
            y_val_original = y_train_np[val_idx]

            # Create model
            model = LGBMRegressor(**lgb_params)

            # Train (sklearn API)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val_transformed)],
                callbacks=[
                    early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
                    log_evaluation(period=100),
                ],
            )

            # Get best_iteration
            best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') else model.n_estimators
            best_iterations.append(best_iter)

            # Predict (in log space)
            val_pred_transformed = model.predict(X_val)

            # Inverse transform to original scale
            if target_transform == "log1p":
                val_pred = np.expm1(val_pred_transformed)
                val_pred = np.maximum(val_pred, 0)  # Prevent negative values
            else:
                val_pred = val_pred_transformed

            oof_predictions[val_idx] = val_pred

            # Test prediction (for ensemble)
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

            # Accumulate feature importance (gain)
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

        # Create output directory
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        # ===== Save OOF predictions (for error analysis) =====
        print("\nSaving OOF predictions...")
        oof_df = pl.DataFrame({
            "id": train_ids,
            "actual": y_train_np,
            "predicted": oof_predictions,
        })
        oof_path = output_dir / f"oof_predictions_{timestamp}.csv"
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
        print("  Top 20 Features:")
        for i, idx in enumerate(sorted_indices[:20]):
            feat_name = ALL_FEATURES[idx]
            # Mark new features
            if feat_name.startswith('tfidf_') or feat_name.startswith('geo_pca_'):
                marker = "★"
            elif feat_name in ACCESS_NUMERIC_FEATURES or feat_name.endswith('_le'):
                marker = "☆"
            else:
                marker = ""
            print(f"    {i+1}. {feat_name}: {feature_importance[idx]:.4f} {marker}")

        # Display new feature importance
        print("\n  New feature importance (exp006):")
        new_features = [f for f in ALL_FEATURES if
                        f.startswith('tfidf_') or f.startswith('geo_pca_') or
                        f in ACCESS_NUMERIC_FEATURES or f.endswith('_le') or
                        f == 'rosen_name1_te']
        for feat in new_features:
            if feat in ALL_FEATURES:
                idx = ALL_FEATURES.index(feat)
                rank = list(sorted_indices).index(idx) + 1
                print(f"    {feat}: {feature_importance[idx]:.4f} (rank: {rank}/{len(ALL_FEATURES)})")

        # Save as JSON
        importance_path = output_dir / f"feature_importance_{timestamp}.json"
        with open(importance_path, "w", encoding="utf-8") as f:
            json.dump(importance_dict, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {importance_path}")
        mlflow.log_artifact(importance_path, artifact_path="feature_importance")

        # Save as CSV (easier to visualize)
        importance_df = pl.DataFrame({
            "feature": ALL_FEATURES,
            "importance": feature_importance,
        }).sort("importance", descending=True)
        importance_csv_path = output_dir / f"feature_importance_{timestamp}.csv"
        importance_df.write_csv(importance_csv_path)
        mlflow.log_artifact(importance_csv_path, artifact_path="feature_importance")

        # ===== Generate submission file =====
        print("\nGenerating submission file...")

        # IDを6桁0埋め文字列に変換
        test_ids_formatted = [f"{int(id_):06d}" for id_ in test_ids]
        # 小数第1位で四捨五入（整数化）
        test_predictions_rounded = np.round(test_predictions).astype(int)

        submission = pl.DataFrame({
            "id": test_ids_formatted,
            "money_room": test_predictions_rounded,
        })

        submission_path = output_dir / f"submission_{timestamp}.csv"
        # ヘッダーなし、6桁0埋めID、整数値で出力
        submission.write_csv(submission_path, include_header=False)

        print(f"  Saved: {submission_path}")

        # Log to MLflow
        mlflow.log_artifact(submission_path, artifact_path="submissions")

        # ===== Experiment complete =====
        mlflow.set_tag("status", "completed")

        # Results summary
        print("\n" + "=" * 60)
        print(f"Experiment Summary ({exp_config['id']}: {exp_config['name']})")
        print("=" * 60)
        print(f"  Baseline (exp005): 14.74%")
        print(f"  This result:       {np.mean(cv_scores):.2f}%")
        improvement = 14.74 - np.mean(cv_scores)
        if improvement > 0:
            print(f"  Improvement:       {improvement:.2f}pt")
        else:
            print(f"  Degradation:       {-improvement:.2f}pt")
        print(f"  Target: < 14.0%")
        print("=" * 60)
        print("\nTraining completed!")


if __name__ == "__main__":
    train_exp006()
