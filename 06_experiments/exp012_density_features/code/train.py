"""
exp012_density_features training script

密度特徴量を追加した実験。低価格帯（特にvery_low密度領域）の予測精度改善を目指す。

使用方法:
    # 基本実行（MSE損失）
    python train.py --objective mse

    # Huber損失
    python train.py --objective huber

    # テストモード（高速実行）
    python train.py --objective mse --test

    # 前処理済み特徴量を読み込み（2回目以降）
    python train.py --objective huber --features-dir outputs/run_mse_xxx/
"""

import sys
import argparse
from pathlib import Path

# Add project root to path (for 04_src)
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "04_src"))
# Add exp012 code directory to path
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

# exp012 components
from preprocessing import preprocess_for_training
from pipeline import FeaturePipeline
from constants import EXP012_FEATURE_COLUMNS, REMOVE_FEATURES


def load_config(config_path: Path, test_mode: bool = False) -> dict:
    """Load configuration file with extends support

    Args:
        config_path: Path to the config file
        test_mode: If True, use params_test instead of params

    Returns:
        Merged configuration dict
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Handle extends (inheritance)
    if "extends" in config:
        base_config_name = config.pop("extends")
        base_config_path = config_path.parent / base_config_name
        base_config = load_config(base_config_path, test_mode=False)  # Don't apply test_mode to base

        # Deep merge: config overrides base_config
        config = deep_merge(base_config, config)

    # テストモードの場合、params_testを使用
    if test_mode and "model" in config and "params_test" in config["model"]:
        config["model"]["params"] = config["model"]["params_test"]
        print("*** TEST MODE: Using params_test ***")

    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries (override takes precedence)"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class _DummyContext:
    """Dummy context manager for test mode (skip MLflow)"""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def get_sample_weight(
    y_train_original: np.ndarray,
    config: dict,
    X_train: pl.DataFrame = None,
    verbose: bool = True,
) -> np.ndarray:
    """Calculate sample weights based on config

    Args:
        y_train_original: Original scale target values
        config: Loss configuration
        X_train: Training features (optional, for column-based weighting)
        verbose: Print details

    Returns:
        Sample weights array

    Config options:
        weight_column: Column name to use for weighting (default: "target" = y)
        weight_transform: Transform to apply ("inverse", "sqrt_inverse", "log_inverse")
        threshold: For threshold-based weighting (万円)
        weight_below/weight_above: Weights for threshold method
    """
    weight_column = config.get("weight_column", "target")
    weight_transform = config.get("weight_transform", "inverse")

    # Get values to weight by
    if weight_column == "target":
        values = y_train_original
        if verbose:
            print(f"  Weight column: target (y)")
    else:
        if X_train is None:
            raise ValueError(f"X_train required for weight_column='{weight_column}'")
        if weight_column not in X_train.columns:
            raise ValueError(f"Column '{weight_column}' not found in X_train")
        values = X_train[weight_column].to_numpy()
        if verbose:
            print(f"  Weight column: {weight_column}")

    # Apply transform
    if weight_transform == "inverse":
        # 逆数重み: 値が小さいほど重み大
        weights = 1.0 / np.maximum(np.abs(values), 1.0)
    elif weight_transform == "sqrt_inverse":
        # 平方根の逆数: より緩やかな重み付け
        weights = 1.0 / np.maximum(np.sqrt(np.abs(values)), 1.0)
    elif weight_transform == "log_inverse":
        # 対数の逆数: さらに緩やかな重み付け
        weights = 1.0 / np.maximum(np.log1p(np.abs(values)), 1.0)
    elif weight_transform == "threshold":
        # 閾値ベース
        threshold = config.get("threshold", 1000) * 10000  # 万円→円
        weight_below = config.get("weight_below", 2.0)
        weight_above = config.get("weight_above", 1.0)
        weights = np.where(values < threshold, weight_below, weight_above)
        if verbose:
            print(f"  Threshold: {threshold/10000:.0f}万円, below={weight_below}, above={weight_above}")
            print(f"  Counts: below={np.sum(values < threshold)}, above={np.sum(values >= threshold)}")
    else:
        raise ValueError(f"Unknown weight_transform: {weight_transform}")

    # Normalize to mean=1 (except threshold)
    if weight_transform != "threshold":
        weights = weights / weights.mean()

    if verbose:
        print(f"  Transform: {weight_transform}")
        print(f"  Weights: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")

    return weights


def train_exp012(
    config_path: str,
    test_mode: bool = False,
    features_dir: str = None,
    cli_overrides: dict = None,
):
    """Train exp012 density_features experiment

    Args:
        config_path: Path to the config YAML file
        test_mode: If True, use test parameters for fast execution
        features_dir: Path to directory with preprocessed features (X_train.parquet, y_train.parquet)
                     If provided, skip preprocessing and load from parquet files
        cli_overrides: Dict of CLI overrides for loss config (e.g., {"objective_type": "huber"})
    """
    # Resolve config path
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = Path(__file__).parent.parent / config_path

    # Load config
    config = load_config(config_file, test_mode=test_mode)
    exp_config = config["experiment"]
    training_config = config["training"]
    model_config = config["model"]
    loss_config = config.get("loss", {})

    # Apply CLI overrides to loss_config
    if cli_overrides:
        for key, value in cli_overrides.items():
            loss_config[key] = value
            if not test_mode:
                print(f"  CLI override: loss.{key} = {value}")

    # Set seed
    seed = training_config["seed"]
    set_seed(seed)

    # Get loss configuration
    objective_type = loss_config.get("objective_type", "mse")

    # Test mode: ultra-lightweight settings
    if test_mode:
        print(f"\n[TEST] objective={objective_type}")
        training_config["n_splits"] = 2
        training_config["early_stopping_rounds"] = 2
        model_config["params"] = {
            "objective": "regression",
            "metric": "l2",
            "boosting_type": "gbdt",
            "learning_rate": 0.5,
            "n_estimators": 5,
            "max_depth": 3,
            "num_leaves": 8,
            "verbose": -1,
            "force_col_wise": True,
        }
    else:
        print(f"\n*** exp012: density_features ***")
        print(f"*** Loss function: {objective_type} ***")

    # MLflow: skip in test mode
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{exp_config['name']}_{timestamp}"

    # Use context manager or dummy context for test mode
    mlflow_context = mlflow.start_run(run_name=run_name) if not test_mode else _DummyContext()

    with mlflow_context:
        if not test_mode:
            print(f"Training started: {run_name}")
            # ===== Set tags =====
            mlflow.set_tag("experiment_type", exp_config["name"])
            mlflow.set_tag("experiment_id", exp_config["id"])
            mlflow.set_tag("model_family", "gbdt")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("base_experiment", exp_config.get("base", "exp011"))
            mlflow.set_tag("target_transform", training_config.get("target_transform", "none"))
            mlflow.set_tag("objective_type", objective_type)

        # ===== Load data =====
        if features_dir:
            # Load preprocessed features from parquet
            features_path = Path(features_dir)
            if not features_path.is_absolute():
                features_path = Path(__file__).parent.parent / features_dir

            print(f"\nLoading preprocessed features from: {features_path}")
            X_train = pl.read_parquet(features_path / "X_train.parquet")
            y_train_df = pl.read_parquet(features_path / "y_train.parquet")
            y_train = y_train_df["target"]

            # Load feature list
            features_json = features_path / "models" / "features.json"
            if features_json.exists():
                with open(features_json, "r") as f:
                    ALL_FEATURES = json.load(f)["features"]
            else:
                ALL_FEATURES = X_train.columns

            print(f"  - X_train: {X_train.shape}")
            print(f"  - y_train: {len(y_train)}")
            print(f"  - Features: {len(ALL_FEATURES)}")

            # Load test data for predictions
            data_config_path = project_root / "03_configs" / "data.yaml"
            with open(data_config_path, "r", encoding="utf-8") as f:
                data_config = yaml.safe_load(f)
            data_config["data"]["test_path"] = str(project_root / data_config["data"]["test_path"])
            loader = DataLoader(config=data_config, add_address_columns=False)
            test = loader.load_test()
            test_ids = test["id"].to_numpy()

            # X_testも読み込み（存在する場合）
            x_test_path = features_path / "X_test.parquet"
            if x_test_path.exists():
                X_test = pl.read_parquet(x_test_path)
            else:
                # X_testがない場合は前処理が必要（テスト用には対応しない）
                raise FileNotFoundError(f"X_test.parquet not found in {features_path}")

            train_ids = np.arange(len(X_train))
            mlflow.set_tag("features_from", str(features_path))

            # CV splits needed for training
            n_splits = training_config["n_splits"]
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            cv_splits = list(cv.split(X_train))

        else:
            # Normal preprocessing
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

            # Test mode: sample data for speed
            if test_mode:
                train = train.sample(n=min(1000, len(train)), seed=seed)
                test = test.head(100)

            # Save IDs before preprocessing
            train_ids = np.arange(len(train))
            test_ids = test["id"].to_numpy()

            if not test_mode:
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
            if not test_mode:
                print("\n" + "=" * 60)
                print("exp012 Preprocessing (with Density Features)")
                print("=" * 60)
            X_train, X_test, y_train, pipeline = preprocess_for_training(
                train, test, cv_splits=cv_splits, verbose=not test_mode
            )

            # Get actual feature list from pipeline
            ALL_FEATURES = pipeline.get_feature_names()

            # 低重要度特徴量を削除（exp010_1で特定した35個）
            if REMOVE_FEATURES and not test_mode:
                use_features = [f for f in ALL_FEATURES if f not in REMOVE_FEATURES]
                print(f"\nFeature Selection: {len(ALL_FEATURES)} -> {len(use_features)} (-{len(REMOVE_FEATURES)})")
                X_train = X_train.select(use_features)
                X_test = X_test.select(use_features)
                ALL_FEATURES = use_features

            # Display pipeline summary
            if not test_mode:
                print("\n" + pipeline.summary())

        # Count feature categories
        n_exp012_features = len([c for c in ALL_FEATURES if c in EXP012_FEATURE_COLUMNS or 'area_age_cat' in c or 'density' in c])
        n_landprice_features = len([c for c in ALL_FEATURES if c.startswith('lp_') or '_lp_' in c])
        n_base_features = len(ALL_FEATURES) - n_landprice_features - n_exp012_features

        if not test_mode:
            print(f"\nTotal features: {len(ALL_FEATURES)}")
            print(f"  - Base features: {n_base_features}")
            print(f"  - Landprice features: {n_landprice_features}")
            print(f"  - exp012 features: {n_exp012_features}")

        # ===== Target variable log transform =====
        target_transform = training_config.get("target_transform", "none")
        y_train_np = y_train.to_numpy()

        if target_transform == "log1p":
            if not test_mode:
                print("\nApplying log1p transform to target variable")
            y_train_transformed = np.log1p(y_train_np)
            if not test_mode:
                print(f"  - Before: mean={y_train_np.mean():.2f}, std={y_train_np.std():.2f}")
                print(f"  - After: mean={y_train_transformed.mean():.4f}, std={y_train_transformed.std():.4f}")
        else:
            y_train_transformed = y_train_np
            if not test_mode:
                print("\nTarget transform: none")

        # ===== Setup loss function =====
        if not test_mode:
            print(f"\nSetting up loss function: {objective_type}")
        sample_weight = None

        lgb_params = model_config["params"].copy()
        lgb_params["random_state"] = seed
        lgb_params["deterministic"] = True

        if objective_type == "mse":
            # Default: regression with L2 loss
            pass
        elif objective_type == "huber":
            lgb_params["objective"] = "huber"
            if not test_mode:
                print("  Using Huber loss (robust to outliers)")
        elif objective_type == "sample_weight":
            # Calculate sample weights (on original scale)
            sample_weight = get_sample_weight(y_train_np, loss_config, X_train, verbose=not test_mode)
        else:
            raise ValueError(f"Unknown objective_type: {objective_type}")

        # Log feature info (skip in test mode)
        if not test_mode:
            mlflow.log_param("n_features", len(ALL_FEATURES))
            mlflow.log_param("n_base_features", n_base_features)
            mlflow.log_param("n_landprice_features", n_landprice_features)
            mlflow.log_param("n_exp012_features", n_exp012_features)
            mlflow.log_param("seed", seed)
            mlflow.log_param("n_splits", n_splits)
            mlflow.log_param("early_stopping_rounds", training_config["early_stopping_rounds"])
            mlflow.log_param("target_transform", target_transform)
            mlflow.log_param("objective_type", objective_type)

            # Log loss config parameters
            mlflow.log_params({f"loss_{k}": v for k, v in loss_config.items()})

            # Save feature list
            log_feature_list(ALL_FEATURES, artifact_path="features.txt")

            # Log parameters
            log_model_params(lgb_params, prefix="lgb")

            print("\nLightGBM parameters:")
            print(f"  - learning_rate: {lgb_params['learning_rate']}")
            print(f"  - max_depth: {lgb_params['max_depth']}")
            print(f"  - num_leaves: {lgb_params['num_leaves']}")
            print(f"  - n_estimators: {lgb_params['n_estimators']} (with early_stopping)")
            print(f"  - objective: {lgb_params.get('objective', 'regression')}")

        # ===== Model training (n-Fold CV) =====
        if not test_mode:
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
            if not test_mode:
                print(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")

            # Split data
            X_tr = X_train_pd.iloc[train_idx]
            X_val = X_train_pd.iloc[val_idx]
            y_tr = y_train_transformed[train_idx]
            y_val_transformed = y_train_transformed[val_idx]
            y_val_original = y_train_np[val_idx]

            # Sample weight for this fold
            fold_sample_weight = None
            if sample_weight is not None:
                fold_sample_weight = sample_weight[train_idx]

            # Create model
            model = LGBMRegressor(**lgb_params)

            # Train
            fit_params = {
                "eval_set": [(X_val, y_val_transformed)],
                "callbacks": [
                    early_stopping(stopping_rounds=early_stopping_rounds, verbose=not test_mode),
                    log_evaluation(period=early_stopping_rounds if not test_mode else 1000),
                ],
            }
            if fold_sample_weight is not None:
                fit_params["sample_weight"] = fold_sample_weight

            model.fit(X_tr, y_tr, **fit_params)

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

            if not test_mode:
                print(f"  Validation MAPE: {mape_score:.4f}%")
                print(f"  Best iteration: {best_iter}")

        # ===== CV Results Summary =====
        if test_mode:
            # Test mode: minimal output
            print(f"[TEST] OK - CV MAPE: {np.mean(cv_scores):.2f}%")
            return {"cv_mape_mean": np.mean(cv_scores), "status": "success"}

        print("\n" + "=" * 60)
        print("Cross Validation Results")
        print("=" * 60)
        print(f"  Objective: {objective_type}")
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
        output_dir = base_output_dir / f"run_{objective_type}_{timestamp}"
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
        print("  Top 20 Features:")
        for i, idx in enumerate(sorted_indices[:20]):
            feat_name = ALL_FEATURES[idx]
            print(f"    {i+1}. {feat_name}: {feature_importance[idx]:.4f}")

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

        # ===== Save preprocessed data as parquet (for reuse and Permutation Importance) =====
        if not features_dir:  # Only save if not loaded from cache
            print("\nSaving preprocessed data...")
            X_train.write_parquet(output_dir / "X_train.parquet")
            X_test.write_parquet(output_dir / "X_test.parquet")
            pl.DataFrame({"target": y_train_np}).write_parquet(output_dir / "y_train.parquet")
            print(f"  Saved: X_train.parquet, X_test.parquet, y_train.parquet")

        # ===== Save test predictions (raw float values for ensemble) =====
        print("\nSaving test predictions...")
        test_pred_df = pl.DataFrame({
            "id": test_ids,
            "predicted": test_predictions,
        })
        test_pred_path = output_dir / "test_predictions.csv"
        test_pred_df.write_csv(test_pred_path)
        print(f"  Saved: {test_pred_path}")
        mlflow.log_artifact(test_pred_path, artifact_path="predictions")

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
        project_submission_path = project_submissions_dir / f"submission_{exp_config['id']}_{objective_type}_{timestamp}.csv"
        submission_df.write_csv(project_submission_path, include_header=False)  # ヘッダーなし
        print(f"  Saved: {project_submission_path}")

        # ===== Experiment complete =====
        mlflow.set_tag("status", "completed")

        # Results summary
        print("\n" + "=" * 60)
        print(f"Experiment Summary ({exp_config['id']}: {exp_config['name']})")
        print("=" * 60)
        print(f"  Objective type: {objective_type}")
        print(f"  Baseline (exp011 MSE): 12.40%")
        print(f"  This result:           {np.mean(cv_scores):.2f}%")
        improvement = 12.40 - np.mean(cv_scores)
        if improvement > 0:
            print(f"  Improvement:           {improvement:.2f}pt")
        else:
            print(f"  Degradation:           {-improvement:.2f}pt")
        print(f"  Features: {len(ALL_FEATURES)}")
        print("=" * 60)
        print("\nTraining completed!")

        return {
            "objective_type": objective_type,
            "cv_mape_mean": np.mean(cv_scores),
            "cv_mape_std": np.std(cv_scores),
            "oof_mape": oof_mape,
            "oof_predictions": oof_predictions,
            "test_predictions": test_predictions,
            "output_dir": str(output_dir),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="exp012 density_features training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic execution
  python train.py --objective mse
  python train.py --objective huber

  # Test mode (fast execution)
  python train.py --objective mse --test

  # Load preprocessed features
  python train.py --objective huber --features-dir outputs/run_mse_xxx/
        """
    )

    # Required arguments
    parser.add_argument("--objective", type=str, required=True,
                        choices=["mse", "huber", "sample_weight"],
                        help="Objective type: mse, huber, sample_weight")

    # Optional arguments
    parser.add_argument("--config", type=str, default="configs/experiment.yaml",
                        help="Path to base config YAML file (default: configs/experiment.yaml)")
    parser.add_argument("--test", action="store_true",
                        help="Use test parameters for fast execution")
    parser.add_argument("--features-dir", type=str, default=None,
                        help="Path to directory with preprocessed features (X_train.parquet, y_train.parquet)")

    # Loss function specific arguments (for sample_weight)
    parser.add_argument("--weight-column", type=str, default=None,
                        help="Column for sample_weight: 'target' (default) or feature column name")
    parser.add_argument("--weight-transform", type=str, default=None,
                        choices=["inverse", "sqrt_inverse", "log_inverse", "threshold"],
                        help="Sample weight transform (default: inverse)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold for threshold-based weighting (万円)")
    parser.add_argument("--weight-below", type=float, default=None,
                        help="Weight for samples below threshold")
    parser.add_argument("--weight-above", type=float, default=None,
                        help="Weight for samples above threshold")

    args = parser.parse_args()

    # Build CLI overrides for config
    cli_overrides = {"objective_type": args.objective}
    if args.weight_column is not None:
        cli_overrides["weight_column"] = args.weight_column
    if args.weight_transform is not None:
        cli_overrides["weight_transform"] = args.weight_transform
    if args.threshold is not None:
        cli_overrides["threshold"] = args.threshold
    if args.weight_below is not None:
        cli_overrides["weight_below"] = args.weight_below
    if args.weight_above is not None:
        cli_overrides["weight_above"] = args.weight_above

    train_exp012(
        config_path=args.config,
        test_mode=args.test,
        features_dir=args.features_dir,
        cli_overrides=cli_overrides
    )
