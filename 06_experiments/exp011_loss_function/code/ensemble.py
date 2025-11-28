"""
exp011_loss_function ensemble script

Phase 1-2の各損失関数モデルの予測をアンサンブルで統合する。

使用方法:
    # 単純平均
    python ensemble.py --level0-dirs outputs/run_mse_* outputs/run_huber_* --method average

    # 重み付き平均（OOFでMAPE最小化）
    python ensemble.py --level0-dirs outputs/run_mse_* outputs/run_huber_* --method weighted

    # スタッキング（Ridge）
    python ensemble.py --level0-dirs outputs/run_mse_* outputs/run_huber_* --method stacking --meta-model ridge

    # スタッキング（LightGBM）
    python ensemble.py --level0-dirs outputs/run_mse_* outputs/run_huber_* --method stacking --meta-model lightgbm
"""

import sys
import argparse
import glob
import json
from pathlib import Path
from datetime import datetime
from typing import Any

# Add project root to path (for 04_src)
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "04_src"))
# Add exp011 code directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow
import numpy as np
import polars as pl
import yaml
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

# Common components (04_src/)
from stacking.trainer import StackingTrainer
from evaluation.metrics import calculate_mape


def load_config(config_path: Path) -> dict:
    """Load configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_meta_model(name: str, params: dict) -> Any:
    """Create meta model from name and parameters."""
    if name == "ridge":
        return Ridge(**params)
    elif name == "lightgbm":
        return LGBMRegressor(**params)
    else:
        raise ValueError(f"Unknown meta model: {name}")


def load_level0_predictions(
    level0_dirs: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load OOF and test predictions from Level 0 models.

    Returns:
        X_oof: OOF predictions matrix (n_train, n_models)
        y_oof: Actual target values (n_train,)
        X_test: Test predictions matrix (n_test, n_models)
        train_ids: Training sample IDs
        test_ids: Test sample IDs
        model_names: List of model names
    """
    # Expand glob patterns
    expanded_dirs = []
    for pattern in level0_dirs:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No directories match pattern: {pattern}")
        expanded_dirs.extend(matches)

    print(f"Loading predictions from {len(expanded_dirs)} Level 0 models:")

    oof_predictions_list = []
    test_predictions_list = []
    model_names = []
    y_oof = None
    train_ids = None
    test_ids = None

    for dir_path in expanded_dirs:
        dir_path = Path(dir_path)
        oof_path = dir_path / "oof_predictions.csv"
        test_path = dir_path / "test_predictions.csv"

        if not oof_path.exists():
            raise FileNotFoundError(f"OOF predictions not found: {oof_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test predictions not found: {test_path}")

        # Extract model name from directory
        model_name = dir_path.name
        model_names.append(model_name)
        print(f"  - {model_name}")

        # Load OOF predictions
        oof_df = pl.read_csv(oof_path)
        if y_oof is None:
            train_ids = oof_df["id"].to_numpy()
            y_oof = oof_df["actual"].to_numpy()
        else:
            assert np.array_equal(train_ids, oof_df["id"].to_numpy()), \
                f"Train IDs mismatch in {model_name}"
            assert np.allclose(y_oof, oof_df["actual"].to_numpy()), \
                f"Actual values mismatch in {model_name}"

        oof_predictions_list.append(oof_df["predicted"].to_numpy())

        # Load test predictions
        test_df = pl.read_csv(test_path)
        if test_ids is None:
            test_ids = test_df["id"].to_numpy()
        else:
            assert np.array_equal(test_ids, test_df["id"].to_numpy()), \
                f"Test IDs mismatch in {model_name}"

        test_predictions_list.append(test_df["predicted"].to_numpy())

    # Stack predictions into matrices
    X_oof = np.column_stack(oof_predictions_list)
    X_test = np.column_stack(test_predictions_list)

    print(f"\nLoaded shapes:")
    print(f"  X_oof: {X_oof.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_oof: {y_oof.shape}")

    return X_oof, y_oof, X_test, train_ids, test_ids, model_names


# =============================================================================
# Ensemble Methods
# =============================================================================


def find_optimal_weights(X_oof: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OOFでMAPE最小化する重みを探索

    Args:
        X_oof: OOF predictions matrix (n_samples, n_models)
        y: Actual target values (n_samples,)

    Returns:
        Optimal weights (n_models,), sum to 1
    """
    n_models = X_oof.shape[1]

    def objective(weights):
        # 正規化
        weights = weights / weights.sum()
        pred = X_oof @ weights
        return calculate_mape(y, pred)

    # 初期値: 均等重み
    x0 = np.ones(n_models) / n_models
    # 制約: 重み >= 0
    bounds = [(0, 1)] * n_models

    result = minimize(objective, x0, bounds=bounds, method='SLSQP')
    return result.x / result.x.sum()


def run_average(
    X_oof: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    model_names: list[str],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """単純平均アンサンブル

    Returns:
        oof_pred: OOF predictions
        test_pred: Test predictions
        info: Additional info (weights, etc.)
    """
    n_models = X_oof.shape[1]
    weights = np.ones(n_models) / n_models

    oof_pred = X_oof @ weights
    test_pred = X_test @ weights

    info = {
        "weights": {name: float(w) for name, w in zip(model_names, weights)},
    }

    return oof_pred, test_pred, info


def run_weighted(
    X_oof: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    model_names: list[str],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """重み付き平均アンサンブル（OOFでMAPE最小化）

    Returns:
        oof_pred: OOF predictions
        test_pred: Test predictions
        info: Additional info (weights, etc.)
    """
    print("\nOptimizing weights to minimize OOF MAPE...")
    weights = find_optimal_weights(X_oof, y)

    oof_pred = X_oof @ weights
    test_pred = X_test @ weights

    print("Optimal weights:")
    for name, w in zip(model_names, weights):
        print(f"  {name}: {w:.4f}")

    info = {
        "weights": {name: float(w) for name, w in zip(model_names, weights)},
    }

    return oof_pred, test_pred, info


def run_stacking(
    X_oof: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    model_names: list[str],
    meta_model: Any,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict, StackingTrainer]:
    """スタッキングアンサンブル

    Returns:
        oof_pred: OOF predictions
        test_pred: Test predictions
        info: Additional info (fold_scores, etc.)
        trainer: Trained StackingTrainer (for model saving)
    """
    trainer = StackingTrainer(
        meta_model=meta_model,
        n_splits=n_splits,
        seed=seed,
    )

    print(f"\nRunning {n_splits}-fold CV for stacking...")
    oof_pred, fold_scores = trainer.fit_predict_oof(X_oof, y)

    print("\nFitting final model on all training data...")
    trainer.fit_final(X_oof, y)
    test_pred = trainer.predict(X_test)

    info = {
        "fold_scores": fold_scores,
        "cv_mape_std": float(np.std(fold_scores)),
    }

    return oof_pred, test_pred, info, trainer


# =============================================================================
# Main
# =============================================================================


def run_ensemble(
    config: dict,
    level0_dirs: list[str],
    method: str,
    meta_model_name: str | None = None,
) -> dict:
    """Run ensemble.

    Args:
        config: Configuration dictionary
        level0_dirs: List of Level 0 model directories
        method: Ensemble method ('average', 'weighted', 'stacking')
        meta_model_name: Meta model name for stacking

    Returns:
        Results dictionary
    """
    exp_config = config["experiment"]
    stacking_config = config["stacking"]

    print("=" * 60)
    print(f"Ensemble Method: {method}")
    print("=" * 60)

    # Load Level 0 predictions
    X_oof, y_oof, X_test, train_ids, test_ids, model_names = load_level0_predictions(
        level0_dirs
    )

    # Run ensemble
    trainer = None
    if method == "average":
        oof_pred, test_pred, info = run_average(X_oof, y_oof, X_test, model_names)
        output_name = "average"

    elif method == "weighted":
        oof_pred, test_pred, info = run_weighted(X_oof, y_oof, X_test, model_names)
        output_name = "weighted"

    elif method == "stacking":
        meta_model_name = meta_model_name or stacking_config["meta_model"]
        meta_params = stacking_config["meta_params"][meta_model_name]
        meta_model = create_meta_model(meta_model_name, meta_params)

        print(f"\nMeta model: {meta_model_name}")
        print(f"  Parameters: {meta_params}")

        cv_config = stacking_config["cv"]
        oof_pred, test_pred, info, trainer = run_stacking(
            X_oof, y_oof, X_test, model_names,
            meta_model=meta_model,
            n_splits=cv_config["n_splits"],
            seed=cv_config["seed"],
        )
        info["meta_model"] = meta_model_name
        info["meta_params"] = meta_params
        output_name = f"stacking_{meta_model_name}"

    else:
        raise ValueError(f"Unknown method: {method}")

    # Calculate OOF MAPE
    oof_mape = calculate_mape(y_oof, oof_pred)

    print("\n" + "=" * 40)
    print(f"{method.upper()} Results")
    print("=" * 40)
    if method == "stacking" and "fold_scores" in info:
        for i, score in enumerate(info["fold_scores"]):
            print(f"  Fold {i + 1}: MAPE = {score:.4f}%")
        print(f"  Std: {info['cv_mape_std']:.4f}%")
    print(f"  Overall OOF MAPE: {oof_mape:.4f}%")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(__file__).parent.parent
        / "outputs"
        / f"run_{output_name}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save OOF predictions
    oof_df = pl.DataFrame({
        "id": train_ids,
        "actual": y_oof,
        "predicted": oof_pred,
    })
    oof_path = output_dir / "oof_predictions.csv"
    oof_df.write_csv(oof_path)
    print(f"\nSaved: {oof_path}")

    # Save test predictions
    test_df = pl.DataFrame({
        "id": test_ids,
        "predicted": test_pred,
    })
    test_path = output_dir / "test_predictions.csv"
    test_df.write_csv(test_path)
    print(f"Saved: {test_path}")

    # Save submission
    submission_df = pl.DataFrame({
        "id": test_ids,
        "price": test_pred.astype(int),
    })
    submission_path = output_dir / "submission.csv"
    submission_df.write_csv(submission_path, include_header=False)
    print(f"Saved: {submission_path}")

    # Copy to project submissions directory
    project_submissions_dir = project_root / "09_submissions"
    project_submissions_dir.mkdir(parents=True, exist_ok=True)
    project_submission_path = (
        project_submissions_dir
        / f"submission_{exp_config['id']}_{output_name}_{timestamp}.csv"
    )
    submission_df.write_csv(project_submission_path, include_header=False)
    print(f"Saved: {project_submission_path}")

    # Save model (stacking only)
    if trainer is not None:
        (output_dir / "models").mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "models" / "meta_model.pkl"
        trainer.save(model_path)
        print(f"Saved: {model_path}")

    # Save metrics
    metrics = {
        "method": method,
        "cv_mape": oof_mape,
        **info,
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {metrics_path}")

    # Save config for reproducibility
    config_output = {
        "method": method,
        "level0_models": model_names,
        "level0_dirs": level0_dirs,
        "timestamp": timestamp,
        **info,
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_output, f, indent=2)
    print(f"Saved: {config_path}")

    # MLflow logging
    mlflow.set_tracking_uri(str(Path(__file__).parent.parent / "mlruns"))
    mlflow.set_experiment(f"{exp_config['id']}_{exp_config['name']}")

    with mlflow.start_run(run_name=f"{output_name}_{timestamp}"):
        # Log parameters
        mlflow.log_param("method", method)
        mlflow.log_param("n_level0_models", len(model_names))
        mlflow.log_param("level0_models", ", ".join(model_names))

        if method == "stacking" and "meta_model" in info:
            mlflow.log_param("meta_model", info["meta_model"])
            mlflow.log_params({f"meta_{k}": v for k, v in info["meta_params"].items()})

        if "weights" in info:
            for name, w in info["weights"].items():
                mlflow.log_param(f"weight_{name}", f"{w:.4f}")

        # Log metrics
        mlflow.log_metric("cv_mape", oof_mape)
        if "fold_scores" in info:
            for i, score in enumerate(info["fold_scores"]):
                mlflow.log_metric(f"fold_{i + 1}_mape", score)
            mlflow.log_metric("cv_mape_std", info["cv_mape_std"])

        # Log artifacts
        mlflow.log_artifact(str(oof_path), artifact_path="predictions")
        mlflow.log_artifact(str(test_path), artifact_path="predictions")
        mlflow.log_artifact(str(submission_path), artifact_path="submission")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        mlflow.log_artifact(str(config_path), artifact_path="config")

    print("\n" + "=" * 60)
    print(f"{method.upper()} Complete!")
    print("=" * 60)
    print(f"  CV MAPE: {oof_mape:.4f}%")
    print(f"  Output: {output_dir}")

    return {
        "method": method,
        "oof_mape": oof_mape,
        "oof_predictions": oof_pred,
        "test_predictions": test_pred,
        "output_dir": str(output_dir),
        **info,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ensemble")
    parser.add_argument(
        "--level0-dirs",
        nargs="+",
        required=True,
        help="Level 0 model directories (supports glob patterns)",
    )
    parser.add_argument(
        "--method",
        choices=["average", "weighted", "stacking"],
        default="stacking",
        help="Ensemble method (default: stacking)",
    )
    parser.add_argument(
        "--meta-model",
        choices=["ridge", "lightgbm"],
        help="Meta model type for stacking (default: from config)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.method == "stacking" and args.meta_model is None:
        # Will use default from config
        pass

    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "experiment.yaml"
    config = load_config(config_path)

    # Run ensemble
    results = run_ensemble(
        config=config,
        level0_dirs=args.level0_dirs,
        method=args.method,
        meta_model_name=args.meta_model,
    )
