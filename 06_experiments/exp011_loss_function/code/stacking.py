"""
exp011_loss_function Phase 3: Stacking

各損失関数モデルのOOF予測をメタ特徴量として、
Ridge回帰でスタッキングを行う。

使用方法:
    cd 06_experiments/exp011_loss_function
    source ../../.venv/bin/activate
    PYTHONPATH=../../04_src:code python code/stacking.py

    # テストモード
    PYTHONPATH=../../04_src:code python code/stacking.py --test
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "04_src"))

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from datetime import datetime
import argparse
import json

from evaluation.metrics import calculate_mape


def load_oof_predictions(outputs_dir: Path) -> dict:
    """Load OOF predictions from all run directories

    Returns:
        Dict[str, pl.DataFrame]: {model_name: oof_df}
    """
    predictions = {}

    for run_dir in sorted(outputs_dir.glob("run_*")):
        oof_path = run_dir / "oof_predictions.csv"
        if oof_path.exists():
            # Extract model name: run_huber_20251128_xxx -> huber
            name = run_dir.name.replace("run_", "").split("_2025")[0]
            df = pl.read_csv(oof_path)
            predictions[name] = df
            print(f"  Loaded: {name} ({len(df)} samples)")

    return predictions


def load_test_predictions(outputs_dir: Path) -> dict:
    """Load test predictions from all run directories

    Returns:
        Dict[str, pl.DataFrame]: {model_name: test_df}
    """
    predictions = {}

    for run_dir in sorted(outputs_dir.glob("run_*")):
        test_path = run_dir / "test_predictions.csv"
        if test_path.exists():
            name = run_dir.name.replace("run_", "").split("_2025")[0]
            df = pl.read_csv(test_path)
            predictions[name] = df
            print(f"  Loaded: {name} ({len(df)} samples)")

    return predictions


def run_stacking(
    outputs_dir: Path,
    meta_model: str = "ridge",
    alpha: float = 1.0,
    n_splits: int = 3,
    seed: int = 42,
    test_mode: bool = False,
):
    """Run stacking with Ridge meta-model

    Args:
        outputs_dir: Path to outputs directory containing run_* folders
        meta_model: Meta model type ("ridge")
        alpha: Ridge regularization parameter
        n_splits: Number of CV folds for meta-model
        seed: Random seed
        test_mode: Quick test mode
    """
    print("=" * 60)
    print("Phase 3: Stacking")
    print("=" * 60)

    # Load OOF predictions
    print("\nLoading OOF predictions...")
    oof_preds = load_oof_predictions(outputs_dir)

    if len(oof_preds) < 2:
        print(f"Error: Need at least 2 models for stacking, found {len(oof_preds)}")
        return

    model_names = list(oof_preds.keys())
    print(f"\nModels for stacking: {model_names}")

    # Build meta-features from OOF predictions
    base_df = oof_preds[model_names[0]]
    y_true = base_df["actual"].to_numpy()
    n_samples = len(y_true)

    # Create meta-feature matrix
    X_meta = np.zeros((n_samples, len(model_names)))
    for i, name in enumerate(model_names):
        X_meta[:, i] = oof_preds[name]["predicted"].to_numpy()

    print(f"\nMeta-features shape: {X_meta.shape}")
    print(f"Target shape: {y_true.shape}")

    # Check correlation between predictions
    print("\nPrediction correlations:")
    for i, name_i in enumerate(model_names):
        for j, name_j in enumerate(model_names):
            if i < j:
                corr = np.corrcoef(X_meta[:, i], X_meta[:, j])[0, 1]
                print(f"  {name_i} vs {name_j}: {corr:.4f}")

    # Calculate individual model MAPEs
    print("\nIndividual model MAPEs:")
    for i, name in enumerate(model_names):
        mape = calculate_mape(y_true, X_meta[:, i])
        print(f"  {name}: {mape:.2f}%")

    # Simple average baseline
    simple_avg = X_meta.mean(axis=1)
    simple_avg_mape = calculate_mape(y_true, simple_avg)
    print(f"\nSimple average MAPE: {simple_avg_mape:.2f}%")

    # Stacking with Ridge
    print(f"\n--- Stacking with {meta_model.upper()} (alpha={alpha}) ---")

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_stacking = np.zeros(n_samples)
    cv_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_meta)):
        X_tr, X_val = X_meta[train_idx], X_meta[val_idx]
        y_tr, y_val = y_true[train_idx], y_true[val_idx]

        # Train meta-model
        if meta_model == "ridge":
            model = Ridge(alpha=alpha, random_state=seed)
        else:
            raise ValueError(f"Unknown meta_model: {meta_model}")

        model.fit(X_tr, y_tr)

        # Predict
        val_pred = model.predict(X_val)
        val_pred = np.maximum(val_pred, 0)  # Ensure non-negative
        oof_stacking[val_idx] = val_pred

        # Calculate MAPE
        fold_mape = calculate_mape(y_val, val_pred)
        cv_scores.append(fold_mape)
        print(f"  Fold {fold_idx + 1}: MAPE = {fold_mape:.2f}%")

    # Final stacking MAPE
    stacking_mape = calculate_mape(y_true, oof_stacking)
    print(f"\nStacking CV MAPE: {np.mean(cv_scores):.2f}% (±{np.std(cv_scores):.2f}%)")
    print(f"Stacking OOF MAPE: {stacking_mape:.2f}%")

    # Train final model on all data
    print("\nTraining final meta-model on all data...")
    final_model = Ridge(alpha=alpha, random_state=seed)
    final_model.fit(X_meta, y_true)

    # Show learned weights
    print(f"\nLearned weights:")
    for i, name in enumerate(model_names):
        print(f"  {name}: {final_model.coef_[i]:.4f}")
    print(f"  Intercept: {final_model.intercept_:.4f}")

    # Load and predict test data
    print("\nLoading test predictions...")
    test_preds = load_test_predictions(outputs_dir)

    # Build test meta-features
    base_test = test_preds[model_names[0]]
    test_ids = base_test["id"].to_numpy()
    n_test = len(test_ids)

    X_test_meta = np.zeros((n_test, len(model_names)))
    for i, name in enumerate(model_names):
        X_test_meta[:, i] = test_preds[name]["predicted"].to_numpy()

    # Predict
    test_pred = final_model.predict(X_test_meta)
    test_pred = np.maximum(test_pred, 0)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = outputs_dir / f"run_stacking_{meta_model}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save OOF predictions
    oof_df = pl.DataFrame({
        "id": np.arange(n_samples),
        "actual": y_true,
        "predicted": oof_stacking,
    })
    oof_df.write_csv(output_dir / "oof_predictions.csv")

    # Save test predictions
    test_df = pl.DataFrame({
        "id": test_ids,
        "predicted": test_pred,
    })
    test_df.write_csv(output_dir / "test_predictions.csv")

    # Save submission
    test_ids_str = [f"{int(id_):06d}" for id_ in test_ids]
    submission_df = pl.DataFrame({
        "id": test_ids_str,
        "price": test_pred.astype(int),
    })
    submission_df.write_csv(output_dir / "submission.csv", include_header=False)

    # Save model info
    model_info = {
        "meta_model": meta_model,
        "alpha": alpha,
        "model_names": model_names,
        "weights": final_model.coef_.tolist(),
        "intercept": float(final_model.intercept_),
        "cv_mape_mean": float(np.mean(cv_scores)),
        "cv_mape_std": float(np.std(cv_scores)),
        "stacking_oof_mape": float(stacking_mape),
        "simple_avg_mape": float(simple_avg_mape),
    }
    with open(output_dir / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    # Also save to project submissions
    project_submissions_dir = project_root / "09_submissions"
    project_submissions_dir.mkdir(parents=True, exist_ok=True)
    submission_df.write_csv(
        project_submissions_dir / f"submission_exp011_stacking_{timestamp}.csv",
        include_header=False
    )

    print(f"\nResults saved to: {output_dir}")

    # Summary
    print("\n" + "=" * 60)
    print("Stacking Results Summary")
    print("=" * 60)
    print(f"  Best single model (huber): 12.17%")
    print(f"  Simple average:            {simple_avg_mape:.2f}%")
    print(f"  Stacking ({meta_model}):          {stacking_mape:.2f}%")

    improvement = 12.17 - stacking_mape
    if improvement > 0:
        print(f"  Improvement vs best:       +{improvement:.2f}pt")
    else:
        print(f"  Degradation vs best:       {improvement:.2f}pt")
    print("=" * 60)

    return {
        "stacking_mape": stacking_mape,
        "simple_avg_mape": simple_avg_mape,
        "cv_scores": cv_scores,
        "weights": final_model.coef_.tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="exp011 Phase 3: Stacking")
    parser.add_argument("--meta-model", type=str, default="ridge",
                        choices=["ridge"],
                        help="Meta model type (default: ridge)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regularization parameter (default: 1.0)")
    parser.add_argument("--n-splits", type=int, default=3,
                        help="Number of CV folds (default: 3)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode")

    args = parser.parse_args()

    outputs_dir = Path(__file__).parent.parent / "outputs"

    run_stacking(
        outputs_dir=outputs_dir,
        meta_model=args.meta_model,
        alpha=args.alpha,
        n_splits=args.n_splits,
        test_mode=args.test,
    )
