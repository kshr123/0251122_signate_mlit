"""
exp007 クイックテスト - 少量データ・高速パラメータで動作確認
"""

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(1, str(project_root / "04_src"))

import numpy as np
import polars as pl
import yaml
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from data.loader import DataLoader
from features.base import set_seed
from evaluation.metrics import calculate_mape

from preprocessing import (
    preprocess_for_training,
    CATEGORICAL_FEATURES,
    TARGET_ENCODING_COLUMNS,
    COUNT_ENCODING_COLUMNS,
    NUMERIC_FEATURES,
    AGE_NUMERIC_FEATURES,
    ACCESS_NUMERIC_FEATURES,
    TFIDF_MAX_FEATURES,
    BUILDING_TAG_SVD_DIM,
    UNIT_TAG_SVD_DIM,
)


def test_quick():
    """少量データでの動作確認"""

    seed = 42
    set_seed(seed)

    # ===== データ読み込み =====
    print("Loading data...")
    data_config_path = project_root / "03_configs" / "data.yaml"
    with open(data_config_path, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    data_config["data"]["train_path"] = str(project_root / data_config["data"]["train_path"])
    data_config["data"]["test_path"] = str(project_root / data_config["data"]["test_path"])
    data_config["data"]["sample_submit_path"] = str(project_root / data_config["data"]["sample_submit_path"])

    loader = DataLoader(config=data_config, add_address_columns=False)
    train_full = loader.load_train()
    test_full = loader.load_test()

    # ===== データサンプリング（10,000件） =====
    n_sample = 10000
    print(f"\nSampling {n_sample} rows for quick test...")
    train = train_full.sample(n=n_sample, seed=seed)
    test = test_full.sample(n=min(2000, len(test_full)), seed=seed)

    print(f"  - Train: {train.shape}")
    print(f"  - Test: {test.shape}")

    # ===== CV splits =====
    n_splits = 2  # 高速化のため2-fold
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_splits = list(cv.split(train))

    # ===== 前処理 =====
    print("\nPreprocessing...")
    X_train, X_test, y_train = preprocess_for_training(train, test, cv_splits=cv_splits)

    ALL_FEATURES = list(X_train.columns)
    print(f"\nFeature count: {len(ALL_FEATURES)}")
    print(f"  - Numeric: {len(NUMERIC_FEATURES)}")
    print(f"  - Age features: {len(AGE_NUMERIC_FEATURES)}")
    print(f"  - Access features: {len(ACCESS_NUMERIC_FEATURES)}")
    print(f"  - TE: {len(TARGET_ENCODING_COLUMNS)}")
    print(f"  - CE: {len(COUNT_ENCODING_COLUMNS)}")
    print(f"  - TF-IDF: {TFIDF_MAX_FEATURES}")
    print(f"  - Geo PCA: 2")
    print(f"  - Building tag SVD: {BUILDING_TAG_SVD_DIM}")
    print(f"  - Unit tag SVD: {UNIT_TAG_SVD_DIM}")

    # ===== Target transform =====
    y_train_np = y_train.to_numpy()
    y_train_transformed = np.log1p(y_train_np)

    # ===== 高速パラメータ =====
    lgb_params = {
        "objective": "regression",
        "metric": "mape",
        "boosting_type": "gbdt",
        "learning_rate": 0.1,  # 高速化
        "n_estimators": 100,   # 少なめ
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 20,
        "random_state": seed,
        "verbose": -1,
        "force_col_wise": True,
    }

    print(f"\nQuick LGB params: lr={lgb_params['learning_rate']}, n_est={lgb_params['n_estimators']}")

    # ===== 学習 =====
    print(f"\n{n_splits}-Fold CV...")

    cv_scores = []
    X_train_pd = X_train.to_pandas()
    X_test_pd = X_test.to_pandas()

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")

        X_tr = X_train_pd.iloc[train_idx]
        X_val = X_train_pd.iloc[val_idx]
        y_tr = y_train_transformed[train_idx]
        y_val_transformed = y_train_transformed[val_idx]
        y_val_original = y_train_np[val_idx]

        model = LGBMRegressor(**lgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val_transformed)],
            callbacks=[
                early_stopping(stopping_rounds=10, verbose=True),
                log_evaluation(period=20),
            ],
        )

        val_pred_transformed = model.predict(X_val)
        val_pred = np.expm1(val_pred_transformed)
        val_pred = np.maximum(val_pred, 0)

        mape_score = calculate_mape(y_val_original, val_pred)
        cv_scores.append(mape_score)
        print(f"  MAPE: {mape_score:.4f}%")

    # ===== 結果 =====
    print("\n" + "=" * 50)
    print("Quick Test Results")
    print("=" * 50)
    print(f"  Mean MAPE: {np.mean(cv_scores):.4f}%")
    print(f"  Std:       {np.std(cv_scores):.4f}%")
    print("=" * 50)
    print("\n✓ Quick test completed successfully!")


if __name__ == "__main__":
    test_quick()
