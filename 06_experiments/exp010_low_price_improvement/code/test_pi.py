"""PI計算の簡易テスト（小データ・1特徴量・1repeat）"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "04_src"))

import numpy as np
import polars as pl
import lightgbm as lgb

run_dir = Path(__file__).parent.parent / "outputs" / "run_20251128_090118"

print("1. Loading model...")
model = lgb.Booster(model_file=str(run_dir / "models" / "model_fold0.txt"))
print("   OK")

print("2. Loading parquet...")
X = pl.read_parquet(run_dir / "X_train.parquet")
y = pl.read_parquet(run_dir / "y_train.parquet")["target"].to_numpy()
print(f"   X: {X.shape}, y: {len(y)}")

print("3. Testing predict on 1000 samples...")
X_small = X.head(1000).to_numpy()
y_small = np.log1p(y[:1000])
pred = model.predict(X_small)
print(f"   pred shape: {pred.shape}")

print("4. Testing MAPE calculation...")
y_pred_orig = np.expm1(pred)
y_true_orig = np.expm1(y_small)
mape = np.mean(np.abs((y_true_orig - y_pred_orig) / y_true_orig)) * 100
print(f"   MAPE: {mape:.2f}%")

print("5. Testing permutation (1 feature, 1 repeat)...")
X_perm = X_small.copy()
X_perm[:, 0] = np.random.permutation(X_perm[:, 0])
pred_perm = model.predict(X_perm)
y_pred_perm = np.expm1(pred_perm)
mape_perm = np.mean(np.abs((y_true_orig - y_pred_perm) / y_true_orig)) * 100
print(f"   MAPE after permutation: {mape_perm:.2f}%")
print(f"   Importance (diff): {mape_perm - mape:.4f}")

print("\nAll tests passed!")
