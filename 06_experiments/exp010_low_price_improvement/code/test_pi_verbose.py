"""PI計算テスト（進捗表示付き）"""
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "04_src"))

import numpy as np
import polars as pl
import lightgbm as lgb

run_dir = Path(__file__).parent.parent / "outputs" / "run_20251128_090118"

print("=" * 50)
print("PI Test (verbose)")
print("=" * 50)

t0 = time.time()
print(f"[{time.time()-t0:.1f}s] Loading model...")
model = lgb.Booster(model_file=str(run_dir / "models" / "model_fold0.txt"))

print(f"[{time.time()-t0:.1f}s] Loading parquet...")
X = pl.read_parquet(run_dir / "X_train.parquet").to_numpy()
y = np.log1p(pl.read_parquet(run_dir / "y_train.parquet")["target"].to_numpy())
print(f"  Shape: X={X.shape}, y={y.shape}")

print(f"[{time.time()-t0:.1f}s] Baseline prediction...")
pred_base = model.predict(X)
y_pred_base = np.expm1(pred_base)
y_true = np.expm1(y)
mape_base = np.mean(np.abs((y_true - y_pred_base) / y_true)) * 100
print(f"  Baseline MAPE: {mape_base:.4f}%")

# 5特徴量だけテスト
n_features = 5
print(f"\n[{time.time()-t0:.1f}s] Testing {n_features} features, 1 repeat each...")

for i in range(n_features):
    t1 = time.time()
    X_perm = X.copy()
    X_perm[:, i] = np.random.permutation(X_perm[:, i])
    pred_perm = model.predict(X_perm)
    y_pred_perm = np.expm1(pred_perm)
    mape_perm = np.mean(np.abs((y_true - y_pred_perm) / y_true)) * 100
    importance = mape_perm - mape_base
    elapsed = time.time() - t1
    print(f"  Feature {i}: importance={importance:.4f}, time={elapsed:.2f}s")

print(f"\n[{time.time()-t0:.1f}s] Done!")
print(f"Estimated time for 262 features x 3 repeats: {elapsed * 262 * 3 / 60:.1f} min")
