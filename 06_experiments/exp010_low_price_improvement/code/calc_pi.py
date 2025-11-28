"""Permutation Importance計算スクリプト（シンプル版）

使用方法:
    python calc_pi.py --run_dir outputs/run_YYYYMMDD_HHMMSS
    python calc_pi.py --run_dir outputs/run_YYYYMMDD_HHMMSS --sample_size 10000
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "04_src"))

import numpy as np
import polars as pl
import lightgbm as lgb

from evaluation.permutation_importance import calculate_permutation_importance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=10000)
    parser.add_argument("--n_repeats", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0, help="使用するfold番号")
    args = parser.parse_args()

    run_dir = Path(__file__).parent.parent / args.run_dir

    # データ読み込み
    print("Loading data...")
    X = pl.read_parquet(run_dir / "X_train.parquet")
    y = pl.read_parquet(run_dir / "y_train.parquet")["target"].to_numpy()
    features = list(X.columns)
    print(f"  X: {X.shape}, y: {len(y)}, features: {len(features)}")

    # モデル読み込み
    model_path = run_dir / "models" / f"model_fold{args.fold}.txt"
    print(f"Loading model: {model_path}")
    model = lgb.Booster(model_file=str(model_path))

    # PI計算
    print(f"\nCalculating PI (sample_size={args.sample_size}, repeats={args.n_repeats})...")
    result = calculate_permutation_importance(
        model=model,
        X=X,
        y=y,
        feature_names=features,
        scoring="mape",
        n_repeats=args.n_repeats,
        sample_size=args.sample_size,
        inverse_transform=lambda x: np.expm1(np.maximum(x, 0)),
    )

    # 保存
    output_path = run_dir / "permutation_importance.csv"
    result.write_csv(output_path)
    print(f"\nSaved: {output_path}")

    # Top 20表示
    print("\nTop 20 important features:")
    for i, row in enumerate(result.head(20).iter_rows(named=True)):
        print(f"  {i+1:2d}. {row['feature']}: {row['importance_mean']:.4f}")


if __name__ == "__main__":
    main()
