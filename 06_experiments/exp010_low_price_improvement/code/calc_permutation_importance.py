"""
Permutation Importanceの計算スクリプト

既存のexp010モデルを使用して、Permutation Importanceを計算する。
Gain重要度が0の特徴量が本当に削除可能かを検証する。

使用方法:
    cd 06_experiments/exp010_low_price_improvement
    python code/calc_permutation_importance.py --run_dir outputs/run_YYYYMMDD_HHMMSS
"""

import sys
import argparse
from pathlib import Path
import json

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "04_src"))
# Add exp010 code directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl
import lightgbm as lgb
import yaml
from datetime import datetime
from sklearn.model_selection import KFold

from data.loader import DataLoader
from features.base import set_seed
from evaluation.permutation_importance import PermutationImportanceCalculator

# exp010 specific modules
from preprocessing import preprocess_for_training


def load_model_and_data(run_dir: Path, seed: int = 42, n_splits: int = 3,
                        target_features: list = None, bottom_pct: float = None):
    """モデルとデータを読み込む

    Args:
        target_features: PI計算対象の特徴量リスト（指定時はこれのみ計算）
        bottom_pct: 重要度下位N%の特徴量のみ計算（0-100）
    """
    # モデル読み込み (LightGBM Boosterテキストファイル)
    models = []
    for model_path in sorted(run_dir.glob("models/model_fold*.txt")):
        booster = lgb.Booster(model_file=str(model_path))
        models.append(booster)
    print(f"Loaded {len(models)} models")

    # 特徴量名取得（feature_importance.csvから）
    fi_path = run_dir / "feature_importance.csv"
    fi_df = pl.read_csv(fi_path)
    feature_names = fi_df["feature"].to_list()

    # PI計算対象の特徴量を絞り込み
    if target_features is not None:
        pi_target_features = [f for f in target_features if f in feature_names]
        print(f"Target features for PI: {len(pi_target_features)} / {len(feature_names)}")
    elif bottom_pct is not None:
        # 重要度下位N%を取得
        n_bottom = int(len(fi_df) * bottom_pct / 100)
        bottom_df = fi_df.sort("importance").head(n_bottom)
        pi_target_features = bottom_df["feature"].to_list()
        print(f"Bottom {bottom_pct}% features for PI: {len(pi_target_features)} / {len(feature_names)}")
    else:
        pi_target_features = feature_names
        print(f"All features for PI: {len(pi_target_features)}")

    # データ読み込み（parquetがあれば使用、なければ前処理実行）
    x_parquet = run_dir / "X_train.parquet"
    y_parquet = run_dir / "y_train.parquet"

    if x_parquet.exists() and y_parquet.exists():
        print("Loading preprocessed data from parquet...")
        X_train = pl.read_parquet(x_parquet)
        y_train = pl.read_parquet(y_parquet)["target"].to_numpy()
        print(f"  Loaded: X_train {X_train.shape}, y_train {len(y_train)}")
    else:
        print("Parquet not found. Running preprocessing...")
        data_config_path = project_root / "03_configs" / "data.yaml"
        with open(data_config_path, "r", encoding="utf-8") as f:
            data_config = yaml.safe_load(f)

        data_config["data"]["train_path"] = str(project_root / data_config["data"]["train_path"])
        data_config["data"]["test_path"] = str(project_root / data_config["data"]["test_path"])
        data_config["data"]["sample_submit_path"] = str(project_root / data_config["data"]["sample_submit_path"])

        loader = DataLoader(config=data_config, add_address_columns=False)
        train = loader.load_train()
        test = loader.load_test()

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        cv_splits = list(cv.split(train))

        X_train, X_test, y_train_pl, pipeline = preprocess_for_training(train, test, cv_splits=cv_splits)
        y_train = y_train_pl.to_numpy()

    # 特徴量名順序を feature_importance.csv に合わせる
    X = X_train.select(feature_names)

    # 目的変数（log1p変換後）
    y = np.log1p(y_train)

    return models, X, y, feature_names, fi_df, pi_target_features


def main():
    parser = argparse.ArgumentParser(description="Calculate Permutation Importance")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to run output directory")
    parser.add_argument("--n_repeats", type=int, default=5,
                        help="Number of permutation repeats")
    parser.add_argument("--scoring", type=str, default="mape",
                        choices=["mape", "mae", "mse", "rmse", "r2"],
                        help="Scoring function")
    parser.add_argument("--bottom_pct", type=float, default=None,
                        help="Only calculate PI for bottom N%% features by Gain importance")
    args = parser.parse_args()

    exp_dir = Path(__file__).parent.parent
    run_dir = exp_dir / args.run_dir if not Path(args.run_dir).is_absolute() else Path(args.run_dir)

    if not run_dir.exists():
        print(f"Error: run_dir not found: {run_dir}")
        return

    print("=" * 70)
    print("Permutation Importance Calculation")
    print("=" * 70)
    print(f"Run directory: {run_dir}")
    print(f"Scoring: {args.scoring}")
    print(f"N repeats: {args.n_repeats}")

    # データ読み込み
    print("\nLoading models and data...")
    models, X, y, feature_names, gain_fi_df, pi_target_features = load_model_and_data(
        run_dir, bottom_pct=args.bottom_pct
    )

    print(f"Features: {len(feature_names)}")
    print(f"PI target features: {len(pi_target_features)}")
    print(f"Samples: {len(y)}")

    # Permutation Importance計算
    print("\nCalculating Permutation Importance...")
    print("(This may take a while...)")

    # カスタムMAPEスコアリング関数（元のスケールで計算）
    # モデルはlog1pスケールで予測するため、MAPEは元のスケール(expm1)で計算する
    def mape_original_scale(y_true_log, y_pred_log):
        """元のスケール（expm1後）でMAPEを計算（大きいほど良い=負値で返す）"""
        y_true = np.expm1(y_true_log)
        y_pred = np.expm1(y_pred_log)
        mask = y_true != 0
        if mask.sum() == 0:
            return 0.0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return -mape  # 大きいほど良い形式

    # スコアリング関数を選択
    if args.scoring == "mape":
        scoring_fn = mape_original_scale
        print("Using MAPE on original scale (expm1)")
    else:
        scoring_fn = args.scoring

    calculator = PermutationImportanceCalculator(
        scoring=scoring_fn,
        n_repeats=args.n_repeats,
        random_state=42,
    )

    # 最初のfoldモデルで計算（全foldだと時間がかかりすぎる）
    # LightGBM Booster用のpredict関数
    def predict_fn(X_np):
        return models[0].predict(X_np)

    # PI計算（target_featuresで対象を限定可能）
    result = calculator.calculate(
        predict_fn=predict_fn,
        X=X,
        y=y,
        feature_names=feature_names,
        target_features=pi_target_features if pi_target_features != feature_names else None,
    )

    # 結果をDataFrameに
    perm_df = result.to_dataframe()

    # Gain重要度と結合
    gain_fi_df = gain_fi_df.rename({"importance": "gain_importance"})
    combined_df = perm_df.join(
        gain_fi_df.select(["feature", "gain_importance"]),
        on="feature",
        how="left"
    )

    # 保存
    output_path = run_dir / "permutation_importance.csv"
    combined_df.write_csv(output_path)
    print(f"\nSaved: {output_path}")

    # 結果表示
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    print(f"\nBaseline score ({args.scoring}): {result.baseline_score:.4f}")

    print("\n--- Top 20 features (Permutation Importance) ---")
    top20 = combined_df.head(20)
    for row in top20.iter_rows(named=True):
        print(f"  {row['feature']:40} perm={row['importance']:8.4f} gain={row['gain_importance']:8.4f}")

    # Gain重要度=0の特徴量のPermutation Importance
    print("\n--- Gain importance = 0 features ---")
    zero_gain = combined_df.filter(pl.col("gain_importance") == 0)
    print(f"Total: {len(zero_gain)} features")

    # Permutation Importanceも低い（削除安全）
    safe_threshold = 0.001  # 0.1% threshold
    safe_to_remove = zero_gain.filter(pl.col("importance") <= safe_threshold)
    print(f"Safe to remove (perm <= {safe_threshold}): {len(safe_to_remove)} features")

    # Permutation Importanceが高い（削除危険）
    risky = zero_gain.filter(pl.col("importance") > safe_threshold)
    if len(risky) > 0:
        print(f"\n⚠️ Risky to remove (perm > {safe_threshold}): {len(risky)} features")
        for row in risky.sort("importance", descending=True).head(10).iter_rows(named=True):
            print(f"  {row['feature']:40} perm={row['importance']:8.4f}")

    # 削除推奨特徴量リスト
    safe_features = result.get_safe_to_remove_features(threshold=safe_threshold, consider_std=True)
    print(f"\n--- Safe to remove features (considering std) ---")
    print(f"Total: {len(safe_features)} features")

    # JSONで保存
    summary = {
        "run_dir": str(run_dir),
        "scoring": args.scoring,
        "n_repeats": args.n_repeats,
        "baseline_score": float(result.baseline_score),
        "n_features": len(feature_names),
        "n_gain_zero": len(zero_gain),
        "n_safe_to_remove": len(safe_features),
        "safe_to_remove_features": safe_features,
        "risky_features": risky["feature"].to_list() if len(risky) > 0 else [],
    }

    summary_path = run_dir / "permutation_importance_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {summary_path}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
