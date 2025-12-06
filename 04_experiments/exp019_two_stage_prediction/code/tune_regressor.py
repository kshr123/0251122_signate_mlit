#!/usr/bin/env python3
"""
exp019 回帰モデルハイパーパラメータチューニング

Optunaを使用してLightGBM Regressorのハイパーパラメータを最適化。
CV MAPEを最小化する。

使用例:
    # 基本実行
    caffeinate -i -s env PYTHONPATH=../../02_src:code ../../.venv/bin/python code/tune_regressor.py \
      --config 003_regressor_lowprice.yaml

    # 共通CV使用
    python code/tune_regressor.py --config 003_regressor_lowprice.yaml \
      --cv-file outputs/common_cv_folds.npy

    # テストモード（高速化）
    python code/tune_regressor.py --config 003_regressor_lowprice.yaml --test
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

# パス設定
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "02_src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import OUTPUT_DIR, EXP_DIR, TARGET_COLUMN, LOW_PRICE_QUANTILE, YEAR_PRICE_THRESHOLDS
from data_manager import DataManager

# Optuna
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


def parse_args() -> argparse.Namespace:
    """コマンドライン引数パース"""
    parser = argparse.ArgumentParser(description="Regressor Hyperparameter Tuning")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config file name (e.g., 003_regressor_lowprice.yaml)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of trials (default: from config)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds (default: from config)",
    )
    parser.add_argument(
        "--cv-file",
        type=str,
        default=None,
        help="Common CV file (e.g., outputs/common_cv_folds.npy)",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default=None,
        help="Directory with pre-computed features",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: use sampled data and fast params",
    )
    return parser.parse_args()


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE計算（元スケール）"""
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def create_log_callback(log_path: Path, trial_number: int, log_interval: int = 100):
    """LightGBM学習進捗ログ用コールバック作成"""
    def callback(env):
        if env.iteration % log_interval == 0:
            val_loss = env.evaluation_result_list[0][2] if env.evaluation_result_list else None
            with open(log_path, "a") as f:
                f.write(f"  [iter {env.iteration:5d}] val_l2={val_loss:.6f}\n")
    return callback


def create_objective(
    X: np.ndarray,
    y: np.ndarray,
    y_original: np.ndarray,
    param_space: dict,
    base_params: dict,
    cv_splits: list[tuple],
    feature_names: list[str],
    progress_log_path: Path,
    target_transform: str = "log1p",
    seed: int = 42,
) -> callable:
    """Optuna目的関数を作成

    Args:
        X: 特徴量（NumPy配列）
        y: 変換済みターゲット（log1pなど）
        y_original: 元スケールターゲット（MAPE計算用）
        param_space: ハイパーパラメータ探索空間
        base_params: ベースLightGBMパラメータ
        cv_splits: CV分割
        feature_names: 特徴量名リスト
        progress_log_path: 進捗ログファイルパス
        target_transform: ターゲット変換方法
        seed: 乱数シード
    """

    def objective(trial: optuna.Trial) -> float:
        start_time = datetime.now()

        # パラメータサンプリング
        params = {}
        for name, space in param_space.items():
            if space["type"] == "float":
                params[name] = trial.suggest_float(
                    name, space["low"], space["high"], log=space.get("log", False)
                )
            elif space["type"] == "int":
                params[name] = trial.suggest_int(
                    name, space["low"], space["high"], log=space.get("log", False)
                )
            elif space["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, space["choices"])

        # 開始ログ
        with open(progress_log_path, "a") as f:
            f.write(f"[{start_time:%H:%M:%S}] Trial {trial.number} started: "
                    f"lr={params.get('learning_rate', 'N/A'):.4f}\n")

        # LightGBMパラメータ構築
        lgb_params = base_params.copy()
        lgb_params.update(params)

        # 学習進捗ログ用コールバック
        log_callback = create_log_callback(progress_log_path, trial.number, log_interval=500)

        # CV学習
        fold_mape = []
        fold_rmse = []
        best_iterations = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            y_val_orig = y_original[val_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
            dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=dtrain)

            # 学習パラメータ
            train_params = lgb_params.copy()
            n_estimators = train_params.pop("n_estimators", 5000)
            early_stopping_rounds = train_params.pop("early_stopping_rounds", 100)

            # 学習
            model = lgb.train(
                train_params,
                dtrain,
                num_boost_round=n_estimators,
                valid_sets=[dval],
                valid_names=["valid"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                    log_callback,
                ],
            )

            # 予測・逆変換
            y_pred_transformed = model.predict(X_val, num_iteration=model.best_iteration)
            if target_transform == "log1p":
                y_pred = np.expm1(y_pred_transformed)
            else:
                y_pred = y_pred_transformed

            # 評価
            mape = calculate_mape(y_val_orig, y_pred)
            rmse = float(np.sqrt(np.mean((y_val_orig - y_pred) ** 2)))
            fold_mape.append(mape)
            fold_rmse.append(rmse)
            best_iterations.append(model.best_iteration)

        cv_mape = np.mean(fold_mape)
        cv_rmse = np.mean(fold_rmse)
        elapsed = (datetime.now() - start_time).total_seconds()

        # 完了ログ
        with open(progress_log_path, "a") as f:
            f.write(f"[{datetime.now():%H:%M:%S}] Trial {trial.number} done: "
                    f"MAPE={cv_mape:.2f}%, RMSE={cv_rmse/1e6:.2f}M, "
                    f"time={elapsed:.1f}s, best_iter={best_iterations}\n")

        # 属性として記録
        trial.set_user_attr("cv_mape", cv_mape)
        trial.set_user_attr("cv_rmse", cv_rmse)

        return cv_mape  # MAPEを最小化

    return objective


def main():
    """メイン処理"""
    args = parse_args()
    test_mode = args.test

    # DataManager初期化
    dm = DataManager(config_name=args.config)
    config = dm.config
    exp_id = config["experiment"]["id"]
    config_name = args.config.replace(".yaml", "")

    # チューニング設定（configから）
    tuning_config = config.get("tuning", {})
    n_trials = args.n_trials or tuning_config.get("n_trials", 100)
    timeout = args.timeout or tuning_config.get("timeout")
    param_space = tuning_config.get("param_space")

    if param_space is None:
        print("ERROR: tuning.param_space not found in config")
        sys.exit(1)

    # テストモード用の軽量パラメータ空間
    if test_mode:
        param_space = {
            "learning_rate": {"type": "float", "low": 0.05, "high": 0.2},
            "num_leaves": {"type": "int", "low": 15, "high": 63},
        }
        n_trials = 5
        timeout = 300
        print("[TEST MODE] Using sampled data and simplified param_space")

    # 出力ディレクトリ
    if test_mode:
        tuning_dir = EXP_DIR / "outputs_test" / config_name / "tuning"
    else:
        tuning_dir = OUTPUT_DIR / config_name / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)

    storage_path = tuning_dir / "optuna.db"
    progress_log_path = tuning_dir / "progress.log"

    print("=" * 60)
    print("exp019 Regressor Hyperparameter Tuning")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Mode: {'TEST' if test_mode else 'FULL'}")
    print(f"Output: {tuning_dir}")
    print()

    # データ準備
    print("Loading raw data...")
    train_raw, test_raw = dm.load_raw_data(test_mode=test_mode)

    # ターゲット取得
    y_original = train_raw[TARGET_COLUMN].to_numpy()

    # CV分割読み込み
    if args.cv_file:
        cv_path = Path(args.cv_file)
        if not cv_path.is_absolute():
            cv_path = EXP_DIR / args.cv_file
        print(f"Loading CV from: {cv_path}")
        fold_data = np.load(cv_path, allow_pickle=True)
        cv_splits_raw = [(d["train_idx"], d["val_idx"]) for d in fold_data]
    else:
        # チューニング用CV（高速化のため少ないsplit）
        n_cv_splits = tuning_config.get("n_cv_splits", 3)
        cv_splits_raw = dm.get_cv_splits(len(train_raw), n_splits=n_cv_splits)
        print(f"Using {n_cv_splits}-fold CV for tuning")

    # 低価格フィルタリング（003_regressor_lowprice用）
    training_config = config.get("training", {})
    data_filter = training_config.get("data_filter", {})
    use_true_label = data_filter.get("use_true_label", False)

    if use_true_label:
        print("Filtering to low-price samples only...")
        # 年度別閾値で低価格判定（DataManagerのcreate_binary_targetを使用）
        is_low_price, threshold_info = dm.create_binary_target(
            y_original, train_df=train_raw, use_yearly_threshold=True
        )
        is_low_price = is_low_price.astype(bool)

        low_price_indices = np.where(is_low_price)[0]
        print(f"  Low-price samples: {len(low_price_indices):,} / {len(train_raw):,} "
              f"({len(low_price_indices)/len(train_raw):.1%})")

        # CV分割をフィルタリング
        cv_splits = []
        for train_idx, val_idx in cv_splits_raw:
            # train_idxとval_idxを低価格のみにフィルタ
            train_idx_filtered = np.intersect1d(train_idx, low_price_indices)
            val_idx_filtered = np.intersect1d(val_idx, low_price_indices)
            cv_splits.append((train_idx_filtered, val_idx_filtered))

        # インデックスマッピング作成（元のインデックス → フィルタ後の連続インデックス）
        # チューニング時はフィルタ済みデータを使うため、インデックスを再マッピング
        filtered_train_raw = train_raw.filter(pl.lit(is_low_price))
        y_original_filtered = y_original[is_low_price]

        # 新しいCV分割（0からの連番に変換）
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(low_price_indices)}
        cv_splits_remapped = []
        for train_idx, val_idx in cv_splits:
            train_new = np.array([old_to_new[i] for i in train_idx])
            val_new = np.array([old_to_new[i] for i in val_idx])
            cv_splits_remapped.append((train_new, val_new))
        cv_splits = cv_splits_remapped

        train_raw = filtered_train_raw
        y_original = y_original_filtered
    else:
        cv_splits = cv_splits_raw

    # 前処理
    print("Preprocessing...")
    X_train, _, _ = dm.preprocess(train_raw, test_raw, cv_splits=cv_splits, verbose=False)
    feature_names = dm.feature_names

    print(f"Features shape: {X_train.shape}")

    # ターゲット変換
    target_config = training_config.get("target", {})
    target_transform = target_config.get("transform", "log1p")
    if target_transform == "log1p":
        y_transformed = np.log1p(y_original)
    else:
        y_transformed = y_original

    print(f"\nCV splits: {len(cv_splits)} folds")
    for i, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"  Fold {i}: train={len(train_idx):,}, val={len(val_idx):,}")

    # NumPy配列に変換
    X_np = X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train

    # ベースパラメータ取得
    base_params = dm.get_model_params(model_type="lgbm", test_mode=test_mode)
    # チューニング対象外のパラメータだけ残す
    for key in list(param_space.keys()):
        base_params.pop(key, None)

    # 目的関数作成
    objective = create_objective(
        X_np, y_transformed, y_original, param_space, base_params, cv_splits, feature_names,
        progress_log_path, target_transform=target_transform, seed=dm.seed
    )

    # Study作成（SQLite保存、再開可能）
    print(f"\nStarting tuning: n_trials={n_trials}, timeout={timeout}s")
    print(f"Storage: {storage_path}")

    study = optuna.create_study(
        study_name=f"{exp_id}_{config_name}",
        storage=f"sqlite:///{storage_path}",
        direction="minimize",  # MAPE最小化
        load_if_exists=True,
        sampler=TPESampler(seed=dm.seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    # 既存の試行数を表示
    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"Resuming from {n_existing} existing trials")
        print(f"Current best MAPE: {study.best_value:.2f}%")

    # 最適化実行
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    # 結果表示
    print("\n" + "=" * 60)
    print("TUNING COMPLETED")
    print("=" * 60)
    print(f"Total trials: {len(study.trials)}")
    print(f"Best MAPE: {study.best_value:.2f}%")

    # 最良トライアルのRMSEも表示
    best_rmse = study.best_trial.user_attrs.get("cv_rmse", "N/A")
    if isinstance(best_rmse, float):
        print(f"Best RMSE: {best_rmse/1e6:.2f}M yen")

    print("\nBest Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 結果保存
    import yaml
    import pandas as pd

    # best_params.yaml
    best_params_path = tuning_dir / "best_params.yaml"
    with open(best_params_path, "w") as f:
        f.write(f"# Best parameters from tuning\n")
        f.write(f"# Trial: {study.best_trial.number}, MAPE: {study.best_value:.2f}%\n\n")
        yaml.dump(study.best_params, f, default_flow_style=False)

    # tuning_history.csv
    history_path = tuning_dir / "tuning_history.csv"
    history = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            record = {
                "trial": trial.number,
                "mape": trial.value,
                "rmse": trial.user_attrs.get("cv_rmse"),
                **trial.params
            }
            history.append(record)
    if history:
        pd.DataFrame(history).to_csv(history_path, index=False)

    print(f"\nResults saved to {tuning_dir}")
    print(f"  - best_params.yaml")
    print(f"  - tuning_history.csv")
    print(f"  - optuna.db (for resume)")
    print(f"  - progress.log")


if __name__ == "__main__":
    main()
