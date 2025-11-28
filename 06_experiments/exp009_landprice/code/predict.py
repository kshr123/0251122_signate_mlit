"""
exp009_landprice 推論スクリプト

学習済みモデルを読み込み、提出ファイルを生成する。
Usage:
    python predict.py                    # 最新のモデルを使用
    python predict.py --timestamp YYYYMMDD_HHMMSS  # 特定のタイムスタンプを指定
"""

import sys
from pathlib import Path
import argparse

# カレントディレクトリを先頭に追加（exp009のpreprocessing.py用）
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(1, str(project_root / "04_src"))

import json
import yaml
import lightgbm as lgb
import polars as pl
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold

# 共通コンポーネント（04_src/）
from data.loader import DataLoader
from features.base import set_seed

# exp009の前処理をインポート
from preprocessing import preprocess_for_training


def find_latest_models(models_dir: Path) -> tuple[list[Path], str]:
    """最新のモデルファイルをタイムスタンプで検索する。"""
    model_files = list(models_dir.glob("model_fold*_*.txt"))
    if not model_files:
        raise FileNotFoundError(f"モデルファイルが見つかりません: {models_dir}")

    # タイムスタンプを抽出し、最新を取得
    timestamps = set()
    for f in model_files:
        # model_fold0_20251127_123456.txt -> 20251127_123456
        parts = f.stem.split("_")
        if len(parts) >= 3:
            ts = "_".join(parts[-2:])
            timestamps.add(ts)

    latest_ts = sorted(timestamps)[-1]

    # このタイムスタンプの全foldを取得
    latest_models = sorted([
        f for f in model_files
        if f.stem.endswith(latest_ts)
    ])

    return latest_models, latest_ts


def load_models(models_dir: Path, timestamp: str = None) -> tuple[list, list[str], str]:
    """学習済みモデルと特徴量リストを読み込む。"""
    if timestamp:
        model_files = sorted(models_dir.glob(f"model_fold*_{timestamp}.txt"))
        if not model_files:
            raise FileNotFoundError(f"指定タイムスタンプのモデルが見つかりません: {timestamp}")
        ts = timestamp
    else:
        model_files, ts = find_latest_models(models_dir)

    print(f"モデル読み込み (timestamp: {ts})...")
    models = []
    for model_path in model_files:
        model = lgb.Booster(model_file=str(model_path))
        models.append(model)
        print(f"  読み込み完了: {model_path.name}")

    # 特徴量リストを読み込み
    features_path = models_dir / f"features_{ts}.json"
    if features_path.exists():
        with open(features_path) as f:
            features = json.load(f)["features"]
        print(f"  特徴量数: {len(features)}")
    else:
        # フォールバック: モデルから特徴量名を取得
        features = None
        print("  警告: features.json が見つかりません。モデルの特徴量名を使用します")

    return models, features, ts


def load_data() -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray]:
    """学習データとテストデータを読み込む。"""
    # 設定ファイルを読み込み
    data_config_path = project_root / "03_configs" / "data.yaml"
    with open(data_config_path, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    # 絶対パスに変換
    data_config["data"]["train_path"] = str(project_root / data_config["data"]["train_path"])
    data_config["data"]["test_path"] = str(project_root / data_config["data"]["test_path"])
    data_config["data"]["sample_submit_path"] = str(project_root / data_config["data"]["sample_submit_path"])

    loader = DataLoader(config=data_config, add_address_columns=False)

    train = loader.load_train()
    test = loader.load_test()
    test_ids = test["id"].to_numpy()

    return train, test, test_ids


def predict_exp009(timestamp: str = None):
    """学習済みモデルを使って予測を生成する。"""

    # シードを固定
    set_seed(42)

    # パス設定
    exp_dir = Path(__file__).resolve().parent.parent
    output_dir = exp_dir / "outputs"
    models_dir = output_dir / "models"

    # モデル読み込み
    models, features, model_ts = load_models(models_dir, timestamp)
    n_models = len(models)
    print(f"\n{n_models}個のモデルを読み込みました")

    # データの読み込み
    print("\n" + "=" * 60)
    print("データの読み込み...")
    print("=" * 60)
    train, test, test_ids = load_data()
    print(f"  Train: {train.shape}")
    print(f"  Test: {test.shape}")

    # CV splits作成（TargetEncoding用）
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_splits = list(cv.split(train))

    # 前処理
    print("\n" + "=" * 60)
    print("前処理...")
    print("=" * 60)
    _, X_test, _ = preprocess_for_training(train, test, cv_splits=cv_splits)

    # 特徴量を選択
    if features:
        X_test = X_test.select(features)
    else:
        # idがあれば除外
        X_test = X_test.drop("id") if "id" in X_test.columns else X_test

    print(f"テストデータ形状: {X_test.shape}")

    # 予測用にpandasに変換
    X_test_pd = X_test.to_pandas()

    # アンサンブル予測
    print("\n" + "=" * 60)
    print("予測を生成中...")
    print("=" * 60)

    test_predictions = np.zeros(len(X_test))

    for fold_idx, model in enumerate(models):
        print(f"  fold {fold_idx} で予測中...")
        pred_transformed = model.predict(X_test_pd)
        # 逆変換 (log1p -> 元の値)
        pred = np.expm1(pred_transformed)
        pred = np.maximum(pred, 0)
        test_predictions += pred / n_models

    # 提出ファイル生成
    print("\n" + "=" * 60)
    print("提出ファイルを生成中...")
    print("=" * 60)

    timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")

    test_ids_formatted = [f"{int(id_):06d}" for id_ in test_ids]
    test_predictions_rounded = np.round(test_predictions).astype(int)

    submission = pl.DataFrame({
        "id": test_ids_formatted,
        "money_room": test_predictions_rounded,
    })

    submission_path = output_dir / f"submission_{timestamp_now}.csv"
    submission.write_csv(submission_path, include_header=False)

    print(f"  保存完了: {submission_path}")

    # 予測統計
    print("\n" + "=" * 60)
    print("予測統計")
    print("=" * 60)
    print(f"  件数:   {len(test_predictions)}")
    print(f"  最小値: {test_predictions.min():,.0f}")
    print(f"  最大値: {test_predictions.max():,.0f}")
    print(f"  平均値: {test_predictions.mean():,.0f}")
    print(f"  中央値: {np.median(test_predictions):,.0f}")
    print("=" * 60)

    print(f"\n提出ファイル: {submission_path}")
    print("完了!")

    return submission_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="exp009の予測を生成")
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="モデルのタイムスタンプ (YYYYMMDD_HHMMSS)。指定しない場合は最新を使用。"
    )
    args = parser.parse_args()

    predict_exp009(timestamp=args.timestamp)
