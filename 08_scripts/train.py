#!/usr/bin/env python3
"""
学習スクリプト（設定ファイルの自動バックアップ + MLflow）

使い方:
    python train.py --name "baseline"
    python train.py --name "target_encode_v2"
    python train.py --restore exp_20251123_140530_baseline

特徴:
    - 実験ごとに設定ファイルを自動バックアップ（06_experiments/configs/）
    - MLflowにも記録（オプション）
    - 過去の実験を簡単に復元・再実行
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from src.utils.config import Config

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️  MLflowがインストールされていません。ローカルバックアップのみ実行します。")


def backup_configs(experiment_name: str = None) -> Path:
    """
    設定ファイルをバックアップ

    Returns:
        バックアップディレクトリのパス
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 実験名がある場合はディレクトリ名に含める
    if experiment_name:
        dir_name = f"exp_{timestamp}_{experiment_name}"
    else:
        dir_name = f"exp_{timestamp}"

    backup_dir = Path("06_experiments/configs") / dir_name
    backup_dir.mkdir(parents=True, exist_ok=True)

    # 全YAMLファイルをコピー
    config_dir = Path("03_configs")
    copied_files = []

    for yaml_file in config_dir.glob("*.yaml"):
        dest = backup_dir / yaml_file.name
        shutil.copy(yaml_file, dest)
        copied_files.append(yaml_file.name)

    print(f"✓ 設定ファイルをバックアップ: {backup_dir}")
    print(f"  ファイル: {', '.join(copied_files)}")

    return backup_dir


def restore_configs(experiment_dir: str):
    """
    過去の実験設定を復元

    Args:
        experiment_dir: 実験ディレクトリ名（例: "exp_20251123_140530_baseline"）
    """
    source_dir = Path("06_experiments/configs") / experiment_dir

    if not source_dir.exists():
        raise FileNotFoundError(f"実験ディレクトリが見つかりません: {source_dir}")

    dest_dir = Path("03_configs")

    print(f"🔄 設定を復元中: {source_dir}")

    for yaml_file in source_dir.glob("*.yaml"):
        dest = dest_dir / yaml_file.name
        shutil.copy(yaml_file, dest)
        print(f"  ✓ {yaml_file.name}")

    print(f"✓ 復元完了。03_configs/ を確認してください。")


def list_experiments():
    """過去の実験一覧を表示"""
    experiments_dir = Path("06_experiments/configs")

    if not experiments_dir.exists():
        print("まだ実験が実行されていません。")
        return

    experiments = sorted(experiments_dir.iterdir(), reverse=True)

    if not experiments:
        print("まだ実験が実行されていません。")
        return

    print("\n📊 過去の実験一覧:")
    print("=" * 80)

    for i, exp_dir in enumerate(experiments[:10], 1):  # 最新10件
        # タイムスタンプと名前を分離
        parts = exp_dir.name.split("_", 3)
        if len(parts) >= 3:
            date = parts[1]
            time = parts[2]
            name = parts[3] if len(parts) > 3 else "(名前なし)"

            # フォーマット
            date_str = f"{date[:4]}-{date[4:6]}-{date[6:]}"
            time_str = f"{time[:2]}:{time[2:4]}:{time[4:]}"

            print(f"{i:2d}. [{date_str} {time_str}] {name}")
            print(f"    └─ {exp_dir.name}")

    print("=" * 80)
    print(f"\n復元: python train.py --restore <実験ディレクトリ名>")


def main():
    parser = argparse.ArgumentParser(description="学習スクリプト")
    parser.add_argument("--name", type=str, help="実験名（例: baseline, target_encode_v2）")
    parser.add_argument("--restore", type=str, help="復元する実験ディレクトリ名")
    parser.add_argument("--list", action="store_true", help="過去の実験一覧を表示")
    parser.add_argument("--no-backup", action="store_true", help="バックアップをスキップ")
    parser.add_argument("--no-mlflow", action="store_true", help="MLflowをスキップ")
    args = parser.parse_args()

    # 実験一覧表示
    if args.list:
        list_experiments()
        return

    # 設定復元
    if args.restore:
        restore_configs(args.restore)
        print("\n次のステップ:")
        print("  1. 03_configs/ の内容を確認")
        print("  2. python train.py --name <実験名> で再実行")
        return

    # 設定読み込み
    cfg = Config("03_configs")

    # 設定バックアップ
    if not args.no_backup:
        backup_dir = backup_configs(args.name)

    # MLflow実験管理
    use_mlflow = MLFLOW_AVAILABLE and not args.no_mlflow

    if use_mlflow:
        mlflow.set_tracking_uri(cfg.get("experiment.tracking_uri", "06_experiments/mlruns"))
        experiment_name = cfg.get("project.name", "real_estate_price_prediction")
        mlflow.set_experiment(experiment_name)

    # 学習実行
    if use_mlflow:
        with mlflow.start_run(run_name=args.name) as run:
            run_id = run.info.run_id
            print(f"\n🚀 実験開始: {args.name or 'unnamed'}")
            print(f"   Run ID: {run_id}")

            # MLflowにパラメータをログ
            mlflow.log_params({
                "model_type": cfg.get("model.type"),
                "random_seed": cfg.get("project.random_seed"),
            })

            # バックアップした設定をMLflowにも保存
            if not args.no_backup:
                for yaml_file in backup_dir.glob("*.yaml"):
                    mlflow.log_artifact(str(yaml_file), "configs")

            # TODO: 実際の学習処理
            # from src.training.trainer import Trainer
            # trainer = Trainer(cfg)
            # metrics = trainer.train()
            # mlflow.log_metrics(metrics)

            # ダミーの結果（実装後は削除）
            print("\n⚠️  学習処理は未実装です（TODO）")
            mlflow.log_metric("rmse", 0.145)

            print(f"\n✓ 実験完了")
            print(f"  設定バックアップ: {backup_dir}")
            print(f"  MLflow Run ID: {run_id}")
    else:
        print(f"\n🚀 実験開始: {args.name or 'unnamed'}")
        print("   MLflowなしモード")

        # TODO: 実際の学習処理
        print("\n⚠️  学習処理は未実装です（TODO）")

        print(f"\n✓ 実験完了")
        if not args.no_backup:
            print(f"  設定バックアップ: {backup_dir}")


if __name__ == "__main__":
    main()
