#!/usr/bin/env python3
"""
MLflowと統合した学習スクリプト（設定ファイルの自動保存）

使い方:
    python train_with_mlflow.py
    python train_with_mlflow.py --experiment-name "baseline_v2"
    python train_with_mlflow.py --restore-run abc123  # 過去の実験を復元
"""

import argparse
import mlflow
import shutil
from datetime import datetime
from pathlib import Path
from src.utils.config import Config


def backup_configs_to_mlflow(run_id: str):
    """全設定ファイルをMLflowに保存"""
    config_dir = Path("03_configs")

    # 各YAMLファイルをログ
    for yaml_file in config_dir.glob("*.yaml"):
        mlflow.log_artifact(str(yaml_file), "configs")

    print(f"✓ 設定ファイルをMLflowに保存: run_id={run_id}")


def restore_configs_from_mlflow(run_id: str):
    """MLflowから設定ファイルを復元"""
    # 一時ディレクトリにダウンロード
    temp_dir = Path(f".temp_restore_{run_id}")
    client = mlflow.tracking.MlflowClient()

    # artifactsをダウンロード
    artifacts = client.list_artifacts(run_id, "configs")

    for artifact in artifacts:
        # ダウンロード
        local_path = client.download_artifacts(run_id, artifact.path)

        # 03_configs/にコピー
        dest = Path("03_configs") / Path(artifact.path).name
        shutil.copy(local_path, dest)
        print(f"✓ 復元: {dest}")

    # 一時ディレクトリ削除
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default=None, help="MLflow実験名")
    parser.add_argument("--restore-run", default=None, help="復元するrun_id")
    args = parser.parse_args()

    # 設定読み込み
    cfg = Config("03_configs")

    # MLflow設定
    mlflow.set_tracking_uri(cfg.get("experiment.tracking_uri", "06_experiments/mlruns"))
    experiment_name = args.experiment_name or cfg.get("project.name")
    mlflow.set_experiment(experiment_name)

    # 過去の実験を復元
    if args.restore_run:
        print(f"🔄 Run ID {args.restore_run} の設定を復元中...")
        restore_configs_from_mlflow(args.restore_run)
        print("✓ 復元完了。03_configs/ を確認してください。")
        return

    # 学習実行
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"🚀 実験開始: run_id={run_id}")

        # 【重要】設定ファイルをMLflowに保存
        backup_configs_to_mlflow(run_id)

        # パラメータをログ
        mlflow.log_params({
            "model_type": cfg.get("model.type"),
            "learning_rate": cfg.get(f"model.{cfg.get('model.type')}.learning_rate"),
            "cv_method": cfg.get("training.cross_validation.method"),
            "random_seed": cfg.get("project.random_seed"),
        })

        # TODO: 実際の学習処理を実装
        # trainer = Trainer(cfg)
        # metrics = trainer.train()
        # mlflow.log_metrics(metrics)

        # ダミーの結果（実装後は削除）
        mlflow.log_metric("rmse", 0.145)
        mlflow.log_metric("mae", 0.098)

        print(f"✓ 実験完了: run_id={run_id}")
        print(f"  設定ファイルは MLflow に保存されました")
        print(f"  復元: python train_with_mlflow.py --restore-run {run_id}")


if __name__ == "__main__":
    main()
