"""
MLflow記録用ヘルパー関数
"""

import mlflow
import polars as pl
import numpy as np
from typing import List, Dict, Any


def log_dataset_info(df: pl.DataFrame, prefix: str = "train") -> None:
    """
    データセットの基本情報をMLflowに記録

    Args:
        df: データフレーム
        prefix: メトリクス名のプレフィックス（train, test等）
    """
    # 基本情報
    mlflow.log_metric(f"{prefix}.n_rows", df.height)
    mlflow.log_metric(f"{prefix}.n_cols", df.width)
    mlflow.log_metric(f"{prefix}.memory_mb", df.estimated_size("mb"))

    # 欠損値情報
    null_counts = df.null_count()
    total_nulls = sum(null_counts.row(0))
    null_ratio = total_nulls / (df.height * df.width) if df.height * df.width > 0 else 0.0

    mlflow.log_metric(f"{prefix}.total_nulls", total_nulls)
    mlflow.log_metric(f"{prefix}.null_ratio", null_ratio)


def log_cv_results(cv_scores: np.ndarray, metric_name: str = "rmse") -> None:
    """
    クロスバリデーション結果をMLflowに記録

    Args:
        cv_scores: CVスコアの配列（各Foldのスコア）
        metric_name: メトリクス名（rmse, mae等）
    """
    # 統計量
    mlflow.log_metric(f"cv_{metric_name}_mean", float(cv_scores.mean()))
    mlflow.log_metric(f"cv_{metric_name}_std", float(cv_scores.std()))
    mlflow.log_metric(f"cv_{metric_name}_min", float(cv_scores.min()))
    mlflow.log_metric(f"cv_{metric_name}_max", float(cv_scores.max()))

    # Fold別スコア
    for i, score in enumerate(cv_scores):
        mlflow.log_metric(f"cv_{metric_name}_fold_{i}", float(score))


def log_feature_list(feature_cols: List[str], artifact_path: str = "features.txt") -> None:
    """
    使用特徴量リストをアーティファクトとして保存

    Args:
        feature_cols: 特徴量カラムのリスト
        artifact_path: 保存先ファイル名
    """
    from pathlib import Path

    temp_file = Path(artifact_path)
    temp_file.write_text("\n".join(feature_cols))

    mlflow.log_artifact(temp_file)
    temp_file.unlink()  # 一時ファイル削除


def log_model_params(params: Dict[str, Any], prefix: str = "") -> None:
    """
    モデルパラメータをMLflowに記録

    Args:
        params: パラメータ辞書
        prefix: パラメータ名のプレフィックス
    """
    for key, value in params.items():
        param_name = f"{prefix}.{key}" if prefix else key

        # 値の型に応じて適切に記録
        if isinstance(value, (int, float, str, bool)):
            mlflow.log_param(param_name, value)
        elif isinstance(value, (list, tuple)):
            mlflow.log_param(param_name, str(value))
        elif isinstance(value, dict):
            # ネストした辞書は再帰的に記録
            log_model_params(value, prefix=param_name)
        else:
            mlflow.log_param(param_name, str(value))
