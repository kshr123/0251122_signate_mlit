"""
MLflow記録ヘルパー関数のテスト
"""

import pytest
import polars as pl
import numpy as np
import mlflow
from pathlib import Path

from training.utils.mlflow_helper import (
    log_dataset_info,
    log_cv_results,
    log_feature_list,
    log_model_params,
)


@pytest.fixture
def sample_dataframe():
    """テスト用のDataFrame"""
    return pl.DataFrame({
        "a": [1, 2, None, 4, 5],
        "b": [1.0, 2.0, 3.0, None, 5.0],
        "c": ["x", "y", "z", "w", "v"],
    })


@pytest.fixture
def mlflow_run():
    """MLflowテスト用Run"""
    mlflow.set_experiment("test_experiment")
    with mlflow.start_run() as run:
        yield run
    mlflow.end_run()


def test_log_dataset_info_logs_basic_metrics(sample_dataframe, mlflow_run):
    """log_dataset_infoが基本メトリクスを記録すること"""
    log_dataset_info(sample_dataframe, prefix="test")

    # Run情報を取得
    run_data = mlflow.get_run(mlflow_run.info.run_id).data

    # 基本情報が記録されていること
    assert "test.n_rows" in run_data.metrics
    assert run_data.metrics["test.n_rows"] == 5

    assert "test.n_cols" in run_data.metrics
    assert run_data.metrics["test.n_cols"] == 3


def test_log_dataset_info_logs_null_info(sample_dataframe, mlflow_run):
    """log_dataset_infoが欠損値情報を記録すること"""
    log_dataset_info(sample_dataframe, prefix="test")

    run_data = mlflow.get_run(mlflow_run.info.run_id).data

    # 欠損値情報が記録されていること
    assert "test.total_nulls" in run_data.metrics
    assert run_data.metrics["test.total_nulls"] == 2  # a列1個 + b列1個

    assert "test.null_ratio" in run_data.metrics


def test_log_cv_results_logs_statistics(mlflow_run):
    """log_cv_resultsが統計量を記録すること"""
    cv_scores = np.array([0.1, 0.2, 0.15, 0.18, 0.12])

    log_cv_results(cv_scores, metric_name="rmse")

    run_data = mlflow.get_run(mlflow_run.info.run_id).data

    # 統計量が記録されていること
    assert "cv_rmse_mean" in run_data.metrics
    assert abs(run_data.metrics["cv_rmse_mean"] - 0.15) < 0.01

    assert "cv_rmse_std" in run_data.metrics
    assert "cv_rmse_min" in run_data.metrics
    assert "cv_rmse_max" in run_data.metrics


def test_log_cv_results_logs_fold_scores(mlflow_run):
    """log_cv_resultsがFold別スコアを記録すること"""
    cv_scores = np.array([0.1, 0.2, 0.15])

    log_cv_results(cv_scores, metric_name="mae")

    run_data = mlflow.get_run(mlflow_run.info.run_id).data

    # Fold別スコアが記録されていること
    assert "cv_mae_fold_0" in run_data.metrics
    assert run_data.metrics["cv_mae_fold_0"] == 0.1

    assert "cv_mae_fold_1" in run_data.metrics
    assert run_data.metrics["cv_mae_fold_1"] == 0.2

    assert "cv_mae_fold_2" in run_data.metrics
    assert run_data.metrics["cv_mae_fold_2"] == 0.15


def test_log_feature_list_creates_artifact(mlflow_run):
    """log_feature_listがアーティファクトを作成すること"""
    feature_cols = ["feature1", "feature2", "feature3"]

    log_feature_list(feature_cols, artifact_path="test_features.txt")

    # アーティファクトが記録されていること
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(mlflow_run.info.run_id)

    artifact_names = [a.path for a in artifacts]
    assert "test_features.txt" in artifact_names


def test_log_model_params_logs_simple_params(mlflow_run):
    """log_model_paramsが単純なパラメータを記録すること"""
    params = {
        "learning_rate": 0.01,
        "num_leaves": 31,
        "seed": 42,
        "objective": "regression",
    }

    log_model_params(params)

    run_data = mlflow.get_run(mlflow_run.info.run_id).data

    # パラメータが記録されていること
    assert run_data.params["learning_rate"] == "0.01"
    assert run_data.params["num_leaves"] == "31"
    assert run_data.params["seed"] == "42"
    assert run_data.params["objective"] == "regression"


def test_log_model_params_logs_nested_params(mlflow_run):
    """log_model_paramsがネストしたパラメータを記録すること"""
    params = {
        "model": {
            "type": "LightGBM",
            "params": {
                "learning_rate": 0.01,
            }
        }
    }

    log_model_params(params)

    run_data = mlflow.get_run(mlflow_run.info.run_id).data

    # ネストしたパラメータが記録されていること
    assert "model.type" in run_data.params
    assert "model.params.learning_rate" in run_data.params
