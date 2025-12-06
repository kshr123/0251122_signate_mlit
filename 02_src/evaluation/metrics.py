"""
評価指標モジュール

MAPE等の評価指標を計算
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_percentage_error,
    precision_score,
    recall_score,
)
from typing import Union


def calculate_mape(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> float:
    """
    MAPE (Mean Absolute Percentage Error) を計算

    sklearn.metrics.mean_absolute_percentage_errorのラッパー関数。
    返り値をパーセンテージ（0-100）に変換する。

    Args:
        y_true: 真値
        y_pred: 予測値

    Returns:
        MAPE（パーセンテージ、0-100の範囲）

    Examples:
        >>> y_true = [100, 200, 300]
        >>> y_pred = [110, 190, 310]
        >>> mape = calculate_mape(y_true, y_pred)
        >>> print(f"MAPE: {mape:.2f}%")
        MAPE: 6.11%

    Notes:
        - sklearnのMAPEは0-1の範囲で返すため、100倍してパーセンテージに変換
        - y_true=0の場合、sklearnが自動的に処理
    """
    # sklearnのMAPEは0-1の範囲で返すため、100倍してパーセンテージに変換
    mape_ratio = mean_absolute_percentage_error(y_true, y_pred)
    return float(mape_ratio * 100)


def calculate_pr_auc(
    y_true: Union[np.ndarray, list],
    y_score: Union[np.ndarray, list],
) -> float:
    """
    PR-AUC (Precision-Recall Area Under Curve) を計算

    不均衡データの分類タスクで推奨される指標。
    ROC-AUCより少数クラスの性能を適切に評価できる。

    Args:
        y_true: 真のラベル（0/1）
        y_score: 正例の予測確率（0-1）

    Returns:
        PR-AUC（0-1の範囲）

    Examples:
        >>> y_true = [0, 0, 1, 1]
        >>> y_score = [0.1, 0.4, 0.35, 0.8]
        >>> pr_auc = calculate_pr_auc(y_true, y_score)
        >>> print(f"PR-AUC: {pr_auc:.4f}")
    """
    return float(average_precision_score(y_true, y_score))


def calculate_classification_metrics(
    y_true: Union[np.ndarray, list],
    y_score: Union[np.ndarray, list],
    threshold: float = 0.5,
) -> dict:
    """
    分類タスクの各種メトリクスを計算

    Args:
        y_true: 真のラベル（0/1）
        y_score: 正例の予測確率（0-1）
        threshold: 正例判定の閾値（デフォルト: 0.5）

    Returns:
        dict: {
            "pr_auc": PR-AUC,
            "f1": F1スコア,
            "recall": 再現率,
            "precision": 適合率,
            "threshold": 使用した閾値,
        }

    Examples:
        >>> y_true = [0, 0, 1, 1]
        >>> y_score = [0.1, 0.4, 0.35, 0.8]
        >>> metrics = calculate_classification_metrics(y_true, y_score, threshold=0.3)
        >>> print(f"Recall: {metrics['recall']:.4f}")
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # 閾値で2値化
    y_pred = (y_score >= threshold).astype(int)

    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
    }
