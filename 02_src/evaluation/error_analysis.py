"""
エラー分析モジュール

予測誤差を多角的に分析する。
- 基本統計量（MAPE/RMSE/MAE等）
- セグメント別分析
- 外れ値検出
- 特徴量ビニング分析
"""

from typing import Union

import numpy as np
import polars as pl
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


class ErrorAnalyzer:
    """
    予測誤差の分析

    Attributes:
        y_true: 真値
        y_pred: 予測値
        residuals: 残差（y_true - y_pred）
        abs_residuals: 絶対残差
        pct_errors: パーセント誤差（for MAPE）
    """

    def __init__(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
    ):
        """
        初期化

        Args:
            y_true: 真値
            y_pred: 予測値
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)

        # 残差計算
        self.residuals = self.y_true - self.y_pred
        self.abs_residuals = np.abs(self.residuals)
        self.pct_errors = (
            np.abs(self.y_true - self.y_pred) / np.abs(self.y_true) * 100
        )

    def calculate_metrics(self) -> dict[str, float]:
        """
        各種誤差指標を計算

        Returns:
            dict: 各種誤差指標
        """
        mape = mean_absolute_percentage_error(self.y_true, self.y_pred) * 100
        rmse = np.sqrt(mean_squared_error(self.y_true, self.y_pred))
        mae = mean_absolute_error(self.y_true, self.y_pred)

        return {
            "mape": mape,
            "rmse": rmse,
            "mae": mae,
            "residual_mean": float(np.mean(self.residuals)),
            "residual_std": float(np.std(self.residuals)),
            "residual_min": float(np.min(self.residuals)),
            "residual_max": float(np.max(self.residuals)),
        }

    def get_residual_stats(self) -> pl.DataFrame:
        """
        残差の詳細統計量を取得

        Returns:
            pl.DataFrame: 統計量のDataFrame
        """
        percentiles = np.percentile(
            self.residuals, [25, 50, 75]
        )

        metrics = self.calculate_metrics()

        stats = {
            "mean": np.mean(self.residuals),
            "std": np.std(self.residuals),
            "min": np.min(self.residuals),
            "25%": percentiles[0],
            "50%": percentiles[1],
            "75%": percentiles[2],
            "max": np.max(self.residuals),
            "mape": metrics["mape"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
        }

        return pl.DataFrame(
            {
                "metric": list(stats.keys()),
                "value": list(stats.values()),
            }
        )

    def analyze_by_segment(
        self,
        segment_col: pl.Series,
        segment_name: str = "segment",
    ) -> pl.DataFrame:
        """
        セグメント別の誤差分析

        Args:
            segment_col: セグメント分類（カテゴリカル変数）
            segment_name: セグメント名（カラム名）

        Returns:
            pl.DataFrame: セグメント別の誤差統計
        """
        # DataFrame作成
        df = pl.DataFrame(
            {
                "segment": segment_col,
                "y_true": self.y_true,
                "y_pred": self.y_pred,
                "residual": self.residuals,
            }
        )

        # セグメント別集計
        result = df.group_by("segment").agg(
            [
                pl.len().alias("count"),
                # MAPE
                (
                    (pl.col("y_true") - pl.col("y_pred")).abs()
                    / pl.col("y_true").abs()
                    * 100
                )
                .mean()
                .alias("mape"),
                # RMSE
                ((pl.col("y_true") - pl.col("y_pred")) ** 2)
                .mean()
                .sqrt()
                .alias("rmse"),
                # MAE
                (pl.col("y_true") - pl.col("y_pred"))
                .abs()
                .mean()
                .alias("mae"),
                # 残差統計
                pl.col("residual").mean().alias("residual_mean"),
                pl.col("residual").std().alias("residual_std"),
            ]
        )

        return result

    def find_outliers(
        self,
        method: str = "std",
        threshold: float = 3.0,
    ) -> np.ndarray:
        """
        予測誤差の外れ値を検出

        Args:
            method: "std" (標準偏差) or "percentile" (パーセンタイル)
            threshold: 閾値

        Returns:
            np.ndarray: 外れ値のインデックス配列

        Raises:
            ValueError: methodが不正な場合
        """
        if method == "std":
            # 標準偏差ベース
            mean = np.mean(self.residuals)
            std = np.std(self.residuals)
            lower = mean - threshold * std
            upper = mean + threshold * std

            outliers = np.where(
                (self.residuals < lower) | (self.residuals > upper)
            )[0]

        elif method == "percentile":
            # パーセンタイルベース
            lower_pct = threshold
            upper_pct = 100 - threshold

            lower = np.percentile(self.residuals, lower_pct)
            upper = np.percentile(self.residuals, upper_pct)

            outliers = np.where(
                (self.residuals < lower) | (self.residuals > upper)
            )[0]

        else:
            raise ValueError(
                f"method must be 'std' or 'percentile', got '{method}'"
            )

        return outliers

    def get_outlier_details(
        self,
        outlier_indices: np.ndarray,
    ) -> pl.DataFrame:
        """
        外れ値の詳細情報を取得

        Args:
            outlier_indices: 外れ値のインデックス

        Returns:
            pl.DataFrame: 外れ値の詳細
        """
        return pl.DataFrame(
            {
                "index": outlier_indices,
                "y_true": self.y_true[outlier_indices],
                "y_pred": self.y_pred[outlier_indices],
                "residual": self.residuals[outlier_indices],
                "abs_residual": self.abs_residuals[outlier_indices],
                "pct_error": self.pct_errors[outlier_indices],
            }
        )

    def analyze_by_feature_bins(
        self,
        feature_values: pl.Series,
        feature_name: str,
        n_bins: int = 10,
    ) -> pl.DataFrame:
        """
        特徴量を区間分割して誤差を分析

        Args:
            feature_values: 特徴量の値
            feature_name: 特徴量名
            n_bins: 分割数

        Returns:
            pl.DataFrame: ビン別の誤差統計
        """
        # DataFrame作成
        df = pl.DataFrame(
            {
                "feature": feature_values,
                "y_true": self.y_true,
                "y_pred": self.y_pred,
                "residual": self.residuals,
            }
        )

        # ビニング（qcut: 各ビンのサンプル数が均等）
        # labels引数は使わず、カテゴリカルをordinalsに変換
        df = df.with_columns(
            pl.col("feature")
            .qcut(n_bins, allow_duplicates=True)
            .cast(pl.Categorical)
            .to_physical()
            .alias("bin_idx")
        )

        # ビン別集計
        result = df.group_by("bin_idx").agg(
            [
                pl.col("feature").min().alias("bin_min"),
                pl.col("feature").max().alias("bin_max"),
                pl.col("feature").mean().alias("bin_center"),
                pl.len().alias("count"),
                # MAPE
                (
                    (pl.col("y_true") - pl.col("y_pred")).abs()
                    / pl.col("y_true").abs()
                    * 100
                )
                .mean()
                .alias("mape"),
                # 残差平均
                pl.col("residual").mean().alias("residual_mean"),
            ]
        )

        # ビン名作成
        result = result.with_columns(
            (
                pl.col("bin_min").round(2).cast(str)
                + "-"
                + pl.col("bin_max").round(2).cast(str)
            ).alias("bin")
        )

        # カラム並べ替え
        result = result.select(
            ["bin", "bin_center", "count", "mape", "residual_mean"]
        ).sort("bin_center")

        return result
