"""
Tests for evaluation.error_analysis module

テスト方針:
- 誤差統計計算（MAPE/RMSE/MAE等）
- セグメント別分析
- 外れ値検出（標準偏差・パーセンタイル）
- 特徴量ビニング分析
"""

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def sample_predictions():
    """サンプルの予測データ"""
    np.random.seed(42)
    n_samples = 100

    # 真値
    y_true = np.random.randint(50000, 150000, size=n_samples).astype(float)

    # 予測値（真値 + ノイズ）
    noise = np.random.randn(n_samples) * 10000
    y_pred = y_true + noise

    return y_true, y_pred


@pytest.fixture
def sample_with_segments(sample_predictions):
    """セグメント付きサンプルデータ"""
    y_true, y_pred = sample_predictions

    # 価格帯セグメント作成
    segments = pl.Series(
        [
            "低価格" if y < 70000 else "中価格" if y < 110000 else "高価格"
            for y in y_true
        ]
    )

    return y_true, y_pred, segments


class TestErrorAnalyzer:
    """ErrorAnalyzerクラスのテスト"""

    def test_init_and_attributes(self, sample_predictions):
        """初期化と属性が正しく設定されること"""
        from evaluation.error_analysis import ErrorAnalyzer

        y_true, y_pred = sample_predictions

        analyzer = ErrorAnalyzer(y_true, y_pred)

        # 基本属性チェック
        assert len(analyzer.y_true) == len(y_true)
        assert len(analyzer.y_pred) == len(y_pred)
        assert len(analyzer.residuals) == len(y_true)
        assert len(analyzer.abs_residuals) == len(y_true)
        assert len(analyzer.pct_errors) == len(y_true)

        # 残差計算チェック
        expected_residuals = y_true - y_pred
        np.testing.assert_array_almost_equal(
            analyzer.residuals, expected_residuals
        )

    def test_calculate_metrics(self, sample_predictions):
        """各種誤差指標が計算できること"""
        from evaluation.error_analysis import ErrorAnalyzer

        y_true, y_pred = sample_predictions

        analyzer = ErrorAnalyzer(y_true, y_pred)
        metrics = analyzer.calculate_metrics()

        # 必須キーチェック
        required_keys = {
            "mape",
            "rmse",
            "mae",
            "residual_mean",
            "residual_std",
            "residual_min",
            "residual_max",
        }
        assert set(metrics.keys()) == required_keys

        # 値の妥当性チェック
        assert metrics["mape"] >= 0
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert isinstance(metrics["residual_mean"], float)
        assert metrics["residual_std"] >= 0

    def test_get_residual_stats(self, sample_predictions):
        """残差統計量が取得できること"""
        from evaluation.error_analysis import ErrorAnalyzer

        y_true, y_pred = sample_predictions

        analyzer = ErrorAnalyzer(y_true, y_pred)
        stats_df = analyzer.get_residual_stats()

        # カラムチェック
        assert set(stats_df.columns) == {"metric", "value"}

        # 統計量の存在確認
        metrics = stats_df["metric"].to_list()
        expected_metrics = [
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
            "mape",
            "rmse",
            "mae",
        ]
        assert set(metrics) == set(expected_metrics)

    def test_analyze_by_segment(self, sample_with_segments):
        """セグメント別分析が動作すること"""
        from evaluation.error_analysis import ErrorAnalyzer

        y_true, y_pred, segments = sample_with_segments

        analyzer = ErrorAnalyzer(y_true, y_pred)
        segment_analysis = analyzer.analyze_by_segment(
            segments, segment_name="price_range"
        )

        # カラムチェック
        expected_cols = {
            "segment",
            "count",
            "mape",
            "rmse",
            "mae",
            "residual_mean",
            "residual_std",
        }
        assert set(segment_analysis.columns) == expected_cols

        # セグメント数チェック
        assert len(segment_analysis) == 3  # 低価格・中価格・高価格

        # 各セグメントのサンプル数チェック
        total_count = segment_analysis["count"].sum()
        assert total_count == len(y_true)

    def test_find_outliers_std(self, sample_predictions):
        """標準偏差ベースで外れ値が検出できること"""
        from evaluation.error_analysis import ErrorAnalyzer

        y_true, y_pred = sample_predictions

        analyzer = ErrorAnalyzer(y_true, y_pred)
        outlier_indices = analyzer.find_outliers(method="std", threshold=3.0)

        # 外れ値が検出されること
        assert isinstance(outlier_indices, np.ndarray)
        assert len(outlier_indices) >= 0

        # 全データよりは少ないこと
        assert len(outlier_indices) < len(y_true)

    def test_find_outliers_percentile(self, sample_predictions):
        """パーセンタイルベースで外れ値が検出できること"""
        from evaluation.error_analysis import ErrorAnalyzer

        y_true, y_pred = sample_predictions

        analyzer = ErrorAnalyzer(y_true, y_pred)
        outlier_indices = analyzer.find_outliers(
            method="percentile", threshold=5.0
        )

        # 約5%が検出されること（両側で合計約10%）
        expected_count = int(len(y_true) * 0.1)
        assert abs(len(outlier_indices) - expected_count) <= 2

    def test_find_outliers_invalid_method(self, sample_predictions):
        """不正なmethodでエラーが発生すること"""
        from evaluation.error_analysis import ErrorAnalyzer

        y_true, y_pred = sample_predictions

        analyzer = ErrorAnalyzer(y_true, y_pred)

        with pytest.raises(ValueError, match="method"):
            analyzer.find_outliers(method="invalid_method")

    def test_get_outlier_details(self, sample_predictions):
        """外れ値の詳細が取得できること"""
        from evaluation.error_analysis import ErrorAnalyzer

        y_true, y_pred = sample_predictions

        analyzer = ErrorAnalyzer(y_true, y_pred)
        outlier_indices = analyzer.find_outliers(method="std", threshold=2.0)
        outlier_details = analyzer.get_outlier_details(outlier_indices)

        # カラムチェック
        expected_cols = {
            "index",
            "y_true",
            "y_pred",
            "residual",
            "abs_residual",
            "pct_error",
        }
        assert set(outlier_details.columns) == expected_cols

        # レコード数チェック
        assert len(outlier_details) == len(outlier_indices)

    def test_analyze_by_feature_bins(self, sample_predictions):
        """特徴量をビニングして誤差分析できること"""
        from evaluation.error_analysis import ErrorAnalyzer

        y_true, y_pred = sample_predictions

        # サンプル特徴量（面積など）
        feature_values = pl.Series(np.random.uniform(20, 100, size=len(y_true)))

        analyzer = ErrorAnalyzer(y_true, y_pred)
        bins_analysis = analyzer.analyze_by_feature_bins(
            feature_values, "area_sqm", n_bins=5
        )

        # カラムチェック
        expected_cols = {
            "bin",
            "bin_center",
            "count",
            "mape",
            "residual_mean",
        }
        assert set(bins_analysis.columns) == expected_cols

        # ビン数チェック
        assert len(bins_analysis) <= 5  # 最大5ビン（データ量によっては少ない可能性）

        # 各ビンのサンプル数合計
        total_count = bins_analysis["count"].sum()
        assert total_count == len(y_true)
