"""
Permutation Importance モジュールのテスト
"""

import numpy as np
import polars as pl
import pytest

from evaluation.permutation_importance import (
    PermutationImportanceCalculator,
    PermutationImportanceResult,
)


class TestPermutationImportanceCalculator:
    """PermutationImportanceCalculator のテスト"""

    def test_init_default(self):
        """デフォルト初期化"""
        calc = PermutationImportanceCalculator()
        assert calc.scoring == "mape"
        assert calc.n_repeats == 5
        assert calc.random_state == 42

    def test_init_custom(self):
        """カスタム初期化"""
        calc = PermutationImportanceCalculator(
            scoring="mae",
            n_repeats=10,
            random_state=123,
        )
        assert calc.scoring == "mae"
        assert calc.n_repeats == 10
        assert calc.random_state == 123

    def test_init_invalid_scoring(self):
        """無効なscoringでエラー"""
        with pytest.raises(ValueError, match="scoring must be one of"):
            PermutationImportanceCalculator(scoring="invalid")

    def test_calculate_basic(self):
        """基本的な計算"""
        # 単純な線形モデルをシミュレート
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 3)
        # y = 2*X0 + 0*X1 + 0*X2 + noise
        y = 2 * X[:, 0] + np.random.randn(n_samples) * 0.1

        # X0だけが重要な特徴量
        def predict_fn(X):
            return 2 * X[:, 0]

        calc = PermutationImportanceCalculator(scoring="mae", n_repeats=3)
        result = calc.calculate(
            predict_fn=predict_fn,
            X=X,
            y=y,
            feature_names=["important", "useless1", "useless2"],
        )

        # X0は重要、X1/X2は不要
        assert result.importances_mean[0] > result.importances_mean[1]
        assert result.importances_mean[0] > result.importances_mean[2]
        # X1/X2はほぼ0
        assert abs(result.importances_mean[1]) < 0.5
        assert abs(result.importances_mean[2]) < 0.5

    def test_calculate_with_polars(self):
        """Polars DataFrameでの計算"""
        np.random.seed(42)
        X_np = np.random.randn(50, 2)
        y = X_np[:, 0] + X_np[:, 1]

        X_pl = pl.DataFrame({
            "feat_a": X_np[:, 0],
            "feat_b": X_np[:, 1],
        })

        def predict_fn(X):
            return X[:, 0] + X[:, 1]

        calc = PermutationImportanceCalculator(scoring="mse", n_repeats=2)
        result = calc.calculate(predict_fn=predict_fn, X=X_pl, y=y)

        # 特徴量名がPolarsから取得される
        assert result.feature_names == ["feat_a", "feat_b"]

    def test_calculate_auto_feature_names(self):
        """特徴量名の自動生成"""
        np.random.seed(42)
        X = np.random.randn(30, 4)
        y = np.random.randn(30)

        def predict_fn(X):
            return np.zeros(len(X))

        calc = PermutationImportanceCalculator(scoring="mae", n_repeats=2)
        result = calc.calculate(predict_fn=predict_fn, X=X, y=y)

        assert result.feature_names == [
            "feature_0", "feature_1", "feature_2", "feature_3"
        ]


class TestPermutationImportanceResult:
    """PermutationImportanceResult のテスト"""

    @pytest.fixture
    def sample_result(self):
        """テスト用のサンプル結果"""
        return PermutationImportanceResult(
            feature_names=["feat_a", "feat_b", "feat_c", "feat_d"],
            importances_mean=np.array([10.0, 5.0, 0.0, -0.5]),
            importances_std=np.array([1.0, 0.5, 0.1, 0.2]),
            importances_raw=np.array([[10, 9, 11], [5, 5, 5], [0, 0, 0], [-0.5, -0.5, -0.5]]),
            baseline_score=-10.0,
            scoring="mape",
        )

    def test_to_dataframe(self, sample_result):
        """DataFrameへの変換"""
        df = sample_result.to_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert "feature" in df.columns
        assert "importance" in df.columns
        assert "importance_std" in df.columns
        assert "importance_pct" in df.columns
        assert len(df) == 4

        # 重要度でソートされている
        assert df["feature"][0] == "feat_a"
        assert df["feature"][1] == "feat_b"

    def test_to_dataframe_normalized(self, sample_result):
        """正規化付きDataFrame変換"""
        df = sample_result.to_dataframe(normalize=True)

        # 正規化されている（合計≈1.0、ただし負値があるので厳密ではない）
        total = df["importance"].sum()
        # 正の重要度のみの合計で正規化されるので、
        # ここでは元の値（10, 5, 0, -0.5）の合計で割られる
        assert abs(total - 1.0) < 0.01

    def test_get_safe_to_remove_features(self, sample_result):
        """削除安全な特徴量の取得"""
        # 閾値0.0で取得
        safe = sample_result.get_safe_to_remove_features(threshold=0.0)

        # feat_c (0.0) と feat_d (-0.5) が対象
        # ただしconsider_std=Trueなので mean+std で判定
        # feat_c: 0.0 + 0.1 = 0.1 > 0 → 対象外
        # feat_d: -0.5 + 0.2 = -0.3 <= 0 → 対象
        assert "feat_d" in safe
        assert "feat_a" not in safe
        assert "feat_b" not in safe

    def test_get_safe_to_remove_features_without_std(self, sample_result):
        """std考慮なしの削除安全特徴量取得"""
        safe = sample_result.get_safe_to_remove_features(
            threshold=0.0, consider_std=False
        )

        # feat_c (0.0) と feat_d (-0.5) が対象
        assert "feat_c" in safe
        assert "feat_d" in safe
        assert "feat_a" not in safe

    def test_get_cumulative_importance_features(self, sample_result):
        """累積重要度による特徴量取得"""
        # 90%の累積重要度
        # 合計: 10 + 5 + 0 + (-0.5) = 14.5
        # feat_a: 10/14.5 = 68.9%
        # feat_a + feat_b: 15/14.5 = 103.4% → 2特徴量で90%超え
        top = sample_result.get_cumulative_importance_features(threshold_pct=90.0)

        assert len(top) == 2
        assert top[0] == "feat_a"
        assert top[1] == "feat_b"


class TestScoringFunctions:
    """スコアリング関数のテスト"""

    def test_mape_scoring(self):
        """MAPEスコアリング"""
        calc = PermutationImportanceCalculator(scoring="mape")

        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])

        # MAPE = mean(|10/100|, |10/200|, |10/300|) * 100
        # = mean(0.1, 0.05, 0.033) * 100 = 6.1%
        score = calc._scoring_fn(y_true, y_pred)

        # 負値で返される（大きいほど良い形式）
        assert score < 0
        assert abs(score + 6.1) < 0.5  # 約-6.1%

    def test_mae_scoring(self):
        """MAEスコアリング"""
        calc = PermutationImportanceCalculator(scoring="mae")

        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 18, 33])

        # MAE = mean(2, 2, 3) = 2.33
        score = calc._scoring_fn(y_true, y_pred)

        assert score < 0
        assert abs(score + 2.33) < 0.1

    def test_r2_scoring(self):
        """R2スコアリング"""
        calc = PermutationImportanceCalculator(scoring="r2")

        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])  # 完璧な予測

        score = calc._scoring_fn(y_true, y_pred)

        assert score == 1.0  # R2は正値


class TestCalculateFromModels:
    """複数モデルからの計算テスト"""

    def test_calculate_from_models(self):
        """複数モデルからの重要度計算"""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] * 2 + X[:, 1]

        # モックモデル
        class MockModel:
            def __init__(self, coef):
                self.coef = coef

            def predict(self, X):
                return X @ self.coef

        models = [
            MockModel(np.array([2.0, 1.0])),
            MockModel(np.array([2.1, 0.9])),
            MockModel(np.array([1.9, 1.1])),
        ]

        calc = PermutationImportanceCalculator(scoring="mae", n_repeats=2)
        result = calc.calculate_from_models(
            models=models,
            X=X,
            y=y,
            feature_names=["x0", "x1"],
        )

        # x0のほうが係数が大きいので重要度が高いはず
        assert result.importances_mean[0] > result.importances_mean[1]
