"""
Tests for evaluation.feature_importance module

テスト方針:
- LightGBMモデルの特徴量重要度計算
- Gain/Split/Permutationの3タイプ
- Top N特徴量抽出
- 複数タイプの比較
"""

import numpy as np
import polars as pl
import pytest
import lightgbm as lgb


@pytest.fixture
def sample_data():
    """サンプルデータ（回帰問題）"""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    X = pl.DataFrame(
        {
            f"feature_{i}": np.random.randn(n_samples)
            for i in range(n_features)
        }
    )
    # 線形関係 + ノイズ
    y = (
        2 * X["feature_0"].to_numpy()
        + 1.5 * X["feature_1"].to_numpy()
        + np.random.randn(n_samples) * 0.1
    )

    return X, y


@pytest.fixture
def trained_lgb_model(sample_data):
    """学習済みLightGBMモデル"""
    X, y = sample_data
    X_np = X.to_numpy()

    train_data = lgb.Dataset(X_np, label=y)
    params = {
        "objective": "regression",
        "metric": "mape",
        "num_leaves": 7,
        "seed": 42,
        "verbose": -1,
        "force_row_wise": True,
    }
    model = lgb.train(params, train_data, num_boost_round=20)

    return model


class TestFeatureImportanceAnalyzer:
    """FeatureImportanceAnalyzerクラスのテスト"""

    def test_calculate_importance_gain(self, trained_lgb_model, sample_data):
        """Gain重要度が計算できること"""
        from evaluation.feature_importance import FeatureImportanceAnalyzer

        X, _ = sample_data
        feature_names = X.columns

        analyzer = FeatureImportanceAnalyzer()
        importance_df = analyzer.calculate_importance(
            model=trained_lgb_model,
            feature_names=feature_names,
            importance_type="gain",
        )

        # カラムチェック
        assert set(importance_df.columns) == {"feature", "importance", "type"}

        # レコード数チェック
        assert len(importance_df) == len(feature_names)

        # 正規化チェック（合計=1.0）
        total = importance_df["importance"].sum()
        assert abs(total - 1.0) < 1e-6

        # typeカラムチェック
        assert importance_df["type"].unique().to_list() == ["gain"]

    def test_calculate_importance_split(self, trained_lgb_model, sample_data):
        """Split重要度が計算できること"""
        from evaluation.feature_importance import FeatureImportanceAnalyzer

        X, _ = sample_data
        feature_names = X.columns

        analyzer = FeatureImportanceAnalyzer()
        importance_df = analyzer.calculate_importance(
            model=trained_lgb_model,
            feature_names=feature_names,
            importance_type="split",
        )

        # typeカラムチェック
        assert importance_df["type"].unique().to_list() == ["split"]

        # Gainと異なることを確認
        gain_df = analyzer.calculate_importance(
            model=trained_lgb_model,
            feature_names=feature_names,
            importance_type="gain",
        )
        # 少なくとも1つの特徴量で重要度が異なる
        assert not (importance_df["importance"] == gain_df["importance"]).all()

    def test_invalid_importance_type(self, trained_lgb_model, sample_data):
        """不正なimportance_typeでエラーが発生すること"""
        from evaluation.feature_importance import FeatureImportanceAnalyzer

        X, _ = sample_data
        feature_names = X.columns

        analyzer = FeatureImportanceAnalyzer()

        with pytest.raises(ValueError, match="importance_type"):
            analyzer.calculate_importance(
                model=trained_lgb_model,
                feature_names=feature_names,
                importance_type="invalid_type",
            )

    def test_calculate_permutation_importance(
        self, trained_lgb_model, sample_data
    ):
        """Permutation Importanceが計算できること"""
        from evaluation.feature_importance import FeatureImportanceAnalyzer

        X, y = sample_data

        analyzer = FeatureImportanceAnalyzer()
        perm_imp_df = analyzer.calculate_permutation_importance(
            model=trained_lgb_model,
            X=X,
            y=y,
            n_repeats=5,
            random_state=42,
        )

        # カラムチェック
        assert set(perm_imp_df.columns) == {"feature", "importance", "type"}

        # typeカラムチェック
        assert perm_imp_df["type"].unique().to_list() == ["permutation"]

        # レコード数チェック
        assert len(perm_imp_df) == len(X.columns)

    def test_get_top_features(self, trained_lgb_model, sample_data):
        """上位N件の特徴量が取得できること"""
        from evaluation.feature_importance import FeatureImportanceAnalyzer

        X, _ = sample_data
        feature_names = X.columns

        analyzer = FeatureImportanceAnalyzer()
        analyzer.calculate_importance(
            model=trained_lgb_model,
            feature_names=feature_names,
            importance_type="gain",
        )

        top3 = analyzer.get_top_features(n=3)

        # レコード数チェック
        assert len(top3) == 3

        # 降順ソートチェック
        importances = top3["importance"].to_list()
        assert importances == sorted(importances, reverse=True)

    def test_get_top_features_before_calculate(self):
        """calculate_importance未実行時にエラーが発生すること"""
        from evaluation.feature_importance import FeatureImportanceAnalyzer

        analyzer = FeatureImportanceAnalyzer()

        with pytest.raises(RuntimeError, match="calculate_importance"):
            analyzer.get_top_features(n=10)

    def test_compare_importance_types_with_permutation(
        self, trained_lgb_model, sample_data
    ):
        """複数タイプの重要度が比較できること（Permutation含む）"""
        from evaluation.feature_importance import FeatureImportanceAnalyzer

        X, y = sample_data
        feature_names = X.columns

        analyzer = FeatureImportanceAnalyzer()
        comparison = analyzer.compare_importance_types(
            model=trained_lgb_model,
            feature_names=feature_names,
            X=X,
            y=y,
        )

        # カラムチェック
        assert set(comparison.columns) == {
            "feature",
            "gain",
            "split",
            "permutation",
        }

        # レコード数チェック
        assert len(comparison) == len(feature_names)

        # 各重要度が正規化されているか確認
        for col in ["gain", "split", "permutation"]:
            total = comparison[col].sum()
            assert abs(total - 1.0) < 1e-6

    def test_compare_importance_types_without_permutation(
        self, trained_lgb_model, sample_data
    ):
        """Permutation未指定時はgain/splitのみ取得できること"""
        from evaluation.feature_importance import FeatureImportanceAnalyzer

        X, _ = sample_data
        feature_names = X.columns

        analyzer = FeatureImportanceAnalyzer()
        comparison = analyzer.compare_importance_types(
            model=trained_lgb_model,
            feature_names=feature_names,
            X=None,
            y=None,
        )

        # カラムチェック（permutationなし）
        assert set(comparison.columns) == {"feature", "gain", "split"}

        # レコード数チェック
        assert len(comparison) == len(feature_names)
