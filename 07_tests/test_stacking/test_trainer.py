"""Tests for StackingTrainer."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from stacking.trainer import StackingTrainer


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 3

    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1 + 100

    return X, y


class TestStackingTrainer:
    """Tests for StackingTrainer class."""

    def test_init(self):
        """Test initialization."""
        model = Ridge(alpha=1.0)
        trainer = StackingTrainer(model, n_splits=3, seed=42)

        assert trainer.meta_model is model
        assert trainer.n_splits == 3
        assert trainer.seed == 42
        assert trainer.fitted_model_ is None

    def test_fit_predict_oof(self, sample_data):
        """Test fit_predict_oof method."""
        X, y = sample_data
        model = Ridge(alpha=1.0)
        trainer = StackingTrainer(model, n_splits=3, seed=42)

        oof_pred, fold_scores = trainer.fit_predict_oof(X, y)

        # Check OOF predictions shape
        assert oof_pred.shape == (len(y),)

        # Check fold scores
        assert len(fold_scores) == 3
        assert all(isinstance(s, float) for s in fold_scores)
        assert all(s >= 0 for s in fold_scores)  # MAPE is non-negative

        # OOF predictions should be reasonable
        assert not np.any(np.isnan(oof_pred))
        assert not np.any(np.isinf(oof_pred))

    def test_fit_final(self, sample_data):
        """Test fit_final method."""
        X, y = sample_data
        model = Ridge(alpha=1.0)
        trainer = StackingTrainer(model, n_splits=3, seed=42)

        trainer.fit_final(X, y)

        assert trainer.fitted_model_ is not None
        # Check that model is fitted (has coef_ attribute for Ridge)
        assert hasattr(trainer.fitted_model_, "coef_")

    def test_predict(self, sample_data):
        """Test predict method."""
        X, y = sample_data
        model = Ridge(alpha=1.0)
        trainer = StackingTrainer(model, n_splits=3, seed=42)

        # Should raise error before fit_final
        with pytest.raises(ValueError, match="fit_final"):
            trainer.predict(X)

        # After fit_final, should work
        trainer.fit_final(X, y)
        predictions = trainer.predict(X)

        assert predictions.shape == (len(y),)
        assert not np.any(np.isnan(predictions))

    def test_save_and_load(self, sample_data):
        """Test save and load methods."""
        X, y = sample_data
        model = Ridge(alpha=1.0)
        trainer = StackingTrainer(model, n_splits=3, seed=42)

        # Should raise error before fit_final
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"

            with pytest.raises(ValueError, match="fit_final"):
                trainer.save(path)

            # After fit_final, should work
            trainer.fit_final(X, y)
            trainer.save(path)

            assert path.exists()

            # Load and compare predictions
            loaded_model = StackingTrainer.load(path)
            original_pred = trainer.predict(X)
            loaded_pred = loaded_model.predict(X)

            np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_custom_scoring_func(self, sample_data):
        """Test custom scoring function."""
        X, y = sample_data

        def rmse(y_true, y_pred):
            return np.sqrt(np.mean((y_true - y_pred) ** 2))

        model = Ridge(alpha=1.0)
        trainer = StackingTrainer(model, n_splits=3, seed=42, scoring_func=rmse)

        oof_pred, fold_scores = trainer.fit_predict_oof(X, y)

        # RMSE should be much smaller than MAPE for this data
        assert len(fold_scores) == 3
        assert all(s < 1.0 for s in fold_scores)  # RMSE should be small

    def test_mape_calculation(self):
        """Test MAPE calculation."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 330.0])

        mape = StackingTrainer._mape(y_true, y_pred)

        # Expected: (|100-110|/100 + |200-190|/200 + |300-330|/300) / 3 * 100
        # = (0.1 + 0.05 + 0.1) / 3 * 100 = 8.33...
        expected = (0.1 + 0.05 + 0.1) / 3 * 100
        assert abs(mape - expected) < 0.01

    def test_original_model_not_modified(self, sample_data):
        """Test that original meta_model is not modified."""
        X, y = sample_data
        model = Ridge(alpha=1.0)
        trainer = StackingTrainer(model, n_splits=3, seed=42)

        # Check that original model has no coef_ before training
        assert not hasattr(model, "coef_")

        trainer.fit_predict_oof(X, y)
        trainer.fit_final(X, y)

        # Original model should still not have coef_
        assert not hasattr(model, "coef_")

    def test_reproducibility(self, sample_data):
        """Test that results are reproducible with same seed."""
        X, y = sample_data
        model1 = Ridge(alpha=1.0)
        model2 = Ridge(alpha=1.0)

        trainer1 = StackingTrainer(model1, n_splits=3, seed=42)
        trainer2 = StackingTrainer(model2, n_splits=3, seed=42)

        oof1, scores1 = trainer1.fit_predict_oof(X, y)
        oof2, scores2 = trainer2.fit_predict_oof(X, y)

        np.testing.assert_array_almost_equal(oof1, oof2)
        np.testing.assert_array_almost_equal(scores1, scores2)
