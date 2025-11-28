"""Stacking trainer for ensemble learning."""

from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold


class StackingTrainer:
    """スタッキングのCV学習を行うトレーナー

    メタモデルはsklearn互換（fit/predictを持つ）であれば何でも使用可能。
    - sklearn.linear_model.Ridge
    - sklearn.linear_model.Lasso
    - lightgbm.LGBMRegressor
    - etc.

    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> trainer = StackingTrainer(Ridge(alpha=1.0), n_splits=3, seed=42)
    >>> oof_pred, fold_scores = trainer.fit_predict_oof(X_meta, y)
    >>> trainer.fit_final(X_meta, y)
    >>> test_pred = trainer.predict(X_test_meta)
    """

    def __init__(
        self,
        meta_model: Any,
        n_splits: int = 3,
        seed: int = 42,
        scoring_func: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ):
        """Initialize StackingTrainer.

        Parameters
        ----------
        meta_model : Any
            sklearn互換モデル（fit/predict を持つ）
        n_splits : int, default=3
            CVのfold数
        seed : int, default=42
            乱数シード
        scoring_func : Callable, optional
            スコア計算関数。signature: (y_true, y_pred) -> float
            Noneの場合はMAPEを使用
        """
        self.meta_model = meta_model
        self.n_splits = n_splits
        self.seed = seed
        self.scoring_func = scoring_func or self._mape
        self.fitted_model_: Any = None

    @staticmethod
    def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    def fit_predict_oof(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, list[float]]:
        """CVでOOF予測を作成し、fold別スコアも返す

        Parameters
        ----------
        X : np.ndarray
            特徴量行列 (n_samples, n_features)
        y : np.ndarray
            目的変数 (n_samples,)

        Returns
        -------
        oof_predictions : np.ndarray
            OOF予測値 (n_samples,)
        fold_scores : list[float]
            fold別スコア
        """
        oof_predictions = np.zeros(len(y))
        fold_scores = []

        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(X)):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            # モデルをcloneして学習（元のモデルは変更しない）
            model = clone(self.meta_model)
            model.fit(X_train, y_train)

            # OOF予測
            pred = model.predict(X_valid)
            oof_predictions[valid_idx] = pred

            # fold別スコア計算
            score = self.scoring_func(y_valid, pred)
            fold_scores.append(score)

        return oof_predictions, fold_scores

    def fit_final(self, X: np.ndarray, y: np.ndarray) -> None:
        """全データで最終モデルを学習

        Parameters
        ----------
        X : np.ndarray
            特徴量行列 (n_samples, n_features)
        y : np.ndarray
            目的変数 (n_samples,)
        """
        self.fitted_model_ = clone(self.meta_model)
        self.fitted_model_.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """テストデータの予測

        Parameters
        ----------
        X : np.ndarray
            特徴量行列 (n_samples, n_features)

        Returns
        -------
        predictions : np.ndarray
            予測値 (n_samples,)

        Raises
        ------
        ValueError
            fit_final() が呼び出されていない場合
        """
        if self.fitted_model_ is None:
            raise ValueError("fit_final() を先に呼び出してください")
        return self.fitted_model_.predict(X)

    def save(self, path: Path) -> None:
        """モデル保存（joblib使用）

        Parameters
        ----------
        path : Path
            保存先パス
        """
        if self.fitted_model_ is None:
            raise ValueError("fit_final() を先に呼び出してください")
        joblib.dump(self.fitted_model_, path)

    @staticmethod
    def load(path: Path) -> Any:
        """モデル読み込み

        Parameters
        ----------
        path : Path
            読み込み元パス

        Returns
        -------
        model : Any
            読み込んだモデル
        """
        return joblib.load(path)
