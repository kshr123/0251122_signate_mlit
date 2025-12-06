"""Stacking trainer for ensemble learning.

内部実装:
  - CVループ: CVRunner
  - 1 fold学習: SklearnFoldTrainer
"""

from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

from evaluation.metrics import calculate_mape
from training.core import CVRunner
from training.fold_trainers import SklearnFoldTrainer


class StackingTrainer:
    """スタッキングのCV学習を行うトレーナー

    内部実装はCVRunnerとSklearnFoldTrainerに委譲。

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
        self.scoring_func = scoring_func or calculate_mape
        self.fitted_model_: Any = None

    def fit_predict_oof(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_splits: list[tuple] | None = None,
    ) -> tuple[np.ndarray, list[float]]:
        """CVでOOF予測を作成し、fold別スコアも返す

        Parameters
        ----------
        X : np.ndarray
            特徴量行列 (n_samples, n_features)
        y : np.ndarray
            目的変数 (n_samples,)
        cv_splits : list[tuple], optional
            CV分割インデックス。Noneの場合はKFoldで作成

        Returns
        -------
        oof_predictions : np.ndarray
            OOF予測値 (n_samples,)
        fold_scores : list[float]
            fold別スコア
        """
        # CV分割
        if cv_splits is None:
            kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            cv_splits = list(kfold.split(X))

        # FoldTrainer作成
        fold_trainer = SklearnFoldTrainer(model_template=self.meta_model)

        # CVRunner実行
        runner = CVRunner()
        cv_result = runner.run(
            X=X,
            y=y,
            cv_splits=cv_splits,
            fold_trainer=fold_trainer,
            scorer=self.scoring_func,
            callbacks=[],
        )

        return cv_result.oof_predictions, cv_result.fold_scores

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
