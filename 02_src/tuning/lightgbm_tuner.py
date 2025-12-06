"""
LightGBMTuner - LightGBM専用チューナー

BaseTunerを継承し、LightGBM固有のパラメータサンプリングと学習を実装。
内部ではLightGBMFoldTrainerを使用。
"""

from training.fold_trainers import FoldTrainer, LightGBMFoldTrainer
from tuning.base import BaseTuner


class LightGBMTuner(BaseTuner):
    """LightGBM専用チューナー

    内部実装はLightGBMFoldTrainerに委譲。
    """

    def __init__(
        self,
        param_space: dict | None = None,
        n_trials: int = 100,
        timeout: int | None = None,
        n_cv_splits: int = 3,
        random_state: int = 42,
        storage_path=None,
        study_name: str = "lgbm_tuning",
        # LightGBM固有
        n_estimators: int = 1000,
        early_stopping_rounds: int = 50,
        lgbm_objective: str = "regression",
        lgbm_metric: str = "l2",
    ):
        """
        Args:
            param_space: 探索空間定義
            n_trials: 試行回数
            timeout: タイムアウト秒
            n_cv_splits: CV分割数
            random_state: 乱数シード
            storage_path: SQLite保存先
            study_name: Study名
            n_estimators: 最大イテレーション数
            early_stopping_rounds: 早期終了ラウンド数
            lgbm_objective: LightGBM目的関数
            lgbm_metric: 評価指標
        """
        super().__init__(
            param_space=param_space,
            n_trials=n_trials,
            timeout=timeout,
            n_cv_splits=n_cv_splits,
            random_state=random_state,
            storage_path=storage_path,
            study_name=study_name,
        )
        self._n_estimators = n_estimators
        self._early_stopping_rounds = early_stopping_rounds
        self._lgbm_objective = lgbm_objective
        self._lgbm_metric = lgbm_metric
        self._random_state = random_state

    def _get_default_param_space(self) -> dict:
        """LightGBMデフォルト探索空間"""
        return {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "num_leaves": {"type": "int", "low": 31, "high": 255},
            "max_depth": {"type": "int", "low": 3, "high": 12},
            "min_child_samples": {"type": "int", "low": 5, "high": 100},
            "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
            "reg_alpha": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        }

    def _create_fold_trainer(self, params: dict) -> FoldTrainer:
        """LightGBMFoldTrainer作成"""
        # LightGBMパラメータ構築
        lgb_params = {
            "objective": self._lgbm_objective,
            "n_estimators": self._n_estimators,
            "random_state": self._random_state,
            "verbose": -1,
            "force_col_wise": True,
            "subsample_freq": 1,
            **params,
        }

        return LightGBMFoldTrainer(
            params=lgb_params,
            target_transform=None,  # 元スケールのまま
            early_stopping_rounds=self._early_stopping_rounds,
            verbose=False,
        )
