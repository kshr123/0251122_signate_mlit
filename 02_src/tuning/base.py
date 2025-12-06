"""
BaseTuner - ハイパーパラメータチューニング基底クラス

Optunaを使用したCV評価ベースのチューニング機能を提供。
各モデル固有のチューナーはこのクラスを継承して実装する。

内部実装:
  - CVループ: CVRunner
  - プルーニング: OptunaPruningCallback
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.model_selection import KFold

from evaluation.metrics import calculate_mape
from training.core import CVRunner, PrunedException
from training.fold_trainers import FoldTrainer, FoldResult
from training.callbacks import OptunaPruningCallback


class BaseTuner(ABC):
    """ハイパーパラメータチューニング基底クラス

    内部実装はCVRunnerとOptunaPruningCallbackに委譲。
    """

    def __init__(
        self,
        param_space: dict | None = None,
        n_trials: int = 100,
        timeout: int | None = None,
        n_cv_splits: int = 3,
        random_state: int = 42,
        storage_path: Path | None = None,
        study_name: str = "tuning",
    ):
        """
        Args:
            param_space: Optuna探索空間定義（YAML形式）。Noneの場合はデフォルト使用
            n_trials: 最大試行回数
            timeout: タイムアウト秒数（None=無制限）
            n_cv_splits: CV分割数
            random_state: 乱数シード
            storage_path: SQLite保存先（None=インメモリ）
            study_name: Optuna study名（resume用）
        """
        self._param_space = param_space or self._get_default_param_space()
        self._n_trials = n_trials
        self._timeout = timeout
        self._n_cv_splits = n_cv_splits
        self._random_state = random_state
        self._storage_path = storage_path
        self._study_name = study_name

        # 結果保存用
        self._study: optuna.Study | None = None
        self._best_params: dict | None = None
        self._history: list[dict] = []

    @abstractmethod
    def _create_fold_trainer(self, params: dict) -> FoldTrainer:
        """モデル固有のFoldTrainer作成

        Args:
            params: サンプリングされたパラメータ

        Returns:
            FoldTrainerインスタンス
        """
        pass

    @abstractmethod
    def _get_default_param_space(self) -> dict:
        """モデル固有のデフォルト探索空間

        Returns:
            param_space定義辞書
        """
        pass

    def _sample_params(self, trial: optuna.Trial) -> dict:
        """param_spaceからパラメータをサンプリング

        Args:
            trial: Optuna trial

        Returns:
            サンプリングされたパラメータ辞書
        """
        params = {}
        for name, config in self._param_space.items():
            param_type = config["type"]

            if param_type == "float":
                params[name] = trial.suggest_float(
                    name,
                    config["low"],
                    config["high"],
                    log=config.get("log", False),
                )
            elif param_type == "int":
                params[name] = trial.suggest_int(
                    name,
                    config["low"],
                    config["high"],
                    log=config.get("log", False),
                )
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(name, config["choices"])
            else:
                raise ValueError(f"Unknown param type: {param_type}")

        return params

    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MAPE計算（evaluation/metrics.pyに委譲）

        Args:
            y_true: 真値
            y_pred: 予測値

        Returns:
            MAPE (%)
        """
        return calculate_mape(y_true, y_pred)

    def _objective(
        self,
        trial: optuna.Trial,
        X: np.ndarray,
        y: np.ndarray,
        cv_splits: list[tuple],
    ) -> float:
        """Optuna目的関数

        Args:
            trial: Optuna trial
            X: 特徴量
            y: ターゲット
            cv_splits: CV分割インデックス

        Returns:
            CV平均MAPE
        """
        params = self._sample_params(trial)
        fold_trainer = self._create_fold_trainer(params)

        # CVRunner + OptunaPruningCallback
        runner = CVRunner()
        callbacks = [OptunaPruningCallback(trial)]

        try:
            cv_result = runner.run(
                X=X,
                y=y,
                cv_splits=cv_splits,
                fold_trainer=fold_trainer,
                scorer=self._calculate_mape,
                callbacks=callbacks,
            )
            cv_mape = cv_result.mean_score
        except PrunedException:
            raise optuna.TrialPruned()

        # 履歴保存
        record = {"trial": trial.number, "mape": cv_mape, **params}
        self._history.append(record)

        return cv_mape

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_splits: list[tuple] | None = None,
    ) -> dict:
        """チューニング実行

        Args:
            X: 特徴量（numpy array）
            y: ターゲット（numpy array）
            cv_splits: CV分割インデックス（Noneの場合はKFoldで作成）

        Returns:
            最良パラメータ辞書
        """
        # CV分割
        if cv_splits is None:
            kf = KFold(
                n_splits=self._n_cv_splits,
                shuffle=True,
                random_state=self._random_state,
            )
            cv_splits = list(kf.split(X))

        # Storage設定
        if self._storage_path is not None:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            storage = f"sqlite:///{self._storage_path}"
        else:
            storage = None

        # Study作成または再開
        self._study = optuna.create_study(
            study_name=self._study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=self._random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        )

        # 最適化実行
        self._study.optimize(
            lambda trial: self._objective(trial, X, y, cv_splits),
            n_trials=self._n_trials,
            timeout=self._timeout,
            show_progress_bar=True,
        )

        self._best_params = self._study.best_params
        return self._best_params

    def save_results(self, output_dir: Path) -> None:
        """結果をファイルに保存

        Args:
            output_dir: 出力ディレクトリ
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # best_params.yaml
        if self._study is not None:
            best_params_path = output_dir / "best_params.yaml"
            header = f"""# Best parameters from tuning
# Trial: {self._study.best_trial.number}, Score: {self._study.best_value:.4f} (MAPE)
# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
            with open(best_params_path, "w") as f:
                f.write(header)
                yaml.dump(self._best_params, f, default_flow_style=False)

        # tuning_history.csv
        if self._history:
            history_path = output_dir / "tuning_history.csv"
            df = pd.DataFrame(self._history)
            df.to_csv(history_path, index=False)

    @property
    def best_params(self) -> dict | None:
        """最良パラメータを取得"""
        return self._best_params

    @property
    def study(self) -> optuna.Study | None:
        """Optuna Studyを取得"""
        return self._study

    # ========== 後方互換メソッド ==========
    # 既存の_create_model, _fit_and_predictを使うサブクラス用

    def _create_model(self, params: dict) -> Any:
        """後方互換: モデル作成（非推奨、_create_fold_trainerを使用）"""
        raise NotImplementedError(
            "Implement _create_fold_trainer instead of _create_model"
        )

    def _fit_and_predict(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        categorical_features: list[int] | None = None,
    ) -> np.ndarray:
        """後方互換: 学習と予測（非推奨、_create_fold_trainerを使用）"""
        raise NotImplementedError(
            "Implement _create_fold_trainer instead of _fit_and_predict"
        )
