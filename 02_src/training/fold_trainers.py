"""FoldTrainer: 1 fold学習を担当するクラス群

Strategy Patternで異なるモデル/学習方法を切り替え可能にする。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.base import clone

from training.transforms import IdentityTransform, TargetTransform


@dataclass
class FoldResult:
    """1 fold学習の結果"""

    predictions: np.ndarray
    model: Any
    feature_importance: np.ndarray | None = None
    best_iteration: int | None = None


class FoldTrainer(ABC):
    """1 fold学習の抽象クラス（Strategy Pattern）

    異なるモデル（LightGBM, XGBoost, sklearn等）に対応するため、
    具体的な学習ロジックはサブクラスで実装する。
    """

    @abstractmethod
    def train_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> FoldResult:
        """1 foldの学習を実行

        Args:
            X_train: 学習用特徴量
            y_train: 学習用ターゲット（元スケール）
            X_val: 検証用特徴量
            y_val: 検証用ターゲット（元スケール）

        Returns:
            FoldResult: 予測値、モデル、特徴量重要度等
        """
        pass


class LightGBMFoldTrainer(FoldTrainer):
    """LightGBM用FoldTrainer

    特徴:
      - early stopping対応
      - ターゲット変換対応（log1p等）
      - カテゴリカル特徴量対応
      - サンプル重み対応
    """

    def __init__(
        self,
        params: dict,
        target_transform: TargetTransform | None = None,
        early_stopping_rounds: int = 200,
        categorical_features: list[int] | str | None = None,
        sample_weight: np.ndarray | None = None,
        verbose: bool = False,
        extra_callbacks: list | None = None,
    ):
        """初期化

        Args:
            params: LightGBMのパラメータ
            target_transform: ターゲット変換（None時はIdentityTransform）
            early_stopping_rounds: early stoppingのラウンド数
            categorical_features: カテゴリカル特徴量のインデックスまたは"auto"
            sample_weight: サンプル重み（全データ分、fold内で分割される）
            verbose: LightGBMの学習ログを表示するか
            extra_callbacks: 追加のLightGBMコールバック
        """
        self.params = params
        self.transform = target_transform or IdentityTransform()
        self.early_stopping_rounds = early_stopping_rounds
        self.categorical_features = categorical_features
        self.sample_weight = sample_weight
        self.verbose = verbose
        self.extra_callbacks = extra_callbacks or []

        # 学習時にfoldのインデックスを受け取るための属性
        self._current_train_idx: np.ndarray | None = None

    def set_train_idx(self, train_idx: np.ndarray) -> None:
        """現在のfoldの学習インデックスを設定（sample_weight用）"""
        self._current_train_idx = train_idx

    def train_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> FoldResult:
        """1 foldの学習を実行

        Args:
            X_train: 学習用特徴量
            y_train: 学習用ターゲット
            X_val: 検証用特徴量（Noneで全データ学習）
            y_val: 検証用ターゲット（Noneで全データ学習）

        Returns:
            FoldResult: 予測値（validation無し時はNone）、モデル、特徴量重要度等
        """
        # ターゲット変換
        y_train_t = self.transform.transform(y_train)

        # DataFrameに変換（LightGBMはDataFrame推奨）
        X_train_df = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train

        # モデル作成
        model = LGBMRegressor(**self.params)

        # サンプル重みの取得
        fold_weight = None
        if self.sample_weight is not None and self._current_train_idx is not None:
            fold_weight = self.sample_weight[self._current_train_idx]

        # Full data training（validation無し）
        if X_val is None or y_val is None:
            model.fit(
                X_train_df,
                y_train_t,
                categorical_feature=self.categorical_features,
                sample_weight=fold_weight,
            )
            return FoldResult(
                predictions=None,
                model=model,
                feature_importance=model.feature_importances_,
                best_iteration=model.n_estimators,
            )

        # CV training（validation有り）
        y_val_t = self.transform.transform(y_val)
        X_val_df = pd.DataFrame(X_val) if not isinstance(X_val, pd.DataFrame) else X_val

        # コールバック構築
        callbacks = [early_stopping(self.early_stopping_rounds, verbose=self.verbose)]
        if self.verbose:
            callbacks.append(log_evaluation(period=self.early_stopping_rounds))
        callbacks.extend(self.extra_callbacks)

        # 学習
        model.fit(
            X_train_df,
            y_train_t,
            eval_set=[(X_val_df, y_val_t)],
            callbacks=callbacks,
            categorical_feature=self.categorical_features,
            sample_weight=fold_weight,
        )

        # 予測（変換後スケール）
        pred_t = model.predict(X_val_df)

        # 逆変換して元スケールに戻す
        pred = self.transform.inverse_transform(pred_t)

        return FoldResult(
            predictions=pred,
            model=model,
            feature_importance=model.feature_importances_,
            best_iteration=model.best_iteration_,
        )


class LightGBMClassifierFoldTrainer(FoldTrainer):
    """LightGBM分類器用FoldTrainer

    特徴:
      - early stopping対応
      - predict_proba()で確率を返す
      - 不均衡データ対応（class_weight）
    """

    def __init__(
        self,
        params: dict,
        early_stopping_rounds: int = 200,
        categorical_features: list[int] | str | None = None,
        verbose: bool = False,
        extra_callbacks: list | None = None,
    ):
        """初期化

        Args:
            params: LightGBMのパラメータ
            early_stopping_rounds: early stoppingのラウンド数
            categorical_features: カテゴリカル特徴量のインデックスまたは"auto"
            verbose: LightGBMの学習ログを表示するか
            extra_callbacks: 追加のLightGBMコールバック
        """
        self.params = params
        self.early_stopping_rounds = early_stopping_rounds
        self.categorical_features = categorical_features
        self.verbose = verbose
        self.extra_callbacks = extra_callbacks or []

    def train_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> FoldResult:
        """1 foldの学習を実行

        Args:
            X_train: 学習用特徴量
            y_train: 学習用ターゲット（0/1のバイナリ）
            X_val: 検証用特徴量（Noneで全データ学習）
            y_val: 検証用ターゲット（Noneで全データ学習）

        Returns:
            FoldResult: 予測確率（正例）、モデル、特徴量重要度等
        """
        # DataFrameに変換（LightGBMはDataFrame推奨）
        X_train_df = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train

        # モデル作成
        model = LGBMClassifier(**self.params)

        # Full data training（validation無し）
        if X_val is None or y_val is None:
            model.fit(
                X_train_df,
                y_train,
                categorical_feature=self.categorical_features,
            )
            return FoldResult(
                predictions=None,
                model=model,
                feature_importance=model.feature_importances_,
                best_iteration=model.n_estimators,
            )

        # CV training（validation有り）
        X_val_df = pd.DataFrame(X_val) if not isinstance(X_val, pd.DataFrame) else X_val

        # コールバック構築
        callbacks = [early_stopping(self.early_stopping_rounds, verbose=self.verbose)]
        if self.verbose:
            callbacks.append(log_evaluation(period=self.early_stopping_rounds))
        callbacks.extend(self.extra_callbacks)

        # 学習
        model.fit(
            X_train_df,
            y_train,
            eval_set=[(X_val_df, y_val)],
            callbacks=callbacks,
            categorical_feature=self.categorical_features,
        )

        # 予測（正例の確率を返す）
        pred_proba = model.predict_proba(X_val_df)[:, 1]

        return FoldResult(
            predictions=pred_proba,
            model=model,
            feature_importance=model.feature_importances_,
            best_iteration=model.best_iteration_,
        )


class SklearnFoldTrainer(FoldTrainer):
    """sklearn互換モデル用FoldTrainer

    スタッキングやシンプルなモデル（Ridge, RandomForest等）に使用。
    sklearn.base.clone()でモデルを複製して学習する。
    """

    def __init__(self, model_template: Any):
        """初期化

        Args:
            model_template: クローン元のモデル（fit前の状態）
        """
        self.model_template = model_template

    def train_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> FoldResult:
        """1 foldの学習を実行"""
        # モデルをクローンして学習
        model = clone(self.model_template)
        model.fit(X_train, y_train)

        # 予測
        pred = model.predict(X_val)

        # 特徴量重要度（存在する場合のみ）
        fi = None
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
        elif hasattr(model, "coef_"):
            fi = np.abs(model.coef_)

        return FoldResult(
            predictions=pred,
            model=model,
            feature_importance=fi,
        )


class CatBoostFoldTrainer(FoldTrainer):
    """CatBoost用FoldTrainer

    特徴:
      - early stopping対応
      - ターゲット変換対応（log1p等）
      - カテゴリカル特徴量の自動処理（cat_features）
    """

    def __init__(
        self,
        params: dict,
        target_transform: TargetTransform | None = None,
        early_stopping_rounds: int = 100,
        cat_features: list[int] | list[str] | None = None,
        verbose: int = 100,
    ):
        """初期化

        Args:
            params: CatBoostのパラメータ
            target_transform: ターゲット変換（None時はIdentityTransform）
            early_stopping_rounds: early stoppingのラウンド数
            cat_features: カテゴリカル特徴量のインデックスまたは名前リスト
            verbose: ログ出力間隔（0で非表示）
        """
        self.params = params.copy()
        self.transform = target_transform or IdentityTransform()
        self.early_stopping_rounds = early_stopping_rounds
        self.cat_features = cat_features
        self.verbose = verbose

    def train_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> FoldResult:
        """1 foldの学習を実行

        Args:
            X_train: 学習用特徴量
            y_train: 学習用ターゲット
            X_val: 検証用特徴量（Noneで全データ学習）
            y_val: 検証用ターゲット（Noneで全データ学習）

        Returns:
            FoldResult: 予測値（validation無し時はNone）、モデル、特徴量重要度等
        """
        # ターゲット変換
        y_train_t = self.transform.transform(y_train)

        # CatBoost Pool作成
        train_pool = Pool(
            data=X_train,
            label=y_train_t,
            cat_features=self.cat_features,
        )

        # モデル作成
        params = {k: v for k, v in self.params.items() if k != "verbose"}
        model = CatBoostRegressor(**params, verbose=self.verbose)

        # Full data training（validation無し）
        if X_val is None or y_val is None:
            model.fit(train_pool)
            return FoldResult(
                predictions=None,
                model=model,
                feature_importance=model.get_feature_importance(),
                best_iteration=params.get("iterations", 1000),
            )

        # CV training（validation有り）
        y_val_t = self.transform.transform(y_val)
        val_pool = Pool(
            data=X_val,
            label=y_val_t,
            cat_features=self.cat_features,
        )

        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=self.early_stopping_rounds,
            use_best_model=True,
        )

        # 予測（変換後スケール）
        pred_t = model.predict(X_val)

        # 逆変換して元スケールに戻す
        pred = self.transform.inverse_transform(pred_t)

        return FoldResult(
            predictions=pred,
            model=model,
            feature_importance=model.get_feature_importance(),
            best_iteration=model.get_best_iteration(),
        )
