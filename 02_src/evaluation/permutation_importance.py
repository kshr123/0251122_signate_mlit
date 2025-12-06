"""
Permutation Importance モジュール

モデル非依存のPermutation Importance計算。
LightGBM, XGBoost, NN等あらゆるモデルで使用可能。

使用方法:
    from evaluation.permutation_importance import calculate_permutation_importance

    # シンプルな使い方
    result_df = calculate_permutation_importance(
        model=model,
        X=X_train,
        y=y_train,
        feature_names=features,
        scoring="mape",
        sample_size=10000,  # サンプリングで高速化
        inverse_transform=np.expm1,  # log1p予測→原スケール
    )
    result_df.write_csv("pi_result.csv")

    # クラスベースの使い方（従来互換）
    calculator = PermutationImportanceCalculator(scoring="mape", n_repeats=5)
    result = calculator.calculate(predict_fn=model.predict, X=X, y=y, feature_names=features)
    df = result.to_dataframe()
"""

from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import polars as pl


# =============================================================================
# シンプルな関数インターフェース（推奨）
# =============================================================================

def calculate_permutation_importance(
    model,
    X: Union[np.ndarray, pl.DataFrame],
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    scoring: str = "mape",
    n_repeats: int = 1,
    sample_size: Optional[int] = None,
    seed: int = 42,
    inverse_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """Permutation Importanceを計算（シンプル版）

    Args:
        model: predict()メソッドを持つ学習済みモデル
        X: 特徴量行列 (numpy or polars)
        y: 目標値（original scale）
        feature_names: 特徴量名リスト（Noneならカラム名or連番）
        scoring: "mape", "mae", "mse", "rmse"
        n_repeats: シャッフル回数
        sample_size: サンプルサイズ（Noneなら全件）
        seed: 乱数シード
        inverse_transform: 予測値の逆変換（例: np.expm1）
        verbose: 進捗表示

    Returns:
        DataFrame: feature, importance_mean, importance_std
    """
    rng = np.random.default_rng(seed)

    # Polars → numpy
    if isinstance(X, pl.DataFrame):
        if feature_names is None:
            feature_names = list(X.columns)
        X = X.to_numpy()

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    n_samples, n_features = X.shape

    # サンプリング
    if sample_size and sample_size < n_samples:
        idx = rng.choice(n_samples, size=sample_size, replace=False)
        X_eval = X[idx].copy()
        y_eval = y[idx]
        if verbose:
            print(f"Sampled {sample_size:,} / {n_samples:,} samples")
    else:
        X_eval = X.copy()
        y_eval = y

    # スコアリング関数
    scoring_fn = _get_scoring_fn(scoring)

    # ベースラインスコア
    pred_base = model.predict(X_eval)
    if inverse_transform:
        pred_base = inverse_transform(pred_base)
    base_score = scoring_fn(y_eval, pred_base)

    if verbose:
        print(f"Baseline {scoring.upper()}: {base_score:.4f}")
        print(f"Calculating PI for {n_features} features (repeats={n_repeats})...")

    # 各特徴量のimportance計算
    results = []
    for i, feat_name in enumerate(feature_names):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_eval.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            pred_perm = model.predict(X_perm)
            if inverse_transform:
                pred_perm = inverse_transform(pred_perm)
            score = scoring_fn(y_eval, pred_perm)
            scores.append(score - base_score)

        results.append({
            "feature": feat_name,
            "importance_mean": np.mean(scores),
            "importance_std": np.std(scores) if n_repeats > 1 else 0.0,
        })

        if verbose and (i + 1) % 50 == 0:
            print(f"  {i + 1} / {n_features} done")

    if verbose:
        print("Done!")

    return pl.DataFrame(results).sort("importance_mean", descending=True)


def _get_scoring_fn(scoring: str) -> Callable:
    """スコアリング関数を取得（値が大きいほど悪い）"""
    def mape(y_true, y_pred):
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def rmse(y_true, y_pred):
        return np.sqrt(mse(y_true, y_pred))

    return {"mape": mape, "mae": mae, "mse": mse, "rmse": rmse}[scoring]


# =============================================================================
# クラスベースインターフェース（従来互換）
# =============================================================================

@dataclass
class PermutationImportanceResult:
    """Permutation Importance計算結果"""

    feature_names: List[str]
    importances_mean: np.ndarray  # 平均重要度
    importances_std: np.ndarray   # 標準偏差
    importances_raw: np.ndarray   # 生データ (n_features, n_repeats)
    baseline_score: float         # シャッフルなしのスコア
    scoring: str                  # 使用したスコアリング

    def to_dataframe(self, normalize: bool = False) -> pl.DataFrame:
        """
        結果をDataFrameに変換

        Args:
            normalize: Trueの場合、重要度を正規化（合計=1.0）

        Returns:
            pl.DataFrame with columns:
                - feature: 特徴量名
                - importance: 平均重要度
                - importance_std: 標準偏差
                - importance_pct: 重要度の割合（%）
        """
        importances = self.importances_mean.copy()

        if normalize:
            total = importances.sum()
            if total > 0:
                importances = importances / total

        total = self.importances_mean.sum()
        importance_pct = (self.importances_mean / total * 100) if total > 0 else np.zeros_like(importances)

        df = pl.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
            "importance_std": self.importances_std,
            "importance_pct": importance_pct,
        })

        return df.sort("importance", descending=True)

    def get_safe_to_remove_features(
        self,
        threshold: float = 0.0,
        consider_std: bool = True
    ) -> List[str]:
        """
        削除しても精度に影響がない特徴量を取得

        Args:
            threshold: 重要度の閾値（この値以下を削除候補）
            consider_std: Trueの場合、mean + std で判定（より保守的）

        Returns:
            削除候補の特徴量名リスト
        """
        if consider_std:
            # mean + std が閾値以下なら安全
            upper_bound = self.importances_mean + self.importances_std
            mask = upper_bound <= threshold
        else:
            mask = self.importances_mean <= threshold

        return [f for f, m in zip(self.feature_names, mask) if m]

    def get_cumulative_importance_features(
        self,
        threshold_pct: float = 95.0
    ) -> List[str]:
        """
        累積重要度で上位N%に入る特徴量を取得

        Args:
            threshold_pct: 累積重要度の閾値（%）

        Returns:
            上位の特徴量名リスト（重要度順）
        """
        # 重要度でソート
        sorted_indices = np.argsort(self.importances_mean)[::-1]
        sorted_importances = self.importances_mean[sorted_indices]
        sorted_names = [self.feature_names[i] for i in sorted_indices]

        # 累積重要度
        total = sorted_importances.sum()
        if total <= 0:
            return sorted_names

        cumsum = np.cumsum(sorted_importances) / total * 100

        # 閾値を超える最初のインデックス
        n_features = np.searchsorted(cumsum, threshold_pct) + 1

        return sorted_names[:n_features]


class PermutationImportanceCalculator:
    """
    モデル非依存のPermutation Importance計算

    任意のpredict関数を受け取り、特徴量をシャッフルして
    精度変化を測定する。
    """

    SUPPORTED_SCORINGS = ["mape", "mae", "mse", "rmse", "r2"]

    def __init__(
        self,
        scoring: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mape",
        n_repeats: int = 5,
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        """
        Args:
            scoring: スコアリング関数（文字列または関数）
                     文字列: "mape", "mae", "mse", "rmse", "r2"
                     関数: (y_true, y_pred) -> float（大きいほど良い形式）
            n_repeats: シャッフル回数
            random_state: 乱数シード
            n_jobs: 並列数（現在は未使用、将来の拡張用）
        """
        if callable(scoring):
            self.scoring = "custom"
            self._scoring_fn = scoring
        else:
            if scoring not in self.SUPPORTED_SCORINGS:
                raise ValueError(
                    f"scoring must be one of {self.SUPPORTED_SCORINGS} or callable, got '{scoring}'"
                )
            self.scoring = scoring
            self._scoring_fn = self._get_scoring_function(scoring)

        self.n_repeats = n_repeats
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _get_scoring_function(self, scoring: str) -> Callable:
        """スコアリング関数を取得（値が大きいほど良い形式に統一）"""

        def mape_score(y_true, y_pred):
            """MAPE (負値、大きいほど良い)"""
            mask = y_true != 0
            if mask.sum() == 0:
                return 0.0
            return -np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

        def mae_score(y_true, y_pred):
            """MAE (負値、大きいほど良い)"""
            return -np.mean(np.abs(y_true - y_pred))

        def mse_score(y_true, y_pred):
            """MSE (負値、大きいほど良い)"""
            return -np.mean((y_true - y_pred) ** 2)

        def rmse_score(y_true, y_pred):
            """RMSE (負値、大きいほど良い)"""
            return -np.sqrt(np.mean((y_true - y_pred) ** 2))

        def r2_score(y_true, y_pred):
            """R2 (正値、大きいほど良い)"""
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            if ss_tot == 0:
                return 0.0
            return 1 - (ss_res / ss_tot)

        scoring_map = {
            "mape": mape_score,
            "mae": mae_score,
            "mse": mse_score,
            "rmse": rmse_score,
            "r2": r2_score,
        }

        return scoring_map[scoring]

    def calculate(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X: Union[np.ndarray, pl.DataFrame],
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        target_features: Optional[List[str]] = None,
    ) -> PermutationImportanceResult:
        """
        Permutation Importanceを計算

        Args:
            predict_fn: 予測関数 (X -> y_pred)
            X: 特徴量 (numpy array or polars DataFrame)
            y: 目的変数
            feature_names: 特徴量名（Noneの場合は自動生成）
            target_features: PI計算対象の特徴量名リスト（指定時はこれのみ計算）

        Returns:
            PermutationImportanceResult
        """
        # Polars → NumPy
        if isinstance(X, pl.DataFrame):
            if feature_names is None:
                feature_names = list(X.columns)
            X = X.to_numpy()

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        n_samples, n_features = X.shape

        # 計算対象の特徴量インデックスを特定
        if target_features is not None:
            target_indices = [feature_names.index(f) for f in target_features if f in feature_names]
            calc_feature_names = [feature_names[i] for i in target_indices]
        else:
            target_indices = list(range(n_features))
            calc_feature_names = feature_names

        # ベースラインスコア
        y_pred_baseline = predict_fn(X)
        baseline_score = self._scoring_fn(y, y_pred_baseline)

        # 乱数生成器
        rng = np.random.RandomState(self.random_state)

        # 各特徴量の重要度を計算
        importances_raw = np.zeros((len(target_indices), self.n_repeats))

        for i, feat_idx in enumerate(target_indices):
            for repeat_idx in range(self.n_repeats):
                # 特徴量をシャッフル
                X_permuted = X.copy()
                X_permuted[:, feat_idx] = rng.permutation(X[:, feat_idx])

                # シャッフル後の予測
                y_pred_permuted = predict_fn(X_permuted)
                permuted_score = self._scoring_fn(y, y_pred_permuted)

                # 重要度 = ベースライン - シャッフル後
                # スコアが大きいほど良いので、ベースラインからの低下量
                importances_raw[i, repeat_idx] = baseline_score - permuted_score

        # 統計量
        importances_mean = importances_raw.mean(axis=1)
        importances_std = importances_raw.std(axis=1)

        return PermutationImportanceResult(
            feature_names=list(calc_feature_names),
            importances_mean=importances_mean,
            importances_std=importances_std,
            importances_raw=importances_raw,
            baseline_score=baseline_score,
            scoring=self.scoring,
        )

    def calculate_from_models(
        self,
        models: List,
        X: Union[np.ndarray, pl.DataFrame],
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        aggregate: str = "mean",
    ) -> PermutationImportanceResult:
        """
        複数モデル（CV fold）からPermutation Importanceを計算

        Args:
            models: 学習済みモデルのリスト（各モデルは.predict()を持つ）
            X: 特徴量
            y: 目的変数
            feature_names: 特徴量名
            aggregate: 集約方法 ("mean" or "median")

        Returns:
            PermutationImportanceResult（集約済み）
        """
        results = []

        for model in models:
            result = self.calculate(
                predict_fn=model.predict,
                X=X,
                y=y,
                feature_names=feature_names,
            )
            results.append(result)

        # 結果を集約
        all_importances = np.stack([r.importances_mean for r in results])

        if aggregate == "mean":
            aggregated_mean = all_importances.mean(axis=0)
        else:  # median
            aggregated_mean = np.median(all_importances, axis=0)

        aggregated_std = all_importances.std(axis=0)

        # 生データは最初の結果を使用（参考用）
        return PermutationImportanceResult(
            feature_names=results[0].feature_names,
            importances_mean=aggregated_mean,
            importances_std=aggregated_std,
            importances_raw=results[0].importances_raw,
            baseline_score=np.mean([r.baseline_score for r in results]),
            scoring=self.scoring,
        )
