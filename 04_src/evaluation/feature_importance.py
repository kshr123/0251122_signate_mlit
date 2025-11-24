"""
特徴量重要度分析モジュール

LightGBMモデルの特徴量重要度を計算・分析する。
- Gain/Split重要度
- Permutation Importance
- 複数タイプの比較
"""

from typing import List, Optional

import numpy as np
import polars as pl
from sklearn.inspection import permutation_importance


class FeatureImportanceAnalyzer:
    """
    特徴量重要度の計算と分析

    Attributes:
        importance_df: 最後に計算した重要度のDataFrame
    """

    def __init__(self):
        self.importance_df: Optional[pl.DataFrame] = None

    def calculate_importance(
        self,
        model,  # LightGBM Booster
        feature_names: List[str],
        importance_type: str = "gain",
    ) -> pl.DataFrame:
        """
        特徴量重要度を計算

        Args:
            model: 学習済みLightGBMモデル
            feature_names: 特徴量名のリスト
            importance_type: "gain" or "split"

        Returns:
            pl.DataFrame with columns: ["feature", "importance", "type"]
            - feature: 特徴量名
            - importance: 重要度（正規化済み、合計=1.0）
            - type: 重要度タイプ（"gain" or "split"）

        Raises:
            ValueError: importance_typeが不正な場合
        """
        if importance_type not in ["gain", "split"]:
            raise ValueError(
                f"importance_type must be 'gain' or 'split', got '{importance_type}'"
            )

        # LightGBMから重要度取得
        importances = model.feature_importance(importance_type=importance_type)

        # 正規化（合計=1.0）
        total = importances.sum()
        if total > 0:
            importances_normalized = importances / total
        else:
            importances_normalized = importances

        # DataFrame作成
        self.importance_df = pl.DataFrame(
            {
                "feature": feature_names,
                "importance": importances_normalized,
                "type": [importance_type] * len(feature_names),
            }
        )

        return self.importance_df

    def calculate_permutation_importance(
        self,
        model,
        X: pl.DataFrame,
        y: np.ndarray,
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> pl.DataFrame:
        """
        Permutation Importanceを計算

        Args:
            model: 学習済みLightGBMモデル
            X: 特徴量（Polars DataFrame）
            y: 目的変数
            n_repeats: シャッフル回数
            random_state: 乱数シード

        Returns:
            pl.DataFrame with columns: ["feature", "importance", "type"]
            - importance: 平均importances（正規化済み）
            - type: "permutation"
        """
        # Polars → NumPy
        X_np = X.to_numpy()

        # LightGBM Boosterのラッパークラス（sklearn互換）
        class LGBMWrapper:
            def __init__(self, booster):
                self.booster = booster

            def predict(self, X):
                return self.booster.predict(X)

            def fit(self, X, y):
                # ダミー実装（permutation_importanceの要件を満たすため）
                return self

        # ラッパーでモデルを包む
        wrapped_model = LGBMWrapper(model)

        # Permutation Importance計算
        result = permutation_importance(
            wrapped_model,
            X_np,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="neg_mean_absolute_percentage_error",
        )

        # 重要度取得（平均値）
        importances = result.importances_mean

        # 正規化
        total = importances.sum()
        if total > 0:
            importances_normalized = importances / total
        else:
            importances_normalized = importances

        # DataFrame作成
        self.importance_df = pl.DataFrame(
            {
                "feature": X.columns,
                "importance": importances_normalized,
                "type": ["permutation"] * len(X.columns),
            }
        )

        return self.importance_df

    def get_top_features(
        self,
        n: int = 20,
    ) -> pl.DataFrame:
        """
        重要度上位N件の特徴量を取得

        Args:
            n: 取得する特徴量数

        Returns:
            pl.DataFrame（importanceで降順ソート済み）

        Raises:
            RuntimeError: calculate_importance未実行の場合
        """
        if self.importance_df is None:
            raise RuntimeError(
                "calculate_importance() or calculate_permutation_importance() "
                "must be called first"
            )

        # 降順ソート → 上位N件
        return self.importance_df.sort("importance", descending=True).head(n)

    def compare_importance_types(
        self,
        model,
        feature_names: List[str],
        X: Optional[pl.DataFrame] = None,
        y: Optional[np.ndarray] = None,
    ) -> pl.DataFrame:
        """
        複数タイプの重要度を比較

        Args:
            model: 学習済みモデル
            feature_names: 特徴量名リスト
            X: Permutation用（Noneの場合はgain/splitのみ）
            y: Permutation用

        Returns:
            pl.DataFrame with columns: ["feature", "gain", "split", "permutation"?]
            - 各列は正規化済み重要度
            - permutationはX/yが与えられた場合のみ
        """
        # Gain重要度
        gain_df = self.calculate_importance(
            model, feature_names, importance_type="gain"
        )
        gain_df = gain_df.select(["feature", pl.col("importance").alias("gain")])

        # Split重要度
        split_df = self.calculate_importance(
            model, feature_names, importance_type="split"
        )
        split_df = split_df.select(
            ["feature", pl.col("importance").alias("split")]
        )

        # Merge
        comparison = gain_df.join(split_df, on="feature", how="left")

        # Permutation Importance（オプション）
        if X is not None and y is not None:
            perm_df = self.calculate_permutation_importance(model, X, y)
            perm_df = perm_df.select(
                ["feature", pl.col("importance").alias("permutation")]
            )
            comparison = comparison.join(perm_df, on="feature", how="left")

        return comparison
