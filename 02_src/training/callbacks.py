"""CV学習用コールバック

Callback: コールバック基底クラス
VerboseCallback: 詳細ログ出力
OptunaPruningCallback: Optunaプルーニング対応
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import optuna


class Callback(ABC):
    """CV学習用コールバックの基底クラス

    Template Method Patternで拡張ポイントを提供。
    デフォルト実装は何もしない（オプショナルなフック）。
    """

    def on_fold_start(self, fold_idx: int) -> None:
        """fold開始時に呼ばれる

        Args:
            fold_idx: 現在のfoldインデックス（0始まり）
        """
        pass

    def on_fold_end(
        self, fold_idx: int, score: float
    ) -> Literal["continue", "prune"] | None:
        """fold終了時に呼ばれる

        Args:
            fold_idx: 現在のfoldインデックス
            score: このfoldの評価スコア

        Returns:
            "prune": CVを中断してPrunedExceptionを発生
            "continue" or None: 次のfoldへ進む
        """
        return None


class VerboseCallback(Callback):
    """学習進捗を表示するコールバック

    各fold終了時にスコアを表示する。
    """

    def __init__(self, n_splits: int, metric_name: str = "MAPE"):
        """初期化

        Args:
            n_splits: 総fold数
            metric_name: 表示するメトリクス名
        """
        self.n_splits = n_splits
        self.metric_name = metric_name
        self.scores: list[float] = []

    def on_fold_start(self, fold_idx: int) -> None:
        """fold開始時: 何もしない（静かに開始）"""
        pass

    def on_fold_end(
        self, fold_idx: int, score: float
    ) -> Literal["continue", "prune"] | None:
        """fold終了時: スコアを表示"""
        self.scores.append(score)
        print(f"  Fold {fold_idx + 1}/{self.n_splits}: {self.metric_name} = {score:.4f}%")
        return None


class OptunaPruningCallback(Callback):
    """Optunaのプルーニングを行うコールバック

    各fold終了時にOptunaにスコアを報告し、
    他のトライアルと比較して劣る場合はプルーニングする。
    """

    def __init__(self, trial: "optuna.Trial"):
        """初期化

        Args:
            trial: Optunaのトライアルオブジェクト
        """
        self.trial = trial

    def on_fold_end(
        self, fold_idx: int, score: float
    ) -> Literal["continue", "prune"] | None:
        """fold終了時: Optunaにスコアを報告し、プルーニング判定"""
        import optuna

        # スコアを報告
        self.trial.report(score, fold_idx)

        # プルーニング判定
        if self.trial.should_prune():
            return "prune"

        return None
