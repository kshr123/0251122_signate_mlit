"""CV分割モジュール（Strategy Pattern）

クロスバリデーションの分割戦略を提供。
- KFoldSplitter: 標準KFold（時系列以外のデータ向け）
- TimeSeriesCVSplitter: 時系列Expanding Window CV

Examples:
    >>> # KFold
    >>> splitter = KFoldSplitter(n_splits=5, random_state=42)
    >>> cv_splits = splitter.split(X)
    >>> for train_idx, val_idx in cv_splits:
    ...     X_train, X_val = X[train_idx], X[val_idx]

    >>> # Time Series CV
    >>> splitter = TimeSeriesCVSplitter(
    ...     time_column="target_ym",
    ...     val_periods=[202101, 202107, 202201, 202207],
    ... )
    >>> cv_splits = splitter.split(df)  # DataFrameを渡す
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import polars as pl
from sklearn.model_selection import KFold, StratifiedKFold


class CVSplitter(ABC):
    """CV分割の抽象クラス（Strategy Pattern）"""

    @abstractmethod
    def split(self, X, y=None, groups=None) -> list[tuple]:
        """データを分割

        Args:
            X: 特徴量
            y: ターゲット（Stratified用、オプション）
            groups: グループ（GroupKFold用、オプション）

        Returns:
            [(train_idx, val_idx), ...] のリスト
        """
        pass

    @property
    @abstractmethod
    def n_splits(self) -> int:
        """分割数を取得"""
        pass


class KFoldSplitter(CVSplitter):
    """標準KFold分割

    Args:
        n_splits: 分割数（デフォルト: 5）
        shuffle: シャッフルするか（デフォルト: True）
        random_state: 乱数シード（デフォルト: 42）
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        self._kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self._n_splits = n_splits

    def split(self, X, y=None, groups=None) -> list[tuple]:
        """データを分割

        Args:
            X: 特徴量（配列またはDataFrame）
            y: 未使用（互換性のため）
            groups: 未使用（互換性のため）

        Returns:
            [(train_idx, val_idx), ...] のリスト
        """
        return list(self._kf.split(X))

    @property
    def n_splits(self) -> int:
        return self._n_splits


class StratifiedKFoldSplitter(CVSplitter):
    """層化KFold分割（分類タスク用）

    クラス比率を維持しながらデータを分割する。
    不均衡データの分類タスクに適している。

    Args:
        n_splits: 分割数（デフォルト: 5）
        shuffle: シャッフルするか（デフォルト: True）
        random_state: 乱数シード（デフォルト: 42）
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        self._skf = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
        self._n_splits = n_splits

    def split(self, X, y=None, groups=None) -> list[tuple]:
        """データを層化分割

        Args:
            X: 特徴量（配列またはDataFrame）
            y: ターゲット（層化に使用、必須）
            groups: 未使用（互換性のため）

        Returns:
            [(train_idx, val_idx), ...] のリスト

        Raises:
            ValueError: yがNoneの場合
        """
        if y is None:
            raise ValueError("StratifiedKFoldSplitter requires y for stratification")
        return list(self._skf.split(X, y))

    @property
    def n_splits(self) -> int:
        return self._n_splits


class TimeSeriesCVSplitter(CVSplitter):
    """時系列Expanding Window CV分割

    過去データで学習し、未来データで検証する時系列CV。
    各fold では val_period より前の全データで学習し、
    val_period のデータで検証する。

    Args:
        time_column: 時間カラム名（デフォルト: "target_ym"）
        val_periods: 検証期間のリスト（例: [202101, 202107, 202201, 202207]）

    Examples:
        >>> splitter = TimeSeriesCVSplitter(
        ...     time_column="target_ym",
        ...     val_periods=[202101, 202107, 202201, 202207],
        ... )
        >>> # Fold 0: train < 202101, val == 202101
        >>> # Fold 1: train < 202107, val == 202107
        >>> # Fold 2: train < 202201, val == 202201
        >>> # Fold 3: train < 202207, val == 202207
    """

    def __init__(
        self,
        time_column: str = "target_ym",
        val_periods: Optional[list[int]] = None,
    ):
        self._time_column = time_column
        self._val_periods = val_periods or []

    def split(self, X, y=None, groups=None) -> list[tuple]:
        """データを時系列分割

        Args:
            X: 特徴量（DataFrameまたはtime_columnを含む配列）
            y: 未使用（互換性のため）
            groups: 未使用（互換性のため）

        Returns:
            [(train_idx, val_idx), ...] のリスト
        """
        # Polars DataFrameの場合
        if isinstance(X, pl.DataFrame):
            time_values = X[self._time_column].to_numpy()
        # NumPy配列や辞書の場合
        elif hasattr(X, "__getitem__"):
            time_values = np.asarray(X[self._time_column])
        else:
            raise ValueError(f"X must be a DataFrame or have __getitem__, got {type(X)}")

        splits = []
        for val_period in self._val_periods:
            train_mask = time_values < val_period
            val_mask = time_values == val_period
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            splits.append((train_idx, val_idx))

        return splits

    @property
    def n_splits(self) -> int:
        return len(self._val_periods)

    @property
    def val_periods(self) -> list[int]:
        """検証期間リストを取得"""
        return self._val_periods


def create_cv_splitter(config: dict) -> CVSplitter:
    """設定からCVSplitterを作成するファクトリ関数

    Args:
        config: CV設定辞書
            strategy: "kfold", "stratified", or "time_series"
            kfold: {n_splits, shuffle, random_state}
            stratified: {n_splits, shuffle, random_state}
            time_series: {time_column, val_periods}

    Returns:
        CVSplitterインスタンス

    Examples:
        >>> config = {
        ...     "strategy": "stratified",
        ...     "stratified": {"n_splits": 5, "random_state": 42},
        ... }
        >>> splitter = create_cv_splitter(config)
    """
    strategy = config.get("strategy", "kfold")

    if strategy == "kfold":
        kfold_config = config.get("kfold", {})
        return KFoldSplitter(
            n_splits=kfold_config.get("n_splits", 5),
            shuffle=kfold_config.get("shuffle", True),
            random_state=kfold_config.get("random_state", 42),
        )
    elif strategy == "stratified":
        stratified_config = config.get("stratified", {})
        return StratifiedKFoldSplitter(
            n_splits=stratified_config.get("n_splits", 5),
            shuffle=stratified_config.get("shuffle", True),
            random_state=stratified_config.get("random_state", 42),
        )
    elif strategy == "time_series":
        ts_config = config.get("time_series", {})
        return TimeSeriesCVSplitter(
            time_column=ts_config.get("time_column", "target_ym"),
            val_periods=ts_config.get("val_periods", []),
        )
    else:
        raise ValueError(f"Unknown CV strategy: {strategy}")
