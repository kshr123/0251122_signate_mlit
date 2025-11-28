"""
次元圧縮Block

PCA, SVD, UMAPによる次元圧縮を行う。
"""

import polars as pl
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from features.base import BaseBlock


class DimensionReductionBlock(BaseBlock):
    """次元圧縮の基底クラス

    複数の数値カラムを低次元空間に圧縮します。

    特徴:
    - trainデータで学習したスケーラー・変換器をtestにも適用
    - 標準化はオプション（デフォルトTrue）
    - 欠損値処理: error（デフォルト）, mean, zero

    Args:
        columns: 次元圧縮対象のカラムリスト
        n_components: 圧縮後の次元数
        standardize: 標準化を行うか（デフォルトTrue）
        handle_missing: 欠損値の処理方法
            - "error": エラーを出す（デフォルト）
            - "mean": trainの平均で補完
            - "zero": 0で補完
        random_state: 乱数シード
        prefix: 出力カラム名のプレフィックス（Noneの場合はデフォルト命名規則）
    """

    def __init__(
        self,
        columns: list[str],
        n_components: int = 2,
        standardize: bool = True,
        handle_missing: str = "error",
        random_state: int = 42,
        prefix: str | None = None,
    ):
        super().__init__()
        self.columns = columns
        self.n_components = n_components
        self.standardize = standardize
        self.handle_missing = handle_missing
        self.random_state = random_state
        self._custom_prefix = prefix  # Noneの場合は_get_prefix()を使用

        self.scaler_ = None
        self.reducer_ = None
        self.fill_values_ = None  # trainの平均値（handle_missing="mean"用）

    def _create_reducer(self, seed: int):
        """次元圧縮器を作成（子クラスで実装）"""
        raise NotImplementedError

    def _get_prefix(self) -> str:
        """プレフィックスを取得（子クラスで実装）"""
        raise NotImplementedError

    def _generate_column_names(self) -> list[str]:
        """出力カラム名を生成"""
        # カスタムprefixが指定されていればそれを使用
        if self._custom_prefix:
            prefix = self._custom_prefix
            return [f"{prefix}_{i}" for i in range(self.n_components)]

        # デフォルト: _get_prefix() + カラム名
        prefix = self._get_prefix()
        n_cols = len(self.columns)

        # カラム数が10以上なら数で表示
        if n_cols >= 10:
            col_suffix = f"{n_cols}cols"
        else:
            col_suffix = "_".join(self.columns)

        return [f"{prefix}_{col_suffix}_{i}" for i in range(self.n_components)]

    def _handle_missing_values(self, pdf: pd.DataFrame, is_fit: bool = False) -> pd.DataFrame:
        """欠損値を処理

        Args:
            pdf: pandas DataFrame
            is_fit: fit時かどうか

        Returns:
            欠損処理後のDataFrame
        """
        has_missing = pdf.isna().any().any()

        if self.handle_missing == "mean" and is_fit:
            # trainの平均を保存（欠損有無に関わらず）
            self.fill_values_ = pdf.mean()

        if not has_missing:
            return pdf

        if self.handle_missing == "error":
            raise ValueError(
                "欠損値が含まれています。"
                "handle_missing='mean'/'zero'を指定するか、"
                "事前にSimpleImputeBlock等で補完してください"
            )
        elif self.handle_missing == "mean":
            # trainの平均で補完
            pdf = pdf.fillna(self.fill_values_)
        elif self.handle_missing == "zero":
            pdf = pdf.fillna(0)

        return pdf

    def fit(self, input_df: pl.DataFrame, y=None) -> pl.DataFrame:
        """trainデータで変換器を学習

        Args:
            input_df: 入力DataFrame（trainデータ）
            y: ターゲット変数（使用しない）

        Returns:
            次元圧縮後のDataFrame
        """
        # Polars → pandas変換
        pdf = input_df.select(self.columns).to_pandas()

        # 欠損値処理
        pdf = self._handle_missing_values(pdf, is_fit=True)

        # 標準化
        if self.standardize:
            self.scaler_ = StandardScaler()
            pdf = pd.DataFrame(
                self.scaler_.fit_transform(pdf),
                columns=self.columns
            )

        # 次元圧縮器を作成・学習
        self.reducer_ = self._create_reducer(self.random_state)

        # n_componentsが特徴数を超える場合は調整
        actual_n_components = min(self.n_components, len(self.columns))
        if hasattr(self.reducer_, 'n_components'):
            self.reducer_.n_components = actual_n_components
        self.n_components = actual_n_components

        self.reducer_.fit(pdf)

        self._fitted = True
        return self._transform(input_df)

    def _transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """学習した変換を適用

        Args:
            input_df: 入力DataFrame

        Returns:
            次元圧縮後のDataFrame
        """

        # Polars → pandas変換
        pdf = input_df.select(self.columns).to_pandas()

        # 欠損値処理（trainの統計量を使用）
        pdf = self._handle_missing_values(pdf, is_fit=False)

        # 標準化（trainで学習したscalerを使用）
        if self.standardize and self.scaler_ is not None:
            pdf = pd.DataFrame(
                self.scaler_.transform(pdf),
                columns=self.columns
            )

        # 次元圧縮
        transformed = self.reducer_.transform(pdf)

        # 出力DataFrame作成
        col_names = self._generate_column_names()
        out_df = pd.DataFrame(transformed, columns=col_names)

        return pl.from_pandas(out_df)


class SVDBlock(DimensionReductionBlock):
    """Truncated SVDによる次元圧縮Block

    TruncatedSVD（特異値分解）を使用して次元圧縮を行います。
    スパースデータや大規模データに適しています。

    Args:
        columns: 次元圧縮対象のカラムリスト
        n_components: 圧縮後の次元数（デフォルト2）
        standardize: 標準化を行うか（デフォルトTrue）
        handle_missing: 欠損値の処理方法

    Examples:
        >>> df = pl.DataFrame({
        ...     "col1": [1.0, 2.0, 3.0],
        ...     "col2": [4.0, 5.0, 6.0],
        ...     "col3": [7.0, 8.0, 9.0],
        ... })
        >>> block = SVDBlock(columns=["col1", "col2", "col3"], n_components=2)
        >>> result = block.fit(df)
        >>> # result.columns = ["svd_col1_col2_col3_0", "svd_col1_col2_col3_1"]
    """

    def _create_reducer(self, seed: int):
        return TruncatedSVD(n_components=self.n_components, random_state=seed)

    def _get_prefix(self) -> str:
        return "svd"


class PCABlock(DimensionReductionBlock):
    """PCA（主成分分析）による次元圧縮Block

    PCA（Principal Component Analysis）を使用して次元圧縮を行います。
    データの分散を最大限保持しながら次元を削減します。

    Args:
        columns: 次元圧縮対象のカラムリスト
        n_components: 圧縮後の次元数（デフォルト2）
        standardize: 標準化を行うか（デフォルトTrue）
        handle_missing: 欠損値の処理方法

    Examples:
        >>> df = pl.DataFrame({
        ...     "col1": [1.0, 2.0, 3.0],
        ...     "col2": [4.0, 5.0, 6.0],
        ... })
        >>> block = PCABlock(columns=["col1", "col2"], n_components=1)
        >>> result = block.fit(df)
    """

    def _create_reducer(self, seed: int):
        return PCA(n_components=self.n_components, random_state=seed)

    def _get_prefix(self) -> str:
        return "pca"


class UMAPBlock(DimensionReductionBlock):
    """UMAPによる次元圧縮Block

    UMAP（Uniform Manifold Approximation and Projection）を使用して次元圧縮を行います。
    非線形な構造を保持した次元削減に適しています。

    注意:
    - umapライブラリが必要（uv pip install umap-learn）
    - 計算コストが高い
    - n_neighbors以上のサンプル数が必要

    Args:
        columns: 次元圧縮対象のカラムリスト
        n_components: 圧縮後の次元数（デフォルト2）
        n_neighbors: 近傍点の数（デフォルト15）
        min_dist: 最小距離（デフォルト0.1）
        metric: 距離関数（デフォルト"euclidean"）
        standardize: 標準化を行うか（デフォルトTrue）
        handle_missing: 欠損値の処理方法

    Examples:
        >>> df = pl.DataFrame({
        ...     "col1": np.random.randn(100).tolist(),
        ...     "col2": np.random.randn(100).tolist(),
        ... })
        >>> block = UMAPBlock(columns=["col1", "col2"], n_components=2, n_neighbors=10)
        >>> result = block.fit(df)
    """

    def __init__(
        self,
        columns: list[str],
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        standardize: bool = True,
        handle_missing: str = "error",
        random_state: int = 42,
    ):
        super().__init__(
            columns=columns,
            n_components=n_components,
            standardize=standardize,
            handle_missing=handle_missing,
            random_state=random_state,
        )
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric

    def _create_reducer(self, seed: int):
        try:
            import umap
        except ImportError:
            raise ImportError(
                "UMAPBlockを使用するにはumap-learnが必要です。"
                "`uv pip install umap-learn`でインストールしてください。"
            )

        return umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=seed,
        )

    def _get_prefix(self) -> str:
        return "umap"
