"""
Multi-hot / One-hot + 次元圧縮Block

タグやカテゴリリストをエンコーディングし、SVDで次元圧縮する。

クラス階層:
- BaseMultiColumnSVDBlock: 抽象基底クラス（SVD処理共通化）
  ├─ MultiColumnOneHotSVDBlock: 単一値カテゴリ × 複数カラム → OneHot → SVD
  └─ MultiColumnMultiHotSVDBlock: 複数タグ × 複数カラム → MultiHot → SVD

- MultiHotSVDBlock: 単一カラム版（MultiColumnMultiHotSVDBlockのラッパー）
"""

from abc import abstractmethod
from typing import List, Callable, Optional

import polars as pl
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

from features.base import BaseBlock


# =============================================================================
# 基底クラス
# =============================================================================

class BaseMultiColumnSVDBlock(BaseBlock):
    """複数カラム → バイナリ行列 → SVD の基底クラス

    子クラスで _encode_fit() と _encode_transform() を実装することで、
    OneHot/MultiHotの両方に対応できる。

    Args:
        columns: 対象カラム名リスト
        n_components: SVD出力次元数
        output_prefix: 出力カラム名のプレフィックス
        random_state: 乱数シード
    """

    def __init__(
        self,
        columns: List[str],
        n_components: int = 10,
        output_prefix: str = 'svd',
        random_state: int = 42,
    ):
        super().__init__()
        self.columns = columns
        self.n_components = n_components
        self.output_prefix = output_prefix
        self.random_state = random_state

        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.feature_names: List[str] = []

    @abstractmethod
    def _encode_fit(self, input_df: pl.DataFrame) -> csr_matrix:
        """入力DataFrameをバイナリ行列に変換（fit時）

        子クラスで実装する。エンコーダのfit_transformを行い、
        スパース行列を返す。
        """
        raise NotImplementedError

    @abstractmethod
    def _encode_transform(self, input_df: pl.DataFrame) -> csr_matrix:
        """入力DataFrameをバイナリ行列に変換（transform時）

        子クラスで実装する。fit済みエンコーダのtransformを行い、
        スパース行列を返す。
        """
        raise NotImplementedError

    def fit(self, input_df: pl.DataFrame, y=None) -> pl.DataFrame:
        """学習データでfit

        Args:
            input_df: 入力DataFrame（trainデータ）
            y: ターゲット変数（使用しない）

        Returns:
            SVD変換後のDataFrame
        """
        # エンコード（子クラスの実装を使用）
        sparse_matrix = self._encode_fit(input_df)

        # n_componentsが最大値を超える場合は調整
        # TruncatedSVD制約: n_components <= min(n_samples, n_features) - 1
        n_samples, n_features = sparse_matrix.shape
        max_components = min(n_samples, n_features) - 1
        actual_n_components = min(self.n_components, max_components)
        if actual_n_components < 1:
            actual_n_components = 1  # 最低1次元は確保

        if actual_n_components != self.n_components:
            self.svd = TruncatedSVD(
                n_components=actual_n_components,
                random_state=self.random_state
            )
            self.n_components = actual_n_components

        self.svd.fit(sparse_matrix)
        self.feature_names = [
            f'{self.output_prefix}_svd_{i}' for i in range(self.n_components)
        ]

        self._fitted = True
        return self._transform(input_df)

    def _transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """学習したエンコーダでSVD変換

        Args:
            input_df: 入力DataFrame

        Returns:
            SVD変換後のDataFrame
        """
        # エンコード（子クラスの実装を使用）
        sparse_matrix = self._encode_transform(input_df)
        svd_result = self.svd.transform(sparse_matrix)

        return pl.DataFrame({
            name: svd_result[:, i]
            for i, name in enumerate(self.feature_names)
        })

    def get_explained_variance_ratio(self) -> float:
        """SVDの累積寄与率を取得"""
        return self.svd.explained_variance_ratio_.sum()


# =============================================================================
# OneHot + SVD（単一値カテゴリ用）
# =============================================================================

class MultiColumnOneHotSVDBlock(BaseMultiColumnSVDBlock):
    """複数カラムの単一値カテゴリ → OneHot → SVD

    各カラムに1つのカテゴリ値が入っている場合に使用。
    複数カラムをまとめてOneHotエンコーディングし、SVDで次元圧縮する。

    Args:
        columns: 対象カラム名リスト
        n_components: SVD出力次元数
        output_prefix: 出力カラム名のプレフィックス
        null_value: 欠損値の置換文字列
        random_state: 乱数シード

    Examples:
        >>> df = pl.DataFrame({
        ...     "road_direction": ["北", "南", "東", None],
        ...     "road_type": ["市道", "国道", "私道", "市道"],
        ... })
        >>> block = MultiColumnOneHotSVDBlock(
        ...     columns=["road_direction", "road_type"],
        ...     n_components=3,
        ...     output_prefix="road",
        ... )
        >>> result = block.fit(df)
        >>> # OneHot: [北, 南, 東, _, 市道, 国道, 私道] → SVD 3次元
    """

    def __init__(
        self,
        columns: List[str],
        n_components: int = 10,
        output_prefix: str = 'onehot',
        null_value: str = "_",
        random_state: int = 42,
    ):
        super().__init__(
            columns=columns,
            n_components=n_components,
            output_prefix=output_prefix,
            random_state=random_state,
        )
        self.null_value = null_value
        self.encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore")

    def _preprocess(self, input_df: pl.DataFrame) -> np.ndarray:
        """欠損値・空文字を置換してnumpy配列に変換"""
        X = input_df.select(self.columns).to_numpy()
        X = np.where(X == None, self.null_value, X)
        X = np.where(X == "", self.null_value, X)
        return X

    def _encode_fit(self, input_df: pl.DataFrame) -> csr_matrix:
        """OneHotEncoder.fit_transform"""
        X = self._preprocess(input_df)
        return self.encoder.fit_transform(X)

    def _encode_transform(self, input_df: pl.DataFrame) -> csr_matrix:
        """OneHotEncoder.transform"""
        X = self._preprocess(input_df)
        return self.encoder.transform(X)

    def get_categories(self) -> List[List[str]]:
        """学習したカテゴリ一覧を取得（カラムごと）"""
        return [list(cats) for cats in self.encoder.categories_]


# =============================================================================
# MultiHot + SVD（複数タグ用）
# =============================================================================

class MultiColumnMultiHotSVDBlock(BaseMultiColumnSVDBlock):
    """複数カラムの複数タグ → MultiHot → SVD

    各カラムに区切り文字で分割された複数タグが入っている場合に使用。
    複数カラムをプレフィックス付きで統合し、Multi-hotエンコーディング後に
    SVDで次元圧縮する。

    Args:
        columns: 対象カラム名リスト
        prefixes: 各カラムに付与するプレフィックス（指定なしの場合はカラム名から生成）
        n_components: SVD出力次元数
        separator: タグの区切り文字
        output_prefix: 出力カラム名のプレフィックス
        random_state: 乱数シード

    Examples:
        >>> df = pl.DataFrame({
        ...     "reform_wet_area": ["浴室/トイレ", "キッチン", ""],
        ...     "reform_interior": ["床/壁", "", "天井"],
        ... })
        >>> block = MultiColumnMultiHotSVDBlock(
        ...     columns=["reform_wet_area", "reform_interior"],
        ...     prefixes=["wet", "int"],
        ...     n_components=5,
        ...     output_prefix="reform",
        ... )
        >>> result = block.fit(df)
        >>> # タグ: ["wet_浴室", "wet_トイレ", "wet_キッチン", "int_床", "int_壁", "int_天井"]
    """

    def __init__(
        self,
        columns: List[str],
        prefixes: Optional[List[str]] = None,
        n_components: int = 10,
        separator: str = '/',
        output_prefix: str = 'multi',
        random_state: int = 42,
    ):
        super().__init__(
            columns=columns,
            n_components=n_components,
            output_prefix=output_prefix,
            random_state=random_state,
        )
        self.prefixes = prefixes or [col.split('_')[-1][:3] for col in columns]
        self.separator = separator

        if len(self.columns) != len(self.prefixes):
            raise ValueError("columns と prefixes の長さが一致しません")

        self.mlb = MultiLabelBinarizer()

    def _parse_tags(self, df: pl.DataFrame) -> List[List[str]]:
        """複数カラムを統合してタグリストを作成"""
        n_samples = len(df)
        tag_lists = [[] for _ in range(n_samples)]

        for col, prefix in zip(self.columns, self.prefixes):
            col_series = df[col].fill_null('').to_list()

            for i, tag_str in enumerate(col_series):
                if tag_str and str(tag_str) not in ('', 'nan', 'None'):
                    tags = [
                        f'{prefix}_{t.strip()}'
                        for t in str(tag_str).split(self.separator) if t.strip()
                    ]
                    tag_lists[i].extend(tags)

        return tag_lists

    def _encode_fit(self, input_df: pl.DataFrame) -> csr_matrix:
        """MultiLabelBinarizer.fit_transform"""
        tag_lists = self._parse_tags(input_df)
        multi_hot = self.mlb.fit_transform(tag_lists)
        return csr_matrix(multi_hot)

    def _encode_transform(self, input_df: pl.DataFrame) -> csr_matrix:
        """MultiLabelBinarizer.transform"""
        tag_lists = self._parse_tags(input_df)
        multi_hot = self.mlb.transform(tag_lists)
        return csr_matrix(multi_hot)

    def get_classes(self) -> List[str]:
        """学習したタグクラス一覧を取得"""
        return list(self.mlb.classes_)


# =============================================================================
# 単一カラム版（後方互換ラッパー）
# =============================================================================

class MultiHotSVDBlock(BaseBlock):
    """Multi-hot + SVD変換Block（単一カラム版）

    区切り文字で分割されたタグカラムをMulti-hotエンコーディングし、
    TruncatedSVDで次元圧縮します。

    内部的にMultiColumnMultiHotSVDBlockを使用。

    Args:
        column: 対象カラム名
        n_components: SVD出力次元数（デフォルト: 10）
        separator: タグの区切り文字（デフォルト: '/'）
        prefix: 出力カラム名のプレフィックス（指定なしの場合はカラム名から生成）
        tag_transformer: タグ変換関数（オプション、例: 'wet_'プレフィックス付与）
        random_state: 乱数シード

    Examples:
        >>> df = pl.DataFrame({
        ...     "tags": ["A/B/C", "B/C/D", "A/D", "", None],
        ... })
        >>> block = MultiHotSVDBlock(column="tags", n_components=2)
        >>> result = block.fit(df)
        >>> # result.columns = ["tags_svd_0", "tags_svd_1"]

    Note:
        - 空文字列やNullは空のタグリストとして扱う
        - 学習データに存在しないタグはtransform時に無視される
        - tag_transformerを使用する場合は内部実装を直接使用（後方互換）
    """

    def __init__(
        self,
        column: str,
        n_components: int = 10,
        separator: str = '/',
        prefix: Optional[str] = None,
        tag_transformer: Optional[Callable[[str], str]] = None,
        random_state: int = 42,
    ):
        super().__init__()
        self.column = column
        self.n_components = n_components
        self.separator = separator
        self.prefix = prefix or column.replace('_id', '').replace('_tag', '') + '_tag'
        self.tag_transformer = tag_transformer
        self.random_state = random_state

        # tag_transformerがある場合は旧実装を使用（後方互換）
        if tag_transformer:
            self._use_legacy = True
            self.mlb = MultiLabelBinarizer()
            self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
            self.feature_names: List[str] = []
        else:
            self._use_legacy = False
            # 内部的にMultiColumnMultiHotSVDBlockを使用
            self._inner = MultiColumnMultiHotSVDBlock(
                columns=[column],
                prefixes=[self.prefix.replace('_tag', '').replace('_svd', '')],
                n_components=n_components,
                separator=separator,
                output_prefix=self.prefix,
                random_state=random_state,
            )

    def _parse_tags_legacy(self, df: pl.DataFrame) -> List[List[str]]:
        """タグカラムをパースしてリストのリストに変換（旧実装）"""
        tag_series = df[self.column].fill_null('').to_list()
        tag_lists = []

        for tag_str in tag_series:
            if tag_str and str(tag_str).strip():
                tags = [t.strip() for t in str(tag_str).split(self.separator) if t.strip()]
                if self.tag_transformer:
                    tags = [self.tag_transformer(t) for t in tags]
            else:
                tags = []
            tag_lists.append(tags)

        return tag_lists

    def fit(self, input_df: pl.DataFrame, y=None) -> pl.DataFrame:
        """学習データでfit"""
        if self._use_legacy:
            return self._fit_legacy(input_df, y)
        else:
            result = self._inner.fit(input_df, y)
            self._fitted = True
            return result

    def _fit_legacy(self, input_df: pl.DataFrame, y=None) -> pl.DataFrame:
        """旧実装のfit（tag_transformer対応）"""
        tag_lists = self._parse_tags_legacy(input_df)
        multi_hot = self.mlb.fit_transform(tag_lists)
        sparse_matrix = csr_matrix(multi_hot)

        actual_n_components = min(self.n_components, sparse_matrix.shape[1])
        if actual_n_components < self.n_components:
            self.svd = TruncatedSVD(n_components=actual_n_components, random_state=self.random_state)
            self.n_components = actual_n_components

        self.svd.fit(sparse_matrix)
        self.feature_names = [f'{self.prefix}_svd_{i}' for i in range(self.n_components)]

        self._fitted = True
        return self._transform(input_df)

    def _transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """学習したタグでMulti-hot + SVD変換"""
        if self._use_legacy:
            return self._transform_legacy(input_df)
        else:
            return self._inner._transform(input_df)

    def _transform_legacy(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """旧実装のtransform（tag_transformer対応）"""
        tag_lists = self._parse_tags_legacy(input_df)
        multi_hot = self.mlb.transform(tag_lists)
        sparse_matrix = csr_matrix(multi_hot)
        svd_result = self.svd.transform(sparse_matrix)

        return pl.DataFrame({
            name: svd_result[:, i]
            for i, name in enumerate(self.feature_names)
        })

    def get_classes(self) -> List[str]:
        """学習したタグクラス一覧を取得"""
        if self._use_legacy:
            return list(self.mlb.classes_)
        else:
            return self._inner.get_classes()

    def get_explained_variance_ratio(self) -> float:
        """SVDの累積寄与率を取得"""
        if self._use_legacy:
            return self.svd.explained_variance_ratio_.sum()
        else:
            return self._inner.get_explained_variance_ratio()
