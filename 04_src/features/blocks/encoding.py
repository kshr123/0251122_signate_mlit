"""
カテゴリ変数エンコーディングBlock

カテゴリ変数を数値に変換（category_encoders使用）
"""

import polars as pl
import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder, CountEncoder, OneHotEncoder
from features.base import BaseBlock


class LabelEncodingBlock(BaseBlock):
    """カテゴリ変数をLabel Encodingで数値に変換するBlock

    category_encoders.OrdinalEncoderを使用して、
    trainデータでfitしたマッピングをtestデータにも適用します。

    特徴:
    - 未知カテゴリ: -1 にエンコード
    - 欠損値(NaN): -2 にエンコード
    - train/testで一貫したマッピングを保証

    Args:
        columns: 対象のカラムリスト

    Examples:
        >>> df_train = pl.DataFrame({
        ...     "cat1": ["A", "B", "A", "C"],
        ...     "cat2": ["X", "Y", "X", "Y"],
        ... })
        >>> df_test = pl.DataFrame({
        ...     "cat1": ["A", "B", "D"],  # "D"は未知カテゴリ
        ...     "cat2": ["X", "Z", None],  # "Z"は未知、Noneは欠損
        ... })
        >>> block = LabelEncodingBlock(columns=["cat1", "cat2"])
        >>> train_result = block.fit(df_train)
        >>> test_result = block.transform(df_test)
        >>> # "D"と"Z"は-1、Noneは-2にエンコードされる
    """

    def __init__(self, columns: list[str]):
        """初期化

        Args:
            columns: 対象のカラムリスト
        """
        super().__init__()
        self.columns = columns
        self.encoder = None

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        """trainデータでエンコーダを学習

        Args:
            input_df: 入力DataFrame（trainデータ）
            y: ターゲット変数（使用しない）

        Returns:
            変換後のDataFrame
        """
        # OrdinalEncoderを初期化
        self.encoder = OrdinalEncoder(
            cols=self.columns,
            handle_unknown='value',   # 未知カテゴリ → -1
            handle_missing='value',   # 欠損値 → -2
        )

        # Polars → pandas変換してfit
        pdf = input_df.select(self.columns).to_pandas()
        self.encoder.fit(pdf)

        self._fitted = True
        return self.transform(input_df)

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """学習したマッピングでカテゴリ変数を数値に変換

        Args:
            input_df: 入力DataFrame（train/testどちらでも可）

        Returns:
            数値化されたDataFrame

        Raises:
            RuntimeError: fit()を先に実行していない場合
        """
        if not self._fitted:
            raise RuntimeError("LabelEncodingBlock: fit()を先に実行してください")

        # Polars → pandas変換してtransform
        pdf = input_df.select(self.columns).to_pandas()
        result_pdf = self.encoder.transform(pdf)

        # pandas → Polars変換して返す
        return pl.from_pandas(result_pdf)


class CountEncodingBlock(BaseBlock):
    """カテゴリ変数をCount Encoding（頻度エンコーディング）で数値に変換するBlock

    category_encoders.CountEncoderを使用して、
    trainデータでfitした頻度をtestデータにも適用します。

    特徴:
    - 各カテゴリの出現頻度（回数）に変換
    - 未知カテゴリ: 0 にエンコード
    - 欠損値(NaN): 1つのカテゴリとして頻度をカウント
    - train/testで一貫したマッピングを保証
    - データリークなし（ターゲット変数を使用しない）

    Args:
        columns: 対象のカラムリスト
        normalize: Trueの場合、頻度を総数で割った割合を返す（デフォルト: False）

    Examples:
        >>> df_train = pl.DataFrame({
        ...     "cat1": ["A", "B", "A", "A", "B", "C"],  # A:3, B:2, C:1
        ... })
        >>> df_test = pl.DataFrame({
        ...     "cat1": ["A", "B", "D"],  # "D"は未知カテゴリ
        ... })
        >>> block = CountEncodingBlock(columns=["cat1"])
        >>> train_result = block.fit(df_train)
        >>> # train_result["cat1"] = [3, 2, 3, 3, 2, 1]
        >>> test_result = block.transform(df_test)
        >>> # test_result["cat1"] = [3, 2, 0]  # "D"は0
    """

    def __init__(self, columns: list[str], normalize: bool = False):
        """初期化

        Args:
            columns: 対象のカラムリスト
            normalize: Trueの場合、頻度を総数で割った割合を返す
        """
        super().__init__()
        self.columns = columns
        self.normalize = normalize
        self.encoder = None

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        """trainデータでエンコーダを学習

        Args:
            input_df: 入力DataFrame（trainデータ）
            y: ターゲット変数（使用しない）

        Returns:
            変換後のDataFrame
        """
        # CountEncoderを初期化
        self.encoder = CountEncoder(
            cols=self.columns,
            handle_unknown=0,      # 未知カテゴリ → 0
            handle_missing='count',  # 欠損値もカウント
            normalize=self.normalize,
        )

        # Polars → pandas変換してfit
        pdf = input_df.select(self.columns).to_pandas()
        self.encoder.fit(pdf)

        self._fitted = True
        return self.transform(input_df)

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """学習した頻度でカテゴリ変数を数値に変換

        Args:
            input_df: 入力DataFrame（train/testどちらでも可）

        Returns:
            頻度エンコードされたDataFrame

        Raises:
            RuntimeError: fit()を先に実行していない場合
        """
        if not self._fitted:
            raise RuntimeError("CountEncodingBlock: fit()を先に実行してください")

        # Polars → pandas変換してtransform
        pdf = input_df.select(self.columns).to_pandas()
        result_pdf = self.encoder.transform(pdf)

        # pandas → Polars変換して返す
        return pl.from_pandas(result_pdf)


class TargetEncodingBlock(BaseBlock):
    """カテゴリ変数をTarget Encoding（ターゲットエンコーディング）で数値に変換するBlock

    Out-of-Fold (OOF) 方式でTarget Encodingを行い、データリークを防止します。

    特徴:
    - trainデータ: CV foldごとにOOF方式で計算（データリーク防止）
    - testデータ: 全trainデータの平均を使用
    - 未知カテゴリ: 全体平均にエンコード
    - 欠損値(NaN): 1つのカテゴリとして平均を計算
    - カラム名には「TE_」プレフィックスが付与される

    Args:
        columns: 対象のカラムリスト
        cv: CVのfold情報（list of (train_idx, valid_idx) tuples）

    Examples:
        >>> from sklearn.model_selection import KFold
        >>> df_train = pl.DataFrame({
        ...     "cat1": ["A", "A", "B", "B", "C", "C"],
        ... })
        >>> y = np.array([100, 200, 300, 400, 500, 600])
        >>> cv = list(KFold(n_splits=3).split(df_train))
        >>> block = TargetEncodingBlock(columns=["cat1"], cv=cv)
        >>> train_result = block.fit(df_train, y=y)
        >>> # train_result["TE_cat1"] にOOF計算値が入る
    """

    def __init__(self, columns: list[str], cv):
        """初期化

        Args:
            columns: 対象のカラムリスト
            cv: CVのfold情報（list of (train_idx, valid_idx) tuples）
        """
        super().__init__()
        self.columns = columns
        self.cv = list(cv)
        self.n_fold = len(self.cv)
        self.mapping_df_ = None
        self.y_mean_ = None

    def _create_mapping(self, input_df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """OOF方式でTarget Encodingのマッピングを作成

        Args:
            input_df: 入力DataFrame（pandas）
            y: ターゲット変数（numpy配列）

        Returns:
            OOF計算されたDataFrame
        """
        self.mapping_df_ = {}
        self.y_mean_ = np.mean(y)
        out_df = pd.DataFrame()
        target = pd.Series(y)

        for col_name in self.columns:
            keys = input_df[col_name].unique()
            X = input_df[col_name]
            oof = np.zeros(len(X), dtype=float)

            for idx_train, idx_valid in self.cv:
                # trainデータのみでカテゴリ別平均を計算
                _df = target.iloc[idx_train].groupby(X.iloc[idx_train]).mean()
                _df = _df.reindex(keys)
                # fold内に存在しないカテゴリは、そのfoldの平均で埋める
                _df = _df.fillna(_df.mean())
                # validation indexにマッピング
                oof[idx_valid] = input_df[col_name].iloc[idx_valid].map(_df.to_dict())

            # NaNを全体平均で埋める
            oof = np.where(np.isnan(oof), self.y_mean_, oof)
            out_df[col_name] = oof

            # testデータ用に全trainデータの平均を保存
            self.mapping_df_[col_name] = target.groupby(X).mean()

        return out_df

    def fit(self, input_df: pl.DataFrame, y=None) -> pl.DataFrame:
        """trainデータでエンコーダを学習（OOF方式）

        Args:
            input_df: 入力DataFrame（trainデータ）
            y: ターゲット変数（必須、numpy配列またはPolars Series）

        Returns:
            OOF計算されたDataFrame（カラム名にTE_プレフィックス付き）

        Raises:
            ValueError: yがNoneの場合
        """
        if y is None:
            raise ValueError("TargetEncodingBlock: ターゲット変数(y)は必須です")

        # yがPolars Seriesの場合、numpy配列に変換
        if isinstance(y, pl.Series):
            y_np = y.to_numpy()
        else:
            y_np = np.asarray(y)

        # Polars → pandas変換
        pdf = input_df.select(self.columns).to_pandas()

        # OOFでマッピング作成
        out_df = self._create_mapping(pdf, y_np)

        # カラム名にTE_プレフィックスを追加
        out_df = out_df.add_prefix('TE_')

        self._fitted = True

        # pandas → Polars変換して返す
        return pl.from_pandas(out_df)

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """全trainデータの平均でカテゴリ変数を数値に変換

        Args:
            input_df: 入力DataFrame（testデータ）

        Returns:
            ターゲットエンコードされたDataFrame（カラム名にTE_プレフィックス付き）

        Raises:
            RuntimeError: fit()を先に実行していない場合
        """
        if not self._fitted:
            raise RuntimeError("TargetEncodingBlock: fit()を先に実行してください")

        # Polars → pandas変換
        pdf = input_df.select(self.columns).to_pandas()

        out_df = pd.DataFrame()
        for c in self.columns:
            # 全trainデータの平均でマッピング、未知カテゴリは全体平均
            out_df[c] = pdf[c].map(self.mapping_df_[c]).fillna(self.y_mean_)

        # カラム名にTE_プレフィックスを追加
        out_df = out_df.add_prefix('TE_')

        # pandas → Polars変換して返す
        return pl.from_pandas(out_df)


class OneHotEncodingBlock(BaseBlock):
    """カテゴリ変数をOne-Hot Encoding（ダミー変数化）で変換するBlock

    category_encoders.OneHotEncoderを使用して、
    trainデータでfitしたカテゴリをtestデータにも適用します。

    注意:
    - **低カーディナリティ向け**: 高カーディナリティの場合は次元爆発に注意。
    - GBDTモデルでは不要な場合が多い（LabelEncodingで十分）。
    - 線形モデルやニューラルネットワークでは有効。

    特徴:
    - 各カテゴリごとに0/1のカラムを生成
    - min_countで低頻度カテゴリを除外可能
    - 未知カテゴリ: すべて0のベクトル
    - 欠損値(NaN): 1つのカテゴリとして扱う
    - train/testで一貫したカラムを保証

    Args:
        columns: 対象のカラムリスト
        min_count: カテゴリの最小出現回数（これ以下は除外、デフォルト: 0）
        use_cat_names: Trueの場合、カラム名にカテゴリ値を使用（デフォルト: True）

    Examples:
        >>> df_train = pl.DataFrame({
        ...     "color": ["red", "red", "red", "blue", "green"],  # red:3, blue:1, green:1
        ... })
        >>> block = OneHotEncodingBlock(columns=["color"], min_count=2)
        >>> result = block.fit(df_train)
        >>> # min_count=2 なので blue, green は除外され、red のみ
    """

    def __init__(self, columns: list[str], min_count: int = 0, use_cat_names: bool = True):
        """初期化

        Args:
            columns: 対象のカラムリスト
            min_count: カテゴリの最小出現回数（これ以下は除外）
            use_cat_names: Trueの場合、カラム名にカテゴリ値を使用
        """
        super().__init__()
        self.columns = columns
        self.min_count = min_count
        self.use_cat_names = use_cat_names
        self.encoder = None
        self.valid_categories_ = {}  # カラムごとの有効カテゴリ

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        """trainデータでエンコーダを学習

        Args:
            input_df: 入力DataFrame（trainデータ）
            y: ターゲット変数（使用しない）

        Returns:
            変換後のDataFrame
        """
        # Polars → pandas変換
        pdf = input_df.select(self.columns).to_pandas()

        # min_count以上のカテゴリのみを抽出
        for col in self.columns:
            vc = pdf[col].value_counts()
            valid_cats = vc[vc > self.min_count].index.tolist()
            self.valid_categories_[col] = valid_cats
            # 有効カテゴリ以外をNaNに置換（encoderが無視するように）
            pdf[col] = pdf[col].where(pdf[col].isin(valid_cats), other=np.nan)

        # OneHotEncoderを初期化
        self.encoder = OneHotEncoder(
            cols=self.columns,
            use_cat_names=self.use_cat_names,
            handle_unknown='value',   # 未知カテゴリ → すべて0
            handle_missing='value',   # 欠損値 → すべて0（NaNカラムは作らない）
        )

        self.encoder.fit(pdf)

        self._fitted = True
        return self.transform(input_df)

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """学習したカテゴリでOne-Hot変換

        Args:
            input_df: 入力DataFrame（train/testどちらでも可）

        Returns:
            One-HotエンコードされたDataFrame

        Raises:
            RuntimeError: fit()を先に実行していない場合
        """
        if not self._fitted:
            raise RuntimeError("OneHotEncodingBlock: fit()を先に実行してください")

        # Polars → pandas変換
        pdf = input_df.select(self.columns).to_pandas()

        # 有効カテゴリ以外をNaNに置換
        for col in self.columns:
            pdf[col] = pdf[col].where(pdf[col].isin(self.valid_categories_[col]), other=np.nan)

        result_pdf = self.encoder.transform(pdf)

        # NaN用カラム（*_nan）を除外
        nan_cols = [c for c in result_pdf.columns if c.endswith('_nan')]
        if nan_cols:
            result_pdf = result_pdf.drop(columns=nan_cols)

        # pandas → Polars変換して返す
        return pl.from_pandas(result_pdf)
