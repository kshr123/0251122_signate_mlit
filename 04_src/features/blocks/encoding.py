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
        return self._transform(input_df)

    def _transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """学習したマッピングでカテゴリ変数を数値に変換

        Args:
            input_df: 入力DataFrame（train/testどちらでも可）

        Returns:
            数値化されたDataFrame
        """
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
        return self._transform(input_df)

    def _transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """学習した頻度でカテゴリ変数を数値に変換

        Args:
            input_df: 入力DataFrame（train/testどちらでも可）

        Returns:
            頻度エンコードされたDataFrame
        """
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


class TopNCategoryLEBlock(LabelEncodingBlock):
    """上位Nカテゴリ + その他でLabel Encodingするブロック

    LabelEncodingBlockを継承し、前処理として「上位N以外→その他」の
    マッピングを追加したブロック。

    特徴:
    - trainデータの頻度で上位Nカテゴリを決定（または直接指定）
    - 上位N以外は全て「その他」にマッピング
    - 未知カテゴリも「その他」として扱う
    - 親クラスのLabelEncodingBlockでエンコード

    Args:
        column: 対象のカラム名（単一カラム）
        top_n: 上位カテゴリ数（デフォルト: 10）
        top_categories: 上位カテゴリを直接指定（指定時はtop_nは無視）
        other_label: その他のラベル（デフォルト: "その他"）
        output_column: 出力カラム名（デフォルト: "{column}_le"）

    Examples:
        >>> df_train = pl.DataFrame({
        ...     "category": ["A", "A", "B", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"],
        ... })
        >>> block = TopNCategoryLEBlock(column="category", top_n=3)
        >>> result = block.fit(df_train)
        >>> # A, B, C が上位3、他は「その他」にまとめてLE
        >>> # top_categories を直接指定することも可能
        >>> block2 = TopNCategoryLEBlock(
        ...     column="category",
        ...     top_categories=["A", "B", "C"]
        ... )
    """

    def __init__(
        self,
        column: str,
        top_n: int = 10,
        top_categories: list[str] | None = None,
        other_label: str = "その他",
        output_column: str | None = None,
    ):
        """初期化

        Args:
            column: 対象のカラム名
            top_n: 上位カテゴリ数
            top_categories: 上位カテゴリを直接指定（指定時はtop_nは無視）
            other_label: その他のラベル
            output_column: 出力カラム名
        """
        # 出力カラム名で親を初期化
        self._output_column = output_column or f"{column}_le"
        super().__init__(columns=[self._output_column])

        self.column = column
        self.top_n = top_n
        self._provided_top_categories = top_categories
        self.other_label = other_label

        self.top_categories_: list[str] | None = None

    def _map_to_top_n(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """上位カテゴリ以外を「その他」にマッピング"""
        # カラムをString型にキャスト（Int64等でも対応可能に）
        col_expr = pl.col(self.column).cast(pl.Utf8)
        return input_df.with_columns(
            pl.when(col_expr.is_in(self.top_categories_))
            .then(col_expr)
            .otherwise(pl.lit(self.other_label))
            .alias(self._output_column)
        )

    def fit(self, input_df: pl.DataFrame, y=None) -> pl.DataFrame:
        """trainデータでエンコーダを学習

        Args:
            input_df: 入力DataFrame（trainデータ）
            y: ターゲット変数（使用しない）

        Returns:
            変換後のDataFrame
        """
        # 上位カテゴリを決定
        if self._provided_top_categories is not None:
            self.top_categories_ = list(self._provided_top_categories)
        else:
            # trainデータの頻度で上位Nを決定
            value_counts = (
                input_df
                .group_by(self.column)
                .len()
                .sort("len", descending=True)
                .head(self.top_n)
            )
            self.top_categories_ = value_counts[self.column].to_list()

        # 上位N + その他にマッピングしてから親のfitを呼ぶ
        mapped_df = self._map_to_top_n(input_df)
        return super().fit(mapped_df, y)

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """学習したマッピングでカテゴリ変数を数値に変換

        Args:
            input_df: 入力DataFrame

        Returns:
            変換後のDataFrame（output_columnのみ）

        Raises:
            RuntimeError: fit()を先に実行していない場合
        """
        if not self._fitted:
            raise RuntimeError("TopNCategoryLEBlock: fit()を先に実行してください")

        # 上位N + その他にマッピングしてから親のtransformを呼ぶ
        mapped_df = self._map_to_top_n(input_df)
        return super().transform(mapped_df)


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
        return self._transform(input_df)

    def _transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """学習したカテゴリでOne-Hot変換

        Args:
            input_df: 入力DataFrame（train/testどちらでも可）

        Returns:
            One-HotエンコードされたDataFrame
        """
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


class MultiKeyTEBlock(BaseBlock):
    """複合キー（複数カラムの組み合わせ）でTarget Encodingを行うBlock

    主キー + 属性カラムの組み合わせでターゲットエンコーディングを行う。
    これにより、カテゴリ毎の属性差を捉える特徴量を生成。
    例: 「カテゴリAの東京都」の平均価格を特徴量化

    フォールバック機能:
    - (primary_key, attr) の組み合わせが存在しない場合
    - → primary_keyのみの平均値を使用
    - → それも存在しない場合はglobal_meanを使用

    特徴:
    - trainデータ: CV foldごとにOOF方式で計算（データリーク防止）
    - testデータ: 全trainデータの平均を使用
    - 未知の組み合わせ: 階層フォールバック

    Args:
        primary_key: 主キーのカラム名
        attr_columns: 属性カラム名のリスト（主キーと組み合わせてTEする）
        cv: CVのfold情報（list of (train_idx, valid_idx) tuples）
        output_prefix: 出力カラム名のプレフィックス（デフォルト: "multikey_te"）

    Examples:
        >>> from sklearn.model_selection import KFold
        >>> df = pl.DataFrame({
        ...     "category": [1, 1, 2, 2, 1, 2],
        ...     "region": ["Tokyo", "Osaka", "Tokyo", "Osaka", "Tokyo", "Tokyo"],
        ...     "type": ["A", "A", "B", "B", "A", "B"],
        ... })
        >>> y = np.array([100, 200, 300, 400, 150, 350])
        >>> cv = list(KFold(n_splits=2).split(df))
        >>> block = MultiKeyTEBlock(
        ...     primary_key="category",
        ...     attr_columns=["region", "type"],
        ...     cv=cv
        ... )
        >>> train_result = block.fit(df, y=y)
        >>> # train_result に multikey_te_region, multikey_te_type カラムが生成
    """

    def __init__(
        self,
        primary_key: str,
        attr_columns: list[str],
        cv,
        output_prefix: str = "multikey_te",
    ):
        """初期化

        Args:
            primary_key: 主キーのカラム名
            attr_columns: 属性カラム名のリスト
            cv: CVのfold情報
            output_prefix: 出力カラム名のプレフィックス
        """
        super().__init__()
        self.primary_key = primary_key
        self.attr_columns = attr_columns
        self.cv = list(cv)
        self.output_prefix = output_prefix

        # fit時に保存する情報
        # {attr_col: {(primary_key_value, attr_value): te_value}}
        self.te_maps: dict[str, dict[tuple, float]] = {}
        # {primary_key_value: mean}
        self.primary_key_means: dict = {}
        self.global_mean: float = 0.0

        # 出力特徴量名
        self.feature_names = [
            f'{output_prefix}_{col}' for col in attr_columns
        ]

    def fit(self, input_df: pl.DataFrame, y=None) -> pl.DataFrame:
        """学習データでfitしてtransform（OOF方式）

        Args:
            input_df: 学習データ（primary_key, attr_columnsを含む）
            y: ターゲット変数

        Returns:
            pl.DataFrame: TE特徴量（カラム数 = len(attr_columns)）
        """
        if y is None:
            raise ValueError("MultiKeyTEBlock: ターゲット変数(y)は必須です")

        n_samples = len(input_df)

        if isinstance(y, pl.Series):
            y_values = y.to_numpy()
        else:
            y_values = np.asarray(y)

        self.global_mean = float(np.mean(y_values))

        # 主キー値を取得
        pk_values = input_df[self.primary_key].to_numpy()

        # 主キー別平均を計算
        pk_df = pl.DataFrame({
            'pk': pk_values,
            'target': y_values,
        })
        pk_means = pk_df.group_by('pk').agg(
            pl.col('target').mean().alias('mean')
        )
        self.primary_key_means = dict(zip(
            pk_means['pk'].to_list(),
            pk_means['mean'].to_list()
        ))

        # 結果を格納する配列
        results = {name: np.full(n_samples, np.nan) for name in self.feature_names}

        # 各属性カラムについてTE計算
        for attr_col, feat_name in zip(self.attr_columns, self.feature_names):
            attr_values = input_df[attr_col].to_numpy()
            self.te_maps[attr_col] = {}

            # Fold毎にOOFでTE計算
            for train_idx, val_idx in self.cv:
                # このfoldの学習データでTE計算
                fold_df = pl.DataFrame({
                    'pk': pk_values[train_idx],
                    'attr': attr_values[train_idx],
                    'target': y_values[train_idx],
                })

                # 主キー×属性のTE
                te_fold = fold_df.group_by(['pk', 'attr']).agg(
                    pl.col('target').mean().alias('te')
                )
                te_map_fold = {
                    (row['pk'], row['attr']): row['te']
                    for row in te_fold.iter_rows(named=True)
                }

                # validation indexに適用
                for i in val_idx:
                    pk = pk_values[i]
                    attr = attr_values[i]
                    key = (pk, attr)

                    # (pk, attr) → pk_mean → global_mean の順でフォールバック
                    te_value = te_map_fold.get(
                        key,
                        self.primary_key_means.get(pk, self.global_mean)
                    )
                    results[feat_name][i] = te_value

            # 全データでのTE（テストデータ用）を計算
            full_df = pl.DataFrame({
                'pk': pk_values,
                'attr': attr_values,
                'target': y_values,
            })
            te_full = full_df.group_by(['pk', 'attr']).agg(
                pl.col('target').mean().alias('te')
            )
            self.te_maps[attr_col] = {
                (row['pk'], row['attr']): row['te']
                for row in te_full.iter_rows(named=True)
            }

        self._fitted = True
        return pl.DataFrame(results)

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """テストデータをtransform

        Args:
            input_df: テストデータ（primary_key, attr_columnsを含む）

        Returns:
            pl.DataFrame: TE特徴量
        """
        if not self._fitted:
            raise RuntimeError(f"{self.__class__.__name__}: fit()を先に実行してください")

        pk_values = input_df[self.primary_key].to_list()

        results = {name: [] for name in self.feature_names}

        for attr_col, feat_name in zip(self.attr_columns, self.feature_names):
            attr_values = input_df[attr_col].to_list()
            te_map = self.te_maps[attr_col]

            for pk, attr in zip(pk_values, attr_values):
                key = (pk, attr)
                te_value = te_map.get(
                    key,
                    self.primary_key_means.get(pk, self.global_mean)
                )
                results[feat_name].append(te_value)

        return pl.DataFrame(results)

    def get_stats(self) -> dict:
        """統計情報を取得

        Returns:
            dict: 以下の統計情報を含む辞書
                - n_combinations: 主キー×属性の組み合わせ数（属性カラム別）
                - primary_key_means: 主キー別平均値
                - global_mean: グローバル平均値
        """
        return {
            'n_combinations': {
                col: len(te_map) for col, te_map in self.te_maps.items()
            },
            'primary_key_means': self.primary_key_means,
            'global_mean': self.global_mean,
        }
