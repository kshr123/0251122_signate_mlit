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
    """統合版Target Encodingブロック

    シングルキー、マルチキー、階層的フォールバックを1つのBlockで対応。
    Out-of-Fold (OOF) 方式でTarget Encodingを行い、データリークを防止。

    特徴:
    - シングルキー / マルチキー（タプル指定）両対応
    - mean / median の集計方法を選択可能
    - オプションで階層的フォールバック
    - 未知カテゴリ: フォールバック → global mean

    Args:
        keys: エンコーディングキー
            - str: 単一カラム（例: "addr1_2"）
            - tuple[str, ...]: 複合キー（例: ("addr1_2", "bukken_type")）
        cv: CVのfold情報（list of (train_idx, valid_idx) tuples）
        agg_func: 集計関数（"mean" or "median"）
        min_samples: 最小サンプル数（これ未満はフォールバック）
        fallback: フォールバック階層（オプション）
            例: [("addr1_1",), ("bukken_type",)]
        prefix: 出力カラム名のプレフィックス

    Examples:
        # シングルキー（従来のTargetEncodingBlock相当）
        >>> block = TargetEncodingBlock(keys="addr1_2", cv=cv)

        # マルチキー（従来のMultiKeyTEBlock相当）
        >>> block = TargetEncodingBlock(
        ...     keys=("addr1_2", "bukken_type"),
        ...     cv=cv,
        ...     agg_func="median"
        ... )

        # 階層フォールバック（従来のHierarchicalTargetEncodingBlock相当）
        >>> block = TargetEncodingBlock(
        ...     keys=("addr1_2", "bukken_type", "age_bin"),
        ...     cv=cv,
        ...     fallback=[("addr1_1", "bukken_type"), ("addr1_2",)],
        ...     min_samples=5
        ... )
    """

    def __init__(
        self,
        keys: str | tuple[str, ...] = None,
        cv=None,
        agg_func: str = "mean",
        min_samples: int = 1,
        fallback: list[tuple[str, ...]] | None = None,
        prefix: str = "TE",
        # 後方互換用引数
        columns: list[str] | None = None,
    ):
        super().__init__()

        # 後方互換: columnsが指定された場合は旧形式として扱う
        if columns is not None:
            # 旧形式: 複数カラムへのシングルキーTE
            self._legacy_mode = True
            self._columns = columns
            self.keys = None  # 複数カラムなのでNone
        else:
            # 新形式: keys引数を使用
            self._legacy_mode = False
            self._columns = None
            if keys is None:
                raise ValueError("TargetEncodingBlock: keys または columns を指定してください")
            # keysをタプルに正規化
            self.keys = (keys,) if isinstance(keys, str) else tuple(keys)

        self.cv = list(cv) if cv else []
        self.agg_func = agg_func
        self.min_samples = min_samples
        # fallbackを正規化（階層順に並べる）
        self.fallback = fallback or []
        self.prefix = prefix

        # 新形式の場合のみ設定
        if not self._legacy_mode:
            # 全階層（プライマリ + フォールバック）
            self._all_levels = [self.keys] + self.fallback
            # 出力カラム名
            self._output_column = self._generate_output_name()
        else:
            self._all_levels = None
            self._output_column = None

        # 学習結果
        self._mappings: dict[tuple, dict[tuple, float]] = {}
        self._counts: dict[tuple, dict[tuple, int]] = {}
        self._global_mean: float = 0.0

        # 旧形式用
        self.mapping_df_ = None
        self.y_mean_ = None

    def _generate_output_name(self) -> str:
        """出力カラム名を生成"""
        keys_str = "_".join(self.keys)
        suffix = f"_{self.agg_func}" if self.agg_func != "mean" else ""
        return f"{self.prefix}_{keys_str}{suffix}"

    def _get_key_values(
        self, df: pd.DataFrame, columns: tuple[str, ...]
    ) -> list[tuple]:
        """DataFrameから指定カラムのキー値タプルのリストを取得"""
        if len(columns) == 1:
            return [(v,) for v in df[columns[0]].values]
        return list(df[list(columns)].itertuples(index=False, name=None))

    def _aggregate(self, values: np.ndarray) -> float:
        """集計関数を適用"""
        if len(values) == 0:
            return np.nan
        if self.agg_func == "mean":
            return float(np.mean(values))
        elif self.agg_func == "median":
            return float(np.median(values))
        else:
            raise ValueError(f"Unknown agg_func: {self.agg_func}")

    def _build_mappings(
        self, pdf: pd.DataFrame, y: np.ndarray
    ) -> tuple[dict[tuple, dict[tuple, float]], dict[tuple, dict[tuple, int]]]:
        """全階層のマッピングを構築"""
        mappings = {}
        counts = {}

        for level_keys in self._all_levels:
            temp_df = pdf[list(level_keys)].copy()
            temp_df["_y"] = y

            grouped = temp_df.groupby(list(level_keys))["_y"]
            if self.agg_func == "mean":
                agg_df = grouped.agg(["mean", "count"])
                agg_df.columns = ["value", "count"]
            else:
                agg_df = grouped.agg(["median", "count"])
                agg_df.columns = ["value", "count"]

            mappings[level_keys] = agg_df["value"].to_dict()
            counts[level_keys] = agg_df["count"].to_dict()

        return mappings, counts

    def _lookup_value(
        self,
        row: pd.Series,
        mappings: dict[tuple, dict[tuple, float]],
        counts: dict[tuple, dict[tuple, int]],
        global_mean: float,
    ) -> float:
        """階層的にルックアップして値を取得"""
        for level_keys in self._all_levels:
            key = tuple(row[c] for c in level_keys)

            # NaNが含まれる場合はスキップ
            if any(pd.isna(k) for k in key):
                continue

            if key in mappings[level_keys]:
                count = counts[level_keys].get(key, 0)
                if count >= self.min_samples:
                    return mappings[level_keys][key]

        return global_mean

    def fit(self, input_df: pl.DataFrame, y=None) -> pl.DataFrame:
        """trainデータでエンコーダを学習（OOF方式）

        Args:
            input_df: 入力DataFrame（trainデータ）
            y: ターゲット変数（必須）

        Returns:
            OOF計算されたDataFrame
        """
        if y is None:
            raise ValueError("TargetEncodingBlock: yは必須です")

        if isinstance(y, pl.Series):
            y_np = y.to_numpy()
        else:
            y_np = np.asarray(y)

        self._global_mean = float(np.mean(y_np))
        self.y_mean_ = self._global_mean  # 後方互換

        # 旧形式: 複数カラムへのシングルキーTE
        if self._legacy_mode:
            return self._fit_legacy(input_df, y_np)

        # 新形式: 統合版TE
        # 必要なカラムを収集
        all_columns = set()
        for level_keys in self._all_levels:
            all_columns.update(level_keys)
        pdf = input_df.select(list(all_columns)).to_pandas()

        n_samples = len(pdf)
        oof = np.full(n_samples, np.nan)

        # OOF計算
        for idx_train, idx_valid in self.cv:
            train_pdf = pdf.iloc[idx_train]
            train_y = y_np[idx_train]

            # このfoldでのマッピング構築
            fold_mappings, fold_counts = self._build_mappings(train_pdf, train_y)
            fold_global_mean = float(np.mean(train_y))

            # validationに適用
            for i in idx_valid:
                row = pdf.iloc[i]
                oof[i] = self._lookup_value(
                    row, fold_mappings, fold_counts, fold_global_mean
                )

        # 全データでマッピング構築（test用）
        self._mappings, self._counts = self._build_mappings(pdf, y_np)

        self._fitted = True
        return pl.DataFrame({self._output_column: oof})

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """testデータを変換

        Args:
            input_df: 入力DataFrame（testデータ）

        Returns:
            エンコードされたDataFrame
        """
        if not self._fitted:
            raise RuntimeError("TargetEncodingBlock: fit()を先に実行してください")

        # 旧形式の場合は専用メソッドを使用
        if self._legacy_mode:
            return self._transform_legacy(input_df)

        # 必要なカラムを収集
        all_columns = set()
        for level_keys in self._all_levels:
            all_columns.update(level_keys)
        pdf = input_df.select(list(all_columns)).to_pandas()

        n_samples = len(pdf)
        values = np.full(n_samples, np.nan)

        for i in range(n_samples):
            row = pdf.iloc[i]
            values[i] = self._lookup_value(
                row, self._mappings, self._counts, self._global_mean
            )

        return pl.DataFrame({self._output_column: values})

    def _fit_legacy(self, input_df: pl.DataFrame, y_np: np.ndarray) -> pl.DataFrame:
        """旧形式（複数カラムへのシングルキーTE）のfit処理

        旧TargetEncodingBlock互換: columns指定で複数カラムに対して
        それぞれシングルキーTEを適用する。
        """
        pdf = input_df.select(self._columns).to_pandas()
        n_samples = len(pdf)

        # OOF計算用の結果配列 (カラム名はTE_{col}形式)
        oof_results = {f"TE_{col}": np.full(n_samples, np.nan) for col in self._columns}

        # 各foldでOOF計算
        for idx_train, idx_valid in self.cv:
            train_pdf = pdf.iloc[idx_train]
            train_y = y_np[idx_train]

            # 各カラムについてマッピング構築
            for col in self._columns:
                temp_df = train_pdf[[col]].copy()
                temp_df["_y"] = train_y

                grouped = temp_df.groupby(col)["_y"]
                if self.agg_func == "mean":
                    mapping = grouped.mean().to_dict()
                else:
                    mapping = grouped.median().to_dict()

                fold_mean = float(np.mean(train_y))

                # validationに適用
                out_col = f"TE_{col}"
                for i in idx_valid:
                    key = pdf.iloc[i][col]
                    if pd.isna(key):
                        oof_results[out_col][i] = fold_mean
                    else:
                        oof_results[out_col][i] = mapping.get(key, fold_mean)

        # 全データでマッピング構築（test用）
        self.mapping_df_ = {}
        for col in self._columns:
            temp_df = pdf[[col]].copy()
            temp_df["_y"] = y_np

            grouped = temp_df.groupby(col)["_y"]
            if self.agg_func == "mean":
                self.mapping_df_[col] = grouped.mean().to_dict()
            else:
                self.mapping_df_[col] = grouped.median().to_dict()

        self._fitted = True
        return pl.DataFrame(oof_results)

    def _transform_legacy(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """旧形式（複数カラムへのシングルキーTE）のtransform処理"""
        pdf = input_df.select(self._columns).to_pandas()
        n_samples = len(pdf)

        results = {}
        for col in self._columns:
            mapping = self.mapping_df_[col]
            values = np.full(n_samples, np.nan)

            for i in range(n_samples):
                key = pdf.iloc[i][col]
                if pd.isna(key):
                    values[i] = self.y_mean_
                else:
                    values[i] = mapping.get(key, self.y_mean_)

            # 出力カラム名はTE_{col}形式
            results[f"TE_{col}"] = values

        return pl.DataFrame(results)

    @property
    def output_column(self) -> str:
        """出力カラム名を取得"""
        return self._output_column


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


# =============================================================================
# 後方互換エイリアス（非推奨）
# =============================================================================
# 以下のクラスは TargetEncodingBlock に統合されました。
# 新規コードでは TargetEncodingBlock を使用してください。
#
# - MultiKeyTEBlock → TargetEncodingBlock(keys=("col1", "col2"), ...)
# - HierarchicalTargetEncodingBlock → TargetEncodingBlock(keys=..., fallback=[...])
# =============================================================================

# 後方互換用エイリアス（将来的に削除予定）
MultiKeyTEBlock = TargetEncodingBlock
HierarchicalTargetEncodingBlock = TargetEncodingBlock
