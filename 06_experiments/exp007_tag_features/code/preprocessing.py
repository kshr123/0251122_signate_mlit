"""
exp007_tag_features 専用の前処理コード

exp006をベースに以下の変更:
1. building_tag_idのMulti-hot + SVD (90 → 15次元)
2. unit_tag_idのMulti-hot + SVD (117 → 30次元)

特徴量数: 119 → 164（+45タグSVD次元）
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "04_src"))

import polars as pl
import numpy as np
from typing import Tuple, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.sparse import csr_matrix

from features.blocks.numeric import NumericBlock
from features.blocks.encoding import (
    CountEncodingBlock,
    TargetEncodingBlock,
)


# =============================================
# 基準値定義（exp005から継承）
# =============================================

AGE_THRESHOLD = 35       # 築年数閾値: 66%ile付近
AREA_THRESHOLD = 80      # 面積閾値: 50%ile付近
MAJOR_CITIES = [13, 14, 23, 27]  # 東京、神奈川、愛知、大阪


# =============================================
# タグ特徴量設定（★exp007で追加）
# =============================================

TAG_COLUMNS = ['building_tag_id', 'unit_tag_id']

# SVD次元数（累積寄与率 ~90%に基づく）
BUILDING_TAG_SVD_DIM = 15  # 90.5%
UNIT_TAG_SVD_DIM = 30      # 86.7%


# =============================================
# 交通アクセス関連カラム定義
# =============================================

# 交通アクセスカラム
ACCESS_COLUMNS = [
    'rosen_name1', 'eki_name1', 'bus_stop1', 'bus_time1', 'walk_distance1',
    'rosen_name2', 'eki_name2', 'bus_stop2', 'bus_time2', 'walk_distance2',
]

# TF-IDF対象のテキストカラム
TFIDF_TEXT_COLUMNS = [
    'rosen_name1', 'rosen_name2', 'eki_name1', 'eki_name2', 'bus_stop1', 'bus_stop2'
]

# TE対象（路線名1のみ）
ROUTE_TE_COLUMNS = ['rosen_name1']

# LE/CE対象
ROUTE_LE_COLUMNS = ['rosen_name1', 'rosen_name2', 'eki_name1', 'eki_name2']
ROUTE_CE_COLUMNS = ['rosen_name1', 'rosen_name2', 'eki_name1', 'eki_name2']

# PCA対象の緯度経度カラム
GEO_COLUMNS = ['lon', 'lat', 'nl', 'el']

# TF-IDF設定
TFIDF_MAX_FEATURES = 20


# =============================================
# 使用するカラム定義（exp005から継承）
# =============================================

# 元データから使用するカラム（68個）
BASE_COLUMNS = [
    # 物件基本情報
    'building_status', 'building_type', 'unit_count', 'lon', 'lat',
    'building_structure', 'total_floor_area', 'floor_count',
    'basement_floor_count', 'year_built', 'building_land_area', 'land_area_all',

    # 土地情報
    'building_land_chimoku', 'land_youto', 'land_toshi', 'land_chisei',
    'land_kenpei', 'land_youseki', 'land_road_cond',

    # 管理情報
    'management_form', 'management_association_flg',

    # 部屋情報
    'room_floor', 'balcony_area', 'dwelling_unit_window_angle',
    'room_count', 'unit_area', 'floor_plan_code',

    # 物件詳細
    'bukken_type', 'flg_investment', 'empty_number',
    'post1', 'post2', 'addr1_1', 'addr1_2',

    # 位置情報
    'nl', 'el', 'snapshot_land_area', 'snapshot_land_shidou',

    # 物件属性
    'house_area', 'flg_new', 'house_kanrinin', 'room_kaisuu',
    'snapshot_window_angle', 'madori_number_all', 'madori_kind_all',

    # 費用情報
    'money_kyoueki', 'money_shuuzen', 'money_shuuzenkikin',
    'money_sonota1', 'money_sonota2', 'money_sonota3',  # → 合計に集約

    # 駐車場
    'parking_money', 'parking_kubun', 'parking_keiyaku',

    # 物件状態
    'genkyo_code', 'usable_status',

    # 周辺施設距離
    'convenience_distance', 'super_distance', 'hospital_distance',
    'park_distance', 'drugstore_distance', 'bank_distance',
    'shopping_street_distance', 'est_other_distance',
]

# 除外カラム（欠損率98%以上）
EXCLUDE_COLUMNS = ['building_area', 'money_hoshou_company']


# =============================================
# エンコーディング設定（exp005から継承 + 拡張）
# =============================================

# ターゲットエンコーディング対象（6個 + 1個）
TARGET_ENCODING_COLUMNS = [
    'addr1_1',          # 都道府県コード (47種類)
    'addr1_2',          # 市区町村コード (126種類)
    'bukken_type',      # 物件タイプ (2種類)
    'land_youto',       # 用途地域 (15種類)
    'land_toshi',       # 都市計画 (6種類)
    'building_age_bin', # 築年数カテゴリ (11種類)
    'rosen_name1',      # 路線名1 (508種類) ★追加
]

# カウントエンコーディング対象（16個 + 4個）
COUNT_ENCODING_COLUMNS = [
    # TE対象（カウントも追加）
    'addr1_1', 'addr1_2', 'bukken_type', 'land_youto', 'land_toshi',
    # その他カテゴリカル
    'building_land_chimoku',  # 地目 (10種類, 欠損52%)
    'building_status',
    'building_type',
    'building_structure',
    'land_chisei',
    'management_form',
    'flg_investment',
    'flg_new',
    'genkyo_code',
    'usable_status',
    'parking_kubun',
    # 路線・駅名（★追加）
    'rosen_name1', 'rosen_name2', 'eki_name1', 'eki_name2',
]


# =============================================
# 数値特徴量（exp005から継承 + 拡張）
# =============================================

# exp003の数値特徴量（lon, lat, nl, elはPCAに置換するため除外）
NUMERIC_FEATURES_BASE = [
    # 数値（lon, lat, nl, elは除外 → geo_pca_0, geo_pca_1に置換）
    'unit_count', 'total_floor_area', 'floor_count',
    'basement_floor_count', 'year_built',  # 変換後
    'building_land_area', 'land_area_all',
    'land_kenpei', 'land_youseki', 'land_road_cond',
    'management_association_flg', 'room_floor', 'balcony_area',
    'dwelling_unit_window_angle', 'room_count', 'unit_area', 'floor_plan_code',
    'empty_number', 'post1', 'post2',
    # 'nl', 'el' は除外 → geo_pca に置換
    'snapshot_land_area', 'snapshot_land_shidou',
    'house_area', 'house_kanrinin', 'room_kaisuu',
    'snapshot_window_angle', 'madori_number_all', 'madori_kind_all',
    'money_kyoueki', 'money_shuuzen', 'money_shuuzenkikin',
    'money_sonota_sum',  # 集約後
    'parking_money', 'parking_keiyaku',
    'convenience_distance', 'super_distance', 'hospital_distance',
    'park_distance', 'drugstore_distance', 'bank_distance',
    'shopping_street_distance', 'est_other_distance',
    # カテゴリカル（元から数値コード）
    'building_status', 'building_type', 'building_structure',
    'building_land_chimoku', 'land_youto', 'land_toshi', 'land_chisei',
    'management_form', 'flg_investment', 'flg_new',
    'genkyo_code', 'usable_status', 'parking_kubun',
    'bukken_type', 'addr1_1', 'addr1_2',
]

# exp005で使用する築年数関連特徴量
AGE_NUMERIC_FEATURES = [
    'building_age',          # 築年数
    'building_age_bin',      # 築年数5年単位カテゴリ
    'old_and_large_flag',    # 築35年以上 & 80㎡以上
    'old_and_rural_flag',    # 築35年以上 & 地方
]

# exp006で追加する交通アクセス数値特徴量
ACCESS_NUMERIC_FEATURES = [
    'walk_time1',            # 徒歩時間1（分）
    'walk_time2',            # 徒歩時間2（分）
    'total_access_time1',    # 総アクセス時間1（分）
    'total_access_time2',    # 総アクセス時間2（分）
]

# 全数値特徴量
NUMERIC_FEATURES = NUMERIC_FEATURES_BASE + AGE_NUMERIC_FEATURES + ACCESS_NUMERIC_FEATURES


# =============================================
# 前処理関数
# =============================================

def transform_year_built(df: pl.DataFrame) -> pl.DataFrame:
    """year_built を YYYYMM → YYYY に変換"""
    return df.with_columns([
        (pl.col('year_built').cast(pl.Utf8).str.slice(0, 4).cast(pl.Int64)).alias('year_built')
    ])


def aggregate_money_sonota(df: pl.DataFrame) -> pl.DataFrame:
    """money_sonota1, money_sonota2, money_sonota3 を合計"""
    return df.with_columns([
        (
            pl.col('money_sonota1').fill_null(0) +
            pl.col('money_sonota2').fill_null(0) +
            pl.col('money_sonota3').fill_null(0)
        ).alias('money_sonota_sum')
    ])


def add_age_features(df: pl.DataFrame) -> pl.DataFrame:
    """築年数関連特徴量を追加"""
    # Step 1: 基本特徴量を追加
    df = df.with_columns([
        # 築年数（year_builtは既にYYYY形式）
        (2024 - pl.col("year_built")).alias("building_age"),

        # 築年数カテゴリ（5年単位、0-10にクリップ）
        ((2024 - pl.col("year_built")) // 5).clip(0, 10).alias("building_age_bin"),

        # 地方フラグ（補助、特徴量には含めない）
        (~pl.col("addr1_1").is_in(MAJOR_CITIES)).cast(pl.Int64).alias("rural_flag"),
    ])

    # Step 2: 交互作用フラグを追加
    df = df.with_columns([
        # 古くて広いフラグ
        (
            (pl.col("building_age") >= AGE_THRESHOLD) &
            (pl.col("house_area") >= AREA_THRESHOLD)
        ).cast(pl.Int64).alias("old_and_large_flag"),

        # 古くて地方フラグ
        (
            (pl.col("building_age") >= AGE_THRESHOLD) &
            (pl.col("rural_flag") == 1)
        ).cast(pl.Int64).alias("old_and_rural_flag"),
    ])

    return df


def add_access_time_features(df: pl.DataFrame) -> pl.DataFrame:
    """交通アクセス時間特徴量を追加"""
    df = df.with_columns([
        # 徒歩時間（分）= 徒歩距離 / 80
        (pl.col('walk_distance1') / 80).alias('walk_time1'),
        (pl.col('walk_distance2') / 80).alias('walk_time2'),
    ])

    df = df.with_columns([
        # 総アクセス時間 = 徒歩時間 + バス時間
        (pl.col('walk_time1') + pl.col('bus_time1').fill_null(0)).alias('total_access_time1'),
        (pl.col('walk_time2') + pl.col('bus_time2').fill_null(0)).alias('total_access_time2'),
    ])

    return df


def preprocess_base(df: pl.DataFrame) -> pl.DataFrame:
    """基本的な前処理"""
    df = transform_year_built(df)
    df = aggregate_money_sonota(df)
    df = add_age_features(df)
    df = add_access_time_features(df)
    return df


class TfidfTransformer:
    """交通アクセステキストのTF-IDF変換"""

    def __init__(self, max_features: int = TFIDF_MAX_FEATURES):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            token_pattern=r'(?u)\b\w+\b',  # 日本語対応
        )
        self.feature_names = []

    def _create_text(self, df: pl.DataFrame) -> List[str]:
        """6カラムを結合してテキスト化（ベクトル化版）"""
        # Polarsで各カラムを文字列化してnullを空文字に置換
        text_cols = []
        for col in TFIDF_TEXT_COLUMNS:
            text_cols.append(
                pl.col(col).cast(pl.Utf8).fill_null('')
            )

        # concat_strで結合（セパレータはスペース）
        combined = df.select(
            pl.concat_str(text_cols, separator=' ').alias('text')
        )
        return combined['text'].to_list()

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """学習データでfitしてtransform"""
        texts = self._create_text(df)
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = [f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]

        # numpy array → polars DataFrame
        tfidf_array = tfidf_matrix.toarray()
        return pl.DataFrame({
            name: tfidf_array[:, i] for i, name in enumerate(self.feature_names)
        })

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """テストデータをtransform"""
        texts = self._create_text(df)
        tfidf_matrix = self.vectorizer.transform(texts)

        tfidf_array = tfidf_matrix.toarray()
        return pl.DataFrame({
            name: tfidf_array[:, i] for i, name in enumerate(self.feature_names)
        })


class LabelEncodingBlock:
    """Label Encoding Block"""

    def __init__(self, columns: List[str]):
        self.columns = columns
        self.encoders = {}

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """学習データでfitしてtransform"""
        result = {}
        for col in self.columns:
            le = LabelEncoder()
            # nullを'__NULL__'に置換してnumpy arrayに変換
            values = df[col].fill_null('__NULL__').to_numpy()
            encoded = le.fit_transform(values)
            self.encoders[col] = le
            result[f'{col}_le'] = encoded
        return pl.DataFrame(result)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """テストデータをtransform（ベクトル化版）"""
        result = {}
        for col in self.columns:
            le = self.encoders[col]
            # クラスのマッピング辞書を作成
            class_to_idx = {c: i for i, c in enumerate(le.classes_)}
            values = df[col].fill_null('__NULL__')
            # mapを使ってベクトル化（未知は-1）
            encoded = values.map_elements(
                lambda v: class_to_idx.get(v, -1),
                return_dtype=pl.Int64
            )
            result[f'{col}_le'] = encoded.to_list()
        return pl.DataFrame(result)


class GeoPCATransformer:
    """緯度経度のPCA次元圧縮"""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.feature_names = [f'geo_pca_{i}' for i in range(n_components)]

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """学習データでfitしてtransform"""
        geo_data = df.select(GEO_COLUMNS).to_numpy()

        # 欠損値を中央値で埋める
        geo_data = np.nan_to_num(geo_data, nan=np.nanmedian(geo_data, axis=0))

        geo_scaled = self.scaler.fit_transform(geo_data)
        geo_pca = self.pca.fit_transform(geo_scaled)

        return pl.DataFrame({
            name: geo_pca[:, i] for i, name in enumerate(self.feature_names)
        })

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """テストデータをtransform"""
        geo_data = df.select(GEO_COLUMNS).to_numpy()

        # 欠損値を中央値で埋める（学習データの統計は使わない簡易版）
        geo_data = np.nan_to_num(geo_data, nan=np.nanmedian(geo_data, axis=0))

        geo_scaled = self.scaler.transform(geo_data)
        geo_pca = self.pca.transform(geo_scaled)

        return pl.DataFrame({
            name: geo_pca[:, i] for i, name in enumerate(self.feature_names)
        })


class TagSVDTransformer:
    """
    タグカラムのMulti-hot + SVD変換

    ★exp007で追加
    - スラッシュ区切りタグをMulti-hot Encoding
    - TruncatedSVDで次元圧縮（スパース行列に最適）
    """

    def __init__(self, column: str, n_components: int, prefix: str = None):
        """
        Args:
            column: タグカラム名 ('building_tag_id' or 'unit_tag_id')
            n_components: SVDの出力次元数
            prefix: 特徴量名のプレフィックス（デフォルトはcolumn名から自動生成）
        """
        self.column = column
        self.n_components = n_components
        self.prefix = prefix or column.replace('_tag_id', '_tag')

        self.mlb = MultiLabelBinarizer()
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.feature_names = [f'{self.prefix}_svd_{i}' for i in range(n_components)]

    def _parse_tags(self, df: pl.DataFrame) -> List[List[str]]:
        """タグカラムをパースしてリストのリストに変換"""
        # nullを空文字に置換してリスト化
        tag_series = df[self.column].fill_null('')

        # スラッシュ区切りでリストに分割
        tag_lists = []
        for tag_str in tag_series.to_list():
            if tag_str and tag_str.strip():
                tags = [t.strip() for t in str(tag_str).split('/') if t.strip()]
            else:
                tags = []
            tag_lists.append(tags)

        return tag_lists

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """学習データでfitしてtransform"""
        tag_lists = self._parse_tags(df)

        # Multi-hot Encoding
        multi_hot = self.mlb.fit_transform(tag_lists)

        # スパース行列に変換してSVD
        sparse_matrix = csr_matrix(multi_hot)
        svd_result = self.svd.fit_transform(sparse_matrix)

        return pl.DataFrame({
            name: svd_result[:, i] for i, name in enumerate(self.feature_names)
        })

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """テストデータをtransform"""
        tag_lists = self._parse_tags(df)

        # Multi-hot Encoding（学習時に見たクラスのみ）
        multi_hot = self.mlb.transform(tag_lists)

        # スパース行列に変換してSVD
        sparse_matrix = csr_matrix(multi_hot)
        svd_result = self.svd.transform(sparse_matrix)

        return pl.DataFrame({
            name: svd_result[:, i] for i, name in enumerate(self.feature_names)
        })

    def get_explained_variance_ratio(self) -> float:
        """SVDの累積寄与率を取得"""
        return self.svd.explained_variance_ratio_.sum()


def get_feature_names(tfidf_dim: int = TFIDF_MAX_FEATURES) -> List[str]:
    """最終的な特徴量名のリストを取得"""
    features = []

    # 数値特徴量
    features.extend(NUMERIC_FEATURES)

    # ターゲットエンコーディング
    for col in TARGET_ENCODING_COLUMNS:
        features.append(f'{col}_te')

    # カウントエンコーディング
    for col in COUNT_ENCODING_COLUMNS:
        features.append(f'{col}_count')

    # Label Encoding（路線・駅名）
    for col in ROUTE_LE_COLUMNS:
        features.append(f'{col}_le')

    # TF-IDF
    for i in range(tfidf_dim):
        features.append(f'tfidf_{i}')

    # Geo PCA
    features.append('geo_pca_0')
    features.append('geo_pca_1')

    # ★exp007追加: タグSVD特徴量
    for i in range(BUILDING_TAG_SVD_DIM):
        features.append(f'building_tag_svd_{i}')
    for i in range(UNIT_TAG_SVD_DIM):
        features.append(f'unit_tag_svd_{i}')

    return features


# カテゴリカル特徴量（LightGBM用）
CATEGORICAL_FEATURES = [
    'building_status', 'building_type', 'building_structure',
    'building_land_chimoku', 'land_youto', 'land_toshi', 'land_chisei',
    'management_form', 'flg_investment', 'flg_new',
    'genkyo_code', 'usable_status', 'parking_kubun',
    'bukken_type', 'addr1_1', 'addr1_2',
    'building_age_bin',
    # 路線・駅名LE
    'rosen_name1_le', 'rosen_name2_le', 'eki_name1_le', 'eki_name2_le',
]


def preprocess_for_training(
    train: pl.DataFrame,
    test: pl.DataFrame,
    cv_splits: list = None,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.Series]:
    """
    学習用の前処理

    Args:
        train: 訓練データ（money_roomカラムを含む）
        test: テストデータ
        cv_splits: CVのfold情報（TargetEncoding用）

    Returns:
        (X_train, X_test, y_train)のタプル
    """
    print("=" * 60)
    print("前処理開始（exp007: tag_features）")
    print("=" * 60)

    # ターゲット変数を分離
    y_train = train["money_room"]
    print(f"\n✓ ターゲット変数分離: {len(y_train)}件")

    # 基本前処理
    print("\n[1/9] 基本前処理")
    train = preprocess_base(train)
    test = preprocess_base(test)
    print("  → year_built: YYYYMM → YYYY")
    print("  → money_sonota_sum: sonota1 + sonota2 + sonota3")
    print("  → 築年数関連特徴量追加（4個）")
    print("  → アクセス時間特徴量追加（4個）")

    # 数値特徴量
    print("\n[2/9] 数値特徴量選択")
    numeric_block = NumericBlock(columns=NUMERIC_FEATURES)
    train_numeric = numeric_block.fit(train)
    test_numeric = numeric_block.transform(test)
    print(f"  → {len(NUMERIC_FEATURES)}個の数値特徴量")

    # ターゲットエンコーディング
    print("\n[3/9] ターゲットエンコーディング")
    if cv_splits is None:
        from sklearn.model_selection import KFold
        cv_splits = list(KFold(n_splits=3, shuffle=True, random_state=42).split(train))

    te_block = TargetEncodingBlock(columns=TARGET_ENCODING_COLUMNS, cv=cv_splits)
    train_te = te_block.fit(train, y=y_train)
    test_te = te_block.transform(test)
    # カラム名を変更
    train_te = train_te.rename({f'TE_{col}': f'{col}_te' for col in TARGET_ENCODING_COLUMNS})
    test_te = test_te.rename({f'TE_{col}': f'{col}_te' for col in TARGET_ENCODING_COLUMNS})
    print(f"  → {len(TARGET_ENCODING_COLUMNS)}個（rosen_name1追加）")

    # カウントエンコーディング
    print("\n[4/9] カウントエンコーディング")
    count_block = CountEncodingBlock(columns=COUNT_ENCODING_COLUMNS)
    train_count = count_block.fit(train)
    test_count = count_block.transform(test)
    train_count = train_count.rename({col: f'{col}_count' for col in COUNT_ENCODING_COLUMNS})
    test_count = test_count.rename({col: f'{col}_count' for col in COUNT_ENCODING_COLUMNS})
    print(f"  → {len(COUNT_ENCODING_COLUMNS)}個（路線・駅名追加）")

    # Label Encoding（路線・駅名）
    print("\n[5/9] ラベルエンコーディング（路線・駅名）")
    le_block = LabelEncodingBlock(columns=ROUTE_LE_COLUMNS)
    train_le = le_block.fit_transform(train)
    test_le = le_block.transform(test)
    print(f"  → {len(ROUTE_LE_COLUMNS)}個")

    # TF-IDF（交通アクセステキスト）
    print("\n[6/9] TF-IDF（交通アクセステキスト）")
    tfidf_transformer = TfidfTransformer(max_features=TFIDF_MAX_FEATURES)
    train_tfidf = tfidf_transformer.fit_transform(train)
    test_tfidf = tfidf_transformer.transform(test)
    print(f"  → {TFIDF_MAX_FEATURES}次元")

    # Geo PCA
    print("\n[7/9] Geo PCA（緯度経度次元圧縮）")
    geo_pca_transformer = GeoPCATransformer(n_components=2)
    train_geo_pca = geo_pca_transformer.fit_transform(train)
    test_geo_pca = geo_pca_transformer.transform(test)
    print(f"  → 4次元 → 2次元")
    print(f"  → 寄与率: {geo_pca_transformer.pca.explained_variance_ratio_}")

    # ★exp007追加: タグSVD特徴量
    print("\n[8/9] タグSVD（building_tag_id）")
    building_tag_transformer = TagSVDTransformer(
        column='building_tag_id',
        n_components=BUILDING_TAG_SVD_DIM,
        prefix='building_tag'
    )
    train_building_tag = building_tag_transformer.fit_transform(train)
    test_building_tag = building_tag_transformer.transform(test)
    building_var_ratio = building_tag_transformer.get_explained_variance_ratio()
    print(f"  → 90タグ → {BUILDING_TAG_SVD_DIM}次元")
    print(f"  → 累積寄与率: {building_var_ratio:.1%}")

    print("\n[9/9] タグSVD（unit_tag_id）")
    unit_tag_transformer = TagSVDTransformer(
        column='unit_tag_id',
        n_components=UNIT_TAG_SVD_DIM,
        prefix='unit_tag'
    )
    train_unit_tag = unit_tag_transformer.fit_transform(train)
    test_unit_tag = unit_tag_transformer.transform(test)
    unit_var_ratio = unit_tag_transformer.get_explained_variance_ratio()
    print(f"  → 117タグ → {UNIT_TAG_SVD_DIM}次元")
    print(f"  → 累積寄与率: {unit_var_ratio:.1%}")

    # 全特徴量を結合
    print("\n[10/9] 全特徴量を結合")
    X_train = pl.concat([
        train_numeric,
        train_te,
        train_count,
        train_le,
        train_tfidf,
        train_geo_pca,
        train_building_tag,
        train_unit_tag,
    ], how="horizontal")

    X_test = pl.concat([
        test_numeric,
        test_te,
        test_count,
        test_le,
        test_tfidf,
        test_geo_pca,
        test_building_tag,
        test_unit_tag,
    ], how="horizontal")

    # 最終特徴量リスト
    ALL_FEATURES = get_feature_names(tfidf_dim=train_tfidf.shape[1])

    # カラム順序を ALL_FEATURES に合わせる
    X_train = X_train.select(ALL_FEATURES)
    X_test = X_test.select(ALL_FEATURES)

    print(f"  → 訓練データ: {X_train.shape}")
    print(f"  → テストデータ: {X_test.shape}")
    print(f"  → 特徴量数: {len(ALL_FEATURES)}個（exp006: 119 → {len(ALL_FEATURES)}）")

    # 新規特徴量の統計を表示
    print("\n[タグSVD特徴量の統計]")
    for prefix, dim in [('building_tag_svd', BUILDING_TAG_SVD_DIM), ('unit_tag_svd', UNIT_TAG_SVD_DIM)]:
        col = f'{prefix}_0'
        if col in X_train.columns:
            mean_val = X_train[col].mean()
            std_val = X_train[col].std()
            print(f"  {prefix}_*: 次元={dim}, {col}平均={mean_val:.4f}, 標準偏差={std_val:.4f}")

    # 検証
    assert X_train.shape[1] == len(ALL_FEATURES), \
        f"特徴量数が一致しません: {X_train.shape[1]} != {len(ALL_FEATURES)}"
    assert X_test.shape[1] == len(ALL_FEATURES), \
        f"特徴量数が一致しません: {X_test.shape[1]} != {len(ALL_FEATURES)}"

    print("\n" + "=" * 60)
    print("前処理完了")
    print("=" * 60)

    return X_train, X_test, y_train


# 最終特徴量リスト（動的に生成）
ALL_FEATURES = get_feature_names()
