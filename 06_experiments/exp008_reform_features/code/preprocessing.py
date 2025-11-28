"""
exp008_reform_features 専用の前処理コード

exp007をベースに以下の変更:
1. リフォーム特徴量（reform_wet_area + reform_interior統合SVD: 7次元）
2. リフォーム経過年数（years_since_wet_reform, years_since_interior_reform）
3. 郵便番号ターゲットエンコーディング（post1_te, post_full_te）
4. 基準年を2024固定 → building_create_dateの年に修正

特徴量数: 164 → 175（+11特徴量）
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "04_src"))

import polars as pl
import numpy as np
from typing import Tuple, List, Optional, Set, Dict
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
# タグ特徴量設定（exp007から継承）
# =============================================

TAG_COLUMNS = ['building_tag_id', 'unit_tag_id']

# SVD次元数（累積寄与率 ~90%に基づく）
BUILDING_TAG_SVD_DIM = 15  # 90.5%
UNIT_TAG_SVD_DIM = 30      # 86.7%


# =============================================
# リフォーム特徴量設定（★exp008で追加）
# =============================================

REFORM_SVD_DIM = 7  # 93.3%（wet + interior統合版）


# =============================================
# 郵便番号TE設定（★exp008で追加）
# =============================================

POST_FULL_MIN_COUNT = 30  # post_fullの最低出現回数（これ未満はpost1_teで補完）


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
# エンコーディング設定（exp008で拡張）
# =============================================

# ターゲットエンコーディング対象（既存7個）- post1は別処理
TARGET_ENCODING_COLUMNS = [
    'addr1_1',          # 都道府県コード (47種類)
    'addr1_2',          # 市区町村コード (126種類)
    'bukken_type',      # 物件タイプ (2種類)
    'land_youto',       # 用途地域 (15種類)
    'land_toshi',       # 都市計画 (6種類)
    'building_age_bin', # 築年数カテゴリ (11種類)
    'rosen_name1',      # 路線名1 (508種類)
]

# 郵便番号TE対象（★exp008追加、別クラスで処理）
POST_TE_COLUMNS = ['post1', 'post_full']

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
# 数値特徴量（exp008で拡張）
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

# ★exp008で追加: リフォーム経過年数特徴量
REFORM_NUMERIC_FEATURES = [
    'years_since_wet_reform',       # 水回りリフォームからの経過年数
    'years_since_interior_reform',  # 内装リフォームからの経過年数
]

# 全数値特徴量
NUMERIC_FEATURES = NUMERIC_FEATURES_BASE + AGE_NUMERIC_FEATURES + ACCESS_NUMERIC_FEATURES + REFORM_NUMERIC_FEATURES


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


def extract_reference_year(df: pl.DataFrame) -> pl.DataFrame:
    """building_create_dateから基準年を抽出"""
    return df.with_columns([
        pl.col('building_create_date').str.slice(0, 4).cast(pl.Int64).alias('reference_year')
    ])


def add_post_full(df: pl.DataFrame) -> pl.DataFrame:
    """★exp008追加: post_full（郵便番号7桁）カラムを追加"""
    return df.with_columns([
        (
            pl.col('post1').cast(pl.Utf8).fill_null('') +
            pl.col('post2').cast(pl.Utf8).fill_null('')
        ).alias('post_full')
    ])


def add_age_features(df: pl.DataFrame) -> pl.DataFrame:
    """築年数関連特徴量を追加（★exp008: 基準年を動的に）"""
    # Step 1: 基本特徴量を追加
    df = df.with_columns([
        # 築年数（★修正: 2024固定 → reference_year）
        (pl.col("reference_year") - pl.col("year_built")).alias("building_age"),

        # 築年数カテゴリ（5年単位、0-10にクリップ）
        ((pl.col("reference_year") - pl.col("year_built")) // 5).clip(0, 10).alias("building_age_bin"),

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


def add_reform_year_features(df: pl.DataFrame) -> pl.DataFrame:
    """★exp008追加: リフォーム経過年数特徴量を追加"""
    df = df.with_columns([
        # reform_wet_area_dateから年を抽出（YYYYMM.0形式）
        # 202107.0 → 2021
        (pl.col('reform_wet_area_date') / 100).floor().cast(pl.Int64).alias('wet_reform_year'),
        (pl.col('reform_interior_date') / 100).floor().cast(pl.Int64).alias('interior_reform_year'),
    ])

    df = df.with_columns([
        # 経過年数 = 基準年 - リフォーム年
        (pl.col('reference_year') - pl.col('wet_reform_year')).alias('years_since_wet_reform'),
        (pl.col('reference_year') - pl.col('interior_reform_year')).alias('years_since_interior_reform'),
    ])

    return df


def preprocess_base(df: pl.DataFrame) -> pl.DataFrame:
    """基本的な前処理"""
    df = transform_year_built(df)
    df = aggregate_money_sonota(df)
    df = extract_reference_year(df)  # ★exp008追加
    df = add_post_full(df)  # ★exp008追加
    df = add_age_features(df)
    df = add_access_time_features(df)
    df = add_reform_year_features(df)  # ★exp008追加
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


class ReformSVDTransformer:
    """
    ★exp008追加: リフォーム特徴量の統合Multi-hot + SVD変換

    reform_wet_area と reform_interior を統合して処理
    - wet_1, wet_2, ... int_1, int_2, ... とプレフィックスで区別
    - Multi-hot 14次元 → SVD 7次元で93.3%の情報を保持
    """

    def __init__(self, n_components: int = REFORM_SVD_DIM):
        self.n_components = n_components
        self.mlb = MultiLabelBinarizer()
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.feature_names = [f'reform_svd_{i}' for i in range(n_components)]

    def _create_reform_tags(self, wet_area: str, interior: str) -> List[str]:
        """wet_area と interior を統合してプレフィックス付きタグリストを作成"""
        tags = []
        if wet_area and str(wet_area) not in ('', 'nan', 'None'):
            tags.extend([f'wet_{t.strip()}' for t in str(wet_area).split('/') if t.strip()])
        if interior and str(interior) not in ('', 'nan', 'None'):
            tags.extend([f'int_{t.strip()}' for t in str(interior).split('/') if t.strip()])
        return tags

    def _parse_tags(self, df: pl.DataFrame) -> List[List[str]]:
        """リフォームカラムをパースしてリストのリストに変換"""
        wet_series = df['reform_wet_area'].fill_null('').to_list()
        int_series = df['reform_interior'].fill_null('').to_list()

        tag_lists = [
            self._create_reform_tags(w, i)
            for w, i in zip(wet_series, int_series)
        ]
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

    def get_classes(self) -> List[str]:
        """学習したタグクラス一覧を取得"""
        return list(self.mlb.classes_)


class PostalCodeTEBlock:
    """
    ★exp008追加: 郵便番号ターゲットエンコーディング

    - post1_te: 郵便番号上3桁のTE
    - post_full_te: 郵便番号7桁のTE（30件未満はpost1_teで補完）
    """

    def __init__(self, min_count: int = POST_FULL_MIN_COUNT, cv: list = None):
        self.min_count = min_count
        self.cv = cv

        # fit時に保存する情報
        self.post1_te_map: Dict[str, float] = {}
        self.post_full_te_map: Dict[str, float] = {}
        self.valid_post_fulls: Set[str] = set()  # 30件以上のpost_full
        self.global_mean: float = 0.0

    def fit(self, df: pl.DataFrame, y: pl.Series) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        学習データでfitしてtransform（OOF方式）

        Returns:
            (post1_te DataFrame, post_full_te DataFrame)
        """
        from sklearn.model_selection import KFold

        n_samples = len(df)
        post1_values = df['post1'].cast(pl.Utf8).fill_null('').to_list()
        post_full_values = df['post_full'].fill_null('').to_list()
        y_values = y.to_numpy()

        self.global_mean = float(np.mean(y_values))

        # post_fullの出現回数を計算
        post_full_counts = df.group_by('post_full').agg(pl.count().alias('count'))
        valid_post_fulls_df = post_full_counts.filter(pl.col('count') >= self.min_count)
        self.valid_post_fulls = set(valid_post_fulls_df['post_full'].to_list())

        # CVでOOFエンコーディング
        if self.cv is None:
            self.cv = list(KFold(n_splits=3, shuffle=True, random_state=42).split(df))

        post1_te_result = np.full(n_samples, np.nan)
        post_full_te_result = np.full(n_samples, np.nan)

        for train_idx, val_idx in self.cv:
            # このfoldの学習データでTE計算
            fold_df = pl.DataFrame({
                'post1': [post1_values[i] for i in train_idx],
                'post_full': [post_full_values[i] for i in train_idx],
                'target': [y_values[i] for i in train_idx],
            })

            # post1のTE
            post1_te_fold = fold_df.group_by('post1').agg(
                pl.col('target').mean().alias('te')
            )
            post1_te_map_fold = dict(zip(
                post1_te_fold['post1'].to_list(),
                post1_te_fold['te'].to_list()
            ))

            # post_fullのTE
            post_full_te_fold = fold_df.group_by('post_full').agg(
                pl.col('target').mean().alias('te')
            )
            post_full_te_map_fold = dict(zip(
                post_full_te_fold['post_full'].to_list(),
                post_full_te_fold['te'].to_list()
            ))

            # validation indexに適用
            for i in val_idx:
                p1 = post1_values[i]
                pf = post_full_values[i]

                # post1_te
                post1_te_result[i] = post1_te_map_fold.get(p1, self.global_mean)

                # post_full_te（30件以上のみ、それ以外はpost1_teで補完）
                if pf in self.valid_post_fulls:
                    post_full_te_result[i] = post_full_te_map_fold.get(pf, post1_te_result[i])
                else:
                    post_full_te_result[i] = post1_te_result[i]

        # 全データでのTE（テストデータ用）を計算
        full_df = pl.DataFrame({
            'post1': post1_values,
            'post_full': post_full_values,
            'target': y_values,
        })

        post1_te_full = full_df.group_by('post1').agg(pl.col('target').mean().alias('te'))
        self.post1_te_map = dict(zip(
            post1_te_full['post1'].to_list(),
            post1_te_full['te'].to_list()
        ))

        post_full_te_full = full_df.group_by('post_full').agg(pl.col('target').mean().alias('te'))
        self.post_full_te_map = dict(zip(
            post_full_te_full['post_full'].to_list(),
            post_full_te_full['te'].to_list()
        ))

        return (
            pl.DataFrame({'post1_te': post1_te_result}),
            pl.DataFrame({'post_full_te': post_full_te_result})
        )

    def transform(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """テストデータをtransform"""
        post1_values = df['post1'].cast(pl.Utf8).fill_null('').to_list()
        post_full_values = df['post_full'].fill_null('').to_list()

        post1_te_result = []
        post_full_te_result = []

        for p1, pf in zip(post1_values, post_full_values):
            # post1_te
            p1_te = self.post1_te_map.get(p1, self.global_mean)
            post1_te_result.append(p1_te)

            # post_full_te（trainで30件以上あったもののみ）
            if pf in self.valid_post_fulls:
                pf_te = self.post_full_te_map.get(pf, p1_te)
            else:
                pf_te = p1_te
            post_full_te_result.append(pf_te)

        return (
            pl.DataFrame({'post1_te': post1_te_result}),
            pl.DataFrame({'post_full_te': post_full_te_result})
        )

    def get_stats(self) -> dict:
        """統計情報を取得"""
        return {
            'post1_unique': len(self.post1_te_map),
            'post_full_unique': len(self.post_full_te_map),
            'valid_post_fulls': len(self.valid_post_fulls),
            'global_mean': self.global_mean,
        }


class AreaRegionalRatioBlock:
    """
    ★exp008追加: 面積の地域平均比率特徴量

    house_area, snapshot_land_area, unit_area について
    都道府県・市区町村単位での平均値との比率を計算（OOF方式）

    比率 = 物件の面積 / 地域平均面積
    - 比率 > 1: 地域平均より広い
    - 比率 < 1: 地域平均より狭い
    """

    AREA_COLUMNS = ['house_area', 'snapshot_land_area', 'unit_area']

    def __init__(self, cv: list = None):
        self.cv = cv

        # fit時に保存する情報（テストデータ用）
        self.pref_means: Dict[str, Dict[int, float]] = {}  # {area_col: {pref_code: mean}}
        self.city_means: Dict[str, Dict[int, float]] = {}  # {area_col: {city_code: mean}}
        self.global_means: Dict[str, float] = {}  # {area_col: global_mean}

        self.feature_names = []
        for col in self.AREA_COLUMNS:
            self.feature_names.append(f'{col}_pref_ratio')
            self.feature_names.append(f'{col}_city_ratio')

    def fit(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        学習データでfitしてtransform（OOF方式）

        Returns:
            6特徴量を含むDataFrame（面積/地域平均の比率）
        """
        from sklearn.model_selection import KFold

        n_samples = len(df)

        # CVの設定
        if self.cv is None:
            self.cv = list(KFold(n_splits=3, shuffle=True, random_state=42).split(df))

        # 地域コードを取得
        pref_codes = df['addr1_1'].to_numpy()
        city_codes = df['addr1_2'].to_numpy()

        # 結果を格納する配列
        results = {name: np.full(n_samples, np.nan) for name in self.feature_names}

        for area_col in self.AREA_COLUMNS:
            area_values = df[area_col].to_numpy()
            self.global_means[area_col] = float(np.nanmean(area_values))

            pref_ratio_col = f'{area_col}_pref_ratio'
            city_ratio_col = f'{area_col}_city_ratio'

            for train_idx, val_idx in self.cv:
                # このfoldの学習データで平均計算
                fold_df = pl.DataFrame({
                    'pref': pref_codes[train_idx],
                    'city': city_codes[train_idx],
                    'area': area_values[train_idx],
                })

                # 都道府県別平均
                pref_mean_fold = fold_df.group_by('pref').agg(
                    pl.col('area').mean().alias('mean')
                )
                pref_mean_map = dict(zip(
                    pref_mean_fold['pref'].to_list(),
                    pref_mean_fold['mean'].to_list()
                ))

                # 市区町村別平均
                city_mean_fold = fold_df.group_by('city').agg(
                    pl.col('area').mean().alias('mean')
                )
                city_mean_map = dict(zip(
                    city_mean_fold['city'].to_list(),
                    city_mean_fold['mean'].to_list()
                ))

                # validation indexに適用（比率を計算）
                for i in val_idx:
                    pref = pref_codes[i]
                    city = city_codes[i]
                    area = area_values[i]

                    pref_mean = pref_mean_map.get(pref, self.global_means[area_col])
                    city_mean = city_mean_map.get(city, self.global_means[area_col])

                    # 比率計算（0除算防止）
                    if pref_mean and pref_mean > 0 and not np.isnan(area):
                        results[pref_ratio_col][i] = area / pref_mean
                    if city_mean and city_mean > 0 and not np.isnan(area):
                        results[city_ratio_col][i] = area / city_mean

            # 全データでの平均（テストデータ用）を計算
            full_df = pl.DataFrame({
                'pref': pref_codes,
                'city': city_codes,
                'area': area_values,
            })

            pref_mean_full = full_df.group_by('pref').agg(pl.col('area').mean().alias('mean'))
            self.pref_means[area_col] = dict(zip(
                pref_mean_full['pref'].to_list(),
                pref_mean_full['mean'].to_list()
            ))

            city_mean_full = full_df.group_by('city').agg(pl.col('area').mean().alias('mean'))
            self.city_means[area_col] = dict(zip(
                city_mean_full['city'].to_list(),
                city_mean_full['mean'].to_list()
            ))

        return pl.DataFrame(results)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """テストデータをtransform"""
        pref_codes = df['addr1_1'].to_list()
        city_codes = df['addr1_2'].to_list()

        results = {name: [] for name in self.feature_names}

        for area_col in self.AREA_COLUMNS:
            area_values = df[area_col].to_list()

            pref_ratio_col = f'{area_col}_pref_ratio'
            city_ratio_col = f'{area_col}_city_ratio'

            for i in range(len(df)):
                pref = pref_codes[i]
                city = city_codes[i]
                area = area_values[i]

                global_mean = self.global_means[area_col]

                pref_mean = self.pref_means[area_col].get(pref, global_mean)
                city_mean = self.city_means[area_col].get(city, global_mean)

                # 比率計算（0除算防止）
                if pref_mean and pref_mean > 0 and area is not None:
                    results[pref_ratio_col].append(area / pref_mean)
                else:
                    results[pref_ratio_col].append(np.nan)

                if city_mean and city_mean > 0 and area is not None:
                    results[city_ratio_col].append(area / city_mean)
                else:
                    results[city_ratio_col].append(np.nan)

        return pl.DataFrame(results)

    def get_stats(self) -> dict:
        """統計情報を取得"""
        return {
            'n_prefs': {col: len(means) for col, means in self.pref_means.items()},
            'n_cities': {col: len(means) for col, means in self.city_means.items()},
            'global_means': self.global_means,
        }


# 面積地域平均比率の特徴量名
AREA_REGIONAL_RATIO_FEATURES = [
    'house_area_pref_ratio', 'house_area_city_ratio',
    'snapshot_land_area_pref_ratio', 'snapshot_land_area_city_ratio',
    'unit_area_pref_ratio', 'unit_area_city_ratio',
]


def get_feature_names(tfidf_dim: int = TFIDF_MAX_FEATURES) -> List[str]:
    """最終的な特徴量名のリストを取得"""
    features = []

    # 数値特徴量
    features.extend(NUMERIC_FEATURES)

    # ターゲットエンコーディング（既存）
    for col in TARGET_ENCODING_COLUMNS:
        features.append(f'{col}_te')

    # ★exp008追加: 郵便番号TE
    features.append('post1_te')
    features.append('post_full_te')

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

    # タグSVD特徴量（exp007）
    for i in range(BUILDING_TAG_SVD_DIM):
        features.append(f'building_tag_svd_{i}')
    for i in range(UNIT_TAG_SVD_DIM):
        features.append(f'unit_tag_svd_{i}')

    # ★exp008追加: リフォームSVD特徴量
    for i in range(REFORM_SVD_DIM):
        features.append(f'reform_svd_{i}')

    # ★exp008追加: 面積地域平均比率特徴量
    features.extend(AREA_REGIONAL_RATIO_FEATURES)

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
    print("前処理開始（exp008: reform_features）")
    print("=" * 60)

    # ターゲット変数を分離
    y_train = train["money_room"]
    print(f"\n✓ ターゲット変数分離: {len(y_train)}件")

    # 基本前処理
    print("\n[1/13] 基本前処理")
    train = preprocess_base(train)
    test = preprocess_base(test)
    print("  → year_built: YYYYMM → YYYY")
    print("  → money_sonota_sum: sonota1 + sonota2 + sonota3")
    print("  → 基準年抽出（building_create_date）★exp008")
    print("  → post_full作成（post1 + post2）★exp008")
    print("  → 築年数関連特徴量追加（4個）- 基準年使用")
    print("  → アクセス時間特徴量追加（4個）")
    print("  → リフォーム経過年数追加（2個）★exp008")

    # 数値特徴量
    print("\n[2/13] 数値特徴量選択")
    numeric_block = NumericBlock(columns=NUMERIC_FEATURES)
    train_numeric = numeric_block.fit(train)
    test_numeric = numeric_block.transform(test)
    print(f"  → {len(NUMERIC_FEATURES)}個の数値特徴量（+2 リフォーム経過年数）")

    # ターゲットエンコーディング（既存カラム）
    print("\n[3/13] ターゲットエンコーディング（既存）")
    if cv_splits is None:
        from sklearn.model_selection import KFold
        cv_splits = list(KFold(n_splits=3, shuffle=True, random_state=42).split(train))

    te_block = TargetEncodingBlock(columns=TARGET_ENCODING_COLUMNS, cv=cv_splits)
    train_te = te_block.fit(train, y=y_train)
    test_te = te_block.transform(test)
    # カラム名を変更
    train_te = train_te.rename({f'TE_{col}': f'{col}_te' for col in TARGET_ENCODING_COLUMNS})
    test_te = test_te.rename({f'TE_{col}': f'{col}_te' for col in TARGET_ENCODING_COLUMNS})
    print(f"  → {len(TARGET_ENCODING_COLUMNS)}個")

    # ★exp008追加: 郵便番号TE
    print("\n[4/13] 郵便番号TE（post1_te, post_full_te）★exp008")
    post_te_block = PostalCodeTEBlock(min_count=POST_FULL_MIN_COUNT, cv=cv_splits)
    train_post1_te, train_post_full_te = post_te_block.fit(train, y=y_train)
    test_post1_te, test_post_full_te = post_te_block.transform(test)
    post_stats = post_te_block.get_stats()
    print(f"  → post1ユニーク数: {post_stats['post1_unique']}")
    print(f"  → post_fullユニーク数: {post_stats['post_full_unique']}")
    print(f"  → 有効post_full数（≥{POST_FULL_MIN_COUNT}件）: {post_stats['valid_post_fulls']}")

    # カウントエンコーディング
    print("\n[5/13] カウントエンコーディング")
    count_block = CountEncodingBlock(columns=COUNT_ENCODING_COLUMNS)
    train_count = count_block.fit(train)
    test_count = count_block.transform(test)
    train_count = train_count.rename({col: f'{col}_count' for col in COUNT_ENCODING_COLUMNS})
    test_count = test_count.rename({col: f'{col}_count' for col in COUNT_ENCODING_COLUMNS})
    print(f"  → {len(COUNT_ENCODING_COLUMNS)}個")

    # Label Encoding（路線・駅名）
    print("\n[6/13] ラベルエンコーディング（路線・駅名）")
    le_block = LabelEncodingBlock(columns=ROUTE_LE_COLUMNS)
    train_le = le_block.fit_transform(train)
    test_le = le_block.transform(test)
    print(f"  → {len(ROUTE_LE_COLUMNS)}個")

    # TF-IDF（交通アクセステキスト）
    print("\n[7/13] TF-IDF（交通アクセステキスト）")
    tfidf_transformer = TfidfTransformer(max_features=TFIDF_MAX_FEATURES)
    train_tfidf = tfidf_transformer.fit_transform(train)
    test_tfidf = tfidf_transformer.transform(test)
    print(f"  → {TFIDF_MAX_FEATURES}次元")

    # Geo PCA
    print("\n[8/13] Geo PCA（緯度経度次元圧縮）")
    geo_pca_transformer = GeoPCATransformer(n_components=2)
    train_geo_pca = geo_pca_transformer.fit_transform(train)
    test_geo_pca = geo_pca_transformer.transform(test)
    print(f"  → 4次元 → 2次元")
    print(f"  → 寄与率: {geo_pca_transformer.pca.explained_variance_ratio_}")

    # タグSVD特徴量（exp007から継承）
    print("\n[9/13] タグSVD（building_tag_id）")
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

    print("\n[10/13] タグSVD（unit_tag_id）")
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

    # ★exp008追加: リフォームSVD特徴量
    print("\n[11/13] リフォームSVD（reform_wet_area + reform_interior統合）★exp008")
    reform_transformer = ReformSVDTransformer(n_components=REFORM_SVD_DIM)
    train_reform = reform_transformer.fit_transform(train)
    test_reform = reform_transformer.transform(test)
    reform_var_ratio = reform_transformer.get_explained_variance_ratio()
    reform_classes = reform_transformer.get_classes()
    print(f"  → 14タグ（wet_1-6 + int_1-6） → {REFORM_SVD_DIM}次元")
    print(f"  → 累積寄与率: {reform_var_ratio:.1%}")
    print(f"  → 学習したクラス: {reform_classes}")

    # ★exp008追加: 面積地域平均比率特徴量
    print("\n[12/13] 面積地域平均比率特徴量 ★exp008")
    area_ratio_block = AreaRegionalRatioBlock(cv=cv_splits)
    train_area_ratio = area_ratio_block.fit(train)
    test_area_ratio = area_ratio_block.transform(test)
    print(f"  → 対象カラム: {AreaRegionalRatioBlock.AREA_COLUMNS}")
    print(f"  → 都道府県比率 × 3 + 市区町村比率 × 3 = 6次元")

    # 全特徴量を結合
    print("\n[13/13] 全特徴量を結合")
    X_train = pl.concat([
        train_numeric,
        train_te,
        train_post1_te,      # ★exp008追加
        train_post_full_te,  # ★exp008追加
        train_count,
        train_le,
        train_tfidf,
        train_geo_pca,
        train_building_tag,
        train_unit_tag,
        train_reform,        # ★exp008追加
        train_area_ratio,    # ★exp008追加
    ], how="horizontal")

    X_test = pl.concat([
        test_numeric,
        test_te,
        test_post1_te,      # ★exp008追加
        test_post_full_te,  # ★exp008追加
        test_count,
        test_le,
        test_tfidf,
        test_geo_pca,
        test_building_tag,
        test_unit_tag,
        test_reform,        # ★exp008追加
        test_area_ratio,    # ★exp008追加
    ], how="horizontal")

    # 最終特徴量リスト
    ALL_FEATURES = get_feature_names(tfidf_dim=train_tfidf.shape[1])

    # カラム順序を ALL_FEATURES に合わせる
    X_train = X_train.select(ALL_FEATURES)
    X_test = X_test.select(ALL_FEATURES)

    print(f"  → 訓練データ: {X_train.shape}")
    print(f"  → テストデータ: {X_test.shape}")
    print(f"  → 特徴量数: {len(ALL_FEATURES)}個（exp007: 164 → {len(ALL_FEATURES)}）")

    # 新規特徴量の統計を表示
    print("\n[exp008追加特徴量の統計]")
    for col in ['years_since_wet_reform', 'years_since_interior_reform', 'post1_te', 'post_full_te']:
        if col in X_train.columns:
            non_null = X_train[col].drop_nulls()
            mean_val = non_null.mean() if len(non_null) > 0 else float('nan')
            null_rate = (len(X_train) - len(non_null)) / len(X_train) * 100
            print(f"  {col}: 平均={mean_val:.2f}, 欠損率={null_rate:.1f}%")

    for i in range(REFORM_SVD_DIM):
        col = f'reform_svd_{i}'
        if col in X_train.columns:
            mean_val = X_train[col].mean()
            std_val = X_train[col].std()
            print(f"  reform_svd_*: 次元={REFORM_SVD_DIM}, reform_svd_0平均={mean_val:.4f}, 標準偏差={std_val:.4f}")
            break

    # 面積比率特徴量の統計
    for col in AREA_REGIONAL_RATIO_FEATURES:
        if col in X_train.columns:
            non_null = X_train[col].drop_nulls()
            mean_val = non_null.mean() if len(non_null) > 0 else float('nan')
            null_rate = (len(X_train) - len(non_null)) / len(X_train) * 100
            print(f"  {col}: 平均={mean_val:.3f}, 欠損率={null_rate:.1f}%")

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
