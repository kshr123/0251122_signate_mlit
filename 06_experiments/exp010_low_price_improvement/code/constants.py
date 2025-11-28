"""
exp010 定数定義

パス定義とカラムリストなど、コードから参照する定数を集約。
ハイパーパラメータや実験設定は experiment.yaml を参照。
"""

from pathlib import Path
from typing import List

# =============================================================================
# 1. パス設定
# =============================================================================

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# データパス
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# 地価公示データ
LANDPRICE_BASE_PATH = EXTERNAL_DATA_DIR / "landprice"

# 実験出力先
EXP_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = EXP_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"


# =============================================================================
# 2. 特徴量カラム定義
# =============================================================================

# -----------------------------------------------------------------------------
# 2.1 元データから使用するカラム（68個）
# -----------------------------------------------------------------------------
BASE_COLUMNS: List[str] = [
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
EXCLUDE_COLUMNS: List[str] = ['building_area', 'money_hoshou_company']

# -----------------------------------------------------------------------------
# 2.2 交通アクセス関連カラム
# -----------------------------------------------------------------------------
ACCESS_COLUMNS: List[str] = [
    'rosen_name1', 'eki_name1', 'bus_stop1', 'bus_time1', 'walk_distance1',
    'rosen_name2', 'eki_name2', 'bus_stop2', 'bus_time2', 'walk_distance2',
]

# TF-IDF対象のテキストカラム
TFIDF_TEXT_COLUMNS: List[str] = [
    'rosen_name1', 'rosen_name2', 'eki_name1', 'eki_name2', 'bus_stop1', 'bus_stop2'
]

# 緯度経度カラム（PCA対象）
GEO_COLUMNS: List[str] = ['lon', 'lat', 'nl', 'el']

# タグカラム
TAG_COLUMNS: List[str] = ['building_tag_id', 'unit_tag_id']

# ステータスカラム（MultiHot形式: 210201/210101/210301...）
STATUS_COLUMNS: List[str] = ['statuses']


# =============================================================================
# 3. エンコーディング設定
# =============================================================================

# -----------------------------------------------------------------------------
# 3.1 ターゲットエンコーディング対象（7個）
# -----------------------------------------------------------------------------
TARGET_ENCODING_COLUMNS: List[str] = [
    'addr1_1',          # 都道府県コード (47種類)
    'addr1_2',          # 市区町村コード (126種類)
    'bukken_type',      # 物件タイプ (2種類)
    'land_youto',       # 用途地域 (15種類)
    'land_toshi',       # 都市計画 (6種類)
    'building_age_bin', # 築年数カテゴリ (11種類)
    'rosen_name1',      # 路線名1 (508種類)
]

# 郵便番号TE対象（別クラスで処理）
POST_TE_COLUMNS: List[str] = ['post1', 'post_full']

# -----------------------------------------------------------------------------
# 3.2 カウントエンコーディング対象（20個）
# -----------------------------------------------------------------------------
COUNT_ENCODING_COLUMNS: List[str] = [
    # TE対象（カウントも追加）
    'addr1_1', 'addr1_2', 'bukken_type', 'land_youto', 'land_toshi',
    # その他カテゴリカル
    'building_land_chimoku', 'building_status', 'building_type',
    'building_structure', 'land_chisei', 'management_form',
    'flg_investment', 'flg_new', 'genkyo_code', 'usable_status', 'parking_kubun',
    # 路線・駅名
    'rosen_name1', 'rosen_name2', 'eki_name1', 'eki_name2',
]

# -----------------------------------------------------------------------------
# 3.3 ラベルエンコーディング対象（路線・駅名）
# -----------------------------------------------------------------------------
ROUTE_LE_COLUMNS: List[str] = ['rosen_name1', 'rosen_name2', 'eki_name1', 'eki_name2']


# =============================================================================
# 4. exp010固有特徴量名
# =============================================================================

EXP010_FEATURE_COLUMNS: List[str] = [
    'lp_area_value',           # 土地価値目安（lp_price × house_area）
    'area_age_category',       # 面積×築年数カテゴリ（0,1,2）
    'area_age_cat_te_pref',    # カテゴリ×都道府県TE
    'area_age_cat_te_youto',   # カテゴリ×用途地域TE
]


# =============================================================================
# 5. 数値特徴量リスト
# =============================================================================

# 基本数値特徴量（lon, lat, nl, elはPCAに置換するため除外）
NUMERIC_FEATURES_BASE: List[str] = [
    'unit_count', 'total_floor_area', 'floor_count',
    'basement_floor_count', 'year_built',
    'building_land_area', 'land_area_all',
    'land_kenpei', 'land_youseki', 'land_road_cond',
    'management_association_flg', 'room_floor', 'balcony_area',
    'dwelling_unit_window_angle', 'room_count', 'unit_area', 'floor_plan_code',
    'empty_number', 'post1', 'post2',
    'snapshot_land_area', 'snapshot_land_shidou',
    'house_area', 'house_kanrinin', 'room_kaisuu',
    'snapshot_window_angle', 'madori_number_all', 'madori_kind_all',
    'money_kyoueki', 'money_shuuzen', 'money_shuuzenkikin',
    'money_sonota_sum',
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

# 築年数関連特徴量
AGE_NUMERIC_FEATURES: List[str] = [
    'building_age',          # 築年数
    'building_age_bin',      # 築年数5年単位カテゴリ
    'old_and_large_flag',    # 築35年以上 & 80㎡以上
    'old_and_rural_flag',    # 築35年以上 & 地方
]

# 交通アクセス数値特徴量
ACCESS_NUMERIC_FEATURES: List[str] = [
    'walk_time1',            # 徒歩時間1（分）
    'walk_time2',            # 徒歩時間2（分）
    'total_access_time1',    # 総アクセス時間1（分）
    'total_access_time2',    # 総アクセス時間2（分）
]

# リフォーム経過年数特徴量
REFORM_NUMERIC_FEATURES: List[str] = [
    'years_since_wet_reform',       # 水回りリフォームからの経過年数
    'years_since_interior_reform',  # 内装リフォームからの経過年数
]

# 全数値特徴量
NUMERIC_FEATURES: List[str] = (
    NUMERIC_FEATURES_BASE +
    AGE_NUMERIC_FEATURES +
    ACCESS_NUMERIC_FEATURES +
    REFORM_NUMERIC_FEATURES
)

# 面積地域平均比率特徴量名
AREA_REGIONAL_RATIO_FEATURES: List[str] = [
    'house_area_pref_ratio', 'house_area_city_ratio',
    'snapshot_land_area_pref_ratio', 'snapshot_land_area_city_ratio',
    'unit_area_pref_ratio', 'unit_area_city_ratio',
]

# カテゴリカル特徴量（LightGBM用）
CATEGORICAL_FEATURES: List[str] = [
    'building_status', 'building_type', 'building_structure',
    'building_land_chimoku', 'land_youto', 'land_toshi', 'land_chisei',
    'management_form', 'flg_investment', 'flg_new',
    'genkyo_code', 'usable_status', 'parking_kubun',
    'bukken_type', 'addr1_1', 'addr1_2',
    'building_age_bin',
    # 路線・駅名LE（LabelEncodingBlockは同じカラム名で出力）
    'rosen_name1', 'rosen_name2', 'eki_name1', 'eki_name2',
]


# =============================================================================
# 6. 削除特徴量（低重要度35個）
# =============================================================================

REMOVE_FEATURES: List[str] = [
    # 重要度0 or 極低
    "basement_floor_count",
    "lp_current_use_le",
    "groupby_bukken_type_lp_price_mean",

    # カウント系（低重要度）
    "building_land_chimoku_count",
    "building_status_count",
    "building_structure_count",
    "building_type_count",
    "bukken_type_count",
    "flg_investment_count",
    "flg_new_count",
    "land_chisei_count",
    "land_toshi_count",
    "management_form_count",
    "parking_kubun_count",
    "usable_status_count",

    # TF-IDF全削除（20個、重要度0）
    "tfidf_0",
    "tfidf_1",
    "tfidf_2",
    "tfidf_3",
    "tfidf_4",
    "tfidf_5",
    "tfidf_6",
    "tfidf_7",
    "tfidf_8",
    "tfidf_9",
    "tfidf_10",
    "tfidf_11",
    "tfidf_12",
    "tfidf_13",
    "tfidf_14",
    "tfidf_15",
    "tfidf_16",
    "tfidf_17",
    "tfidf_18",
    "tfidf_19",
]
