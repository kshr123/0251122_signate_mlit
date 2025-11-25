"""
exp001_baseline 専用の前処理コード（Block System使用）

このファイルは、Blockシステムを使って明示的に特徴量エンジニアリングを記述しています。
- NumericBlock: 数値特徴量をそのまま使用
- TargetYmBlock: target_ymを年・月に分解
- LabelEncodingBlock: カテゴリカル特徴量を数値化
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "04_src"))

import polars as pl
from typing import Tuple

from features.blocks.numeric import NumericBlock
from features.blocks.temporal import TargetYmBlock
from features.blocks.encoding import LabelEncodingBlock


# =============================================
# 使用する特徴量の明示的なリスト
# =============================================

# 数値特徴量（96個）
NUMERIC_FEATURES = [
    # 物件情報
    "building_id", "building_status", "building_type", "unit_count",
    "lon", "lat", "building_structure", "total_floor_area", "building_area",
    "floor_count", "basement_floor_count", "year_built",

    # 土地情報
    "building_land_area", "land_area_all", "unit_area_min", "unit_area_max",
    "building_land_chimoku", "land_youto", "land_toshi", "land_chisei",
    "land_area_kind", "land_setback_flg", "land_setback", "land_kenpei",
    "land_youseki", "land_road_cond", "building_area_kind",

    # 管理情報
    "management_form", "management_association_flg",
    "reform_exterior_date", "reform_common_area_date",

    # 部屋情報
    "unit_id", "room_floor", "balcony_area", "dwelling_unit_window_angle",
    "room_count", "unit_area", "floor_plan_code",
    "reform_date", "reform_wet_area_date", "reform_interior_date",

    # 物件詳細
    "bukken_id", "bukken_type", "flg_investment", "empty_number",
    "post1", "post2", "addr1_1", "addr1_2",

    # 位置・交通
    "nl", "el", "bus_time1", "walk_distance1", "bus_time2", "walk_distance2",
    "traffic_car",

    # 土地面積詳細
    "snapshot_land_area", "snapshot_land_shidou",
    "land_shidou_a", "land_shidou_b", "land_mochibun_a", "land_mochibun_b",

    # 物件属性
    "house_area", "flg_new", "house_kanrinin", "room_kaisuu",
    "snapshot_window_angle", "madori_number_all", "madori_kind_all",

    # 費用情報
    "money_kyoueki", "money_kyoueki_tax", "money_rimawari_now",
    "money_shuuzen", "money_shuuzenkikin",
    "money_sonota1", "money_sonota2", "money_sonota3",

    # 駐車場情報
    "parking_money", "parking_money_tax", "parking_kubun",
    "parking_distance", "parking_number", "parking_keiyaku",

    # 物件状態
    "genkyo_code", "usable_status", "usable_date",

    # 周辺施設距離
    "school_ele_distance", "school_jun_distance",
    "convenience_distance", "super_distance", "hospital_distance",
    "park_distance", "drugstore_distance", "bank_distance",
    "shopping_street_distance", "est_other_distance",
]

# カテゴリカル特徴量（8個）- カーディナリティ < 50 のみ
CATEGORICAL_FEATURES = [
    "building_name_ruby",      # 建物名（読み）
    "reform_exterior",          # 外装リフォーム
    "name_ruby",                # 名称（読み）
    "school_ele_code",          # 小学校コード
    "school_jun_code",          # 中学校コード
    "money_hoshou_company",     # 保証会社
    "free_rent_duration",       # フリーレント期間
    "free_rent_gen_timing",     # フリーレント開始タイミング
]

# 生成する特徴量（2個）
GENERATED_FEATURES = ["target_year", "target_month"]

# 全特徴量（106個）
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + GENERATED_FEATURES


# =============================================
# 前処理関数（Block System使用）
# =============================================

def preprocess_for_training(
    train: pl.DataFrame,
    test: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.Series]:
    """
    学習用の前処理（Block Systemを使用）

    処理フロー:
    1. TargetYmBlock: target_ym → target_year, target_month
    2. NumericBlock: 数値特徴量を選択
    3. LabelEncodingBlock: カテゴリカル特徴量を数値化
    4. 特徴量を結合

    Args:
        train: 訓練データ（money_roomカラムを含む）
        test: テストデータ

    Returns:
        (X_train, X_test, y_train)のタプル
        - X_train: 訓練データの特徴量（106カラム）
        - X_test: テストデータの特徴量（106カラム）
        - y_train: 訓練データのターゲット（money_room）
    """
    print("=" * 60)
    print("前処理開始（Block System）")
    print("=" * 60)

    # ターゲット変数を分離
    y_train = train["money_room"]
    print(f"✓ ターゲット変数分離: {len(y_train)}件")

    # 1. TargetYmBlock: target_ym → target_year, target_month
    print("\n[1/3] TargetYmBlock: target_ym分解")
    target_ym_block = TargetYmBlock(source_col="target_ym")
    train_ym = target_ym_block.fit(train)
    test_ym = target_ym_block.transform(test)
    print(f"  → 生成: target_year, target_month")

    # 2. NumericBlock: 数値特徴量を選択
    print("\n[2/3] NumericBlock: 数値特徴量選択")
    numeric_block = NumericBlock(columns=NUMERIC_FEATURES)
    train_numeric = numeric_block.fit(train)
    test_numeric = numeric_block.transform(test)
    print(f"  → 選択: {len(NUMERIC_FEATURES)}個の数値特徴量")

    # 3. LabelEncodingBlock: カテゴリカル特徴量を数値化
    print("\n[3/3] LabelEncodingBlock: カテゴリカル特徴量を数値化")
    encoding_block = LabelEncodingBlock(columns=CATEGORICAL_FEATURES)
    train_categorical = encoding_block.fit(train)
    test_categorical = encoding_block.transform(test)
    print(f"  → 変換: {len(CATEGORICAL_FEATURES)}個のカテゴリカル特徴量")

    # 4. 特徴量を結合（横方向）
    print("\n[結合] 全特徴量を結合")
    X_train = pl.concat([train_numeric, train_categorical, train_ym], how="horizontal")
    X_test = pl.concat([test_numeric, test_categorical, test_ym], how="horizontal")

    print(f"  → 訓練データ: {X_train.shape}")
    print(f"  → テストデータ: {X_test.shape}")
    print(f"  → 特徴量数: {len(ALL_FEATURES)}個")

    # 検証: カラム数が正しいか
    assert X_train.shape[1] == len(ALL_FEATURES), \
        f"特徴量数が一致しません: {X_train.shape[1]} != {len(ALL_FEATURES)}"
    assert X_test.shape[1] == len(ALL_FEATURES), \
        f"特徴量数が一致しません: {X_test.shape[1]} != {len(ALL_FEATURES)}"

    print("\n" + "=" * 60)
    print("前処理完了")
    print("=" * 60)

    return X_train, X_test, y_train


def preprocess_for_prediction(test: pl.DataFrame) -> pl.DataFrame:
    """
    推論用の前処理

    注意: 訓練時と同じBlockインスタンスを使う必要があります。
    このため、推論時は訓練時に保存したBlockを読み込む必要があります。

    Args:
        test: テストデータ

    Returns:
        特徴量行列（106カラム）
    """
    raise NotImplementedError(
        "推論用の前処理は未実装です。訓練時に保存したBlockを読み込む必要があります。"
    )
