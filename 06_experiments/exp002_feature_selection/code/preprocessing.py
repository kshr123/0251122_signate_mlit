"""
exp002_feature_selection 専用の前処理コード（Block System使用）

このファイルでは以下のカラムを除外：
1. 削除フラグが立っているカラム（13個）
2. 欠損率95%以上のカラム（31個）
3. その他不要カラム（id, target_ym）
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
# 除外カラム定義
# =============================================

# 削除フラグカラム（コンペ運営指定、13個）
DROP_FLAG_COLUMNS = {
    "building_id",           # No.3: 棟ID（AUTO_INCREMENT）
    "building_create_date",  # No.5: 作成日時
    "building_modify_date",  # No.6: 修正日時
    "building_name",         # No.8: 建物名
    "building_name_ruby",    # No.9: 建物名フリガナ
    "homes_building_name",   # No.10: HOME'S建物名
    "homes_building_name_ruby",  # No.11: HOME'S建物名フリガナ
    "full_address",          # No.13: 住所（全住所文字列）
    "snapshot_create_date",  # No.69: 作成日時
    "new_date",              # No.70: 公開日時
    "snapshot_modify_date",  # No.71: 修正日時
    "school_ele_name",       # No.130: 小学校名
    "school_jun_name",       # No.133: 中学校名
}

# 欠損率95%以上のカラム（31個）
HIGH_MISSING_COLUMNS = {
    "building_name_ruby",       # 100.00% (削除フラグとも重複)
    "name_ruby",                # 100.00%
    "school_ele_code",          # 100.00%
    "school_jun_code",          # 100.00%
    "money_hoshou_company",     # 100.00%
    "free_rent_duration",       # 100.00%
    "free_rent_gen_timing",     # 100.00%
    "traffic_car",              # 100.00%
    "reform_etc",               # 99.88%
    "reform_place_other",       # 99.76%
    "reform_place",             # 99.72%
    "reform_date",              # 99.65%
    "reform_common_area",       # 99.37%
    "reform_common_area_date",  # 99.37%
    "money_sonota_str3",        # 99.15%
    "money_sonota3",            # 99.00%
    "reform_exterior_other",    # 98.77%
    "money_shuuzenkikin",       # 98.44%
    "reform_wet_area_other",    # 98.16%
    "building_area",            # 98.07%
    "money_rimawari_now",       # 98.02%
    "parking_keiyaku",          # 97.91%
    "money_sonota_str2",        # 97.33%
    "money_sonota2",            # 97.17%
    "land_shidou_a",            # 96.30%
    "land_shidou_b",            # 96.19%
    "usable_date",              # 95.97%
    "reform_exterior",          # 95.42%
    "renovation_etc",           # 95.14%
    "renovation_date",          # 95.14%
    "reform_exterior_date",     # 95.06%
}

# その他除外カラム
OTHER_DROP_COLUMNS = {
    "id",                    # 行ID（予測に不要）
    "target_ym",             # TargetYmBlockで分解するので元カラムは不要
    "money_room",            # ターゲット変数
}

# 除外対象の全カラム
ALL_DROP_COLUMNS = DROP_FLAG_COLUMNS | HIGH_MISSING_COLUMNS | OTHER_DROP_COLUMNS


# =============================================
# 使用する特徴量の明示的なリスト（除外後）
# =============================================

# 数値特徴量（除外カラムを除いた78個）
NUMERIC_FEATURES = [
    # 物件情報
    "building_status", "building_type", "unit_count",
    "lon", "lat", "building_structure", "total_floor_area",
    "floor_count", "basement_floor_count", "year_built",

    # 土地情報
    "building_land_area", "land_area_all", "unit_area_min", "unit_area_max",
    "building_land_chimoku", "land_youto", "land_toshi", "land_chisei",
    "land_area_kind", "land_setback_flg", "land_setback", "land_kenpei",
    "land_youseki", "land_road_cond", "building_area_kind",

    # 管理情報
    "management_form", "management_association_flg",

    # 部屋情報
    "unit_id", "room_floor", "balcony_area", "dwelling_unit_window_angle",
    "room_count", "unit_area", "floor_plan_code",
    "reform_wet_area_date", "reform_interior_date",

    # 物件詳細
    "bukken_id", "bukken_type", "flg_investment", "empty_number",
    "post1", "post2", "addr1_1", "addr1_2",

    # 位置・交通
    "nl", "el", "bus_time1", "walk_distance1", "bus_time2", "walk_distance2",

    # 土地面積詳細
    "snapshot_land_area", "snapshot_land_shidou",
    "land_mochibun_a", "land_mochibun_b",

    # 物件属性
    "house_area", "flg_new", "house_kanrinin", "room_kaisuu",
    "snapshot_window_angle", "madori_number_all", "madori_kind_all",

    # 費用情報
    "money_kyoueki", "money_kyoueki_tax",
    "money_shuuzen",
    "money_sonota1",

    # 駐車場情報
    "parking_money", "parking_money_tax", "parking_kubun",
    "parking_distance", "parking_number",

    # 物件状態
    "genkyo_code", "usable_status",

    # 周辺施設距離
    "school_ele_distance", "school_jun_distance",
    "convenience_distance", "super_distance", "hospital_distance",
    "park_distance", "drugstore_distance", "bank_distance",
    "shopping_street_distance", "est_other_distance",
]

# カテゴリカル特徴量（全て欠損率95%以上で除外されたため空）
CATEGORICAL_FEATURES = []

# 生成する特徴量（2個）
GENERATED_FEATURES = ["target_year", "target_month"]

# 全特徴量
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
    3. (カテゴリカル特徴量は全て欠損率95%以上のため除外)
    4. 特徴量を結合

    Args:
        train: 訓練データ（money_roomカラムを含む）
        test: テストデータ

    Returns:
        (X_train, X_test, y_train)のタプル
    """
    print("=" * 60)
    print("前処理開始（exp002: 特徴量選択）")
    print("=" * 60)

    # 除外カラム情報を表示
    print(f"\n📋 除外カラム数:")
    print(f"  - 削除フラグ: {len(DROP_FLAG_COLUMNS)}個")
    print(f"  - 欠損率95%以上: {len(HIGH_MISSING_COLUMNS)}個")
    print(f"  - その他: {len(OTHER_DROP_COLUMNS)}個")
    print(f"  - 合計除外（重複除く）: {len(ALL_DROP_COLUMNS)}個")

    # ターゲット変数を分離
    y_train = train["money_room"]
    print(f"\n✓ ターゲット変数分離: {len(y_train)}件")

    # 1. TargetYmBlock: target_ym → target_year, target_month
    print("\n[1/2] TargetYmBlock: target_ym分解")
    target_ym_block = TargetYmBlock(source_col="target_ym")
    train_ym = target_ym_block.fit(train)
    test_ym = target_ym_block.transform(test)
    print(f"  → 生成: target_year, target_month")

    # 2. NumericBlock: 数値特徴量を選択
    print("\n[2/2] NumericBlock: 数値特徴量選択")
    numeric_block = NumericBlock(columns=NUMERIC_FEATURES)
    train_numeric = numeric_block.fit(train)
    test_numeric = numeric_block.transform(test)
    print(f"  → 選択: {len(NUMERIC_FEATURES)}個の数値特徴量")

    # 特徴量を結合（横方向）
    print("\n[結合] 全特徴量を結合")
    X_train = pl.concat([train_numeric, train_ym], how="horizontal")
    X_test = pl.concat([test_numeric, test_ym], how="horizontal")

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
