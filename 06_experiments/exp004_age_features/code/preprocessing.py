"""
exp004_age_features 専用の前処理コード

exp003をベースに、築年数関連特徴量を追加:
1. building_age: 築年数（2024 - year_built）
2. building_age_bin: 築年数5年単位カテゴリ（0-10）
3. old_building_flag: 築35年以上フラグ
4. old_and_large_flag: 築35年以上 & 80㎡以上フラグ
5. old_and_rural_flag: 築35年以上 & 地方フラグ
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "04_src"))

import polars as pl
import numpy as np
from typing import Tuple, List

from features.blocks.numeric import NumericBlock
from features.blocks.encoding import (
    CountEncodingBlock,
    TargetEncodingBlock,
)


# =============================================
# 基準値定義（eda_threshold_analysis.ipynbより）
# =============================================

AGE_THRESHOLD = 35       # 築年数閾値: 66%ile付近
AREA_THRESHOLD = 80      # 面積閾値: 50%ile付近
MAJOR_CITIES = [13, 14, 23, 27]  # 東京、神奈川、愛知、大阪


# =============================================
# 使用するカラム定義（exp003から継承）
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
# エンコーディング設定（exp003から継承）
# =============================================

# ターゲットエンコーディング対象（6個）
TARGET_ENCODING_COLUMNS = [
    'addr1_1',          # 都道府県コード (47種類)
    'addr1_2',          # 市区町村コード (126種類)
    'bukken_type',      # 物件タイプ (2種類)
    'land_youto',       # 用途地域 (15種類)
    'land_toshi',       # 都市計画 (6種類)
    'building_age_bin', # 築年数カテゴリ (11種類) ※exp004追加
]

# カウントエンコーディング対象（14個）
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
]


# =============================================
# 数値特徴量（exp003から継承 + 新規追加）
# =============================================

# exp003の数値特徴量
NUMERIC_FEATURES_BASE = [
    # 数値
    'unit_count', 'lon', 'lat', 'total_floor_area', 'floor_count',
    'basement_floor_count', 'year_built',  # 変換後
    'building_land_area', 'land_area_all',
    'land_kenpei', 'land_youseki', 'land_road_cond',
    'management_association_flg', 'room_floor', 'balcony_area',
    'dwelling_unit_window_angle', 'room_count', 'unit_area', 'floor_plan_code',
    'empty_number', 'post1', 'post2',
    'nl', 'el', 'snapshot_land_area', 'snapshot_land_shidou',
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

# exp004で追加する数値特徴量
AGE_NUMERIC_FEATURES = [
    'building_age',          # 築年数
    'building_age_bin',      # 築年数5年単位カテゴリ
    'old_building_flag',     # 築35年以上フラグ
    'old_and_large_flag',    # 築35年以上 & 80㎡以上
    'old_and_rural_flag',    # 築35年以上 & 地方
]

# 全数値特徴量
NUMERIC_FEATURES = NUMERIC_FEATURES_BASE + AGE_NUMERIC_FEATURES


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
    """築年数関連特徴量を追加（exp004の新規特徴量）"""
    # Step 1: 基本特徴量を追加
    df = df.with_columns([
        # 築年数（year_builtは既にYYYY形式）
        (2024 - pl.col("year_built")).alias("building_age"),

        # 築年数カテゴリ（5年単位、0-10にクリップ）
        ((2024 - pl.col("year_built")) // 5).clip(0, 10).alias("building_age_bin"),

        # 築35年以上フラグ
        ((2024 - pl.col("year_built")) >= AGE_THRESHOLD).cast(pl.Int64).alias("old_building_flag"),

        # 地方フラグ（補助、特徴量には含めない）
        (~pl.col("addr1_1").is_in(MAJOR_CITIES)).cast(pl.Int64).alias("rural_flag"),
    ])

    # Step 2: 交互作用フラグを追加（building_ageとrural_flagを使用）
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


def preprocess_base(df: pl.DataFrame) -> pl.DataFrame:
    """基本的な前処理（year_built変換、money_sonota集約、築年数特徴量追加）"""
    df = transform_year_built(df)
    df = aggregate_money_sonota(df)
    df = add_age_features(df)
    return df


def get_feature_names() -> List[str]:
    """最終的な特徴量名のリストを取得"""
    features = []

    # 数値特徴量（カテゴリカル含む + 新規追加）
    features.extend(NUMERIC_FEATURES)

    # ターゲットエンコーディング（5個）
    for col in TARGET_ENCODING_COLUMNS:
        features.append(f'{col}_te')

    # カウントエンコーディング（16個）
    for col in COUNT_ENCODING_COLUMNS:
        features.append(f'{col}_count')

    return features


# 最終特徴量リスト
ALL_FEATURES = get_feature_names()

# カテゴリカル特徴量（LightGBM用）- 元データのカテゴリカルカラム + building_age_bin
CATEGORICAL_FEATURES = [
    'building_status', 'building_type', 'building_structure',
    'building_land_chimoku', 'land_youto', 'land_toshi', 'land_chisei',
    'management_form', 'flg_investment', 'flg_new',
    'genkyo_code', 'usable_status', 'parking_kubun',
    'bukken_type', 'addr1_1', 'addr1_2',
    'building_age_bin',  # 新規追加
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
    print("前処理開始（exp004: age_features）")
    print("=" * 60)

    # ターゲット変数を分離
    y_train = train["money_room"]
    print(f"\n✓ ターゲット変数分離: {len(y_train)}件")

    # 基本前処理（year_built変換、money_sonota集約、築年数特徴量追加）
    print("\n[1/5] 基本前処理")
    train = preprocess_base(train)
    test = preprocess_base(test)
    print("  → year_built: YYYYMM → YYYY")
    print("  → money_sonota_sum: sonota1 + sonota2 + sonota3")
    print("  → 築年数関連特徴量追加（5個）")

    # 数値特徴量
    print("\n[2/5] 数値特徴量選択")
    numeric_block = NumericBlock(columns=NUMERIC_FEATURES)
    train_numeric = numeric_block.fit(train)
    test_numeric = numeric_block.transform(test)
    print(f"  → {len(NUMERIC_FEATURES)}個の数値特徴量（築年数関連含む）")

    # ターゲットエンコーディング
    print("\n[3/5] ターゲットエンコーディング")
    if cv_splits is None:
        from sklearn.model_selection import KFold
        cv_splits = list(KFold(n_splits=3, shuffle=True, random_state=42).split(train))

    te_block = TargetEncodingBlock(columns=TARGET_ENCODING_COLUMNS, cv=cv_splits)
    train_te = te_block.fit(train, y=y_train)
    test_te = te_block.transform(test)
    # カラム名を変更（TE_ → _te）
    train_te = train_te.rename({f'TE_{col}': f'{col}_te' for col in TARGET_ENCODING_COLUMNS})
    test_te = test_te.rename({f'TE_{col}': f'{col}_te' for col in TARGET_ENCODING_COLUMNS})
    print(f"  → {len(TARGET_ENCODING_COLUMNS)}個: {TARGET_ENCODING_COLUMNS}")

    # カウントエンコーディング
    print("\n[4/5] カウントエンコーディング")
    count_block = CountEncodingBlock(columns=COUNT_ENCODING_COLUMNS)
    train_count = count_block.fit(train)
    test_count = count_block.transform(test)
    train_count = train_count.rename({col: f'{col}_count' for col in COUNT_ENCODING_COLUMNS})
    test_count = test_count.rename({col: f'{col}_count' for col in COUNT_ENCODING_COLUMNS})
    print(f"  → {len(COUNT_ENCODING_COLUMNS)}個")

    # 全特徴量を結合
    print("\n[5/5] 全特徴量を結合")
    X_train = pl.concat([
        train_numeric,
        train_te,
        train_count,
    ], how="horizontal")

    X_test = pl.concat([
        test_numeric,
        test_te,
        test_count,
    ], how="horizontal")

    # カラム順序を ALL_FEATURES に合わせる
    X_train = X_train.select(ALL_FEATURES)
    X_test = X_test.select(ALL_FEATURES)

    print(f"  → 訓練データ: {X_train.shape}")
    print(f"  → テストデータ: {X_test.shape}")
    print(f"  → 特徴量数: {len(ALL_FEATURES)}個")

    # 新規特徴量の統計を表示
    print("\n[新規特徴量の統計]")
    for col in AGE_NUMERIC_FEATURES:
        if col in X_train.columns:
            mean_val = X_train[col].mean()
            null_pct = X_train[col].null_count() / len(X_train) * 100
            print(f"  {col}: 平均={mean_val:.2f}, 欠損率={null_pct:.1f}%")

    # 検証
    assert X_train.shape[1] == len(ALL_FEATURES), \
        f"特徴量数が一致しません: {X_train.shape[1]} != {len(ALL_FEATURES)}"
    assert X_test.shape[1] == len(ALL_FEATURES), \
        f"特徴量数が一致しません: {X_test.shape[1]} != {len(ALL_FEATURES)}"

    print("\n" + "=" * 60)
    print("前処理完了")
    print("=" * 60)

    return X_train, X_test, y_train
