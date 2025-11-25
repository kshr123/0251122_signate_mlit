"""
exp003_baseline_v2 専用の前処理コード

特徴量エンジニアリング:
1. year_built: YYYYMM → YYYY (最初の4文字)
2. money_sonota_sum: money_sonota1 + money_sonota2 + money_sonota3
3. ターゲットエンコーディング: addr1_1, addr1_2, bukken_type, land_youto, land_toshi
4. カウントエンコーディング: 14カラム（カテゴリカルカラムすべて）

※ ラベルエンコーディングは不要（元データが既に数値コード）
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
# 使用するカラム定義（SPEC.mdに基づく）
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
# エンコーディング設定
# =============================================

# ターゲットエンコーディング対象（5個）
TARGET_ENCODING_COLUMNS = [
    'addr1_1',      # 都道府県コード (47種類)
    'addr1_2',      # 市区町村コード (126種類)
    'bukken_type',  # 物件タイプ (2種類)
    'land_youto',   # 用途地域 (15種類)
    'land_toshi',   # 都市計画 (6種類)
]

# カウントエンコーディング対象（14個）
# ※ラベルエンコーディングは不要（元データが既に数値コード）
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
# 数値特徴量（変換後）
# =============================================

# そのまま使用する数値特徴量（元データのカテゴリカルも含む）
NUMERIC_FEATURES = [
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


def preprocess_base(df: pl.DataFrame) -> pl.DataFrame:
    """基本的な前処理（year_built変換、money_sonota集約）"""
    df = transform_year_built(df)
    df = aggregate_money_sonota(df)
    return df


def get_feature_names() -> List[str]:
    """最終的な特徴量名のリストを取得"""
    features = []

    # 数値特徴量（カテゴリカル含む）
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

# カテゴリカル特徴量（LightGBM用）- 元データのカテゴリカルカラム
CATEGORICAL_FEATURES = [
    'building_status', 'building_type', 'building_structure',
    'building_land_chimoku', 'land_youto', 'land_toshi', 'land_chisei',
    'management_form', 'flg_investment', 'flg_new',
    'genkyo_code', 'usable_status', 'parking_kubun',
    'bukken_type', 'addr1_1', 'addr1_2',
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
    print("前処理開始（exp003: baseline_v2）")
    print("=" * 60)

    # ターゲット変数を分離
    y_train = train["money_room"]
    print(f"\n✓ ターゲット変数分離: {len(y_train)}件")

    # 基本前処理（year_built変換、money_sonota集約）
    print("\n[1/4] 基本前処理")
    train = preprocess_base(train)
    test = preprocess_base(test)
    print("  → year_built: YYYYMM → YYYY")
    print("  → money_sonota_sum: sonota1 + sonota2 + sonota3")

    # 数値特徴量
    print("\n[2/4] 数値特徴量選択")
    numeric_block = NumericBlock(columns=NUMERIC_FEATURES)
    train_numeric = numeric_block.fit(train)
    test_numeric = numeric_block.transform(test)
    print(f"  → {len(NUMERIC_FEATURES)}個の数値特徴量（カテゴリカル含む）")

    # ターゲットエンコーディング
    print("\n[3/4] ターゲットエンコーディング")
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
    print("\n[4/4] カウントエンコーディング")
    count_block = CountEncodingBlock(columns=COUNT_ENCODING_COLUMNS)
    train_count = count_block.fit(train)
    test_count = count_block.transform(test)
    train_count = train_count.rename({col: f'{col}_count' for col in COUNT_ENCODING_COLUMNS})
    test_count = test_count.rename({col: f'{col}_count' for col in COUNT_ENCODING_COLUMNS})
    print(f"  → {len(COUNT_ENCODING_COLUMNS)}個")

    # 全特徴量を結合
    print("\n[結合] 全特徴量を結合")
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

    # 検証
    assert X_train.shape[1] == len(ALL_FEATURES), \
        f"特徴量数が一致しません: {X_train.shape[1]} != {len(ALL_FEATURES)}"
    assert X_test.shape[1] == len(ALL_FEATURES), \
        f"特徴量数が一致しません: {X_test.shape[1]} != {len(ALL_FEATURES)}"

    print("\n" + "=" * 60)
    print("前処理完了")
    print("=" * 60)

    return X_train, X_test, y_train
