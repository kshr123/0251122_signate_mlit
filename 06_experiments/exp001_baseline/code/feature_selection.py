"""
exp001_baseline - 特徴量選択ロジック

このファイルは、149個の元特徴量から106個の特徴量を選択した過程を記録しています。
"""

import polars as pl
from typing import List, Tuple

# =============================================
# 特徴量選択の設定
# =============================================

# カテゴリカル変数のカーディナリティ閾値
CARDINALITY_THRESHOLD = 50

# 必ず除外するカラム（システムカラム、ターゲット変数）
EXCLUDE_COLUMNS = ["id", "money_room", "target_ym"]


# =============================================
# 特徴量選択関数
# =============================================

def select_features(df: pl.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    データフレームから使用する特徴量を選択

    選択基準:
    1. システムカラム（id, money_room, target_ym）は除外
    2. 数値特徴量は全て使用
    3. カテゴリカル特徴量はカーディナリティ < 50 のみ使用
    4. target_ym から target_year, target_month を生成

    Parameters
    ----------
    df : pl.DataFrame
        入力データフレーム

    Returns
    -------
    numeric_features : List[str]
        数値特徴量のリスト
    categorical_features : List[str]
        カテゴリカル特徴量のリスト（低カーディナリティのみ）
    generated_features : List[str]
        生成する特徴量のリスト
    """

    # 1. 除外カラムを除く
    available_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]

    # 2. 数値特徴量とカテゴリカル特徴量を分類
    numeric_features = []
    categorical_features = []

    for col in available_cols:
        dtype = df[col].dtype

        # 数値型
        if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                     pl.Float32, pl.Float64]:
            numeric_features.append(col)

        # カテゴリカル型または文字列型
        elif dtype in [pl.Categorical, pl.Utf8]:
            # カーディナリティをチェック
            n_unique = df[col].n_unique()

            if n_unique < CARDINALITY_THRESHOLD:
                categorical_features.append(col)
            else:
                # 高カーディナリティは除外
                print(f"除外: {col} (cardinality={n_unique} >= {CARDINALITY_THRESHOLD})")

    # 3. 生成特徴量
    generated_features = ["target_year", "target_month"]

    return numeric_features, categorical_features, generated_features


def get_excluded_high_cardinality_features(df: pl.DataFrame) -> List[Tuple[str, int]]:
    """
    高カーディナリティで除外されたカテゴリカル特徴量のリストを取得

    Parameters
    ----------
    df : pl.DataFrame
        入力データフレーム

    Returns
    -------
    excluded : List[Tuple[str, int]]
        除外された特徴量とそのカーディナリティのリスト
    """

    excluded = []

    for col in df.columns:
        if col in EXCLUDE_COLUMNS:
            continue

        dtype = df[col].dtype

        if dtype in [pl.Categorical, pl.Utf8]:
            n_unique = df[col].n_unique()

            if n_unique >= CARDINALITY_THRESHOLD:
                excluded.append((col, n_unique))

    return excluded


# =============================================
# 実行例（検証用）
# =============================================

if __name__ == "__main__":
    """
    このスクリプトを実行すると、元データから選択された特徴量を確認できます。

    実行方法:
        python code/feature_selection.py
    """

    from pathlib import Path
    import sys

    # プロジェクトルートをパスに追加
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.append(str(project_root))

    from src.data.loader import DataLoader

    # データ読み込み
    loader = DataLoader()
    train, test = loader.load_train_test()

    print("=" * 60)
    print("特徴量選択結果")
    print("=" * 60)

    # 特徴量選択
    numeric_feats, categorical_feats, generated_feats = select_features(train)

    print(f"\n✅ 数値特徴量: {len(numeric_feats)}個")
    print(f"✅ カテゴリカル特徴量: {len(categorical_feats)}個")
    print(f"✅ 生成特徴量: {len(generated_feats)}個")
    print(f"✅ 合計: {len(numeric_feats) + len(categorical_feats) + len(generated_feats)}個")

    print(f"\n📋 カテゴリカル特徴量（カーディナリティ < {CARDINALITY_THRESHOLD}）:")
    for feat in categorical_feats:
        n_unique = train[feat].n_unique()
        print(f"  - {feat}: {n_unique}")

    # 除外された高カーディナリティ特徴量
    excluded = get_excluded_high_cardinality_features(train)

    print(f"\n❌ 除外された高カーディナリティ特徴量: {len(excluded)}個")
    for feat, cardinality in sorted(excluded, key=lambda x: x[1], reverse=True):
        print(f"  - {feat}: {cardinality}")

    print("\n" + "=" * 60)
    print("特徴量選択完了")
    print("=" * 60)
