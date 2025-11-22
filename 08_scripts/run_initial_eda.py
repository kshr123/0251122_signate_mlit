"""
初期EDAを実行するスクリプト

notebooks/01_eda/01_initial_eda.ipynb の主要部分を実行します
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import polars as pl

# 自作モジュール
from src.data.loader import DataLoader
from src.utils.config import load_config
from src.eda.profiler import (
    get_basic_info,
    get_missing_info,
    get_dtype_summary,
    get_numerical_summary,
    get_categorical_summary,
    get_duplicate_info,
    print_profile_summary,
)

# Polars設定
pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_cols(15)


def main():
    print("=" * 80)
    print("初期EDA実行")
    print("=" * 80)

    # データ読み込み
    print("\n[1] データ読み込み中...")
    data_config = load_config("data")
    loader = DataLoader(data_config)

    train = loader.load_train()
    test = loader.load_test()

    print(f"✓ 訓練データ: {train.shape}")
    print(f"✓ テストデータ: {test.shape}")

    # プロファイルサマリー
    print("\n" + "=" * 80)
    print("[2] 訓練データのプロファイル")
    print("=" * 80)
    print_profile_summary(train, name="訓練データ")

    # 基本情報
    print("\n" + "=" * 80)
    print("[3] 詳細情報")
    print("=" * 80)

    train_info = get_basic_info(train)
    print("\n[基本情報]")
    print(f"Shape: {train_info['shape']}")
    print(f"Columns: {len(train_info['columns'])}")
    print(f"Memory: {train_info['memory_mb']} MB")

    # データ型サマリー
    print("\n[データ型サマリー]")
    dtype_summary = get_dtype_summary(train)
    print(dtype_summary)

    # 欠損値（上位20件）
    print("\n[欠損値 - 上位20件]")
    missing_info = get_missing_info(train)
    print(missing_info.head(20))

    # 欠損値50%以上のカラム
    high_missing = missing_info.filter(pl.col("null_ratio") >= 0.5)
    print(f"\n[欠損値50%以上のカラム: {high_missing.height}件]")
    if high_missing.height > 0:
        print(high_missing)

    # 数値カラムの統計量
    print("\n[数値カラムの統計量]")
    numerical_summary = get_numerical_summary(train)
    print(numerical_summary)

    # カテゴリカラムのサマリー
    print("\n[カテゴリカラムのサマリー（ユニーク数50以下、上位20件）]")
    categorical_summary = get_categorical_summary(train, max_unique=50)
    print(categorical_summary.head(20))

    # 重複情報
    print("\n[重複行情報]")
    train_dup_info = get_duplicate_info(train)
    print(f"総行数: {train_dup_info['total_rows']:,}")
    print(f"ユニーク行数: {train_dup_info['unique_rows']:,}")
    print(f"重複行数: {train_dup_info['duplicate_rows']:,}")
    print(f"重複割合: {train_dup_info['duplicate_ratio']:.2%}")

    # 訓練データとテストデータの比較
    print("\n[カラム構成の比較]")
    train_cols = set(train.columns)
    test_cols = set(test.columns)

    common_cols = train_cols & test_cols
    train_only = train_cols - test_cols
    test_only = test_cols - train_cols

    print(f"共通カラム数: {len(common_cols)}")
    print(f"訓練データのみ: {train_only}")
    print(f"テストデータのみ: {test_only}")

    # ターゲット変数の基本統計
    if "money_room" in train.columns:
        print("\n" + "=" * 80)
        print("[4] ターゲット変数（money_room）の基本統計")
        print("=" * 80)

        target_stats = train.select(pl.col("money_room")).describe()
        print(target_stats)

        # 欠損値確認
        null_count = train["money_room"].null_count()
        null_ratio = null_count / train.height
        print(f"\n欠損値数: {null_count}")
        print(f"欠損割合: {null_ratio:.2%}")

        # パーセンタイル
        print("\n[パーセンタイル別の値]")
        percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        for p in percentiles:
            value = train["money_room"].quantile(p)
            print(f"{p*100:5.1f}%: {value:>12,.0f}")

    print("\n" + "=" * 80)
    print("初期EDA完了！")
    print("=" * 80)
    print("\n次のステップ:")
    print("- Jupyter Notebookで詳細な可視化を確認")
    print("  → jupyter notebook notebooks/01_eda/01_initial_eda.ipynb")
    print("- ターゲット変数の詳細分析")
    print("  → jupyter notebook notebooks/01_eda/02_target_analysis.ipynb")


if __name__ == "__main__":
    main()
