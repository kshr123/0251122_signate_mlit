"""
データプロファイリングユーティリティ

データの基本情報、欠損値、統計量などを取得する汎用関数を提供
"""

from typing import Dict, Any, List
import polars as pl


def get_basic_info(df: pl.DataFrame) -> Dict[str, Any]:
    """
    データフレームの基本情報を取得

    Args:
        df: Polarsデータフレーム

    Returns:
        基本情報の辞書
            - shape: (行数, 列数)
            - columns: カラム名リスト
            - dtypes: カラムごとのデータ型
            - memory_mb: メモリ使用量（MB）
    """
    return {
        "shape": (df.height, df.width),
        "columns": df.columns,
        "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
        "memory_mb": round(df.estimated_size("mb"), 2),
    }


def get_missing_info(df: pl.DataFrame) -> pl.DataFrame:
    """
    欠損値情報を取得

    Args:
        df: Polarsデータフレーム

    Returns:
        欠損値情報のDataFrame
            - column: カラム名
            - null_count: 欠損値の数
            - null_ratio: 欠損値の割合
    """
    null_counts = df.null_count()

    missing_info = pl.DataFrame({
        "column": df.columns,
        "null_count": null_counts.row(0),
        "null_ratio": [count / df.height for count in null_counts.row(0)],
    }).sort("null_count", descending=True)

    return missing_info


def get_dtype_summary(df: pl.DataFrame) -> pl.DataFrame:
    """
    データ型ごとのカラム数を集計

    Args:
        df: Polarsデータフレーム

    Returns:
        データ型集計のDataFrame
            - dtype: データ型
            - count: カラム数
            - columns: カラム名のリスト
    """
    dtype_groups: Dict[str, List[str]] = {}

    for col, dtype in zip(df.columns, df.dtypes):
        dtype_str = str(dtype)
        if dtype_str not in dtype_groups:
            dtype_groups[dtype_str] = []
        dtype_groups[dtype_str].append(col)

    summary = pl.DataFrame({
        "dtype": list(dtype_groups.keys()),
        "count": [len(cols) for cols in dtype_groups.values()],
        "columns": [", ".join(cols[:5]) + ("..." if len(cols) > 5 else "")
                   for cols in dtype_groups.values()],
    }).sort("count", descending=True)

    return summary


def get_numerical_summary(df: pl.DataFrame) -> pl.DataFrame:
    """
    数値カラムの基本統計量を取得

    Args:
        df: Polarsデータフレーム

    Returns:
        数値カラムの統計量DataFrame
    """
    # 数値カラムのみ抽出
    numerical_cols = [
        col for col, dtype in zip(df.columns, df.dtypes)
        if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                     pl.Float32, pl.Float64]
    ]

    if not numerical_cols:
        return pl.DataFrame()

    # 統計量を計算
    summary = df.select(numerical_cols).describe()

    return summary


def get_categorical_summary(df: pl.DataFrame, max_unique: int = 50) -> pl.DataFrame:
    """
    カテゴリカラムの情報を取得

    Args:
        df: Polarsデータフレーム
        max_unique: カテゴリとみなす最大ユニーク数

    Returns:
        カテゴリカラムの情報DataFrame
            - column: カラム名
            - unique_count: ユニーク値の数
            - top_value: 最頻値
            - top_freq: 最頻値の出現回数
    """
    # 文字列型カラムまたはユニーク数が少ないカラム
    categorical_info = []

    for col in df.columns:
        # 文字列型、またはユニーク数が少ない数値型
        unique_count = df[col].n_unique()

        if unique_count <= max_unique or df[col].dtype == pl.Utf8:
            # 最頻値を取得
            value_counts = df[col].value_counts().sort("count", descending=True)
            top_value = value_counts[0, col] if value_counts.height > 0 else None
            top_freq = value_counts[0, "count"] if value_counts.height > 0 else 0

            categorical_info.append({
                "column": col,
                "unique_count": unique_count,
                "top_value": str(top_value),
                "top_freq": top_freq,
            })

    if not categorical_info:
        return pl.DataFrame()

    return pl.DataFrame(categorical_info).sort("unique_count", descending=True)


def get_duplicate_info(df: pl.DataFrame) -> Dict[str, Any]:
    """
    重複行の情報を取得

    Args:
        df: Polarsデータフレーム

    Returns:
        重複情報の辞書
            - total_rows: 総行数
            - unique_rows: ユニーク行数
            - duplicate_rows: 重複行数
            - duplicate_ratio: 重複行の割合
    """
    total_rows = df.height
    unique_rows = df.unique().height
    duplicate_rows = total_rows - unique_rows

    return {
        "total_rows": total_rows,
        "unique_rows": unique_rows,
        "duplicate_rows": duplicate_rows,
        "duplicate_ratio": round(duplicate_rows / total_rows, 4) if total_rows > 0 else 0,
    }


def print_profile_summary(df: pl.DataFrame, name: str = "DataFrame") -> None:
    """
    データプロファイルのサマリーを表示

    Args:
        df: Polarsデータフレーム
        name: データフレームの名前
    """
    print("=" * 80)
    print(f"{name} - データプロファイル")
    print("=" * 80)

    # 基本情報
    print("\n[基本情報]")
    info = get_basic_info(df)
    print(f"  行数: {info['shape'][0]:,}")
    print(f"  列数: {info['shape'][1]:,}")
    print(f"  メモリ使用量: {info['memory_mb']} MB")

    # データ型サマリー
    print("\n[データ型]")
    dtype_summary = get_dtype_summary(df)
    print(dtype_summary)

    # 欠損値サマリー（上位10件）
    print("\n[欠損値 - 上位10件]")
    missing_info = get_missing_info(df)
    print(missing_info.head(10))

    # 重複情報
    print("\n[重複行]")
    dup_info = get_duplicate_info(df)
    print(f"  総行数: {dup_info['total_rows']:,}")
    print(f"  ユニーク行数: {dup_info['unique_rows']:,}")
    print(f"  重複行数: {dup_info['duplicate_rows']:,}")
    print(f"  重複割合: {dup_info['duplicate_ratio']:.2%}")

    print("\n" + "=" * 80)
