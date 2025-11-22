"""
データ可視化ユーティリティ

Polarsデータフレームの可視化を支援する汎用関数を提供
"""

from typing import List, Optional, Tuple
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns


def plot_missing_heatmap(df: pl.DataFrame, figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    欠損値のヒートマップを表示

    Args:
        df: Polarsデータフレーム
        figsize: 図のサイズ
    """
    # 欠損値をbooleanに変換
    missing_data = df.select([
        pl.col(col).is_null().alias(col) for col in df.columns
    ]).to_pandas()  # matplotlibはpandasを使用

    plt.figure(figsize=figsize)
    sns.heatmap(missing_data.T, cbar=True, cmap="viridis", yticklabels=True)
    plt.title("欠損値パターン（黄色=欠損）")
    plt.xlabel("行インデックス")
    plt.ylabel("カラム")
    plt.tight_layout()
    plt.show()


def plot_distribution(
    df: pl.DataFrame,
    column: str,
    bins: int = 50,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    数値カラムの分布をヒストグラムで表示

    Args:
        df: Polarsデータフレーム
        column: 表示するカラム名
        bins: ビン数
        figsize: 図のサイズ
    """
    data = df[column].drop_nulls().to_pandas()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ヒストグラム
    axes[0].hist(data, bins=bins, edgecolor="black", alpha=0.7)
    axes[0].set_title(f"{column} - ヒストグラム")
    axes[0].set_xlabel(column)
    axes[0].set_ylabel("頻度")
    axes[0].grid(alpha=0.3)

    # 箱ひげ図
    axes[1].boxplot(data, vert=True)
    axes[1].set_title(f"{column} - 箱ひげ図")
    axes[1].set_ylabel(column)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_categorical_distribution(
    df: pl.DataFrame,
    column: str,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    カテゴリカラムの分布を棒グラフで表示

    Args:
        df: Polarsデータフレーム
        column: 表示するカラム名
        top_n: 表示する上位件数
        figsize: 図のサイズ
    """
    # 値のカウント
    value_counts = (
        df[column]
        .value_counts()
        .sort("count", descending=True)
        .head(top_n)
        .to_pandas()
    )

    plt.figure(figsize=figsize)
    plt.bar(range(len(value_counts)), value_counts["count"])
    plt.xticks(
        range(len(value_counts)),
        value_counts[column],
        rotation=45,
        ha="right"
    )
    plt.title(f"{column} - 上位{top_n}件の分布")
    plt.xlabel(column)
    plt.ylabel("件数")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(
    df: pl.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    相関行列のヒートマップを表示

    Args:
        df: Polarsデータフレーム
        columns: 対象カラム（Noneの場合は全数値カラム）
        figsize: 図のサイズ
    """
    # 数値カラムのみ抽出
    if columns is None:
        numerical_cols = [
            col for col, dtype in zip(df.columns, df.dtypes)
            if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                        pl.Float32, pl.Float64]
        ]
    else:
        numerical_cols = columns

    if len(numerical_cols) < 2:
        print("相関行列を表示するには2つ以上の数値カラムが必要です")
        return

    # 相関行列を計算（pandasを使用）
    corr_matrix = df.select(numerical_cols).to_pandas().corr()

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True if len(numerical_cols) <= 15 else False,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        square=True,
        linewidths=0.5
    )
    plt.title("相関行列")
    plt.tight_layout()
    plt.show()


def plot_target_vs_feature(
    df: pl.DataFrame,
    target: str,
    feature: str,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    ターゲット変数と特徴量の関係を散布図で表示

    Args:
        df: Polarsデータフレーム
        target: ターゲット変数のカラム名
        feature: 特徴量のカラム名
        figsize: 図のサイズ
    """
    data = df.select([target, feature]).drop_nulls().to_pandas()

    plt.figure(figsize=figsize)
    plt.scatter(data[feature], data[target], alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(f"{target} vs {feature}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_time_series(
    df: pl.DataFrame,
    time_col: str,
    value_col: str,
    agg_func: str = "mean",
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """
    時系列データをプロット

    Args:
        df: Polarsデータフレーム
        time_col: 時系列カラム名
        value_col: 値のカラム名
        agg_func: 集約関数（"mean", "median", "sum", "count"）
        figsize: 図のサイズ
    """
    # 時系列ごとに集約
    if agg_func == "mean":
        ts_data = df.group_by(time_col).agg(pl.col(value_col).mean()).sort(time_col)
    elif agg_func == "median":
        ts_data = df.group_by(time_col).agg(pl.col(value_col).median()).sort(time_col)
    elif agg_func == "sum":
        ts_data = df.group_by(time_col).agg(pl.col(value_col).sum()).sort(time_col)
    elif agg_func == "count":
        ts_data = df.group_by(time_col).agg(pl.col(value_col).count()).sort(time_col)
    else:
        raise ValueError(f"Unknown aggregation function: {agg_func}")

    ts_pandas = ts_data.to_pandas()

    plt.figure(figsize=figsize)
    plt.plot(ts_pandas[time_col], ts_pandas[value_col], marker="o")
    plt.xlabel(time_col)
    plt.ylabel(f"{value_col} ({agg_func})")
    plt.title(f"{value_col} の時系列推移")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
