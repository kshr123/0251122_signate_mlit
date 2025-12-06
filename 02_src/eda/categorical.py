"""カテゴリ変数分析モジュール

カーディナリティ分類、Target Encoding効果予測等
"""

import polars as pl
from typing import List, Dict


def classify_cardinality(
    df: pl.DataFrame,
    categorical_cols: List[str]
) -> Dict[str, List[tuple]]:
    """
    カテゴリ変数をカーディナリティで分類

    Parameters
    ----------
    df : pl.DataFrame
        データフレーム
    categorical_cols : list[str]
        カテゴリカラムリスト

    Returns
    -------
    dict
        {
            'low': [(col, n_unique), ...],    # < 10
            'medium': [(col, n_unique), ...], # 10-50
            'high': [(col, n_unique), ...]    # > 50
        }
        各リストはn_uniqueの降順でソート

    Examples
    --------
    >>> df = pl.DataFrame({
    ...     'cat_low': ['A', 'B', 'A', 'B', 'C'],
    ...     'cat_medium': list(range(20)),
    ...     'cat_high': list(range(100))
    ... })
    >>> result = classify_cardinality(df, ['cat_low', 'cat_medium', 'cat_high'])
    >>> len(result['low'])
    1
    >>> len(result['medium'])
    1
    >>> len(result['high'])
    1
    """
    low = []
    medium = []
    high = []

    for col in categorical_cols:
        n_unique = df[col].n_unique()

        if n_unique < 10:
            low.append((col, n_unique))
        elif n_unique <= 50:
            medium.append((col, n_unique))
        else:
            high.append((col, n_unique))

    # 各リストをn_uniqueの降順でソート
    low.sort(key=lambda x: x[1], reverse=True)
    medium.sort(key=lambda x: x[1], reverse=True)
    high.sort(key=lambda x: x[1], reverse=True)

    return {
        'low': low,
        'medium': medium,
        'high': high
    }


def calculate_target_encoding_potential(
    df: pl.DataFrame,
    categorical_col: str,
    target_col: str
) -> float:
    """
    Target Encodingの効果を予測（カテゴリ間の標準偏差）

    Parameters
    ----------
    df : pl.DataFrame
        データフレーム
    categorical_col : str
        カテゴリカラム名
    target_col : str
        ターゲット変数名

    Returns
    -------
    float
        カテゴリ別平均値の標準偏差（大きいほど効果大）
        カテゴリが1つ以下の場合は0.0を返す

    Examples
    --------
    >>> df = pl.DataFrame({
    ...     'category': ['A', 'A', 'B', 'B'],
    ...     'target': [1.0, 2.0, 100.0, 200.0]
    ... })
    >>> potential = calculate_target_encoding_potential(df, 'category', 'target')
    >>> potential > 50  # Aの平均=1.5, Bの平均=150 → 標準偏差が大きい
    True
    """
    # カテゴリごとの平均値を計算
    cat_means = df.group_by(categorical_col).agg(
        pl.col(target_col).mean().alias('mean_target')
    )

    # カテゴリが1つ以下の場合は0
    if len(cat_means) <= 1:
        return 0.0

    # 平均値の標準偏差を計算
    std = cat_means['mean_target'].std()

    # NaNの場合は0
    if std is None:
        return 0.0

    return float(std)


def get_category_target_stats(
    df: pl.DataFrame,
    categorical_col: str,
    target_col: str,
    top_n: int = 10
) -> pl.DataFrame:
    """
    カテゴリ別のターゲット変数統計量を取得

    Parameters
    ----------
    df : pl.DataFrame
        データフレーム
    categorical_col : str
        カテゴリカラム名
    target_col : str
        ターゲット変数名
    top_n : int, default=10
        頻度上位N件を返す

    Returns
    -------
    pl.DataFrame
        columns: [categorical_col, 'count', 'mean_target', 'std_target']
        sorted by count descending

    Examples
    --------
    >>> df = pl.DataFrame({
    ...     'category': ['A', 'A', 'B', 'B', 'C'],
    ...     'target': [1.0, 2.0, 10.0, 20.0, 100.0]
    ... })
    >>> stats = get_category_target_stats(df, 'category', 'target')
    >>> stats.shape[0]
    3
    """
    stats = df.group_by(categorical_col).agg([
        pl.len().alias('count'),
        pl.col(target_col).mean().alias('mean_target'),
        pl.col(target_col).std().alias('std_target')
    ]).sort('count', descending=True).head(top_n)

    return stats
