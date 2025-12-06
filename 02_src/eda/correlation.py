"""相関分析モジュール

ターゲット変数との相関計算、多重共線性チェック等
"""

import polars as pl
import numpy as np
from typing import List


def calculate_correlations(
    df: pl.DataFrame,
    target_col: str,
    numeric_cols: List[str]
) -> pl.DataFrame:
    """
    ターゲット変数との相関を計算

    Parameters
    ----------
    df : pl.DataFrame
        データフレーム
    target_col : str
        ターゲット変数名
    numeric_cols : list[str]
        数値カラムリスト

    Returns
    -------
    pl.DataFrame
        columns: ['feature', 'correlation']
        sorted by abs(correlation) descending

    Examples
    --------
    >>> df = pl.DataFrame({
    ...     'x1': [1, 2, 3, 4, 5],
    ...     'x2': [5, 4, 3, 2, 1],
    ...     'target': [1, 2, 3, 4, 5]
    ... })
    >>> result = calculate_correlations(df, 'target', ['x1', 'x2'])
    >>> result['feature'].to_list()
    ['x1', 'x2']
    """
    correlations = []

    for col in numeric_cols:
        # 欠損値を除外して相関計算
        valid_data = df.select([col, target_col]).drop_nulls()

        if len(valid_data) > 1:
            corr = np.corrcoef(
                valid_data[col].to_numpy(),
                valid_data[target_col].to_numpy()
            )[0, 1]

            # NaN（分散が0の場合等）は0とする
            if np.isnan(corr):
                corr = 0.0

            correlations.append({
                'feature': col,
                'correlation': corr
            })
        else:
            correlations.append({
                'feature': col,
                'correlation': 0.0
            })

    result_df = pl.DataFrame(correlations)

    # 絶対値でソート
    result_df = result_df.with_columns(
        pl.col('correlation').abs().alias('abs_corr')
    ).sort('abs_corr', descending=True).drop('abs_corr')

    return result_df


def find_high_correlation_pairs(
    df: pl.DataFrame,
    numeric_cols: List[str],
    threshold: float = 0.8
) -> List[tuple]:
    """
    多重共線性の可能性がある特徴量ペアを検出

    Parameters
    ----------
    df : pl.DataFrame
        データフレーム
    numeric_cols : list[str]
        数値カラムリスト
    threshold : float, default=0.8
        相関係数の閾値（絶対値）

    Returns
    -------
    list[tuple]
        [(col1, col2, correlation), ...]
        abs(correlation) > threshold のペアリスト
        abs(correlation) の降順でソート

    Examples
    --------
    >>> df = pl.DataFrame({
    ...     'x1': [1, 2, 3, 4, 5],
    ...     'x2': [1, 2, 3, 4, 5],  # x1と完全相関
    ...     'x3': [5, 4, 3, 2, 1]
    ... })
    >>> pairs = find_high_correlation_pairs(df, ['x1', 'x2', 'x3'])
    >>> len(pairs)
    1
    >>> pairs[0][0], pairs[0][1]  # x1とx2のペア
    ('x1', 'x2')
    """
    high_corr_pairs = []

    # 全ペアの相関を計算
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col1 = numeric_cols[i]
            col2 = numeric_cols[j]

            # 欠損値を除外して相関計算
            valid_data = df.select([col1, col2]).drop_nulls()

            if len(valid_data) > 1:
                corr = np.corrcoef(
                    valid_data[col1].to_numpy(),
                    valid_data[col2].to_numpy()
                )[0, 1]

                # NaNは0とする
                if np.isnan(corr):
                    corr = 0.0

                # 閾値を超えるペアを記録
                if abs(corr) > threshold:
                    high_corr_pairs.append((col1, col2, corr))

    # abs(correlation)の降順でソート
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    return high_corr_pairs
