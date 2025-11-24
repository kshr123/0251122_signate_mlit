# エラー分析 仕様書

**作成日**: 2025-11-24
**対象モジュール**: `04_src/evaluation/error_analysis.py`

---

## 📋 概要

予測誤差を多角的に分析するモジュール。
残差の統計分析、セグメント別分析、外れ値検出などを提供。

---

## 🎯 要件

### 機能要件

1. **基本的な誤差統計**
   - 残差の平均、標準偏差、最大・最小
   - MAPE、RMSE、MAE（複数指標）
   - 残差分布の可視化用データ

2. **セグメント別分析**
   - 価格帯別（低価格・中価格・高価格）
   - 任意のカテゴリ変数別（都道府県、物件タイプなど）
   - カスタムセグメント分割

3. **外れ値検出**
   - 予測誤差が大きいサンプルの特定
   - 残差の標準偏差ベース（±3σなど）
   - パーセンタイルベース（上位/下位5%など）

4. **特徴量別エラー傾向**
   - 各特徴量における残差の関係
   - ビニング分析（連続値を区間に分割）

### 非機能要件

- Polars DataFrame入力対応
- 再利用可能な設計
- 型ヒント必須

---

## 📐 仕様

### クラス設計

```python
class ErrorAnalyzer:
    """
    予測誤差の分析

    Attributes:
        y_true: 真値
        y_pred: 予測値
        residuals: 残差（y_true - y_pred）
        abs_residuals: 絶対残差
        pct_errors: パーセント誤差（for MAPE）
    """
```

### メソッド仕様

#### 1. `__init__()`

```python
def __init__(
    self,
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
):
    """
    初期化

    Args:
        y_true: 真値
        y_pred: 予測値

    Examples:
        >>> analyzer = ErrorAnalyzer(y_true, y_pred)
    """
```

#### 2. `calculate_metrics()`

```python
def calculate_metrics(self) -> dict[str, float]:
    """
    各種誤差指標を計算

    Returns:
        dict: {
            "mape": float,           # Mean Absolute Percentage Error (%)
            "rmse": float,           # Root Mean Squared Error
            "mae": float,            # Mean Absolute Error
            "residual_mean": float,  # 残差の平均（バイアス確認）
            "residual_std": float,   # 残差の標準偏差
            "residual_min": float,   # 残差の最小値
            "residual_max": float,   # 残差の最大値
        }

    Examples:
        >>> metrics = analyzer.calculate_metrics()
        >>> print(f"MAPE: {metrics['mape']:.2f}%")
    """
```

#### 3. `get_residual_stats()`

```python
def get_residual_stats(self) -> pl.DataFrame:
    """
    残差の詳細統計量を取得

    Returns:
        pl.DataFrame: 統計量のDataFrame
        Columns: ["metric", "value"]

    Examples:
        >>> stats_df = analyzer.get_residual_stats()
        >>> print(stats_df)
        shape: (10, 2)
        ┌──────────────┬──────────┐
        │ metric       ┆ value    │
        │ ---          ┆ ---      │
        │ str          ┆ f64      │
        ╞══════════════╪══════════╡
        │ mean         ┆ -123.45  │
        │ std          ┆ 5432.1   │
        │ min          ┆ -15000.0 │
        │ 25%          ┆ -2000.0  │
        │ 50%          ┆ -100.0   │
        │ 75%          ┆ 1800.0   │
        │ max          ┆ 20000.0  │
        │ mape         ┆ 15.23    │
        │ rmse         ┆ 6789.0   │
        │ mae          ┆ 3456.0   │
        └──────────────┴──────────┘
    """
```

#### 4. `analyze_by_segment()`

```python
def analyze_by_segment(
    self,
    segment_col: pl.Series,
    segment_name: str = "segment",
) -> pl.DataFrame:
    """
    セグメント別の誤差分析

    Args:
        segment_col: セグメント分類（カテゴリカル変数）
        segment_name: セグメント名（カラム名）

    Returns:
        pl.DataFrame with columns:
        - segment: セグメント名
        - count: サンプル数
        - mape: MAPE (%)
        - rmse: RMSE
        - mae: MAE
        - residual_mean: 残差平均
        - residual_std: 残差標準偏差

    Examples:
        >>> # 価格帯別分析
        >>> price_segments = pl.when(df["money_room"] < 50000).then("低価格")\\
        ...     .when(df["money_room"] < 100000).then("中価格")\\
        ...     .otherwise("高価格")
        >>> segment_errors = analyzer.analyze_by_segment(
        ...     price_segments, segment_name="price_range"
        ... )
    """
```

#### 5. `find_outliers()`

```python
def find_outliers(
    self,
    method: str = "std",
    threshold: float = 3.0,
) -> np.ndarray:
    """
    予測誤差の外れ値を検出

    Args:
        method: "std" (標準偏差) or "percentile" (パーセンタイル)
        threshold:
            - method="std": 標準偏差の倍数（default=3.0 → ±3σ）
            - method="percentile": パーセンタイル（default=3.0 → 上位/下位3%）

    Returns:
        np.ndarray: 外れ値のインデックス配列

    Raises:
        ValueError: methodが不正な場合

    Examples:
        >>> # ±3σ外れ値
        >>> outlier_indices = analyzer.find_outliers(method="std", threshold=3.0)
        >>> print(f"外れ値数: {len(outlier_indices)}")

        >>> # 上位/下位5%
        >>> outlier_indices = analyzer.find_outliers(method="percentile", threshold=5.0)
    """
```

#### 6. `get_outlier_details()`

```python
def get_outlier_details(
    self,
    outlier_indices: np.ndarray,
) -> pl.DataFrame:
    """
    外れ値の詳細情報を取得

    Args:
        outlier_indices: 外れ値のインデックス

    Returns:
        pl.DataFrame with columns:
        - index: サンプルインデックス
        - y_true: 真値
        - y_pred: 予測値
        - residual: 残差
        - abs_residual: 絶対残差
        - pct_error: パーセント誤差 (%)

    Examples:
        >>> outlier_indices = analyzer.find_outliers()
        >>> outlier_details = analyzer.get_outlier_details(outlier_indices)
        >>> print(outlier_details.sort("abs_residual", descending=True))
    """
```

#### 7. `analyze_by_feature_bins()`

```python
def analyze_by_feature_bins(
    self,
    feature_values: pl.Series,
    feature_name: str,
    n_bins: int = 10,
) -> pl.DataFrame:
    """
    特徴量を区間分割して誤差を分析

    Args:
        feature_values: 特徴量の値
        feature_name: 特徴量名
        n_bins: 分割数

    Returns:
        pl.DataFrame with columns:
        - bin: 区間（例: "50000-60000"）
        - bin_center: 区間中央値
        - count: サンプル数
        - mape: MAPE (%)
        - residual_mean: 残差平均

    Examples:
        >>> # 面積別の誤差傾向
        >>> area_analysis = analyzer.analyze_by_feature_bins(
        ...     df["area_sqm"], "area_sqm", n_bins=10
        ... )
    """
```

---

## 🧪 テストケース

### 1. `test_init_and_attributes`
- 初期化時にy_true/y_predが正しく保存されること
- residuals/abs_residuals/pct_errorsが正しく計算されること

### 2. `test_calculate_metrics`
- MAPE/RMSE/MAEが正しく計算されること
- 返り値が辞書型で全キーを含むこと

### 3. `test_get_residual_stats`
- 統計量DataFrameが正しい形式で返されること
- 平均・標準偏差・パーセンタイルが含まれること

### 4. `test_analyze_by_segment`
- セグメント別集計が正しく動作すること
- 各セグメントのMAPE/RMSE/MAEが計算されること

### 5. `test_find_outliers_std`
- 標準偏差ベースで外れ値が検出できること
- threshold=3.0で適切な数が検出されること

### 6. `test_find_outliers_percentile`
- パーセンタイルベースで外れ値が検出できること
- threshold=5.0で約5%が検出されること

### 7. `test_find_outliers_invalid_method`
- 不正なmethodでValueErrorが発生すること

### 8. `test_get_outlier_details`
- 外れ値の詳細DataFrameが正しく返されること
- y_true/y_pred/residualが含まれること

### 9. `test_analyze_by_feature_bins`
- 特徴量をビニングして誤差分析できること
- n_bins数の区間に分割されること

---

## 📊 出力形式

### メトリクス（辞書）

```python
{
    "mape": 15.23,
    "rmse": 6789.0,
    "mae": 3456.0,
    "residual_mean": -123.45,
    "residual_std": 5432.1,
    "residual_min": -15000.0,
    "residual_max": 20000.0,
}
```

### セグメント別分析

```
shape: (3, 7)
┌───────────┬───────┬───────┬─────────┬─────────┬────────────────┬──────────────┐
│ segment   ┆ count ┆ mape  ┆ rmse    ┆ mae     ┆ residual_mean  ┆ residual_std │
│ ---       ┆ ---   ┆ ---   ┆ ---     ┆ ---     ┆ ---            ┆ ---          │
│ str       ┆ u32   ┆ f64   ┆ f64     ┆ f64     ┆ f64            ┆ f64          │
╞═══════════╪═══════╪═══════╪═════════╪═════════╪════════════════╪══════════════╡
│ 低価格    ┆ 3000  ┆ 18.5  ┆ 5000.0  ┆ 2500.0  ┆ -200.0         ┆ 4800.0       │
│ 中価格    ┆ 7000  ┆ 14.2  ┆ 6500.0  ┆ 3200.0  ┆ -100.0         ┆ 5200.0       │
│ 高価格    ┆ 2345  ┆ 12.8  ┆ 9500.0  ┆ 4500.0  ┆ 50.0           ┆ 7800.0       │
└───────────┴───────┴───────┴─────────┴─────────┴────────────────┴──────────────┘
```

### 外れ値詳細

```
shape: (150, 6)
┌───────┬──────────┬──────────┬───────────┬──────────────┬───────────┐
│ index ┆ y_true   ┆ y_pred   ┆ residual  ┆ abs_residual ┆ pct_error │
│ ---   ┆ ---      ┆ ---      ┆ ---       ┆ ---          ┆ ---       │
│ u32   ┆ f64      ┆ f64      ┆ f64       ┆ f64          ┆ f64       │
╞═══════╪══════════╪══════════╪═══════════╪══════════════╪═══════════╡
│ 1234  ┆ 80000.0  ┆ 120000.0 ┆ -40000.0  ┆ 40000.0      ┆ 50.0      │
│ 5678  ┆ 150000.0 ┆ 95000.0  ┆ 55000.0   ┆ 55000.0      ┆ 36.7      │
│ ...   ┆ ...      ┆ ...      ┆ ...       ┆ ...          ┆ ...       │
└───────┴──────────┴──────────┴───────────┴──────────────┴───────────┘
```

---

## 🔄 使用例

```python
from evaluation.error_analysis import ErrorAnalyzer

# 初期化
analyzer = ErrorAnalyzer(y_true, y_pred)

# 基本メトリクス
metrics = analyzer.calculate_metrics()
print(f"MAPE: {metrics['mape']:.2f}%")
print(f"RMSE: {metrics['rmse']:.2f}")

# 残差統計
stats_df = analyzer.get_residual_stats()
print(stats_df)

# セグメント別分析（価格帯）
price_segments = pl.when(df["money_room"] < 50000).then("低価格")\
    .when(df["money_room"] < 100000).then("中価格")\
    .otherwise("高価格")

segment_errors = analyzer.analyze_by_segment(price_segments, "price_range")
print(segment_errors)

# 外れ値検出
outlier_indices = analyzer.find_outliers(method="std", threshold=3.0)
outlier_details = analyzer.get_outlier_details(outlier_indices)
print(f"外れ値数: {len(outlier_indices)}")
print(outlier_details.sort("abs_residual", descending=True).head(10))

# 特徴量別分析
area_analysis = analyzer.analyze_by_feature_bins(
    df["area_sqm"], "area_sqm", n_bins=10
)
print(area_analysis)
```

---

## 🚀 今後の拡張

- 時系列別エラー分析（target_ym別）
- 複数モデルの誤差比較
- 相関分析（特徴量と残差の関係）

---

**更新日**: 2025-11-24
