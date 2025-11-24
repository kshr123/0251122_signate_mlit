# 特徴量重要度分析 仕様書

**作成日**: 2025-11-24
**対象モジュール**: `04_src/evaluation/feature_importance.py`

---

## 📋 概要

モデルの特徴量重要度を計算・可視化するモジュール。
GBDT系モデルに対応し、gain/split/permutationの3種類の重要度計算をサポート。

---

## 🎯 要件

### 機能要件

1. **LightGBM対応**
   - `model.feature_importance(importance_type='gain')` - Gain重要度
   - `model.feature_importance(importance_type='split')` - Split重要度
   - Permutation Importance（sklearn使用）

2. **再利用性**
   - 異なるモデルタイプに対応可能な設計
   - 複数モデルの重要度を比較可能

3. **可読性**
   - 特徴量名付きでDataFrame形式で返却
   - Top N特徴量の抽出機能

### 非機能要件

- SHAP不使用（要件により除外）
- Polars DataFrame入力に対応
- 型ヒント必須

---

## 📐 仕様

### クラス設計

```python
class FeatureImportanceAnalyzer:
    """
    特徴量重要度の計算

    Attributes:
        importance_df: 重要度のDataFrame（feature, importance, typeカラム）
    """
```

### メソッド仕様

#### 1. `calculate_importance()`

```python
def calculate_importance(
    self,
    model,  # LightGBM Booster
    feature_names: List[str],
    importance_type: str = "gain",
) -> pl.DataFrame:
    """
    特徴量重要度を計算

    Args:
        model: 学習済みLightGBMモデル
        feature_names: 特徴量名のリスト
        importance_type: "gain" or "split"

    Returns:
        pl.DataFrame with columns: ["feature", "importance", "type"]
        - feature: 特徴量名
        - importance: 重要度（正規化済み、合計=1.0）
        - type: 重要度タイプ（"gain" or "split"）

    Raises:
        ValueError: importance_typeが不正な場合

    Examples:
        >>> analyzer = FeatureImportanceAnalyzer()
        >>> importance_df = analyzer.calculate_importance(
        ...     model, feature_names, importance_type="gain"
        ... )
        >>> print(importance_df.head())
        shape: (5, 3)
        ┌─────────────┬────────────┬──────┐
        │ feature     ┆ importance ┆ type │
        │ ---         ┆ ---        ┆ ---  │
        │ str         ┆ f64        ┆ str  │
        ╞═════════════╪════════════╪══════╡
        │ area_sqm    ┆ 0.234      ┆ gain │
        │ distance    ┆ 0.187      ┆ gain │
        │ ...         ┆ ...        ┆ ...  │
        └─────────────┴────────────┴──────┘
    """
```

#### 2. `calculate_permutation_importance()`

```python
def calculate_permutation_importance(
    self,
    model,
    X: pl.DataFrame,
    y: np.ndarray,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pl.DataFrame:
    """
    Permutation Importanceを計算

    Args:
        model: 学習済みLightGBMモデル
        X: 特徴量（Polars DataFrame）
        y: 目的変数
        n_repeats: シャッフル回数
        random_state: 乱数シード

    Returns:
        pl.DataFrame with columns: ["feature", "importance", "type"]
        - importance: 平均importances（降順ソート済み）
        - type: "permutation"

    Examples:
        >>> analyzer = FeatureImportanceAnalyzer()
        >>> perm_imp = analyzer.calculate_permutation_importance(
        ...     model, X_val, y_val
        ... )
    """
```

#### 3. `get_top_features()`

```python
def get_top_features(
    self,
    n: int = 20,
) -> pl.DataFrame:
    """
    重要度上位N件の特徴量を取得

    Args:
        n: 取得する特徴量数

    Returns:
        pl.DataFrame（importanceで降順ソート済み）

    Raises:
        RuntimeError: calculate_importance未実行の場合

    Examples:
        >>> top_features = analyzer.get_top_features(n=10)
    """
```

#### 4. `compare_importance_types()`

```python
def compare_importance_types(
    self,
    model,
    feature_names: List[str],
    X: Optional[pl.DataFrame] = None,
    y: Optional[np.ndarray] = None,
) -> pl.DataFrame:
    """
    複数タイプの重要度を比較

    Args:
        model: 学習済みモデル
        feature_names: 特徴量名リスト
        X: Permutation用（Noneの場合はgain/splitのみ）
        y: Permutation用

    Returns:
        pl.DataFrame with columns: ["feature", "gain", "split", "permutation"?]
        - 各列は正規化済み重要度
        - permutationはX/yが与えられた場合のみ

    Examples:
        >>> comparison = analyzer.compare_importance_types(
        ...     model, feature_names, X_val, y_val
        ... )
    """
```

---

## 🧪 テストケース

### 1. `test_calculate_importance_gain`
- LightGBMモデルでgain重要度が計算できること
- 返り値が["feature", "importance", "type"]カラムを持つこと
- importanceの合計が1.0になること（正規化確認）

### 2. `test_calculate_importance_split`
- split重要度が計算できること
- gainとは異なる値になること

### 3. `test_invalid_importance_type`
- 不正なimportance_typeでValueErrorが発生すること

### 4. `test_calculate_permutation_importance`
- Permutation Importanceが計算できること
- n_repeatsが反映されること

### 5. `test_get_top_features`
- 上位N件が正しく取得できること
- importanceで降順ソートされていること

### 6. `test_get_top_features_before_calculate`
- calculate_importance未実行時にRuntimeErrorが発生すること

### 7. `test_compare_importance_types`
- gain/split/permutationが1つのDataFrameに統合されること
- X/y未指定時はpermutationカラムが含まれないこと

---

## 📊 出力形式

### 基本形式（単一タイプ）

```
shape: (57, 3)
┌──────────────────┬────────────┬──────┐
│ feature          ┆ importance ┆ type │
│ ---              ┆ ---        ┆ ---  │
│ str              ┆ f64        ┆ str  │
╞══════════════════╪════════════╪══════╡
│ area_sqm         ┆ 0.234      ┆ gain │
│ distance_station ┆ 0.187      ┆ gain │
│ target_year      ┆ 0.123      ┆ gain │
│ ...              ┆ ...        ┆ ...  │
└──────────────────┴────────────┴──────┘
```

### 比較形式

```
shape: (57, 4)
┌──────────────────┬────────┬────────┬─────────────┐
│ feature          ┆ gain   ┆ split  ┆ permutation │
│ ---              ┆ ---    ┆ ---    ┆ ---         │
│ str              ┆ f64    ┆ f64    ┆ f64         │
╞══════════════════╪════════╪════════╪═════════════╡
│ area_sqm         ┆ 0.234  ┆ 0.198  ┆ 0.215       │
│ distance_station ┆ 0.187  ┆ 0.203  ┆ 0.192       │
│ target_year      ┆ 0.123  ┆ 0.145  ┆ 0.134       │
│ ...              ┆ ...    ┆ ...    ┆ ...         │
└──────────────────┴────────┴────────┴─────────────┘
```

---

## 🔄 使用例

```python
from evaluation.feature_importance import FeatureImportanceAnalyzer

# 初期化
analyzer = FeatureImportanceAnalyzer()

# Gain重要度
importance_df = analyzer.calculate_importance(
    model=lgb_model,
    feature_names=X_train.columns,
    importance_type="gain",
)

# Top 20特徴量
top20 = analyzer.get_top_features(n=20)
print(top20)

# 複数タイプ比較
comparison = analyzer.compare_importance_types(
    model=lgb_model,
    feature_names=X_train.columns,
    X=X_val,
    y=y_val,
)
print(comparison)
```

---

## 🚀 今後の拡張

- XGBoost/CatBoost対応
- 重要度の可視化機能（visualizer.pyに委譲）
- CV全体での重要度平均計算

---

**更新日**: 2025-11-24
