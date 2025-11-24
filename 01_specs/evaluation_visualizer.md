# 評価可視化 仕様書

**作成日**: 2025-11-24
**対象モジュール**: `04_src/evaluation/visualizer.py`

---

## 📋 概要

評価結果を可視化するモジュール。
予測vs実測、残差分布、特徴量重要度などの標準的なプロットを提供。

---

## 🎯 要件

### 機能要件

1. **予測vs実測プロット**
   - 散布図（予測値 vs 真値）
   - 対角線（y=x）の追加
   - MAPE/RMSEの表示

2. **残差プロット**
   - ヒストグラム（分布確認）
   - Q-Qプロット（正規性確認）
   - 残差vs予測値（パターン確認）

3. **特徴量重要度プロット**
   - 横棒グラフ（Top N表示）
   - 複数タイプ比較（gain/split/permutation）

4. **セグメント別誤差プロット**
   - 棒グラフ（セグメント別MAPE）
   - エラーバー表示

5. **統合レポート**
   - 複数プロットを1ページにまとめて保存
   - 実験記録用

### 非機能要件

- matplotlib/seaborn使用
- 日本語フォント対応
- 保存機能（PNG/PDF）
- 再利用可能な設計

---

## 📐 仕様

### クラス設計

```python
class EvaluationVisualizer:
    """
    評価結果の可視化

    Attributes:
        figsize: デフォルトのfigureサイズ
        style: プロットスタイル
    """
```

### メソッド仕様

#### 1. `__init__()`

```python
def __init__(
    self,
    figsize: tuple[int, int] = (10, 6),
    style: str = "seaborn-v0_8-darkgrid",
):
    """
    初期化

    Args:
        figsize: デフォルトのfigureサイズ
        style: matplotlibスタイル
    """
```

#### 2. `plot_prediction_vs_actual()`

```python
def plot_prediction_vs_actual(
    self,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "予測値 vs 実測値",
    save_path: Optional[str] = None,
) -> None:
    """
    予測値vs実測値の散布図

    Args:
        y_true: 真値
        y_pred: 予測値
        title: プロットタイトル
        save_path: 保存先パス（Noneの場合は表示のみ）

    Plot features:
        - 散布図（半透明）
        - y=x対角線（赤破線）
        - MAPE/RMSE表示（テキストボックス）
        - グリッド表示

    Examples:
        >>> visualizer = EvaluationVisualizer()
        >>> visualizer.plot_prediction_vs_actual(
        ...     y_true, y_pred,
        ...     save_path="06_experiments/exp001_baseline/pred_vs_actual.png"
        ... )
    """
```

#### 3. `plot_residuals_distribution()`

```python
def plot_residuals_distribution(
    self,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "残差分布",
    save_path: Optional[str] = None,
) -> None:
    """
    残差のヒストグラム

    Args:
        y_true: 真値
        y_pred: 予測値
        title: プロットタイトル
        save_path: 保存先パス

    Plot features:
        - ヒストグラム（bins=50）
        - 正規分布曲線の重ね合わせ
        - 平均・標準偏差の表示
        - ゼロ線（垂直）

    Examples:
        >>> visualizer.plot_residuals_distribution(y_true, y_pred)
    """
```

#### 4. `plot_residuals_qq()`

```python
def plot_residuals_qq(
    self,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "残差 Q-Qプロット",
    save_path: Optional[str] = None,
) -> None:
    """
    残差のQ-Qプロット（正規性確認）

    Args:
        y_true: 真値
        y_pred: 予測値
        title: プロットタイトル
        save_path: 保存先パス

    Plot features:
        - Q-Qプロット（scipy.stats.probplot使用）
        - 対角線（正規分布ならこの線上）

    Examples:
        >>> visualizer.plot_residuals_qq(y_true, y_pred)
    """
```

#### 5. `plot_residuals_vs_predicted()`

```python
def plot_residuals_vs_predicted(
    self,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "残差 vs 予測値",
    save_path: Optional[str] = None,
) -> None:
    """
    残差vs予測値の散布図（パターン確認）

    Args:
        y_true: 真値
        y_pred: 予測値
        title: プロットタイトル
        save_path: 保存先パス

    Plot features:
        - 散布図
        - ゼロ線（水平）
        - パターンがないことが理想

    Examples:
        >>> visualizer.plot_residuals_vs_predicted(y_true, y_pred)
    """
```

#### 6. `plot_feature_importance()`

```python
def plot_feature_importance(
    self,
    importance_df: pl.DataFrame,
    top_n: int = 20,
    title: str = "特徴量重要度",
    save_path: Optional[str] = None,
) -> None:
    """
    特徴量重要度の横棒グラフ

    Args:
        importance_df: 重要度DataFrame (columns: ["feature", "importance"])
        top_n: 表示する特徴量数
        title: プロットタイトル
        save_path: 保存先パス

    Plot features:
        - 横棒グラフ（降順ソート済み上位N件）
        - 値ラベル表示

    Examples:
        >>> visualizer.plot_feature_importance(importance_df, top_n=20)
    """
```

#### 7. `plot_importance_comparison()`

```python
def plot_importance_comparison(
    self,
    comparison_df: pl.DataFrame,
    top_n: int = 20,
    title: str = "特徴量重要度比較",
    save_path: Optional[str] = None,
) -> None:
    """
    複数タイプの重要度比較（横並び棒グラフ）

    Args:
        comparison_df: 比較DataFrame (columns: ["feature", "gain", "split", ...])
        top_n: 表示する特徴量数
        title: プロットタイトル
        save_path: 保存先パス

    Plot features:
        - グループ化横棒グラフ
        - 凡例表示

    Examples:
        >>> visualizer.plot_importance_comparison(comparison_df, top_n=15)
    """
```

#### 8. `plot_segment_errors()`

```python
def plot_segment_errors(
    self,
    segment_df: pl.DataFrame,
    metric: str = "mape",
    title: str = "セグメント別誤差",
    save_path: Optional[str] = None,
) -> None:
    """
    セグメント別誤差の棒グラフ

    Args:
        segment_df: セグメント別分析結果
                   (columns: ["segment", "mape", "rmse", "mae", ...])
        metric: 表示する指標（"mape", "rmse", "mae"）
        title: プロットタイトル
        save_path: 保存先パス

    Plot features:
        - 棒グラフ
        - エラーバー（標準偏差があれば）
        - 値ラベル表示

    Examples:
        >>> visualizer.plot_segment_errors(segment_df, metric="mape")
    """
```

#### 9. `create_evaluation_report()`

```python
def create_evaluation_report(
    self,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    importance_df: Optional[pl.DataFrame] = None,
    segment_df: Optional[pl.DataFrame] = None,
    save_path: str = "evaluation_report.png",
) -> None:
    """
    評価レポート（複数プロットを1ページに統合）

    Args:
        y_true: 真値
        y_pred: 予測値
        importance_df: 特徴量重要度（オプション）
        segment_df: セグメント別分析（オプション）
        save_path: 保存先パス

    Layout:
        2x2 or 2x3 グリッド:
        - [0, 0]: 予測vs実測
        - [0, 1]: 残差分布
        - [1, 0]: 残差Q-Q
        - [1, 1]: 残差vs予測値
        - [0, 2]: 特徴量重要度（あれば）
        - [1, 2]: セグメント別誤差（あれば）

    Examples:
        >>> visualizer.create_evaluation_report(
        ...     y_true, y_pred,
        ...     importance_df=importance_df,
        ...     save_path="06_experiments/exp001_baseline/report.png"
        ... )
    """
```

---

## 🧪 テストケース

### 1. `test_init`
- 初期化時にfigsize/styleが設定されること

### 2. `test_plot_prediction_vs_actual`
- プロットが生成されること（エラーが出ないこと）
- save_path指定時にファイルが保存されること

### 3. `test_plot_residuals_distribution`
- ヒストグラムが生成されること

### 4. `test_plot_residuals_qq`
- Q-Qプロットが生成されること

### 5. `test_plot_residuals_vs_predicted`
- 散布図が生成されること

### 6. `test_plot_feature_importance`
- 横棒グラフが生成されること
- top_nが反映されること

### 7. `test_plot_importance_comparison`
- 比較グラフが生成されること

### 8. `test_plot_segment_errors`
- セグメント別棒グラフが生成されること

### 9. `test_create_evaluation_report`
- 統合レポートが生成されること
- ファイル保存されること

**Note**: 可視化のテストは「エラーが出ないこと」「ファイルが保存されること」を確認する軽量テストとする

---

## 🔄 使用例

```python
from evaluation.visualizer import EvaluationVisualizer
from evaluation.error_analysis import ErrorAnalyzer
from evaluation.feature_importance import FeatureImportanceAnalyzer

# 初期化
visualizer = EvaluationVisualizer()

# 1. 予測vs実測
visualizer.plot_prediction_vs_actual(
    y_true, y_pred,
    save_path="06_experiments/exp001_baseline/pred_vs_actual.png"
)

# 2. 残差分析
visualizer.plot_residuals_distribution(y_true, y_pred)
visualizer.plot_residuals_qq(y_true, y_pred)
visualizer.plot_residuals_vs_predicted(y_true, y_pred)

# 3. 特徴量重要度
importance_analyzer = FeatureImportanceAnalyzer()
importance_df = importance_analyzer.calculate_importance(model, feature_names)
visualizer.plot_feature_importance(importance_df, top_n=20)

# 4. 統合レポート
visualizer.create_evaluation_report(
    y_true, y_pred,
    importance_df=importance_df,
    save_path="06_experiments/exp001_baseline/report.png"
)
```

---

## 🚀 今後の拡張

- 学習曲線プロット
- CV Fold別スコア比較
- アンサンブルの多様性プロット

---

**更新日**: 2025-11-24
