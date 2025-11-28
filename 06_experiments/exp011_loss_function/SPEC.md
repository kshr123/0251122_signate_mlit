# exp011 仕様書: 損失関数の比較実験

損失関数・サンプル重み付けの効果を検証する実験。

---

## 背景・課題

exp010では低価格帯（特に広面積×築古）の予測精度が課題：

| セグメント | MAPE |
|-----------|------|
| 全体 | 12.48% |
| 1000万以下 | 20.75% |
| 100㎡以上×35年以上 | 16.60% |

現状の設定：
- ターゲット変換: `log1p(price)`
- 目的関数: `l2`（MSE）

---

## 実験フェーズ

### Phase 1: パラメータ変更のみ（実装コスト小）

| # | 実験名 | 設定 | 期待効果 |
|---|--------|------|----------|
| 1 | huber | `objective="huber"` | 外れ値の影響軽減 |
| 2 | quantile | `objective="quantile", alpha=0.5` | 中央値予測で安定 |
| 3 | sample_weight | `sample_weight=1/y` | 低価格帯重視 |

### Phase 2: カスタム目的関数（実装コスト中）

| # | 実験名 | 設定 | 期待効果 |
|---|--------|------|----------|
| 4 | mape | MAPE直接最適化 | 評価指標と一致 |
| 5 | focal | 難サンプル重視 | 低価格帯・広×古改善 |

### Phase 3: アンサンブル・スタッキング

Phase 1-2の結果を見て、効果的なモデルを組み合わせる。

#### 3-1. 単純アンサンブル
| 方式 | 説明 | 優先度 |
|------|------|--------|
| 単純平均 | 各モデルの予測を等しく平均 | 最初に試す |
| 重み付き平均 | OOFでMAPE最小化する重みを探索 | 次に試す |

#### 3-2. スタッキング
| メタモデル | 説明 | 優先度 |
|------------|------|--------|
| Ridge | L2正則化線形回帰 | 最初に試す |
| LightGBM | 非線形メタモデル | Ridgeで効果あれば |

**スタッキング設計**:
- **Level 0**: MSE, Huber, Quantile 等の各モデル（Phase 1-2の結果）
- **Level 1**: Level 0のOOF予測を特徴量としてメタモデルを学習
- **バリデーション**: Level 0のOOF予測に対して新たに3-Fold CVでLevel 1を評価
- **予測時**: Level 0で各モデル予測 → Level 1で統合

---

## 実装設計

### ディレクトリ構成

```
# 共通コンポーネント（04_src）
04_src/stacking/
├── __init__.py
└── trainer.py             # StackingTrainer（CVロジック）

# 実験固有（code/）
code/
├── train.py               # Phase 1-2: 損失関数別学習
├── ensemble.py            # Phase 3: スタッキング実行
├── objectives.py          # カスタム目的関数（mape, focal）
└── constants.py           # パス定義

configs/
└── experiment.yaml        # 全設定（stacking設定を追加）
```

### 設計方針

- **メタモデルはsklearn互換モデルを直接使用**（ラッパークラス不要）
- sklearn/lightgbmは `fit/predict` を持つため、そのまま渡せる
- StackingTrainerのみを共通コンポーネントとして04_srcに配置

### 共通コンポーネント設計

#### StackingTrainer（04_src/stacking/trainer.py）
```python
from pathlib import Path
from typing import Any
import numpy as np
import joblib
from sklearn.model_selection import KFold


class StackingTrainer:
    """スタッキングのCV学習を行うトレーナー

    メタモデルはsklearn互換（fit/predictを持つ）であれば何でも使用可能。
    - sklearn.linear_model.Ridge
    - sklearn.linear_model.Lasso
    - lightgbm.LGBMRegressor
    - etc.
    """

    def __init__(
        self,
        meta_model: Any,  # sklearn互換モデル（fit/predict を持つ）
        n_splits: int = 3,
        seed: int = 42,
    ):
        self.meta_model = meta_model
        self.n_splits = n_splits
        self.seed = seed
        self.fitted_model_ = None  # fit_final後に設定

    def fit_predict_oof(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, list[float]]:
        """CVでOOF予測を作成し、fold別スコアも返す

        Returns:
            oof_predictions: OOF予測値 (n_samples,)
            fold_scores: fold別MAPEスコア
        """
        pass

    def fit_final(self, X: np.ndarray, y: np.ndarray) -> None:
        """全データで最終モデルを学習

        学習済みモデルは self.fitted_model_ に保存される
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """テストデータの予測"""
        if self.fitted_model_ is None:
            raise ValueError("fit_final() を先に呼び出してください")
        return self.fitted_model_.predict(X)

    def save(self, path: Path) -> None:
        """モデル保存（joblib使用）"""
        joblib.dump(self.fitted_model_, path)

    @classmethod
    def load(cls, path: Path) -> Any:
        """モデル読み込み"""
        return joblib.load(path)
```

### 実験固有コンポーネント設計

#### ensemble.py（code/）

3つのアンサンブル方式を`--method`で切り替え：

| method | 説明 | 追加引数 |
|--------|------|----------|
| `average` | 単純平均 | なし |
| `weighted` | 重み付き平均（OOFでMAPE最小化） | なし |
| `stacking` | スタッキング | `--meta-model ridge/lightgbm` |

```python
# 実行例
# 単純平均
# caffeinate -i env PYTHONPATH=../../04_src:code python code/ensemble.py \
#   --level0-dirs outputs/run_mse_* outputs/run_huber_* \
#   --method average

# 重み付き平均
# caffeinate -i env PYTHONPATH=../../04_src:code python code/ensemble.py \
#   --level0-dirs outputs/run_mse_* outputs/run_huber_* \
#   --method weighted

# スタッキング
# caffeinate -i env PYTHONPATH=../../04_src:code python code/ensemble.py \
#   --level0-dirs outputs/run_mse_* outputs/run_huber_* \
#   --method stacking --meta-model ridge

from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from stacking.trainer import StackingTrainer

def find_optimal_weights(oof_preds: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OOFでMAPE最小化する重みを探索"""
    n_models = oof_preds.shape[1]

    def objective(weights):
        weights = weights / weights.sum()  # 正規化
        pred = oof_preds @ weights
        return calculate_mape(y, pred)

    # 初期値: 均等重み
    x0 = np.ones(n_models) / n_models
    # 制約: 重み >= 0
    bounds = [(0, 1)] * n_models

    result = minimize(objective, x0, bounds=bounds, method='SLSQP')
    return result.x / result.x.sum()

def run_average(X_oof, y, X_test):
    """単純平均"""
    oof_pred = X_oof.mean(axis=1)
    test_pred = X_test.mean(axis=1)
    return oof_pred, test_pred, None  # weightsなし

def run_weighted(X_oof, y, X_test):
    """重み付き平均（最適化）"""
    weights = find_optimal_weights(X_oof, y)
    oof_pred = X_oof @ weights
    test_pred = X_test @ weights
    return oof_pred, test_pred, weights

def run_stacking(X_oof, y, X_test, meta_model, n_splits, seed):
    """スタッキング"""
    trainer = StackingTrainer(meta_model, n_splits=n_splits, seed=seed)
    oof_pred, fold_scores = trainer.fit_predict_oof(X_oof, y)
    trainer.fit_final(X_oof, y)
    test_pred = trainer.predict(X_test)
    return oof_pred, test_pred, trainer
```

### 実装の依存関係・フロー

```
Step 1: 04_src/stacking/trainer.py
        └── StackingTrainer
            依存: sklearn.model_selection.KFold, joblib

            │
            ▼
Step 2: code/ensemble.py（実験固有）
        └── メイン実行スクリプト
            依存: StackingTrainer, sklearn.linear_model.Ridge
            入力: outputs/run_*/oof_predictions.csv, test_predictions.csv
```

### 実行フロー

```
Phase 1-2: train.py
    │
    ├── outputs/run_mse_*/
    │   ├── oof_predictions.csv      # id, actual, predicted
    │   └── test_predictions.csv     # id, predicted
    │
    ├── outputs/run_huber_*/
    │   └── ...
    │
    └── outputs/run_sample_weight_*/
        └── ...
            │
            ▼
Phase 3: ensemble.py --method {average|weighted|stacking}
    │
    ├── 1. Level 0予測読み込み
    │      oof_predictions.csv × N個 → X_oof (n_samples, N)
    │      test_predictions.csv × N個 → X_test
    │
    ├── 2. アンサンブル実行（methodで分岐）
    │      ├── average:  均等重み平均
    │      ├── weighted: scipy.optimizeで最適重み探索
    │      └── stacking: StackingTrainerでCV学習
    │
    ├── 3. OOF MAPE計算
    │
    └── 4. 出力保存
           outputs/run_{method}_*/
           ├── oof_predictions.csv
           ├── test_predictions.csv
           ├── submission.csv
           ├── metrics.json
           ├── config.json
           └── (stacking時のみ) models/meta_model.pkl
```

### 設定ファイル（experiment.yaml に追加）

```yaml
stacking:
  meta_model: "ridge"           # ridge / lightgbm
  meta_params:
    ridge:
      alpha: 1.0
    lightgbm:
      n_estimators: 100
      learning_rate: 0.1
      max_depth: 3
  cv:
    n_splits: 3
    seed: 42
```

### 出力形式

```
outputs/run_stacking_ridge_YYYYMMDD_HHMMSS/
├── oof_predictions.csv       # id, actual, predicted
├── test_predictions.csv      # id, predicted
├── submission.csv            # id, price（整数）
├── models/
│   └── meta_model.pkl        # 全データで学習したメタモデル
├── metrics.json              # cv_mape, fold別スコア
└── config.json               # 再現性用（使用したLevel0モデル、パラメータ）
```

### 実行方法

```bash
cd 06_experiments/exp011_loss_function
source ../../.venv/bin/activate

# Phase 1-2（損失関数別）
caffeinate -i env PYTHONPATH=../../04_src:code python code/train.py --objective mse
caffeinate -i env PYTHONPATH=../../04_src:code python code/train.py --objective huber --features-dir outputs/run_mse_*/
caffeinate -i env PYTHONPATH=../../04_src:code python code/train.py --objective sample_weight --weight-transform inverse --features-dir outputs/run_mse_*/

# Phase 3（アンサンブル）
# 単純平均
caffeinate -i env PYTHONPATH=../../04_src:code python code/ensemble.py \
  --level0-dirs outputs/run_mse_* outputs/run_huber_* outputs/run_sample_weight_* \
  --method average

# 重み付き平均
caffeinate -i env PYTHONPATH=../../04_src:code python code/ensemble.py \
  --level0-dirs outputs/run_mse_* outputs/run_huber_* outputs/run_sample_weight_* \
  --method weighted

# スタッキング（Ridge）
caffeinate -i env PYTHONPATH=../../04_src:code python code/ensemble.py \
  --level0-dirs outputs/run_mse_* outputs/run_huber_* outputs/run_sample_weight_* \
  --method stacking --meta-model ridge
```

---

## 評価指標

- 全体MAPE
- 1000万以下MAPE
- 100㎡以上×35年以上MAPE

---

## 期待効果

| セグメント | 現状MAPE | 目標 |
|-----------|----------|------|
| 全体 | 12.48% | 12.0%以下 |
| 1000万以下 | 20.75% | 18%以下 |
| 100㎡以上×35年以上 | 16.60% | 14%以下 |

---

**最終更新**: 2025-11-28（Phase 3 スタッキング設計追加）
