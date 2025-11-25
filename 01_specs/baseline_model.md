# ベースラインモデル 仕様書

Version: 1.0.0
Last Updated: 2025-11-24
Author: Claude Code

---

## 目次

1. [目的](#1-目的)
2. [評価指標](#2-評価指標)
3. [クロスバリデーション](#3-クロスバリデーション)
4. [モデル](#4-モデル)
5. [特徴量戦略](#5-特徴量戦略)
6. [再現性](#6-再現性)
7. [実装仕様](#7-実装仕様)
8. [成功基準](#8-成功基準)

---

## 1. 目的

### 1.1 ゴール

**シンプル・高速・再現性確保のベースラインモデル構築**

### 1.2 成功基準

- ✅ MAPE（Mean Absolute Percentage Error）でCV評価できる
- ✅ LightGBMモデルが訓練できる
- ✅ 3-Fold CVで評価できる
- ✅ 同じシードで実行すると同じスコアになる（再現性）
- ✅ 提出ファイルが生成できる
- ✅ MLflowに実験が記録される
- ✅ 実行時間 < 5分

---

## 2. 評価指標

### 2.1 MAPE (Mean Absolute Percentage Error)

**定義**:
```
MAPE = (1/n) * Σ |y_true - y_pred| / |y_true| * 100
```

**実装方針**:
- `sklearn.metrics.mean_absolute_percentage_error`を使用
- LightGBMのカスタムメトリクスとして実装
- パーセンテージ表記（0-100%）

**注意点**:
- `y_true = 0`の場合にゼロ除算発生 → `epsilon`（小さい値）を加算
- 外れ値に敏感な指標

---

## 3. クロスバリデーション

### 3.1 手法

**KFold (ランダム分割)**

**設定**:
```python
from sklearn.model_selection import KFold

cv = KFold(
    n_splits=3,
    shuffle=True,
    random_state=SEED
)
```

**理由**:
- ベースラインはシンプルさ優先
- 時系列性（`target_ym`）は後続実験で考慮（TimeSeriesSplit）

### 3.2 評価フロー

```
Train → 3-Fold CV → 各FoldでMAPE計算 → 平均・標準偏差
```

---

## 4. モデル

### 4.1 LightGBM

**パラメータ**:
```python
params = {
    "objective": "regression",
    "metric": "mape",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "seed": SEED,
    "verbose": -1,
    "force_row_wise": True,
}
```

**訓練設定**:
- `num_boost_round=100`（固定）
- Early stopping: なし
- カテゴリカル特徴量: `categorical_feature`で自動処理

---

## 5. 特徴量戦略

### 5.1 使用する特徴量

#### 1. 数値データ（前処理なし）
- `area_sqm`: 専有面積
- `distance_station`: 駅距離
- `year_built`: 建築年
- `building_age`: 築年数（計算済み）
- `floor_current`: 所在階
- その他の数値カラム

#### 2. 低カーディナリティカテゴリ（ラベルエンコーディング）
**基準**: ユニーク数 < 50

- `prefecture_code`: 都道府県コード
- `structure_type`: 構造
- `direction`: 方位
- `floor_plan_type`: 間取りタイプ
- その他の低カーディナリティカラム

#### 3. target_ym分解
```python
target_year = target_ym // 100
target_month = target_ym % 100
```

### 5.2 使用しないもの（後回し）

- ❌ 高カーディナリティ（`city_code`, `city_name`, `station_name`等）
- ❌ テキスト特徴量（`remarks`等）
- ❌ 複雑な集約・外部データ
- ❌ 高度な特徴量エンジニアリング

### 5.3 欠損値処理

**方針**: シンプルに固定値で埋める

- 数値カラム: `-999`
- カテゴリカルカラム: `"missing"`

**理由**:
- LightGBMは欠損値を扱えるが、ベースラインは明示的に処理
- 後続実験で高度な補完手法を試す

---

## 6. 再現性

### 6.1 SeedManager

**対象ライブラリ**:
- Python標準ライブラリ（`random`）
- NumPy
- `PYTHONHASHSEED`
- LightGBM（`params["seed"]`）

**実装**:
```python
import random
import os
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
```

### 6.2 検証方法

1. 同じシードで2回実行
2. CVスコア（平均・標準偏差）が完全一致
3. 提出ファイルのハッシュ値が一致

---

## 7. 実装仕様

### 7.1 モジュール構成

```
04_src/
├── features/
│   └── base.py              # SeedManager
├── preprocessing/
│   └── simple.py            # SimplePreprocessor
├── evaluation/
│   └── metrics.py           # MAPE計算
└── training/
    ├── train_baseline.py    # ベースライン訓練スクリプト
    └── utils/
        └── mlflow_helper.py # 既存
```

### 7.2 SimplePreprocessor

**責務**: データの基本的な前処理

**インターフェース**:
```python
class SimplePreprocessor:
    def __init__(self, numeric_fill_value=-999, categorical_fill_value="missing"):
        pass

    def fit(self, df: pl.DataFrame) -> "SimplePreprocessor":
        """統計量を学習（カーディナリティ計算等）"""
        pass

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """前処理を適用"""
        pass

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """fit + transform"""
        pass
```

**処理内容**:
1. target_ym分解（year, month）
2. 欠損値補完（数値: -999, カテゴリ: "missing"）
3. 低カーディナリティカラム抽出（<50）
4. 特徴量カラムの選択

### 7.3 訓練スクリプト (train_baseline.py)

**フロー**:
```
1. シード固定
2. MLflow実験開始
3. データ読み込み
4. 前処理
5. 3-Fold CV
   - 各Foldで訓練・予測
   - MAPE計算
6. MLflow記録
   - パラメータ
   - メトリクス（CV統計量）
   - アーティファクト（提出ファイル、特徴量リスト）
7. 提出ファイル生成
```

**MLflow記録項目**:

- **Parameters**:
  - `seed`: シード値
  - `model_type`: "LightGBM"
  - `n_splits`: 3
  - `num_boost_round`: 100
  - `n_features`: 特徴量数
  - LightGBMパラメータ（learning_rate等）

- **Metrics**:
  - `cv_mape_mean`: MAPE平均
  - `cv_mape_std`: MAPE標準偏差
  - `cv_mape_min`: MAPE最小値
  - `cv_mape_max`: MAPE最大値
  - `cv_mape_fold_0`, `cv_mape_fold_1`, `cv_mape_fold_2`: Fold別MAPE
  - `train_size`: 訓練データサイズ
  - `test_size`: テストデータサイズ

- **Artifacts**:
  - `submission_{timestamp}.csv`: 提出ファイル
  - `features.txt`: 使用特徴量リスト
  - `model/`: LightGBMモデル（オプション）

- **Tags**:
  - `experiment_type`: "baseline"
  - `model_family`: "gbdt"
  - `status`: "completed"

---

## 8. 成功基準

### 8.1 機能面

- [ ] 訓練が正常に完了する
- [ ] 3-Fold CVでMAPEが計算できる
- [ ] 提出ファイルが生成される（`id`, `money_room`カラム）
- [ ] MLflowに実験が記録される

### 8.2 再現性

- [ ] 同じシードで2回実行して同じスコアになる
- [ ] 提出ファイルのハッシュ値が一致する

### 8.3 パフォーマンス

- [ ] 実行時間 < 5分
- [ ] メモリ使用量が適切（< 8GB）

### 8.4 品質

- [ ] 全テストが通過する
- [ ] コードがPEP8準拠（black, ruff）
- [ ] 型ヒントが適切に付与されている

---

## 変更履歴

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-24 | Claude Code | 初版作成 |

---

## 参考資料

- [特徴量エンジニアリングルール](../.claude/feature_engineering_rules.md)
- [MLflow実験記録仕様書](./mlflow_experiment.md)
- [LightGBM公式ドキュメント](https://lightgbm.readthedocs.io/)
