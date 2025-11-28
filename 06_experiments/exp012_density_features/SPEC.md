# exp012: 密度特徴量の追加

## 概要

「近傍物件数」を需要・人口密度の代理変数として導入し、低価格帯の予測精度改善を目指す。

**この実験では、共通コンポーネントの04_src移動も同時に実施する。**

---

## Phase 0: 共通コンポーネント化（前提作業）

exp011で成熟したコンポーネントを04_srcに移動し、exp012から使用する。

### 移動対象

| コンポーネント | 移動元 | 移動先 | 状態 |
|---------------|--------|--------|------|
| FeaturePipeline | exp011/code/pipeline.py | 04_src/features/pipeline.py | **新規追加** |
| deep_merge | exp011/code/train.py | 04_src/utils/config.py | **追加** |

### 移動しないもの（実験固有）

| コンポーネント | 理由 |
|---------------|------|
| load_config (exp011版) | extends継承、test_modeは実験固有 |
| get_sample_weight | 実験固有のサンプル重み付け |
| PostalCodeTEBlock等 | 実験固有Block |

### 0-1. FeaturePipeline移動

**移動先**: `04_src/features/pipeline.py`

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List
import polars as pl

@dataclass
class BlockInfo:
    name: str
    block: Any
    input_columns: List[str]
    description: str = ""
    output_columns: List[str] = field(default_factory=list)

class FeaturePipeline:
    """特徴量変換パイプライン"""

    def add_block(self, name, block, input_columns, description="") -> "FeaturePipeline": ...
    def fit_transform(self, df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame: ...
    def transform(self, df: pl.DataFrame) -> pl.DataFrame: ...
    def get_feature_names(self) -> List[str]: ...
    def summary(self) -> str: ...
```

**使用方法（移動後）**:
```python
from features.pipeline import FeaturePipeline
from features.blocks.encoding import TargetEncodingBlock

pipeline = FeaturePipeline()
pipeline.add_block("te", TargetEncodingBlock(...), ["city"], "市区町村TE")
X_train = pipeline.fit_transform(train_df, y_train)
X_test = pipeline.transform(test_df)
```

### 0-2. deep_merge追加

**追加先**: `04_src/utils/config.py`

```python
def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries (override takes precedence)"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
```

### 0-3. objectives確認

`04_src/objectives/base.py` に既に存在（exp011と同等）:
- `BaseObjective`
- `MAPEObjective`
- `FocalObjective`

→ exp012では04_srcからimportする

### 0-4. ルール更新

`.claude/04_feature_engineering_rules.md`:
```diff
- ### FeaturePipeline（code/pipeline.py）
+ ### FeaturePipeline（04_src/features/pipeline.py）
```

### 0-5. テスト追加

```
07_tests/test_features/test_pipeline.py
├── test_feature_pipeline_init
├── test_add_block
├── test_fit_transform
├── test_transform_before_fit_raises
└── test_summary

07_tests/test_utils/test_config.py
├── test_deep_merge_basic
├── test_deep_merge_nested
└── test_deep_merge_override
```

### exp012でのインポートパターン

```python
# 04_src（共通）
from features.pipeline import FeaturePipeline
from features.blocks.encoding import TargetEncodingBlock
from objectives.base import MAPEObjective
from utils.config import deep_merge

# 実験固有
from exp012_features import DensityBinBlock
from constants import NUMERIC_COLUMNS
```

---

## Phase 1: 密度特徴量の追加

## 背景

exp011の分析で以下が判明：
- 物件数が少ない地域ほど予測誤差が大きい（郵便番号内物件数1-5件: MAPE 15.4% vs 101件+: 10.5%）
- **駅物件数はMAPEとの相関がほぼない**（全パーセンタイルで約12%）→ カテゴリ化の価値なし
- 郵便番号物件数 vs 駅物件数の相関は0.19と低い

### MAPE分布の分析結果

#### 郵便番号物件数 vs MAPE（パーセンタイル別）

| パーセンタイル | 物件数範囲 | N | MAPE |
|---------------|-----------|-------|------|
| 0-10%ile | 1-5件 | 42,032 | **15.41%** |
| 10-20%ile | 6-9件 | 34,481 | 12.99% |
| 20-30%ile | 10-13件 | 32,679 | 12.68% |
| 30-40%ile | 14-19件 | 36,461 | 12.38% |
| 40-50%ile | 20-27件 | 38,109 | 12.07% |
| 50-60%ile | 28-37件 | 32,920 | 11.73% |
| 60-70%ile | 38-53件 | 37,654 | 11.49% |
| 70-80%ile | 54-78件 | 35,089 | 11.19% |
| 80-90%ile | 79-128件 | 38,499 | 10.73% |
| 90-100%ile | 129-8171件 | 35,900 | **10.52%** |

→ **0-10%ile（1-5件）のMAPEが特に高い**（15.41%）

#### 駅物件数 vs MAPE（パーセンタイル別）

| パーセンタイル | 物件数範囲 | N | MAPE |
|---------------|-----------|-------|------|
| 0-10%ile | 1-303件 | 35,613 | 12.40% |
| 10-30%ile | 304-637件 | 71,233 | 12.48% |
| 30-50%ile | 638-1131件 | 71,249 | 12.25% |
| 50-70%ile | 1132-2247件 | 71,233 | 12.05% |
| 70-90%ile | 2248-5148件 | 71,233 | 11.91% |
| 90-100%ile | 5149-24282件 | 35,616 | 11.85% |

→ **MAPEの差が小さい（11.85%-12.48%）**ためカテゴリ化の価値なし

## 追加特徴量

| # | 特徴量名 | 説明 | 既存との関係 |
|---|----------|------|--------------|
| ① | post_full_count | フル郵便番号の物件数 | **新規** |
| ② | post_full_density_bin | ①のパーセンタイルベースビン分け（4カテゴリ） | **新規** |
| ③ | post_full_density_bin_TE | ②でグループしてTE | **新規** |
| ③' | post_full_density_bin_agg | ②でグループして集計 | **新規** |
| ④ | area_age_category | 面積×築年数カテゴリ | **拡張**（3→4カテゴリ） |
| ⑤ | rosen_name1_count | 駅物件数（カウントのみ） | **既存流用**（カテゴリ化なし） |

### 削除した特徴量（分析の結果）

| 特徴量 | 削除理由 |
|--------|----------|
| rosen_count_bin | MAPEとの相関がない（全パーセンタイルで約12%） |
| rosen_bin_TE | 同上 |
| rosen_bin_agg | 同上 |

### ③ post_full_density_bin_TE vs 既存 post_full_te の違い

| 特徴量 | グループ単位 | 説明 |
|--------|--------------|------|
| post_full_te（既存） | 郵便番号（31,691種類） | 各郵便番号の平均価格 |
| post_full_density_bin_TE（新規） | 密度ビン（4種類） | 密度レベル別の平均価格 |

→ 異なる情報を持つため両方有効

## ビン定義

### 郵便番号物件数ビン (post_full_density_bin)

**パーセンタイルベース**（train/testで分布が一致）

| カテゴリ | パーセンタイル | 物件数範囲(参考) | N | MAPE |
|----------|---------------|-----------------|-------|------|
| very_low | 0-10%ile | 1-5件 | 42,032 (11.5%) | **15.41%** |
| low | 10-30%ile | 6-13件 | 67,160 (18.4%) | 12.84% |
| medium | 30-70%ile | 14-53件 | 145,144 (39.9%) | 11.91% |
| high | 70-100%ile | 54-8171件 | 109,488 (30.1%) | 10.81% |

**実装方法**:
```python
# fitでパーセンタイル境界を計算
boundaries = [0, 10, 30, 70, 100]  # パーセンタイル
thresholds = np.percentile(train_counts, boundaries[1:-1])  # [5, 14, 54]

# transformでビン分け
bins = np.digitize(counts, thresholds)  # 0, 1, 2, 3
```

**ポイント**:
- train時にパーセンタイル境界を学習し、testに適用
- 固定閾値ではなく、trainのパーセンタイルから閾値を算出
- これによりtrain/testで分布が一致

### ④ area_age_category の定義（拡張）

| カテゴリ | 条件 | 期待N | 期待MAPE |
|----------|------|-------|----------|
| cat0 | それ以外 | 332,868 | 11.8% |
| cat1 | 100㎡+ AND 35年+ | 27,520 | 15.4% |
| cat2 | 150㎡+ AND 45年+ | 2,148 | 18.2% |
| cat3 | 200㎡+ AND 45年+ | 1,388 | 20.5% |

判定順序: cat3 → cat2 → cat1 → cat0（上位条件を優先）

**既存との違い:** 既存は3カテゴリ（cat0-2）、新規は4カテゴリ（cat3追加）

## モデル設定

### ベースライン（MSE）
- 損失関数: MSE（regression）
- exp010の特徴量 + 上記追加特徴量

### Huber
- 損失関数: Huber
- exp010の特徴量 + 上記追加特徴量

## 評価指標

- MAPE（Mean Absolute Percentage Error）
- セグメント別MAPE（area_age_categoryごと）
- 大外れ率（APE > 50%の割合）

## 期待効果

- 全体MAPE: 12.17% → 11.8%程度（0.3-0.5pt改善）
- cat3セグメント: 20.5% → 18%程度（2-3pt改善）
- 大外れ率: 2.2% → 1.8%程度

## ディレクトリ構造

```
exp012_density_features/
├── SPEC.md              # この仕様書
├── README.md            # 実験サマリー
├── configs/
│   └── experiment.yaml  # 全ハイパーパラメータ
├── code/
│   ├── train.py         # エントリーポイント
│   ├── features.py      # 密度特徴量の実装
│   └── constants.py     # パス・カラムリスト
├── outputs/             # 実行結果
└── notebooks/           # 分析ノートブック
```

## 実行方法

```bash
cd 06_experiments/exp012_density_features
source ../../.venv/bin/activate

# ベースライン（MSE）
PYTHONPATH=../../04_src:code python code/train.py --objective mse

# Huber
PYTHONPATH=../../04_src:code python code/train.py --objective huber

# テストモード
PYTHONPATH=../../04_src:code python code/train.py --objective huber --test
```
