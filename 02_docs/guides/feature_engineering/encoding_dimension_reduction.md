# エンコーディング・次元圧縮 ガイド

> **目的**: カテゴリ変数のエンコーディング手法と次元圧縮手法の使い分け、実装方法、注意点をまとめる。

---

## 1. エンコーディング手法

カテゴリ変数を数値に変換する手法。

### 1.1 手法一覧と使い分け

| 手法 | カーディナリティ | 特徴 | 適用場面 |
|------|-----------------|------|---------|
| **Label Encoding** | 低〜高 | 順序なしカテゴリを整数に変換 | 木系モデル全般 |
| **Count Encoding** | 中〜高 | 出現頻度を特徴量化 | 頻度が重要な場合 |
| **Target Encoding** | 中〜高 | ターゲット平均を特徴量化 | ターゲットとの相関が高い場合 |
| **One-Hot Encoding** | **低のみ** | ダミー変数化 | 線形モデル、低カーディナリティ |

### 1.2 各手法の詳細

#### Label Encoding

```python
from features.blocks.encoding import LabelEncodingBlock

block = LabelEncodingBlock(columns=["prefecture", "city"])
train_encoded = block.fit(train_df)
test_encoded = block.transform(test_df)
```

**特徴:**
- シンプル、高速
- 木系モデルに適合（分岐で利用可能）
- 線形モデルには不向き（順序を仮定してしまう）

**注意点:**
- 未知カテゴリ: -1で埋める
- fit時に学習したマッピングをtransformで使用

---

#### Count Encoding

```python
from features.blocks.encoding import CountEncodingBlock

block = CountEncodingBlock(columns=["prefecture", "building_type"])
train_encoded = block.fit(train_df)
test_encoded = block.transform(test_df)
```

**特徴:**
- カテゴリの出現頻度を数値化
- 高カーディナリティでも1カラムで表現可能
- 頻度が予測に重要な場合に有効

**注意点:**
- 未知カテゴリ: 0で埋める
- train/testで頻度分布が異なる場合は注意

---

#### Target Encoding

```python
from features.blocks.encoding import TargetEncodingBlock
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=42)
block = TargetEncodingBlock(
    columns=["prefecture", "city"],
    cv=cv.split(train_df)
)
train_encoded = block.fit(train_df, y=train_df["target"])
test_encoded = block.transform(test_df)
```

**特徴:**
- ターゲットとの関係を直接エンコード
- 高カーディナリティでも効果的
- **データリーク防止のためOOF方式を使用**

**OOF (Out-of-Fold) 方式:**
```
Train時:
  Fold1のvalid → Fold2〜5のtrainから計算した平均でエンコード
  Fold2のvalid → Fold1,3〜5のtrainから計算した平均でエンコード
  ...

Test時:
  全trainデータから計算した平均でエンコード
```

**注意点:**
- **CVオブジェクトを渡す必要がある**
- 未知カテゴリ: 全体平均（y_mean_）で埋める
- カテゴリ内サンプル数が少ないとノイズが大きい → smoothing検討

---

#### One-Hot Encoding

```python
from features.blocks.encoding import OneHotEncodingBlock

block = OneHotEncodingBlock(
    columns=["building_type", "structure"],
    min_count=10,      # 10件未満のカテゴリは除外
    use_cat_names=True # カラム名にカテゴリ値を含める
)
train_encoded = block.fit(train_df)
test_encoded = block.transform(test_df)
```

**特徴:**
- 線形モデルに適合
- カテゴリ間の関係を仮定しない

**注意点:**
- **次元爆発**: カーディナリティが高いと特徴量数が爆発
  - 目安: カーディナリティ < 50
- `min_count`で低頻度カテゴリを除外可能
- 未知カテゴリ: すべて0のベクトル

---

### 1.3 エンコーディング選択フローチャート

```
カテゴリ変数
    │
    ├─ カーディナリティ ≤ 50?
    │       │
    │       ├─ Yes → 線形モデル? → Yes → One-Hot Encoding
    │       │                    → No  → Label Encoding
    │       │
    │       └─ No → ターゲットと相関あり? → Yes → Target Encoding
    │                                      → No  → Count Encoding or Label Encoding
    │
    └─ 木系モデル? → Yes → Label Encoding（まず試す）
                   → No  → Target Encoding or Count Encoding
```

---

## 2. 次元圧縮手法

高次元データを低次元に圧縮する手法。

### 2.1 手法一覧と使い分け

| 手法 | 特徴 | 計算コスト | 適用場面 |
|------|------|-----------|---------|
| **PCA** | 線形、分散最大化 | 低 | 数値特徴量の圧縮、前処理 |
| **SVD** | PCA類似、スパース対応 | 低 | 大規模・スパースデータ |
| **UMAP** | 非線形、局所構造保持 | 高 | 可視化、非線形構造の捕捉 |

### 2.2 各手法の詳細

#### PCA (Principal Component Analysis)

```python
from features.blocks.dimension_reduction import PCABlock

block = PCABlock(
    columns=["feat1", "feat2", "feat3", "feat4", "feat5"],
    n_components=2,
    standardize=True,      # 標準化（推奨）
    handle_missing="mean"  # 欠損は平均で補完
)
train_pca = block.fit(train_df)
test_pca = block.transform(test_df)
```

**特徴:**
- 分散を最大化する方向に射影
- 線形変換のため解釈しやすい
- 計算が高速

**適用場面:**
- 多重共線性の解消
- ノイズ除去
- 特徴量圧縮

---

#### SVD (Truncated SVD)

```python
from features.blocks.dimension_reduction import SVDBlock

block = SVDBlock(
    columns=["feat1", "feat2", "feat3"],
    n_components=2,
    standardize=True,
    handle_missing="error"  # デフォルト: 欠損があればエラー
)
train_svd = block.fit(train_df)
test_svd = block.transform(test_df)
```

**特徴:**
- PCAと類似だがスパースデータに対応
- 中心化を行わない（スパース構造を保持）

**適用場面:**
- テキストデータ（TF-IDF行列）
- One-Hot後の高次元スパースデータ
- 大規模データ

**PCAとの違い:**
| | PCA | SVD |
|---|-----|-----|
| 中心化 | する | しない |
| スパース対応 | △ | ◎ |
| 解釈 | 主成分 | 特異ベクトル |

---

#### UMAP

```python
from features.blocks.dimension_reduction import UMAPBlock

block = UMAPBlock(
    columns=["feat1", "feat2", "feat3", "feat4", "feat5"],
    n_components=2,
    n_neighbors=15,   # 近傍点数（大きい→大域構造重視）
    min_dist=0.1,     # 点間の最小距離
    standardize=True,
    handle_missing="mean"
)
train_umap = block.fit(train_df)
test_umap = block.transform(test_df)
```

**特徴:**
- 非線形な構造を捕捉
- 局所的な関係を保持
- t-SNEより高速

**適用場面:**
- 可視化（2D/3D）
- クラスタリング前の次元圧縮
- 非線形構造の特徴量化

**注意点:**
- 計算コストが高い
- `n_neighbors`以上のサンプル数が必要
- 再現性のため`random_state`を固定

**パラメータの影響:**
| パラメータ | 小さい値 | 大きい値 |
|-----------|---------|---------|
| `n_neighbors` | 局所構造重視 | 大域構造重視 |
| `min_dist` | 密集したクラスタ | 分散したクラスタ |

---

### 2.3 次元圧縮選択フローチャート

```
高次元データ
    │
    ├─ スパースデータ? → Yes → SVD
    │
    ├─ 線形で十分? → Yes → PCA
    │
    ├─ 非線形構造あり? → Yes → UMAP
    │
    └─ 可視化目的? → Yes → UMAP (n_components=2)
                   → No  → PCA（まず試す）
```

---

## 3. 共通の注意点

### 3.1 sklearn互換 fit/transform/fit_transform パターン【必須】

**すべてのTransformerは以下の3メソッドを必ず実装すること:**

```python
class SomeTransformer:
    def fit(self, df: pl.DataFrame) -> "SomeTransformer":
        """学習データでパラメータを学習"""
        # 学習ロジック
        return self  # 必ず self を返す

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """fitしてtransform（sklearn互換）"""
        return self.fit(df).transform(df)  # 必ずこの実装

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """変換を適用"""
        # 変換ロジック
        return result_df
```

**呼び出し方（厳守）:**
```python
# ✅ 正しい呼び出し方
transformer = SomeTransformer(...)
train_result = transformer.fit_transform(train_df)  # trainはfit_transform
test_result = transformer.transform(test_df)        # testはtransformのみ

# ❌ 誤り（冗長）
transformer = SomeTransformer(...)
transformer.fit(train_df)
train_result = transformer.transform(train_df)  # fit_transformを使え
test_result = transformer.transform(test_df)

# ❌ 誤り（データリーク）
transformer = SomeTransformer(...)
all_result = transformer.fit_transform(concat([train_df, test_df]))  # testを含めてfit
```

**パラメータを持たない変換でも3メソッドを実装する理由:**

1. **抽象化・一貫性**: 呼び出し元コードがすべてのTransformerを同じインターフェースで扱える
2. **将来の拡張性**: 後からスムージングや正規化を追加する際にインターフェース変更が不要
3. **パイプライン統合**: sklearn Pipelineや自作パイプラインに組み込み可能

```python
# ✅ 正しい（パラメータなしでも3メソッド実装）
class SimpleRatioTransformer:
    def fit(self, df: pl.DataFrame = None) -> "SimpleRatioTransformer":
        """fitで何もしなくてもメソッドは必要"""
        return self

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """fitしてtransform（sklearn互換）"""
        return self.fit(df).transform(df)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(...)

# 使い方（パラメータなしでも同じパターン）
transformer = SimpleRatioTransformer()
train_result = transformer.fit_transform(train_df)  # trainはfit_transform
test_result = transformer.transform(test_df)        # testはtransformのみ

# ❌ 誤り（fit_transformがない）
class SimpleRatioTransformer:
    def fit(self, df):
        return self
    def transform(self, df):
        return df.with_columns(...)  # fit_transformメソッドがない
```

### 3.2 欠損値の扱い

次元圧縮は欠損値を直接扱えないため、事前処理が必要:

```python
# 方法1: 事前に補完（推奨）
from features.blocks.numeric import SimpleImputeBlock

impute_block = SimpleImputeBlock(columns=cols, strategy="mean")
train_imputed = impute_block.fit(train_df)
test_imputed = impute_block.transform(test_df)

pca_block = PCABlock(columns=cols, n_components=5)
train_pca = pca_block.fit(train_imputed)

# 方法2: Block内で補完
pca_block = PCABlock(columns=cols, n_components=5, handle_missing="mean")
train_pca = pca_block.fit(train_df)  # 内部で平均補完
```

### 3.3 標準化

次元圧縮前の標準化は**ほぼ必須**:

```python
# スケールが異なるデータ
# col1: 0〜1, col2: 0〜10000

# 標準化なし → col2が支配的になる
block = PCABlock(columns=["col1", "col2"], standardize=False)

# 標準化あり → 公平に扱われる（推奨）
block = PCABlock(columns=["col1", "col2"], standardize=True)
```

### 3.4 n_componentsの決め方

**累積寄与率で決める:**
```python
from sklearn.decomposition import PCA
import numpy as np

pca = PCA()
pca.fit(X_train)

# 累積寄与率
cumsum = np.cumsum(pca.explained_variance_ratio_)
# 95%をカバーする成分数
n_components = np.argmax(cumsum >= 0.95) + 1
```

**経験則:**
- 探索的: 2〜3成分（可視化）
- 特徴量圧縮: 元の次元数の10〜50%
- 累積寄与率: 80〜95%をカバー

---

## 4. 実装済みBlock一覧

### エンコーディング (`features/blocks/encoding.py`)

| Block | 説明 | テスト数 |
|-------|------|---------|
| `LabelEncodingBlock` | ラベルエンコーディング | 6 |
| `CountEncodingBlock` | カウントエンコーディング | - |
| `TargetEncodingBlock` | OOF Target Encoding | 10 |
| `OneHotEncodingBlock` | One-Hot (min_count対応) | 12 |

### 集計 (`features/blocks/aggregation.py`)

| Block | 説明 | テスト数 |
|-------|------|---------|
| `GroupByAggBlock` | カテゴリ別数値統計 | 9 |
| `CategoryNuniqueBlock` | カテゴリ間nunique | 8 |

### 次元圧縮 (`features/blocks/dimension_reduction.py`)

| Block | 説明 | テスト数 |
|-------|------|---------|
| `SVDBlock` | TruncatedSVD | 9 |
| `PCABlock` | 主成分分析 | 3 |
| `UMAPBlock` | 非線形次元圧縮 | 3 |

---

## 5. 参考リンク

- [scikit-learn: Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [scikit-learn: Decomposition](https://scikit-learn.org/stable/modules/decomposition.html)
- [category_encoders documentation](https://contrib.scikit-learn.org/category_encoders/)
- [UMAP documentation](https://umap-learn.readthedocs.io/)

---

**最終更新**: 2025-11-25
