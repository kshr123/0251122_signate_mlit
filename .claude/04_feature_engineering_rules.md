# 特徴量エンジニアリング ルール

> fit/transform分離とデータリーク防止

---

## 📋 基本原則

1. **Polarsファースト**: NumPy連携時は`to_numpy()`
2. **不変性**: 新しいDataFrameを返す（元を上書きしない）
3. **データリーク防止**: trainでfit、testでtransformのみ

```python
block.fit(train_df, y_train)
train_out = block.transform(train_df)
test_out = block.transform(test_df)  # ✅ transformのみ
```

---

## 🏗️ アーキテクチャ

### BaseBlock（04_src/features/base.py）

```python
class BaseBlock(ABC):
    def fit(self, X, y=None) -> pl.DataFrame:
        """パラメータ学習 + 変換"""
        self._fitted = True
        return self._transform(X)

    def transform(self, X) -> pl.DataFrame:
        """学習済みパラメータで変換"""
        if not self._fitted:
            raise RuntimeError("fit()を先に実行してください")
        return self._transform(X)

    def _transform(self, X) -> pl.DataFrame:
        """実際の変換ロジック（子クラスでオーバーライド）"""
        raise NotImplementedError()
```

**設計意図**: `fit()` と `transform()` は `_transform()` を呼び出す。子クラスは `_transform()` をオーバーライドすることで、継承時のポリモーフィズム問題を回避。

### FeaturePipeline（code/pipeline.py）

```python
pipeline = FeaturePipeline([Block1(), Block2(), ...])
pipeline.fit(X, y).transform(X)  # 各Blockの結果を横結合
```

---

## 📁 ファイル構成

| 場所 | 用途 |
|------|------|
| `04_src/features/blocks/` | 汎用Block（encoding, aggregation, text等） |
| `code/pipeline.py` | Blockの組み合わせ定義 |
| `code/expXXX_features.py` | 実験固有Block |

---

## 📝 Block使用方針

### 原則: 04_srcの共通Blockをそのまま使う

```python
# pipeline.py
from features.blocks.encoding import TargetEncodingBlock, CountEncodingBlock

pipeline = FeaturePipeline([
    TargetEncodingBlock(cols=["city"], cv_splits=cv_splits),
    CountEncodingBlock(cols=["station"]),
])
```

### 共通Blockで対応できない場合のみ実験固有Blockを作成

| パターン | 用途 | 注意点 |
|----------|------|--------|
| **継承** | 共通Blockを拡張 | `_transform()`をオーバーライド |
| **コンポジション** | 共通Blockを内部で使用 | - |

```python
# expXXX_features.py - 継承例（_transform()をオーバーライド）
class PostalCountBlock(CountEncodingBlock):
    def __init__(self, column="post_full", output_column="post_full_count"):
        super().__init__(columns=[column])
        self._input_column = column
        self._output_column = output_column

    def _transform(self, input_df):
        result = super()._transform(input_df)
        return result.rename({self._input_column: self._output_column})
```

```python
# expXXX_features.py - 新規Block例（stateless）
class AreaAgeCategoryBlock(BaseBlock):
    def _transform(self, input_df):
        # statelessなので_transform()のみ実装
        area = input_df["house_area"].to_numpy()
        age = input_df["building_age"].to_numpy()
        category = np.zeros(len(input_df), dtype=np.int32)
        # ... カテゴリ計算 ...
        return pl.DataFrame({"area_age_category": category})
```

```python
# expXXX_features.py - 新規Block例（stateful: fit時にパラメータ学習）
class DensityBinBlock(BaseBlock):
    def fit(self, input_df, y=None):
        # パーセンタイル閾値を学習
        counts = input_df[self._column].to_numpy()
        self._thresholds = np.percentile(counts, [10, 30, 70])
        self._fitted = True
        return self._transform(input_df)

    def _transform(self, input_df):
        counts = input_df[self._column].to_numpy()
        bins = np.digitize(counts, self._thresholds)
        return pl.DataFrame({f"{self._column}_bin": bins})
```

---

## ⚠️ 禁止事項

- ❌ train+testを結合してfit（データリーク）
- ❌ 元DataFrameを上書き（不変性違反）
- ❌ fit統計量をtransformで未使用（リーク）
- ❌ 子クラスで`transform()`をオーバーライド（`_transform()`を使う）

---

**最終更新**: 2025-11-29
