# 共通特徴量コンポーネント仕様書

> **目的**: 実験で再利用可能な特徴量処理Blockを定義します。

---

## 📋 設計原則

### 1. Blockベース設計
- **BaseBlockを継承したクラスとして実装**
- 各処理を独立したBlockとして実装
- 小さく、テストしやすいBlock

### 2. 不変性
- 元のDataFrameを変更しない
- 新しいDataFrameを返す（`.copy()`を使用）

### 3. fit/transform分離
- `fit()`: 統計量の学習（trainデータのみ使用）
- `transform()`: 学習した統計量で変換（train/test両方に適用）
- データリーク防止

### 4. 明示的なインターフェース
- `__init__()`で対象カラムを明示
- デフォルト引数は最小限

### 5. Polars対応
- 入力: `pl.DataFrame`
- 出力: `pl.DataFrame`
- pandas互換性は内部で変換（必要に応じて）

---

## 🏗️ BaseBlock定義

すべてのBlockの基底クラス:

```python
# 04_src/features/base.py
import polars as pl

class BaseBlock:
    """特徴量Blockの基底クラス"""

    def __init__(self):
        self._fitted = False

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        """統計量を学習し、変換結果を返す

        Args:
            input_df: 入力DataFrame（trainデータ）
            y: ターゲット変数（Target Encodingなどで使用）

        Returns:
            変換後のDataFrame
        """
        self._fitted = True
        return self.transform(input_df)

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """学習した統計量で変換

        Args:
            input_df: 入力DataFrame（train/testどちらでも可）

        Returns:
            変換後のDataFrame

        Raises:
            RuntimeError: fit()を先に実行していない場合
        """
        if not self._fitted:
            raise RuntimeError(f"{self.__class__.__name__}: fit()を先に実行してください")
        raise NotImplementedError()
```

**重要なルール**:
- `fit()`は**trainデータのみ**で実行
- `transform()`は**train/test両方**で実行可能
- `fit()`前に`transform()`を呼ぶと`RuntimeError`

---

## 🚨 重要: Blockの組み合わせは実験ごとに行う

**共通コンポーネント（04_src/features/）には、個別のBlockのみを実装します。**

FeaturePipelineのような抽象化されたパイプラインクラスは**作成しません**。

### 理由
- 抽象化されすぎて実験の内容が見えづらくなる
- 実験ごとに異なるBlock組み合わせロジックが必要
- 実験ディレクトリを見れば完全に理解できる状態を維持

### Blockの組み合わせ方（実験コード内で行う）

```python
# 06_experiments/exp001_baseline/code/preprocessing.py
from src.features.blocks.numeric import NumericBlock
from src.features.blocks.temporal import TargetYmBlock
from src.features.blocks.encoding import LabelEncodingBlock

# Blockのリスト作成
blocks = [
    NumericBlock(columns=NUMERIC_FEATURES),
    TargetYmBlock(source_col="target_ym"),
    LabelEncodingBlock(columns=CATEGORICAL_FEATURES),
]

# 訓練データで fit & transform
feature_dfs = []
for block in blocks:
    feature_dfs.append(block.fit(train, y=train["money_room"]))
X_train = pl.concat(feature_dfs, how="horizontal")

# テストデータで transform
feature_dfs = []
for block in blocks:
    feature_dfs.append(block.transform(test))
X_test = pl.concat(feature_dfs, how="horizontal")
```

**ポイント**:
- 各実験で明示的にBlockを組み合わせる
- どのBlockを使ったか一目瞭然
- 実験の再現性が高い

---

## 🎯 ベースライン実装対象

### Priority 1: ベースラインに必須

1. **base.py** - BaseBlock（既存拡張）
2. **blocks/numeric.py** - NumericBlock
3. **blocks/temporal.py** - TargetYmBlock
4. **blocks/encoding.py** - LabelEncodingBlock

**注意**: FeaturePipelineは作成しません（実験固有のロジック）

---

## 1. blocks/numeric.py - 数値特徴量

### 1.1 NumericBlock

**目的**: 数値特徴量をそのまま返す（前処理なし）

**クラス定義**:
```python
class NumericBlock(BaseBlock):
    """数値特徴量をそのまま返すBlock"""

    def __init__(self, columns: list[str]):
        """
        Args:
            columns: 対象の数値カラムリスト
        """
        super().__init__()
        self.columns = columns

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        """統計量の学習（不要なのでそのままtransform）"""
        self._fitted = True
        return self.transform(input_df)

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """指定されたカラムをそのまま返す"""
        if not self._fitted:
            raise RuntimeError("NumericBlock: fit()を先に実行してください")
        return input_df.select(self.columns)
```

**テストケース**:
```python
# 正常系
df = pl.DataFrame({
    "num1": [1, 2, 3],
    "num2": [1.5, 2.5, 3.5],
    "cat": ["A", "B", "C"]
})

block = NumericBlock(columns=["num1", "num2"])
result = block.fit(df)

assert result.columns == ["num1", "num2"]
assert result.shape == (3, 2)

# fit前のtransform
block2 = NumericBlock(columns=["num1"])
try:
    block2.transform(df)
    assert False, "RuntimeErrorが発生すべき"
except RuntimeError:
    pass  # 期待通り
```

---

## 2. blocks/temporal.py - 時系列特徴量

### 2.1 TargetYmBlock

**目的**: YYYYMMフォーマットの列を年・月に分解

**クラス定義**:
```python
class TargetYmBlock(BaseBlock):
    """target_ymを年・月に分解するBlock"""

    def __init__(self, source_col: str = "target_ym"):
        """
        Args:
            source_col: 分解する列名（デフォルト: "target_ym"）
        """
        super().__init__()
        self.source_col = source_col

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        """統計量の学習（不要なのでそのままtransform）"""
        self._fitted = True
        return self.transform(input_df)

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """YYYYMMを年・月に分解"""
        if not self._fitted:
            raise RuntimeError("TargetYmBlock: fit()を先に実行してください")

        return input_df.select([
            (pl.col(self.source_col) // 100).alias("target_year"),
            (pl.col(self.source_col) % 100).alias("target_month"),
        ])
```

**処理**:
```python
# YYYYMM → 年・月
year = target_ym // 100  # 202301 → 2023
month = target_ym % 100  # 202301 → 1
```

**制約**:
- `source_col`は整数型（Int64等）
- YYYYMMフォーマット（例: 202301, 202412）

**テストケース**:
```python
# 正常系
df = pl.DataFrame({
    "target_ym": [202301, 202312, 202401]
})

block = TargetYmBlock()
result = block.fit(df)

assert result["target_year"].to_list() == [2023, 2023, 2024]
assert result["target_month"].to_list() == [1, 12, 4]

# 不変性テスト
original_data = df["target_ym"].to_list()
_ = block.transform(df)
assert df["target_ym"].to_list() == original_data  # 変更されていない
```

---

## 3. blocks/encoding.py - エンコーディング

### 3.1 LabelEncodingBlock

**目的**: カテゴリカル変数を数値に変換（Categorical → ordinal）

**クラス定義**:
```python
class LabelEncodingBlock(BaseBlock):
    """カテゴリカル変数をラベルエンコーディングするBlock"""

    def __init__(self, columns: list[str]):
        """
        Args:
            columns: エンコードする列名のリスト
        """
        super().__init__()
        self.columns = columns

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        """統計量の学習（カテゴリの一覧を記録）"""
        self._fitted = True
        return self.transform(input_df)

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        """カテゴリカル→数値変換"""
        if not self._fitted:
            raise RuntimeError("LabelEncodingBlock: fit()を先に実行してください")

        result = input_df.select(self.columns)

        for col in self.columns:
            if col not in result.columns:
                continue

            dtype = result[col].dtype

            # Categorical型
            if dtype == pl.Categorical:
                result = result.with_columns(
                    pl.col(col).to_physical().alias(col)
                )
            # Utf8型（文字列）
            elif dtype == pl.Utf8:
                result = result.with_columns(
                    pl.col(col).cast(pl.Categorical).to_physical().alias(col)
                )
            # 数値型はスキップ

        return result
```

**処理**:
```python
# Polars Categorical → 物理値（0, 1, 2, ...）
for col in columns:
    if df[col].dtype == pl.Categorical:
        df = df.with_columns(
            pl.col(col).to_physical().alias(col)
        )
```

**テストケース**:
```python
# Categorical型の場合
df = pl.DataFrame({
    "cat1": pl.Series(["A", "B", "A"], dtype=pl.Categorical),
    "cat2": pl.Series(["X", "Y", "X"], dtype=pl.Categorical),
})

block = LabelEncodingBlock(columns=["cat1", "cat2"])
result = block.fit(df)

assert result["cat1"].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt32]
assert result["cat2"].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt32]

# Utf8型の場合
df2 = pl.DataFrame({
    "str_col": ["未実施", "実施", "未実施"]
})

block2 = LabelEncodingBlock(columns=["str_col"])
result2 = block2.fit(df2)

assert result2["str_col"].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt32]
```

---

## 🧪 TDD実装順序

### Phase 1: base.py拡張
1. テスト作成（Red）
   - `test_base_block_fit_transform()`
   - `test_base_block_not_fitted_error()`

2. 実装（Green）
   - `BaseBlock`拡張（`_fitted`フラグ追加）

3. リファクタリング（Refactor）

**注意**: FeaturePipelineのテストは作成しません

### Phase 2: blocks/numeric.py
1. テスト作成（Red）
   - `test_numeric_block_normal()`
   - `test_numeric_block_not_fitted_error()`
   - `test_numeric_block_immutability()`

2. 実装（Green）
   - `NumericBlock`

3. リファクタリング（Refactor）

### Phase 3: blocks/temporal.py
1. テスト作成（Red）
   - `test_target_ym_block_normal()`
   - `test_target_ym_block_custom_column()`
   - `test_target_ym_block_not_fitted_error()`
   - `test_target_ym_block_immutability()`

2. 実装（Green）
   - `TargetYmBlock`

3. リファクタリング（Refactor）

### Phase 4: blocks/encoding.py
1. テスト作成（Red）
   - `test_label_encoding_categorical()`
   - `test_label_encoding_utf8()`
   - `test_label_encoding_numeric_skip()`
   - `test_label_encoding_not_fitted_error()`
   - `test_label_encoding_immutability()`

2. 実装（Green）
   - `LabelEncodingBlock`

3. リファクタリング（Refactor）

---

## 📦 モジュール構成

```
04_src/features/
├── __init__.py
├── base.py                  # BaseBlock, SeedManager
└── blocks/
    ├── __init__.py
    ├── numeric.py           # NumericBlock ← NEW
    ├── temporal.py          # TargetYmBlock ← NEW
    └── encoding.py          # LabelEncodingBlock ← NEW
```

**注意**: FeaturePipelineは含まれません（実験固有のロジック）

---

## 🔗 実験での使用例

```python
# 06_experiments/exp001_baseline/code/preprocessing.py
from src.features.blocks.numeric import NumericBlock
from src.features.blocks.temporal import TargetYmBlock
from src.features.blocks.encoding import LabelEncodingBlock
import polars as pl

# 数値特徴量リスト
NUMERIC_FEATURES = [
    "building_id", "building_status", "lon", "lat", ...
]

# カテゴリカル特徴量リスト（低カーディナリティのみ）
CATEGORICAL_FEATURES = [
    "building_name_ruby", "reform_exterior", "name_ruby", ...
]

def preprocess_for_training(train: pl.DataFrame, test: pl.DataFrame):
    """実験固有の前処理ロジック"""

    # Blockリスト作成
    blocks = [
        NumericBlock(columns=NUMERIC_FEATURES),
        TargetYmBlock(source_col="target_ym"),
        LabelEncodingBlock(columns=CATEGORICAL_FEATURES),
    ]

    # 訓練データ処理（fit & transform）
    feature_dfs = []
    for block in blocks:
        feature_dfs.append(block.fit(train, y=train["money_room"]))
    X_train = pl.concat(feature_dfs, how="horizontal")

    # テストデータ処理（transform）
    feature_dfs = []
    for block in blocks:
        feature_dfs.append(block.transform(test))
    X_test = pl.concat(feature_dfs, how="horizontal")

    # NumPy変換（LightGBM用）
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()
    y_train_np = train["money_room"].to_numpy()

    return X_train_np, X_test_np, y_train_np
```

**重要なポイント**:
- **FeaturePipelineは使わない** - 実験コード内で明示的にBlockを組み合わせる
- 各Blockの使用が一目瞭然
- 実験の再現性が高い（このファイルを見れば完全に理解できる）

---

## ✅ 受け入れ基準

### 各Block
- [ ] テストが全てパス（Red → Green）
- [ ] fit前のtransformで`RuntimeError`
- [ ] 不変性のテストがパス
- [ ] 型ヒントが正しい
- [ ] Docstringが記述されている

### 全体
- [ ] 3つのBlockモジュール（numeric, temporal, encoding）が実装完了
- [ ] base.pyにBaseBlockが実装済み
- [ ] exp001で使用して動作確認（FeaturePipelineは使わない）
- [ ] exp001のCV結果が再現できる（MAPE 28.34% ± 0.09%）

### 実装しないもの
- [ ] FeaturePipeline（実験固有のロジックとして各実験で明示的に実装）

---

## 📝 今後の拡張（Priority 2以降）

### Priority 2: 精度向上
- **FrequencyEncodingBlock** - 頻度エンコーディング
- **TargetEncodingBlock** - ターゲットエンコーディング（CVあり）
- **StandardScalerBlock** - 標準化
- **CategoryNumBlock** - カテゴリカル×数値の集約

### Priority 3: 高度な特徴量
- **PCABlock** - 主成分分析
- **InteractionBlock** - 交互作用特徴量
- **OneHotEncodingBlock** - One-Hotエンコーディング

---

**作成日**: 2025-11-24
**対象実験**: exp001_baseline（ベースラインモデル）
