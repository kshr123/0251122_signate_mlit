# 特徴量エンジニアリング - コーディングルール

> **このドキュメントについて**: 特徴量作成時の実装方針とコーディング規約を定めます。

---

## 📋 基本方針

### 1. **Polarsファースト、pandas互換性**
- データフレーム操作は **Polars** を優先
- 必要に応じてpandasに変換可能な設計
- 既存コード（DataLoader）との一貫性を保つ

### 2. **不変性の原則**
```python
# ❌ 悪い例：元のDataFrameを変更
def transform(self, df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(...)  # dfを上書き
    return df

# ✅ 良い例：新しいDataFrameを返す
def transform(self, df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(...)
```

### 3. **データリーク防止**
```python
# ❌ 悪い例：testデータでfit
block.fit(test_df)  # NG!

# ✅ 良い例：trainでfit、testでtransform
block.fit(train_df)
train_transformed = block.transform(train_df)
test_transformed = block.transform(test_df)
```

### 4. **再現性の確保**
- すべての乱数生成は `SeedManager` を使用
- Pipeline実行時にシード固定
- 実験記録にシード値を含める

---

## 🏗️ アーキテクチャ

### 3つの核心クラス

1. **BaseBlock**: すべての特徴量ブロックの抽象基底クラス
   - `fit(df, target)`: 訓練データで統計量を学習
   - `transform(df)`: 特徴量を変換
   - `fit_transform(df, target)`: 上記2つを連続実行

2. **FeaturePipeline**: 複数のBlockを組み合わせて実行
   - Blockのリストを受け取り、順次実行
   - 出力は水平結合（hstack）
   - シード管理も担当

3. **SeedManager**: 再現性のためのシード管理
   - Python標準ライブラリ（`random`）
   - NumPy
   - Polars
   - `PYTHONHASHSEED`（ハッシュの順序固定）
   - PyTorch（使用する場合）

**重要な設計原則**:
- **fit/transform分離**: データリーク防止
- **不変性**: 元のDataFrameを変更しない
- **再現性**: すべての乱数生成をSeedManager経由

---

## 📝 命名規則

### Block名
- `{処理内容}Block` 形式
- 例: `NumericBlock`, `SimpleImputeBlock`, `FrequencyEncodingBlock`

### 変換後のカラム名
- 元のカラム名 + サフィックス
- 例: `area_sqm` → `area_sqm_log` (対数変換)
- 例: `prefecture_code` → `prefecture_code_freq` (頻度エンコーディング)

### ファイル構成
```
04_src/features/
├── __init__.py
├── base.py                  # BaseBlock, FeaturePipeline, SeedManager
├── blocks/
│   ├── __init__.py
│   ├── numeric.py          # NumericBlock
│   ├── impute.py           # SimpleImputeBlock, KnnImputeBlock
│   ├── encoding.py         # FrequencyEncodingBlock, TargetEncodingBlock
│   ├── scaling.py          # StandardScalerBlock, MinMaxScalerBlock
│   └── temporal.py         # TargetYmBlock
└── pipelines/
    ├── __init__.py
    └── baseline.py         # ベースラインモデル用のPipeline
```

---

## 🧪 テスト方針（TDD）

### 各Blockに必須のテスト

- **fit/transform分離**: fit後にtransform可能
- **fit前のtransformでエラー**: `RuntimeError`
- **不変性**: 元のDataFrameが変更されない
- **データリーク防止**: trainとtestで異なるデータでtransform可能
- **欠損値・外れ値への対応**: エッジケースをテスト

---

## 🎯 ベースライン作成方針

**目的**: シンプル・高速・再現性確保

### 使用する特徴量
1. **数値データそのまま** - 前処理不要な数値カラム
2. **低カーディナリティ → ラベルエンコーディング** - ユニーク数 < 50程度
3. **target_ym分解** - 年・月・季節フラグ

### 使用しないもの（後回し）
- 高カーディナリティ（city_name等）
- テキスト特徴量
- 複雑な集約・外部データ

### モデル
- **LightGBM**（デフォルトパラメータ）
- Time-Series Split検証
- seed固定で再現性確保

### 成果物
- 提出ファイル作成できる
- CVスコア取得できる
- 実行時間 < 5分

---

## 🚀 実装優先度（ベースライン後）

### Priority 1: ベースラインモデルに必須
- NumericBlock, SimpleImputeBlock, FrequencyEncodingBlock, TargetYmBlock

### Priority 2: 精度向上
- TargetEncodingBlock, StandardScalerBlock, AggregationBlock, OutlierClipBlock

### Priority 3: 高度な特徴量
- PCABlock, DistanceBlock, InteractionBlock

詳細は `01_specs/features.md` を参照してください。

---

## 🔗 既存コードとの統合

### DataLoaderとの連携

```python
loader = DataLoader(config, add_address_columns=True)
train = loader.load_train()  # prefecture_name, city_nameが自動追加

pipeline = FeaturePipeline(blocks=[...])
X_train = pipeline.fit_transform(train)
```

---

## ⚠️ 注意事項

### やってはいけないこと

❌ **trainとtestを結合してfit** → データリーク
❌ **元のDataFrameを上書き** → 不変性の原則違反
❌ **シード固定なしの乱数使用** → 再現性なし

### やるべきこと

✅ **trainでfit、testでtransform**
✅ **新しいDataFrameを返す**（`df.with_columns(...)` で新規作成）
✅ **SeedManagerでシード固定**

---

**最終更新**: 2025-11-23
