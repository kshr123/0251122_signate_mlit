# Hashing Trick 完全ガイド

> 超高カーディナリティ変数を固定次元に圧縮する技術

---

## 📖 Hashing Trickとは

**定義**: ハッシュ関数を使って、カテゴリ値を**固定サイズの整数（インデックス）**に変換する手法

**目的**: 高カーディナリティ変数（数万〜数百万種類）を**メモリ効率的**にエンコードする

---

## 🔑 基本原理

### 仕組み

```python
# ハッシュ関数で整数に変換 → 固定次元数で割る

hash("桜マンション") = 1234567
1234567 % 32 = 15  → index 15 に割り当て

hash("緑ハイツ") = 8901234
8901234 % 32 = 7   → index 7 に割り当て

hash("青アパート") = 5678915
5678915 % 32 = 15  → index 15 に割り当て（衝突）
```

### 視覚的な理解

```
元データ（69,370種類の建物名）
↓
ハッシュ関数
↓
32次元のベクトル
[0, 0, 0, 0, 0, 0, 0, 1, 0, ..., 0, 1, 0]
       index 7↑           index 15↑
```

---

## ⚙️ 実装方法

### scikit-learnでの実装

```python
from sklearn.feature_extraction import FeatureHasher
import polars as pl

# データ準備
train = pl.DataFrame({
    'building_name': [
        '桜マンション',
        '緑ハイツ',
        '青アパート',
        '桜マンション',  # 重複
        '赤レジデンス'
    ],
    'price': [100000, 80000, 90000, 105000, 95000]
})

# Hashing Trick適用（32次元）
hasher = FeatureHasher(n_features=32, input_type='string')
building_hashed = hasher.transform(train['building_name'])

print(building_hashed.shape)  # (5, 32)
print(type(building_hashed))  # scipy.sparse.csr_matrix

# 疎行列 → 密行列（必要な場合）
building_hashed_dense = building_hashed.toarray()

# モデルに渡す
# X = building_hashed_dense
```

### カスタム実装（One-Hot版）

```python
def hash_trick_onehot(values: list, n_features: int = 32) -> list:
    """
    シンプルなHashing Trick実装（One-Hot版）

    Args:
        values: カテゴリ値のリスト
        n_features: 出力次元数

    Returns:
        One-Hotベクトルのリスト
    """
    result = []

    for val in values:
        # ハッシュ値を計算
        idx = hash(val) % n_features

        # One-Hotベクトル作成
        vec = [0] * n_features
        vec[idx] = 1
        result.append(vec)

    return result

# 使用例
building_names = ['桜マンション', '緑ハイツ', '青アパート']
hashed = hash_trick_onehot(building_names, n_features=32)

print(len(hashed[0]))  # 32
print(hashed[0])       # [0, 0, ..., 1, ..., 0]
```

### カスタム実装（カウント版）

```python
def hash_trick_count(values: list, n_features: int = 32) -> list:
    """
    Hashing Trick（カウント版）
    同じインデックスに複数のカテゴリが衝突した場合、カウントを増やす

    Args:
        values: カテゴリ値のリスト
        n_features: 出力次元数

    Returns:
        カウントベクトルのリスト
    """
    from collections import defaultdict

    result = []

    for val in values:
        vec = [0] * n_features

        # 複数のハッシュ値（衝突軽減）
        for i in range(3):  # 3つのハッシュ関数
            idx = hash((val, i)) % n_features
            vec[idx] += 1

        result.append(vec)

    return result
```

### Polarsとの統合

```python
import polars as pl
from sklearn.feature_extraction import FeatureHasher
import numpy as np

def add_hashed_features(
    df: pl.DataFrame,
    cat_col: str,
    n_features: int = 32,
    prefix: str = None
) -> pl.DataFrame:
    """
    PolarsデータフレームにHashed特徴量を追加

    Args:
        df: データフレーム
        cat_col: ハッシュ化するカテゴリカラム
        n_features: 出力次元数
        prefix: カラム名プレフィックス（デフォルト: f"{cat_col}_hash"）

    Returns:
        Hashed特徴量が追加されたデータフレーム
    """
    if prefix is None:
        prefix = f"{cat_col}_hash"

    # FeatureHasher適用
    hasher = FeatureHasher(n_features=n_features, input_type='string')
    hashed = hasher.transform(df[cat_col]).toarray()

    # DataFrameに追加
    hash_cols = {
        f"{prefix}_{i}": hashed[:, i]
        for i in range(n_features)
    }

    return df.with_columns([
        pl.Series(name, values)
        for name, values in hash_cols.items()
    ])

# 使用例
train_hashed = add_hashed_features(
    train,
    cat_col='building_name',
    n_features=32
)

# 結果
# building_name | price  | building_name_hash_0 | building_name_hash_1 | ...
# --------------|--------|---------------------|---------------------|----
# 桜マンション   | 100000 |         0           |         0           | ...
```

---

## ✅ メリット

### 1. メモリ効率が極めて高い

```python
# 比較: building_name（69,370種類）

# One-Hot Encoding
# → 69,370次元 → 約540KB/サンプル（float64）
# → 100万サンプル → 540GB！

# Hashing Trick（256次元）
# → 256次元 → 約2KB/サンプル
# → 100万サンプル → 2GB
```

### 2. 未知カテゴリに自動対応

```python
# 訓練データ
train_buildings = ['マンションA', 'マンションB', 'マンションC']

# テストデータ（未知カテゴリ）
test_buildings = ['マンションD', 'マンションE']

# One-Hot Encoding
# → エラーまたは全0ベクトル

# Hashing Trick
# → 自動的にエンコード
hash('マンションD') % 32 = 18  # index 18に割り当て
hash('マンションE') % 32 = 23  # index 23に割り当て
```

### 3. 計算速度が速い

```python
# One-Hot: カテゴリ辞書を事前構築 → O(n)
# Hashing: ハッシュ計算のみ → O(1)

# オンライン学習に最適
for new_sample in stream:
    hashed = hash(new_sample) % n_features
    # すぐに学習可能
```

### 4. 実装が簡単

```python
# One-Hot: fit() でカテゴリ辞書作成 → transform()
encoder = OneHotEncoder()
encoder.fit(train_categories)
encoded = encoder.transform(test_categories)

# Hashing: fit不要、直接transform
hasher = FeatureHasher(n_features=32)
encoded = hasher.transform(categories)  # fit不要！
```

---

## ❌ デメリット

### 1. ハッシュ衝突（Collision）

**問題**: 異なるカテゴリが同じインデックスに割り当てられる

```python
hash("桜マンション") % 32 = 15
hash("青アパート")   % 32 = 15  # 衝突！

# 両方ともindex 15に割り当て
# → 2つのカテゴリが区別できない
```

**影響**:
- 情報の損失
- わずかな精度低下

**対策**:
```python
# 1. 次元数を増やす
n_features = 256  # または 512, 1024

# 衝突確率 = 約 1/n_features
# n=32  → 約3%
# n=256 → 約0.4%

# 2. 複数のハッシュ関数を使う
for i in range(3):  # 3つのハッシュ
    idx = hash((value, i)) % n_features
    vec[idx] = 1
```

### 2. 解釈性の喪失

```python
# One-Hot
feature_15 = "桜マンション"  # 明確

# Hashing
feature_15 = ???  # 何のカテゴリか不明
# "桜マンション" と "青アパート" が衝突している可能性
```

**影響**:
- 特徴量重要度が解釈できない
- デバッグが困難

**対策**:
```python
# 解釈性が必要な場合はHashing Trickを使わない
# → Target EncodingやFrequency Encodingを使う
```

### 3. わずかな精度低下

```python
# 衝突により情報損失
# → 他の手法より1〜3%精度が下がることがある

# 精度が最優先の場合は他手法を検討
```

---

## 📊 次元数の選び方

### 経験則

```python
n_unique = 69_370  # カーディナリティ

# 1. 保守的（衝突を避ける）
n_features = n_unique // 10  # 約7,000次元
# 衝突確率 ≈ 1.4%

# 2. バランス型
n_features = int(n_unique ** 0.5)  # 約263次元
# 衝突確率 ≈ 38%

# 3. 積極的（メモリ優先）
n_features = 128  # または 256
# 衝突確率 ≈ 100%（ほぼ確実に衝突）

# 4. 実用的な選択
n_features = 256  # または 512
# メモリ効率と精度のバランス
```

### 次元数と衝突確率の関係

| 次元数 | カーディナリティ | 衝突確率（近似） |
|--------|----------------|-----------------|
| 32     | 69,370        | ~100%          |
| 128    | 69,370        | ~100%          |
| 256    | 69,370        | ~99.6%         |
| 512    | 69,370        | ~99.3%         |
| 1,024  | 69,370        | ~98.5%         |
| 2,048  | 69,370        | ~97%           |
| 10,000 | 69,370        | ~85%           |

**注**: 上記は誕生日のパラドックスで計算した近似値

---

## 🎯 使いどころ

### ✅ 推奨される場面

#### 1. 超高カーディナリティ（数万〜数百万種類）

```python
# ユーザーID: 100万種類
# URL: 500万種類
# テキストのn-gram: 無限

# → Hashing Trick一択
```

#### 2. オンライン学習

```python
# ストリーミングデータ
# 新しいカテゴリが次々と出現
# 事前に全カテゴリを知ることができない

# → Hashing Trickで自動対応
```

#### 3. メモリ制約が厳しい

```python
# 組み込み機器
# エッジデバイス
# ストリーミング処理

# → 固定サイズのメモリで処理可能
```

#### 4. 解釈性が不要

```python
# ディープラーニング
# アンサンブルモデル（多数の特徴量）
# 中間特徴量

# → 解釈性より精度・効率重視
```

### ❌ 推奨されない場面

#### 1. 低〜中カーディナリティ

```python
# < 100種類
# → One-Hot Encodingで十分
# → 情報損失のリスクを避ける
```

#### 2. 解釈性が重要

```python
# ビジネス意思決定に使う
# 特徴量重要度を確認したい
# デバッグが必要

# → Target EncodingやFrequency Encodingを使う
```

#### 3. 高精度が最優先

```python
# コンペの上位入賞狙い
# 本番システムの精度改善

# → Target Encoding、複数手法の併用
```

---

## 🔬 実践例（このプロジェクト）

### 対象変数

```python
# 04_categorical_analysis.ipynb の結果

高カーディナリティ変数:
  - statuses:           232,339種類  ← 超高
  - unit_tag_id:        209,158種類  ← 超高
  - building_id:        175,577種類  ← 超高
  - full_address:       172,933種類  ← 超高
  - building_name:       69,370種類  ← 高
  - addr2_name:          64,822種類  ← 高
```

### 実装例

```python
from sklearn.feature_extraction import FeatureHasher
import polars as pl

# データ読み込み
train = pl.read_parquet('data/processed/train.parquet')

# Hashing Trick適用（256次元）
hasher = FeatureHasher(n_features=256, input_type='string')

# building_name をハッシュ化
building_hashed = hasher.transform(train['building_name'])

print(f"元の次元数: 69,370")
print(f"Hashing後: {building_hashed.shape[1]}")  # 256
print(f"メモリ削減率: {256/69370:.2%}")  # 0.37%

# モデリングに使用
from sklearn.linear_model import Ridge
import numpy as np

X = building_hashed
y = train['money_room'].to_numpy()

model = Ridge()
model.fit(X, y)
```

### 複数カラムのハッシュ化

```python
# 複数の高カーディナリティ変数をハッシュ化
hash_cols = [
    'building_name',
    'full_address',
    'addr2_name'
]

n_features = 128  # 各カラム128次元

hashed_features = []

for col in hash_cols:
    hasher = FeatureHasher(n_features=n_features, input_type='string')
    hashed = hasher.transform(train[col])
    hashed_features.append(hashed)

# 結合
from scipy.sparse import hstack
X_all = hstack(hashed_features)

print(X_all.shape)  # (n_samples, 128*3 = 384)
```

---

## 📊 他手法との比較

| 手法 | カーディナリティ | メモリ | 未知カテゴリ | 解釈性 | 精度 | 計算速度 |
|------|----------------|--------|-------------|--------|------|---------|
| **One-Hot** | 低（<100） | 大 | ❌ | ✅ 高 | ✅ 高 | ○ |
| **Target Encoding** | 中〜高 | 小 | △ 要対策 | ○ 中 | ✅ 高 | ○ |
| **Frequency Encoding** | 中〜高 | 小 | △ 要対策 | ○ 中 | ○ 中 | ✅ 高 |
| **Leave-One-Out** | 高 | 小 | △ 要対策 | △ 低 | ✅ 高 | △ 遅 |
| **Hashing Trick** | 超高 | 極小 | ✅ 自動 | ❌ 低 | △ 中 | ✅ 高 |

---

## 💡 実践的なTips

### Tip 1: Target Encodingとの併用

```python
# Hashing Trickだけでなく、Target Encodingも使う
# → モデルが補完しあう

df = df.with_columns([
    # Hashing Trick（256次元）
    # ...hashed features...

    # Target Encoding（1次元）
    pl.col('building_name').alias('building_name_te')
])

# 両方使うことで精度向上
```

### Tip 2: 適切な次元数の実験

```python
# グリッドサーチで最適な次元数を探す
from sklearn.model_selection import cross_val_score

for n_features in [64, 128, 256, 512, 1024]:
    hasher = FeatureHasher(n_features=n_features, input_type='string')
    X_hashed = hasher.transform(train['building_name'])

    score = cross_val_score(model, X_hashed, y, cv=5).mean()
    print(f"n_features={n_features}: {score:.4f}")

# 出力例:
# n_features=64:   0.7234
# n_features=128:  0.7456
# n_features=256:  0.7498  ← 最適
# n_features=512:  0.7501
# n_features=1024: 0.7502
```

### Tip 3: 複数ハッシュ関数で衝突軽減

```python
def multi_hash_trick(values, n_features=32, n_hash_functions=3):
    """
    複数のハッシュ関数を使って衝突を軽減

    Args:
        n_hash_functions: 使用するハッシュ関数の数
    """
    result = []

    for val in values:
        vec = [0] * n_features

        # 複数のハッシュ関数
        for i in range(n_hash_functions):
            idx = hash((val, i)) % n_features
            vec[idx] = 1

        result.append(vec)

    return result

# 衝突確率が低減される
```

### Tip 4: カテゴリ集約との組み合わせ

```python
# 上位N件は個別にエンコード、残りをHashing

top_n = 100
top_buildings = (
    train['building_name']
    .value_counts()
    .head(top_n)
    .struct.field('building_name')
)

# 上位100件: One-Hot or Target Encoding
# 残り: Hashing Trick

df = df.with_columns([
    pl.when(pl.col('building_name').is_in(top_buildings))
    .then(pl.col('building_name'))  # そのまま
    .otherwise(pl.lit('_other_'))   # その他
    .alias('building_name_top100')
])

# building_name_top100: One-Hot（100次元）
# building_name（全体）: Hashing（256次元）
```

---

## 📚 参考資料

### 論文・記事

- [Feature Hashing for Large Scale Multitask Learning](https://arxiv.org/abs/0902.2206) (Weinberger et al., 2009)
- Kaggleでの活用事例: Click-Through Rate予測コンペ等

### 実装ライブラリ

- scikit-learn: `FeatureHasher`
- Vowpal Wabbit: 高速ハッシュベース学習
- xLearn: CTR予測向けライブラリ

---

## 📖 関連ドキュメント

- [cardinality_guide.md](./cardinality_guide.md) - カーディナリティ完全ガイド
- [target_encoding_guide.md](./target_encoding_guide.md) - Target Encoding完全ガイド
- [04_categorical_analysis.ipynb](../05_notebooks/01_eda/04_categorical_analysis.ipynb) - カテゴリ変数分析

---

**最終更新**: 2025-11-23
