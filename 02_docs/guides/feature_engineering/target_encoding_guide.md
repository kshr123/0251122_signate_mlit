# Target Encoding 完全ガイド

> カテゴリ変数をターゲット統計量でエンコードする強力な手法

---

## 📖 Target Encodingとは

**定義**: カテゴリを**そのカテゴリのターゲット平均値**で置き換える手法

**別名**: Mean Encoding, Likelihood Encoding

**特徴**:
- カテゴリ変数を**1次元の数値**に変換
- ターゲットとの**関係性を直接エンコード**
- 高カーディナリティ変数に特に有効

---

## 🔑 基本原理

### 仕組み

```python
# 元データ
room_count | money_room (target)
-----------|-------------------
    1      |   50,000
    1      |   55,000
    2      |   80,000
    2      |   85,000
    3      |  120,000
    3      |  125,000

# Step 1: カテゴリ別のターゲット平均を計算
room_count | mean_target
-----------|-------------
    1      |   52,500
    2      |   82,500
    3      |  122,500

# Step 2: カテゴリを平均値で置き換え
room_count_encoded | money_room
-------------------|------------
     52,500        |   50,000
     52,500        |   55,000
     82,500        |   80,000
     82,500        |   85,000
    122,500        |  120,000
    122,500        |  125,000

# これをモデルの特徴量として使用
```

### なぜ有効か？

```python
# カテゴリ変数のまま
room_count = [1, 2, 3, 4]
# モデルは「1 < 2 < 3 < 4」という大小関係しか学習できない

# Target Encoding後
room_count_encoded = [52500, 82500, 122500, 182500]
# モデルは「1部屋は安い、4部屋は高い」という
# ターゲットとの関係を直接学習できる

# → 予測精度が大幅に向上！
```

---

## ⚙️ 実装方法

### 基本実装（ナイーブ版）

```python
import polars as pl

def target_encode_naive(
    df: pl.DataFrame,
    cat_col: str,
    target_col: str
) -> pl.DataFrame:
    """
    ナイーブなTarget Encoding（訓練データのみ）

    ⚠️ 警告: データリーク（過学習）が発生するため本番では使用不可
    """
    # カテゴリ別平均を計算
    cat_means = (
        df.group_by(cat_col)
        .agg(pl.col(target_col).mean().alias(f"{cat_col}_te"))
    )

    # 元のデータフレームにマージ
    df_encoded = df.join(cat_means, on=cat_col, how='left')

    return df_encoded

# 使用例
train = pl.DataFrame({
    'room_count': [1, 1, 2, 2, 3, 3],
    'money_room': [50000, 55000, 80000, 85000, 120000, 125000]
})

train_encoded = target_encode_naive(train, 'room_count', 'money_room')
print(train_encoded)
```

### 正しい実装（クロスバリデーション版）

```python
from sklearn.model_selection import KFold
import numpy as np

def target_encode_cv(
    df: pl.DataFrame,
    cat_col: str,
    target_col: str,
    n_folds: int = 5,
    alpha: float = 10.0
) -> pl.Series:
    """
    クロスバリデーションを使った正しいTarget Encoding

    Args:
        df: データフレーム
        cat_col: カテゴリカラム
        target_col: ターゲットカラム
        n_folds: Fold数
        alpha: 正則化パラメータ（スムージング）

    Returns:
        エンコードされた値のSeries
    """
    encoded = np.zeros(len(df))
    global_mean = df[target_col].mean()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(df):
        # 訓練Foldで平均を計算
        train_fold = df[train_idx]

        cat_stats = (
            train_fold.group_by(cat_col)
            .agg([
                pl.col(target_col).mean().alias('mean'),
                pl.col(target_col).count().alias('count')
            ])
        )

        # バリデーションFoldに適用
        val_fold = df[val_idx]

        for row_idx, cat_value in enumerate(val_fold[cat_col]):
            # カテゴリの統計量を取得
            stats = cat_stats.filter(pl.col(cat_col) == cat_value)

            if stats.height > 0:
                mean = stats['mean'][0]
                count = stats['count'][0]

                # スムージング適用
                smoothed = (count * mean + alpha * global_mean) / (count + alpha)
                encoded[val_idx[row_idx]] = smoothed
            else:
                # 未知カテゴリは全体平均
                encoded[val_idx[row_idx]] = global_mean

    return pl.Series(f"{cat_col}_te", encoded)

# 使用例
train['room_count_te'] = target_encode_cv(
    train,
    cat_col='room_count',
    target_col='money_room',
    n_folds=5
)
```

### テストデータへの適用

```python
def apply_target_encoding(
    train: pl.DataFrame,
    test: pl.DataFrame,
    cat_col: str,
    target_col: str,
    alpha: float = 10.0
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    訓練データでTarget Encodingを学習し、テストデータに適用

    Args:
        train: 訓練データ
        test: テストデータ
        cat_col: カテゴリカラム
        target_col: ターゲットカラム
        alpha: スムージングパラメータ

    Returns:
        (train_encoded, test_encoded)
    """
    global_mean = train[target_col].mean()

    # 訓練データで統計量を計算
    cat_stats = (
        train.group_by(cat_col)
        .agg([
            pl.col(target_col).mean().alias('mean'),
            pl.col(target_col).count().alias('count')
        ])
    )

    # スムージング適用
    cat_stats = cat_stats.with_columns(
        ((pl.col('count') * pl.col('mean') + alpha * global_mean) / (pl.col('count') + alpha))
        .alias(f"{cat_col}_te")
    )

    # 訓練データにマージ（クロスバリデーション版を使用推奨）
    train_encoded = train.join(
        cat_stats.select([cat_col, f"{cat_col}_te"]),
        on=cat_col,
        how='left'
    )

    # テストデータにマージ
    test_encoded = test.join(
        cat_stats.select([cat_col, f"{cat_col}_te"]),
        on=cat_col,
        how='left'
    )

    # 未知カテゴリは全体平均で埋める
    test_encoded = test_encoded.with_columns(
        pl.col(f"{cat_col}_te").fill_null(global_mean)
    )

    return train_encoded, test_encoded
```

---

## ✅ メリット

### 1. 高い予測精度

```python
# カテゴリとターゲットの関係を直接数値化
# → モデルが学習しやすい
# → 特にTree-basedモデル（LightGBM、XGBoost）で効果大
```

### 2. 次元数が増えない

```python
# One-Hot Encoding
# 47都道府県 → 47次元

# Target Encoding
# 47都道府県 → 1次元

# → メモリ効率が良い
# → 学習速度が速い
```

### 3. 高カーディナリティに対応

```python
# building_name（69,370種類）

# One-Hot: 不可能（次元爆発）
# Target Encoding: 可能（1次元）
```

### 4. 順序関係を保持

```python
# 1部屋: 52,500円
# 2部屋: 82,500円
# 3部屋: 122,500円

# → 自然な順序関係が数値に反映される
```

---

## ❌ デメリットと対策

### 1. データリーク（過学習）

**問題**:
```python
# 訓練データで直接エンコードすると...
train['room_count_te'] = train.group_by('room_count')['money_room'].mean()

# → 訓練データでは完璧に予測できる
# → テストデータでは精度が下がる（過学習）
```

**対策**:
```python
# クロスバリデーションでエンコード
# → 各Foldでは「他のFoldの平均」を使う
# → データリークを防ぐ

encoded = target_encode_cv(train, 'room_count', 'money_room', n_folds=5)
```

### 2. 未知カテゴリ

**問題**:
```python
# 訓練データ: room_count = [1, 2, 3, 4]
# テストデータ: room_count = [1, 2, 3, 5]  # 5は未知

# → 5部屋のエンコード値がない
```

**対策**:
```python
# 全体平均で埋める
global_mean = train['money_room'].mean()

test['room_count_te'] = test['room_count'].map(encoding_dict)
test['room_count_te'] = test['room_count_te'].fill_null(global_mean)
```

### 3. サンプル数が少ないカテゴリ

**問題**:
```python
# 28部屋: サンプル数12件、平均1.5億円
# → サンプルが少なく、平均が不安定

# 訓練データとテストデータで大きく異なる可能性
```

**対策**:
```python
# スムージング（Bayesian平均）
# サンプル数が少ないカテゴリは全体平均に近づける

smoothed = (count * category_mean + alpha * global_mean) / (count + alpha)

# alpha=10の場合:
# サンプル数12件 → 全体平均の影響 約45%
# サンプル数100件 → 全体平均の影響 約9%
```

---

## 🎯 Target Encoding効果の予測

### 指標の定義

**Target Encoding効果** = カテゴリ別ターゲット平均値の標準偏差

```python
def calculate_target_encoding_potential(
    df: pl.DataFrame,
    category_col: str,
    target_col: str
) -> float:
    """
    Target Encodingの効果を予測

    Returns:
        カテゴリ別ターゲット平均値の標準偏差
        （値が大きいほど効果的）
    """
    # カテゴリごとの平均を計算
    category_means = (
        df.group_by(category_col)
        .agg(pl.col(target_col).mean().alias("mean_target"))
        .drop_nulls()
    )

    # 平均値の標準偏差
    std_of_means = category_means["mean_target"].std()

    return std_of_means
```

### 解釈

```python
# 標準偏差が大きい
# → カテゴリ間でターゲットが大きく異なる
# → Target Encodingが有効
# → 予測精度向上が期待できる

# 標準偏差が小さい
# → カテゴリ間でターゲットがほぼ同じ
# → Target Encodingの効果は薄い
```

### 実例（このプロジェクト）

```python
# 04_categorical_analysis.ipynb の結果

[Target Encoding効果予測 - 上位5件]
カラム名                | ユニーク数 | 標準偏差
-----------------------|-----------|-------------
room_count             |        28 | 34,579,045  ← 最高！
madori_number_all      |        32 | 33,062,546
reform_exterior        |         7 | 17,664,376
parking_money_tax      |         5 | 16,789,196
parking_kubun          |         6 | 14,193,758

# room_count の詳細
room_count | count  | mean_target
-----------|--------|---------------
    28     |    12  | 150,000,000   ← 28部屋: 超高額
    27     |    18  | 142,000,000
    ...    |  ...   |  ...
     3     | 15,420 |  35,000,000
     2     | 45,890 |  25,000,000
     1     | 89,234 |  18,000,000   ← 1部屋: 安い

# カテゴリ間の差が約1.3億円
# → Target Encodingが超有効！
```

---

## 🔬 スムージング（平滑化）

### 概念

サンプル数が少ないカテゴリの平均を**全体平均に近づける**ことで、過学習を防ぐ。

### Bayesian平均

```python
smoothed_mean = (n * category_mean + alpha * global_mean) / (n + alpha)

# n: カテゴリのサンプル数
# alpha: 正則化パラメータ（一般的に 5〜20）
```

### 効果

```python
# 例: 全体平均 = 25,000,000円

# サンプル数100件、平均30,000,000円
smoothed = (100 * 30M + 10 * 25M) / (100 + 10)
         = (3,000M + 250M) / 110
         = 29,545,455  # ほぼ元の平均

# サンプル数5件、平均80,000,000円（外れ値の可能性）
smoothed = (5 * 80M + 10 * 25M) / (5 + 10)
         = (400M + 250M) / 15
         = 43,333,333  # 全体平均に引き寄せられた

# → サンプル数が少ないほど全体平均に近づく
# → 過学習を防ぐ
```

### alphaパラメータの選び方

```python
# alpha = 0: スムージングなし（過学習リスク大）
# alpha = 5: 弱いスムージング
# alpha = 10: 中程度のスムージング（推奨）
# alpha = 20: 強いスムージング
# alpha = 100: 非常に強いスムージング（全体平均に近づきすぎ）

# グリッドサーチで最適値を探す
for alpha in [5, 10, 15, 20]:
    encoded = target_encode_cv(train, 'room_count', 'money_room', alpha=alpha)
    score = cross_val_score(model, encoded, y, cv=5).mean()
    print(f"alpha={alpha}: {score:.4f}")
```

---

## 💡 実践的なTips

### Tip 1: 複数のTarget統計量を使う

```python
# 平均だけでなく、他の統計量も特徴量に

cat_stats = df.group_by('room_count').agg([
    pl.col('money_room').mean().alias('room_count_mean'),    # 平均
    pl.col('money_room').median().alias('room_count_median'),  # 中央値
    pl.col('money_room').std().alias('room_count_std'),      # 標準偏差
    pl.col('money_room').min().alias('room_count_min'),      # 最小値
    pl.col('money_room').max().alias('room_count_max'),      # 最大値
    pl.col('money_room').count().alias('room_count_count')   # 件数
])

# すべてを特徴量として使う
# → モデルがより豊富な情報を学習
```

### Tip 2: カテゴリ組み合わせのTarget Encoding

```python
# 単一カテゴリだけでなく、組み合わせも

# 例: 都道府県 × 建物構造
df = df.with_columns(
    (pl.col('prefecture') + '_' + pl.col('building_structure'))
    .alias('pref_structure')
)

# pref_structure をTarget Encoding
# → より細かい粒度で関係性を捉える
```

### Tip 3: 他のエンコーディングとの併用

```python
# Target Encodingだけでなく、複数手法を併用

df = df.with_columns([
    # Target Encoding
    pl.col('room_count').alias('room_count_te'),

    # Frequency Encoding
    pl.col('room_count')
    .value_counts()
    .struct.field('count')
    .alias('room_count_freq'),

    # One-Hot Encoding（部屋数が少ないので可能）
    # ...
])

# モデルが自動的に有効な特徴量を選択
```

### Tip 4: クロスバリデーション戦略の統一

```python
# Target Encodingとモデル学習で同じFold分割を使う

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 1. Target Encoding
encoded = target_encode_cv(train, 'room_count', 'money_room', kf=kf)

# 2. モデル学習も同じFoldで
for train_idx, val_idx in kf.split(train):
    # ...
```

---

## 📊 このプロジェクトでの活用

### 対象変数

```python
# Target Encoding効果が高い変数（上位10件）

カラム名                | カーディナリティ | 効果（標準偏差）
-----------------------|----------------|----------------
room_count             |       28       | 34,579,045
madori_number_all      |       32       | 33,062,546
reform_exterior        |        7       | 17,664,376
parking_money_tax      |        5       | 16,789,196
parking_kubun          |        6       | 14,193,758
building_structure     |       13       | 13,389,693
building_type          |       16       | 13,074,501
genkyo_code            |        6       | 12,263,027
basement_floor_count   |       18       | 11,646,776
traffic_car            |        3       |  8,629,124
```

### 実装例

```python
import polars as pl
from sklearn.model_selection import KFold

# データ読み込み
train = pl.read_parquet('data/processed/train.parquet')

# Target Encoding対象カラム
te_cols = [
    'room_count',
    'madori_number_all',
    'reform_exterior',
    'parking_money_tax',
    'parking_kubun'
]

# クロスバリデーションでTarget Encoding
target_col = 'money_room'

for col in te_cols:
    train = train.with_columns(
        target_encode_cv(train, col, target_col, n_folds=5, alpha=10)
    )

# モデリング
from lightgbm import LGBMRegressor

# 元のカテゴリカラムとTarget Encodedカラムの両方を使う
feature_cols = te_cols + [f"{col}_te" for col in te_cols]

X = train.select(feature_cols).to_pandas()
y = train[target_col].to_numpy()

model = LGBMRegressor()
model.fit(X, y)
```

---

## 📚 参考資料

### 論文・記事

- [A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems](https://dl.acm.org/doi/10.1145/507533.507538) (Micci-Barreca, 2001)
- Kaggleでの活用事例多数（特にClick-Through Rate予測、広告コンペ等）

### 実装ライブラリ

- category_encoders: `TargetEncoder`
- feature-engine: `MeanEncoder`
- 自作実装推奨（クロスバリデーション制御のため）

---

## 📖 関連ドキュメント

- [cardinality_guide.md](./cardinality_guide.md) - カーディナリティ完全ガイド
- [hashing_trick_guide.md](./hashing_trick_guide.md) - Hashing Trick完全ガイド
- [04_categorical_analysis.ipynb](../05_notebooks/01_eda/04_categorical_analysis.ipynb) - Target Encoding効果の分析
- [04_src/eda/categorical.py](../04_src/eda/categorical.py) - Target Encoding効果計算関数（TDD済み）

---

**最終更新**: 2025-11-23
