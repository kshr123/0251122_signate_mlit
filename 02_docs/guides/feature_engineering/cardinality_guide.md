# カーディナリティ完全ガイド

> カテゴリ変数のユニーク値数の理解と活用

---

## 📖 カーディナリティとは

**定義**: カテゴリ変数が持つ**ユニークな値の数**

```python
# 例
gender = ['男', '女', '男', '女', '男']
# ユニーク値: ['男', '女'] → カーディナリティ = 2

prefecture = ['東京', '大阪', '愛知', '東京', '福岡', ...]
# ユニーク値: 47種類 → カーディナリティ = 47

building_id = [1, 2, 3, 4, ..., 175577]
# ユニーク値: 175,577種類 → カーディナリティ = 175,577
```

---

## 📊 カーディナリティの分類

### 低カーディナリティ（Low Cardinality）

**定義**: ユニーク値が少ない（一般的に **< 10**）

**特徴**:
- 各カテゴリに十分なサンプル数がある
- カテゴリ間の比較が容易
- そのままエンコーディングしやすい

**例**:
```python
# 性別
['男', '女']  # 2種類

# 建物構造
['木造', '鉄筋コンクリート', '鉄骨造', 'RC造', 'SRC造']  # 5種類

# 駐車場区分
['有', '無', '月極', '空き無し', '敷地内', '近隣']  # 6種類
```

**推奨エンコーディング**:
- ✅ **One-Hot Encoding**（ダミー変数化）
- ✅ Label Encoding

**実装例**:
```python
import polars as pl

# One-Hot Encoding
df = pl.DataFrame({'構造': ['木造', 'RC造', '木造', 'SRC造']})

# Polarsでダミー変数化
df_encoded = df.to_dummies('構造')

# 結果
# 構造_木造 | 構造_RC造 | 構造_SRC造
# ---------|----------|----------
#    1     |    0     |    0
#    0     |    1     |    0
#    1     |    0     |    0
#    0     |    0     |    1
```

---

### 中カーディナリティ（Medium Cardinality）

**定義**: ユニーク値が中程度（一般的に **10〜50**）

**特徴**:
- カテゴリによってサンプル数にばらつきがある
- One-Hotだと次元が多くなりすぎる
- Target Encodingが効果的

**例**:
```python
# 都道府県
['北海道', '青森', '岩手', ..., '沖縄']  # 47種類

# 間取り種類
['1K', '1DK', '1LDK', '2K', '2DK', '2LDK', '3LDK', ...]  # 32種類

# 部屋数
[1, 2, 3, 4, 5, 6, 7, ..., 28]  # 28種類
```

**推奨エンコーディング**:
- ✅ **Target Encoding**（ターゲット平均値でエンコード）
- ✅ Frequency Encoding（出現頻度でエンコード）
- ○ One-Hot Encoding（次元数が許容範囲なら）

**実装例**:
```python
# Target Encoding（クロスバリデーション版）
def target_encode_cv(df, cat_col, target_col, n_folds=5):
    from sklearn.model_selection import KFold

    encoded = pl.Series([None] * len(df))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(df):
        # 訓練データで平均を計算
        train_means = (
            df[train_idx]
            .group_by(cat_col)
            .agg(pl.col(target_col).mean())
        )

        # バリデーションデータに適用
        # ...（実装詳細は省略）

    return encoded

# Frequency Encoding
df = df.with_columns(
    pl.col('prefecture')
    .value_counts()
    .struct.field('count')
    .alias('prefecture_freq')
)
```

---

### 高カーディナリティ（High Cardinality）

**定義**: ユニーク値が非常に多い（一般的に **> 50**）

**特徴**:
- 多くのカテゴリが少数サンプルしかない
- One-Hotは次元爆発で不可能
- 未知カテゴリ対策が必須

**例**:
```python
# 物件ID
[1, 2, 3, ..., 175577]  # 175,577種類

# 住所（市区町村）
['千代田区', '中央区', '港区', ..., '那覇市']  # 64,822種類

# 建物名
['〇〇マンション', '△△ハイツ', ...]  # 69,370種類
```

**推奨エンコーディング**:
- ✅ **Leave-One-Out Encoding**
- ✅ **Frequency Encoding**
- ✅ **Hashing Trick**（超高カーディナリティ）
- ✅ カテゴリ集約（上位N件 + "その他"）

**実装例**:
```python
# Leave-One-Out Encoding
def loo_encode(df, cat_col, target_col):
    # 全体平均
    global_mean = df[target_col].mean()

    # カテゴリ別の合計とカウント
    cat_stats = df.group_by(cat_col).agg([
        pl.col(target_col).sum().alias('sum'),
        pl.col(target_col).count().alias('count')
    ])

    # 各行について、自分自身を除いた平均を計算
    # encoded = (sum - value) / (count - 1)
    # ...

# Frequency Encoding
df = df.with_columns(
    pl.col('building_name')
    .value_counts()
    .struct.field('count')
    .alias('building_name_freq')
)

# カテゴリ集約
top_n = 50
top_buildings = (
    df['building_name']
    .value_counts()
    .head(top_n)
    .struct.field('building_name')
)

df = df.with_columns(
    pl.when(pl.col('building_name').is_in(top_buildings))
    .then(pl.col('building_name'))
    .otherwise(pl.lit('その他'))
    .alias('building_name_grouped')
)
```

---

## 🎯 このプロジェクトでの実例

### 分析結果（04_categorical_analysis.ipynb）

```
カテゴリカラム数: 126件

低カーディナリティ（<10）: 28件
中カーディナリティ（10-50）: 11件
高カーディナリティ（>50）: 87件
```

### 低カーディナリティの例

```
カラム名                  | ユニーク数
-------------------------|----------
target_ym                |    8
land_chisei              |    8
reform_exterior          |    7
land_toshi               |    6
house_kanrinin           |    6
parking_kubun            |    6
genkyo_code              |    6
management_form          |    5
parking_money_tax        |    5
land_area_kind           |    4
```

### 中カーディナリティの例

```
カラム名                  | ユニーク数
-------------------------|----------
addr1_1                  |   47
madori_number_all        |   32
room_count               |   28
basement_floor_count     |   18
building_type            |   16
land_youto               |   15
building_structure       |   13
```

### 高カーディナリティの例

```
カラム名                  | ユニーク数
-------------------------|----------
statuses                 | 232,339
unit_tag_id              | 209,158
building_id              | 175,577
full_address             | 172,933
snapshot_modify_date     | 170,780
homes_building_name      | 144,790
building_tag_id          | 127,515
building_name            |  69,370
addr2_name               |  64,822
```

---

## ⚠️ カーディナリティが重要な理由

### 1. エンコーディング手法の選択

```python
# ❌ 高カーディナリティでOne-Hot Encoding
building_id (175,577種類) → 175,577次元の疎行列
# → メモリ不足、学習不可能

# ✅ 適切な手法
# → Target Encoding、Frequency Encoding等
```

### 2. 過学習リスク

高カーディナリティ変数の問題:
- 訓練データにしか存在しないカテゴリが多い
- テストデータで未知カテゴリに対処できない
- 特定のカテゴリに過度に適合しやすい

**対策**:
```python
# 1. クロスバリデーションでエンコード
# 2. 正則化を強める
# 3. カテゴリを集約（上位N件 + その他）
```

### 3. 計算コスト

```python
# One-Hot Encoding: 次元数 = カーディナリティ
# 175,577次元 → メモリ使用量・学習時間が爆発

# Target Encoding: 次元数 = 1
# 1次元 → 効率的
```

### 4. 未知カテゴリへの対応

```python
# 訓練データ
buildings = ['マンションA', 'マンションB', 'マンションC']

# テストデータ
new_buildings = ['マンションD', 'マンションE']  # 未知カテゴリ

# One-Hot: エンコード不可（全0ベクトル）
# Target: 全体平均で埋める
# Hashing: 自動的にエンコード
```

---

## 📋 エンコーディング手法の選択フローチャート

```
カーディナリティを確認
    ↓
< 10種類？
    Yes → One-Hot Encoding
    ↓ No
    ↓
< 50種類？
    Yes → Target Encoding（推奨）
          または Frequency Encoding
    ↓ No
    ↓
< 1000種類？
    Yes → Leave-One-Out Encoding
          または Frequency Encoding
          または カテゴリ集約
    ↓ No
    ↓
超高カーディナリティ（> 1000）
    → Hashing Trick
    → Frequency Encoding
    → カテゴリ集約（上位N件）
```

---

## 📊 カーディナリティの確認方法

### Polarsでの実装

```python
import polars as pl

def check_cardinality(df: pl.DataFrame, col: str) -> dict:
    """カーディナリティ情報を取得"""
    n_unique = df[col].n_unique()
    n_total = df.height
    top_10 = df[col].value_counts().head(10)

    return {
        'column': col,
        'n_unique': n_unique,
        'n_total': n_total,
        'ratio': n_unique / n_total,
        'top_10': top_10
    }

# 使用例
info = check_cardinality(train, 'building_name')
print(f"ユニーク数: {info['n_unique']}")
print(f"比率: {info['ratio']:.2%}")
```

### 分類関数（テスト済み）

```python
# 04_src/eda/categorical.py
def classify_cardinality(
    df: pl.DataFrame,
    categorical_cols: list[str],
    low_threshold: int = 10,
    medium_threshold: int = 50
) -> dict[str, list[tuple[str, int]]]:
    """
    カテゴリカラムをカーディナリティで分類

    Returns:
        {
            'low': [(col, n_unique), ...],
            'medium': [(col, n_unique), ...],
            'high': [(col, n_unique), ...]
        }
    """
    result = {'low': [], 'medium': [], 'high': []}

    for col in categorical_cols:
        n_unique = df[col].n_unique()

        if n_unique < low_threshold:
            result['low'].append((col, n_unique))
        elif n_unique < medium_threshold:
            result['medium'].append((col, n_unique))
        else:
            result['high'].append((col, n_unique))

    # 降順ソート
    for key in result:
        result[key].sort(key=lambda x: x[1], reverse=True)

    return result
```

---

## 💡 実践的なTips

### Tip 1: カーディナリティは相対的

```python
# サンプル数1,000のデータセット
# カーディナリティ100 → 高い（平均10サンプル/カテゴリ）

# サンプル数100万のデータセット
# カーディナリティ100 → 低い（平均10,000サンプル/カテゴリ）

# 判断基準: n_unique / n_total の比率も見る
```

### Tip 2: Target Encodingとの相性

```python
# 中カーディナリティ × 高Target Encoding効果
# → 最も効果的な特徴量になる可能性

# このプロジェクトでの例
room_count:
  - カーディナリティ: 28（中）
  - Target Encoding効果: 34,579,045（最高）
  - → 超有効な特徴量！
```

### Tip 3: 複数手法の併用

```python
# 同じカテゴリ変数を複数の方法でエンコード
df = df.with_columns([
    # Target Encoding
    pl.col('building_name').alias('building_name_te'),

    # Frequency Encoding
    pl.col('building_name')
    .value_counts()
    .struct.field('count')
    .alias('building_name_freq'),

    # 集約版（上位100件 + その他）
    pl.col('building_name').alias('building_name_grouped')
])

# モデルが自動的に有効な特徴量を選択
```

---

## 📚 関連ドキュメント

- [target_encoding_guide.md](./target_encoding_guide.md) - Target Encoding完全ガイド
- [hashing_trick_guide.md](./hashing_trick_guide.md) - Hashing Trick完全ガイド
- [04_categorical_analysis.ipynb](../05_notebooks/01_eda/04_categorical_analysis.ipynb) - カーディナリティ分析の実装
- [04_src/eda/categorical.py](../04_src/eda/categorical.py) - カーディナリティ分類関数

---

**最終更新**: 2025-11-23
