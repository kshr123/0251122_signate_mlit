# exp004: 築年数関連特徴量の強化

## 概要
- **目的**: 低価格帯（~1000万円）の予測精度改善
- **ベースライン**: exp003 (CV MAPE: 27.47%)
- **アプローチ**: 築年数の非線形性と交互作用を捉える特徴量追加

## 背景（exp003のエラー分析より）

### 問題点
- 低価格帯（~1000万円）の **97.8%が過大予測**
- 築40年以上の物件が低価格帯の53%を占める
- 築年数とAPEの関係:
  - 5-10年: 17.2%（最良）
  - 30-40年: 29.8%
  - 50年以上: 34-82%（大幅悪化）

### 仮説
1. 築年数の影響は線形ではなく、特定の年数で急激に価値が下がる
2. 「古い × 広い」「古い × 地方」の物件は特に安くなる傾向

---

## 基準値の決定根拠

### データ分布（eda_threshold_analysis.ipynb）

| 指標 | 中央値(50%ile) | 66%ile | 75%ile |
|------|---------------|--------|--------|
| 築年数 | 30年 | 37年 | 42年 |
| 面積 | 78.7㎡ | 92.7㎡ | 101.6㎡ |

### 組み合わせの効果検証

| 組み合わせ | APE差 | 件数 | 低価格帯での出現率 |
|-----------|-------|------|-------------------|
| 築35年 & 80㎡以上 | **+11.7pt** | 53,686 (14.8%) | 24.5% (x1.7倍) |
| 築35年 & 地方 | **+12.0pt** | 70,492 (19.4%) | 46.6% (x2.4倍) |

### 決定した基準値

```python
AGE_THRESHOLD = 35      # 築年数: 66%ile付近
AREA_THRESHOLD = 80     # 面積: 50%ile付近
```

**選定理由**:
- 築35年: 66%ile（上位1/3が該当）、APE悪化が顕著に始まる
- 面積80㎡: 50%ile付近、組み合わせで15%程度が該当（適度な粒度）

---

## 新規特徴量

### 1. 築年数（building_age）
```python
building_age = 2024 - year_built
```

**理由**: 築年数の直接的な影響を捉える（year_builtは既にYYYY形式）

### 2. 築年数のカテゴリ化（5年単位）
```python
building_age_bin = (building_age // 5).clip(0, 10)
# 0: 0-5年, 1: 5-10年, ..., 10: 50年以上
```

**理由**:
- 築年数とAPEの関係が非線形
- LightGBMがカテゴリとして扱うことで非線形性を捉える

### 3. 築35年以上フラグ
```python
old_building_flag = (building_age >= 35).cast(int)
```

**理由**:
- 66%ile付近でAPE悪化が顕著
- 古い物件特有の価格決定要因をモデルが学習しやすくする

### 4. 交互作用特徴量（フラグの組み合わせ）

#### 4-1. 古くて広い物件フラグ
```python
old_and_large_flag = ((building_age >= 35) & (house_area >= 80)).cast(int)
```

**理由**:
- APE差 +11.7pt（効果大）
- 低価格帯で1.7倍出現
- 「古くて広い＝郊外の安い物件」パターンを捉える

#### 4-2. 古くて地方の物件フラグ
```python
# 地方フラグ: 東京(13)、神奈川(14)、愛知(23)、大阪(27) 以外
MAJOR_CITIES = [13, 14, 23, 27]
rural_flag = (~addr1_1.is_in(MAJOR_CITIES)).cast(int)

old_and_rural_flag = ((building_age >= 35) & (rural_flag == 1)).cast(int)
```

**理由**:
- APE差 +12.0pt（最大効果）
- 低価格帯で2.4倍出現
- 「地方で古い＝特に安い」パターンを捉える

---

## 特徴量一覧（exp003からの差分）

### 追加特徴量（5個）
| 特徴量名 | 説明 | 型 |
|----------|------|-----|
| `building_age` | 築年数（2024 - year_built） | 数値 |
| `building_age_bin` | 築年数5年単位カテゴリ（0-10） | カテゴリカル |
| `old_building_flag` | 築35年以上フラグ | バイナリ |
| `old_and_large_flag` | 築35年以上 & 80㎡以上フラグ | バイナリ |
| `old_and_rural_flag` | 築35年以上 & 地方フラグ | バイナリ |

### 補助特徴量（特徴量として直接使用しない）
| 特徴量名 | 説明 |
|----------|------|
| `rural_flag` | 地方フラグ（4大都市圏以外=1） |

### 特徴量総数
- exp003: 83個
- exp004: 83 + 5 = **88個**

---

## 実装方針

### ディレクトリ構成
```
06_experiments/exp004_age_features/
├── SPEC.md                    # この仕様書
├── code/
│   ├── preprocessing.py       # 前処理（exp003ベース + 新特徴量）
│   └── train.py               # 訓練スクリプト
├── outputs/                   # 出力ファイル
└── notebooks/
    └── eda_threshold_analysis.ipynb  # 基準値決定の分析
```

### preprocessing.py の変更点

exp003の`preprocessing.py`をベースに以下を追加:

```python
# 基準値
AGE_THRESHOLD = 35
AREA_THRESHOLD = 80
MAJOR_CITIES = [13, 14, 23, 27]  # 東京、神奈川、愛知、大阪

def add_age_features(df: pl.DataFrame) -> pl.DataFrame:
    """築年数関連特徴量を追加"""
    return df.with_columns([
        # 築年数（year_builtは既にYYYY形式）
        (2024 - pl.col("year_built")).alias("building_age"),

        # 築年数カテゴリ（5年単位、0-10）
        ((2024 - pl.col("year_built")) // 5).clip(0, 10).alias("building_age_bin"),

        # 築35年以上フラグ
        ((2024 - pl.col("year_built")) >= AGE_THRESHOLD).cast(pl.Int64).alias("old_building_flag"),

        # 地方フラグ
        (~pl.col("addr1_1").is_in(MAJOR_CITIES)).cast(pl.Int64).alias("rural_flag"),
    ]).with_columns([
        # 交互作用フラグ
        (
            (pl.col("building_age") >= AGE_THRESHOLD) &
            (pl.col("house_area") >= AREA_THRESHOLD)
        ).cast(pl.Int64).alias("old_and_large_flag"),

        (
            (pl.col("building_age") >= AGE_THRESHOLD) &
            (pl.col("rural_flag") == 1)
        ).cast(pl.Int64).alias("old_and_rural_flag"),
    ])
```

### カテゴリカル特徴量の更新
```python
CATEGORICAL_FEATURES = [
    # exp003のカテゴリカル特徴量
    ...
    # 追加
    'building_age_bin',
]
```

---

## 期待される効果

| 特徴量 | 期待効果 |
|--------|----------|
| `building_age` | 築年数の直接的な影響を捉える |
| `building_age_bin` | 非線形な減価パターンを捉える |
| `old_building_flag` | 古い物件の特別な価格決定要因 |
| `old_and_large_flag` | 「古くて広い＝安い」パターン |
| `old_and_rural_flag` | 「地方で古い＝特に安い」パターン |

**目標**: CV MAPE 27.0% 以下（0.5pt以上の改善）

---

## 検証項目

1. 全体のCV MAPEの改善
2. **低価格帯（~1000万）のAPE改善**（最重要）
3. 新規特徴量の重要度
4. 築年数帯別のAPE変化
5. 地方 vs 都市部のAPE変化

---

## リスクと対策

| リスク | 対策 |
|--------|------|
| 過学習 | 特徴量数が少ないため低リスク |
| 欠損値 | year_built欠損時はbuilding_age関連すべてnull |
| 効果なし | 個別特徴量の寄与を分析し、効果のないものは削除 |
