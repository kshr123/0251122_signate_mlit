# exp008_reform_features

## 概要

exp007をベースに、リフォーム情報と郵便番号ターゲットエンコーディングを追加する実験。
また、築年数の基準年を2024固定からデータ作成日時（building_create_date）に修正。

- **ベース**: exp007_tag_features (CV MAPE: 13.62%)
- **目標**: CV MAPE < 13.5%（0.1pt以上の改善）

## 変更点

### 1. リフォーム特徴量の追加

#### 元データの分析結果

| カラム | 欠損率 | 有効件数 | ユニークタグ | データ形式 |
|--------|--------|----------|--------------|------------|
| reform_wet_area | 77.2% | 82,921 | 1〜6 | "1/2/3/4" |
| reform_interior | 76.2% | 86,442 | 1〜6 | "2/4/3" |
| reform_wet_area_date | 77.0% | 83,673 | 370 | 201807.0 |
| reform_interior_date | 75.2% | 90,266 | 384 | 201907.0 |

**発見**: wet_area と interior は同じタグ体系（1〜6）を使用

#### 処理方法

| 特徴量 | 処理 | 次元 | 累積寄与率 |
|--------|------|------|------------|
| reform_svd | wet + interior 統合 Multi-hot → SVD | 7 | 93.3% |
| years_since_wet_reform | 経過年数計算 | 1 | - |
| years_since_interior_reform | 経過年数計算 | 1 | - |

**統合版の利点**:
- wet_1, wet_2, ... int_1, int_2, ... とプレフィックスで区別
- Multi-hot 14次元 → SVD 7次元で93.3%の情報を保持
- 分離版（10次元）より効率的

### 2. 郵便番号ターゲットエンコーディング

| カラム | 処理 | 備考 |
|--------|------|------|
| post1 | ターゲットエンコーディング | 上3桁（地域ブロック） |
| post_full | ターゲットエンコーディング | 7桁（post1 + post2）、件数30未満はpost1_teにフォールバック |

**階層フォールバック方式**:
- post_full（7桁郵便番号）のTE値を算出
- trainで件数30未満のpost_fullは、post1のTE値で代替
- testでは、trainのvalid_post_fulls（30件以上）に含まれるかで判定

### 3. 面積の地域平均比率特徴量

| 特徴量名 | 計算方法 | 備考 |
|----------|----------|------|
| house_area_pref_ratio | house_area / 都道府県別平均 | 地域水準との比較 |
| house_area_city_ratio | house_area / 市区町村別平均 | 地域水準との比較 |
| snapshot_land_area_pref_ratio | snapshot_land_area / 都道府県別平均 | 地域水準との比較 |
| snapshot_land_area_city_ratio | snapshot_land_area / 市区町村別平均 | 地域水準との比較 |
| unit_area_pref_ratio | unit_area / 都道府県別平均 | 地域水準との比較 |
| unit_area_city_ratio | unit_area / 市区町村別平均 | 地域水準との比較 |

**ポイント**:
- 面積そのものではなく、地域平均との比率を特徴量化
- 比率 > 1: その地域の平均より広い物件
- 比率 < 1: その地域の平均より狭い物件
- OOF方式で地域平均を計算しリーク防止

### 4. 基準年の修正

| 特徴量 | 修正前 | 修正後 |
|--------|--------|--------|
| building_age | 2024 - year_built | building_create_date年 - year_built |
| years_since_wet_reform | - | building_create_date年 - reform_wet_area_date年 |
| years_since_interior_reform | - | building_create_date年 - reform_interior_date年 |

## 追加特徴量

| カテゴリ | 特徴量名 | 個数 |
|----------|----------|------|
| リフォームSVD | reform_svd_0 ~ 6 | 7 |
| 経過年数 | years_since_wet_reform | 1 |
| 経過年数 | years_since_interior_reform | 1 |
| 郵便番号TE | post1_te | 1 |
| 郵便番号TE | post_full_te | 1 |
| 面積比率 | house_area_pref_ratio, house_area_city_ratio | 2 |
| 面積比率 | snapshot_land_area_pref_ratio, snapshot_land_area_city_ratio | 2 |
| 面積比率 | unit_area_pref_ratio, unit_area_city_ratio | 2 |
| **合計** | | **17** |

## 特徴量数

- exp007: 164個
- 追加特徴量: 17個
- **合計**: 164 + 17 = **181個**

## 実装詳細

### 統合Multi-hot → SVD処理

```python
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

def create_reform_tags(wet_area, interior):
    """wet_area と interior を統合してプレフィックス付きタグリストを作成"""
    tags = []
    if wet_area and str(wet_area) != 'nan':
        tags.extend([f'wet_{t}' for t in str(wet_area).split('/')])
    if interior and str(interior) != 'nan':
        tags.extend([f'int_{t}' for t in str(interior).split('/')])
    return tags

# タグリスト作成
tag_lists = [create_reform_tags(w, i) for w, i in zip(df['reform_wet_area'], df['reform_interior'])]

# Multi-hot Encoding
mlb = MultiLabelBinarizer()
multi_hot = mlb.fit_transform(tag_lists)  # (n_samples, 14)

# SVDで次元圧縮
svd = TruncatedSVD(n_components=7, random_state=42)
reform_svd = svd.fit_transform(csr_matrix(multi_hot))  # (n_samples, 7)
```

### 経過年数計算

```python
# building_create_dateから年を抽出 (YYYY-MM-DD形式)
reference_year = pl.col('building_create_date').str.slice(0, 4).cast(pl.Int32)

# reform_wet_area_dateから年を抽出 (YYYYMM.0形式)
reform_year = (pl.col('reform_wet_area_date') / 100).floor().cast(pl.Int32)

# 経過年数
years_since_wet_reform = reference_year - reform_year
```

### 欠損値の扱い

| 特徴量タイプ | 欠損時の処理 |
|--------------|--------------|
| SVD特徴量 | 空リスト → 全0ベクトル → SVD後も0付近 |
| 経過年数 | null のまま（LightGBMが自動処理） |
| 郵便番号TE | グローバル平均で補完 |

## ファイル構成

```
exp008_reform_features/
├── README.md              # この仕様書
├── code/
│   ├── preprocessing.py   # 前処理コード
│   └── train.py          # 学習スクリプト
├── configs/
│   └── params.yaml       # パラメータ設定
├── outputs/              # 出力ファイル
└── notebooks/            # 分析用ノートブック
```

## 期待される効果

### リフォーム特徴量
1. **リフォーム内容**: 水回り・内装のリフォーム種別が家賃に影響
2. **リフォーム時期**: 最近リフォームした物件ほど高家賃

### 郵便番号TE
- 郵便番号上3桁で地域の平均家賃を直接表現
- geo_pca と相補的に働くことを期待

### 基準年修正
- データ作成日時を基準にすることで、時系列的な整合性が向上

## 注意事項

- SVDのfit/transformはtrain/testで分離（trainでfit、testはtransformのみ）
- ターゲットエンコーディングはOOF方式でリーク防止
- 欠損値は特徴量タイプに応じて適切に処理

## 実験履歴

| 実験 | CV MAPE | 特徴量数 | 備考 |
|------|---------|----------|------|
| exp005 | 14.74% | 88 | ベースライン |
| exp006 | 14.19% | 119 | +路線・駅・アクセス時間・geo_pca |
| exp007 | 13.62% | 164 | +タグSVD |
| **exp008** | **TBD** | 174 | +リフォームSVD・経過年数・郵便番号TE・基準年修正 |

## 結果サマリー

（実験後に記入）
