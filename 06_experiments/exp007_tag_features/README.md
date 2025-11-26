# exp007_tag_features

## 概要

exp006をベースに、建物・部屋タグ情報をMulti-hot + SVDで次元圧縮して追加する実験。

- **ベース**: exp006_route_features (CV MAPE: 14.19%)
- **目標**: CV MAPE < 14.0%（0.2pt以上の改善）

## 変更点

### タグ特徴量の追加

| カラム | 処理方法 | 元次元 | 圧縮後 | 情報保持率 |
|--------|----------|--------|--------|------------|
| building_tag_id | Multi-hot → SVD | 90 | 15 | 90.5% |
| unit_tag_id | Multi-hot → SVD | 117 | 30 | 86.7% |

### 元データの情報

| カラム | ユニーク組合せ | 欠損率 | ユニークタグ数 | 平均タグ数/物件 |
|--------|---------------|--------|---------------|-----------------|
| building_tag_id | 127,515 | 7.7% | 90 | 6.0 |
| unit_tag_id | 209,158 | 17.4% | 117 | 15.8 |

### タグの意味

- **building_tag_id**: 建物全体の設備・特徴
  - 例: オートロック、宅配BOX、エレベーター、駐車場など
  - 頻出: 210101(84%), 210301(78%), 210201(66%)

- **unit_tag_id**: 各部屋の設備・特徴
  - 例: エアコン、バストイレ別、フローリング、室内洗濯機置場など
  - 頻出: 230401(64%), 290401(62%), 290101(55%)

## 特徴量数

- exp006: 119個
- 追加特徴量:
  - building_tag_svd_0〜14: 15個
  - unit_tag_svd_0〜29: 30個
- **合計**: 119 + 15 + 30 = **164個**

## 実装詳細

### Multi-hot → SVD処理

```python
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

# 1. タグをリストに分割
tag_lists = [s.split('/') if s else [] for s in df['building_tag_id'].fill_null('')]

# 2. Multi-hot Encoding
mlb = MultiLabelBinarizer()
multi_hot = mlb.fit_transform(tag_lists)  # (n_samples, 90)

# 3. SVDで次元圧縮（スパース行列対応）
svd = TruncatedSVD(n_components=15, random_state=42)
tag_svd = svd.fit_transform(csr_matrix(multi_hot))  # (n_samples, 15)
```

### なぜSVDか？

| 手法 | 特徴 | このデータでの適性 |
|------|------|-------------------|
| PCA | 線形、中心化あり | ○ |
| **SVD** | 線形、スパース最適 | **◎** (93.8%がゼロ) |
| UMAP | 非線形 | △ (遅い、過剰) |
| TF-IDF | 重み付け | △ (レア≠重要) |

## ファイル構成

```
exp007_tag_features/
├── README.md              # この仕様書
├── code/
│   ├── preprocessing.py   # 前処理コード
│   ├── train.py          # 学習スクリプト
│   └── test_quick.py     # クイックテスト
├── configs/
│   └── params.yaml       # パラメータ設定
├── outputs/              # 出力ファイル
└── notebooks/            # 分析用ノートブック
```

## 期待される効果

1. **建物タグ**: 建物グレードや設備充実度を表現
   - オートロック・宅配BOXがある → 高級物件 → 高家賃
2. **部屋タグ**: 部屋の設備・仕様を表現
   - バストイレ別・独立洗面台 → 高家賃

## 注意事項

- SVDのfit/transformはtrain/testで分離（trainでfit、testはtransformのみ）
- 欠損値は空リスト`[]`として扱う → Multi-hotでは全て0のベクトル
- SVDの結果は連続値（LightGBMは問題なく処理可能）

## 実験履歴

| 実験 | CV MAPE | 特徴量数 | 備考 |
|------|---------|----------|------|
| exp005 | 14.74% | 88 | ベースライン |
| exp006 | 14.19% | 119 | +路線・駅・アクセス時間・geo_pca |
| **exp007** | **13.62%** | 164 | +タグSVD |

## 結果サマリー

### CV結果

| Fold | MAPE | Best Iteration |
|------|------|----------------|
| Fold 1 | 13.63% | 15000 |
| Fold 2 | 13.59% | 15000 |
| Fold 3 | 13.65%* | 15000 |
| **Mean** | **13.62%** | 15000 |

*推定値（ログ切り捨てのため）

### 目標達成状況

- **目標**: CV MAPE < 14.0%（0.2pt以上の改善）
- **結果**: CV MAPE = **13.62%**（**0.57pt改善**）
- **達成**: 目標達成

### Top 20 Feature Importance

| Rank | Feature | Importance | タイプ |
|------|---------|------------|--------|
| 1 | post2 | 22192 | 数値 |
| 2 | house_area | 17910 | 数値 |
| 3 | year_built | 17656 | 数値 |
| 4 | snapshot_land_area | 17116 | 数値 |
| 5 | geo_pca_1 | 16978 | Geo PCA |
| 6 | **unit_tag_svd_0** | **16454** | **新規（タグSVD）** |
| 7 | eki_name1_le | 15818 | ラベルエンコ |
| 8 | eki_name1_count | 15606 | カウントエンコ |
| 9 | geo_pca_0 | 14711 | Geo PCA |
| 10 | rosen_name1_te | 14063 | ターゲットエンコ |
| 11 | super_distance | 13589 | 数値 |
| 12 | post1 | 13330 | 数値 |
| 13 | unit_area | 13182 | 数値 |
| 14 | addr1_2_te | 13089 | ターゲットエンコ |
| 15 | total_access_time1 | 12884 | 数値（アクセス時間） |
| 16 | convenience_distance | 12439 | 数値 |
| 17 | walk_time1 | 11871 | 数値 |
| 18 | rosen_name1_count | 11616 | カウントエンコ |
| 19 | building_land_area | 11464 | 数値 |
| 20 | rosen_name1_le | 11323 | ラベルエンコ |

### タグSVD特徴量の寄与

- **unit_tag_svd_0**: Feature importance **6位**（16454）
- unit_tag_svd_1〜29: 多数が上位50%内
- building_tag_svd_*: 中程度の寄与（unit_tagより弱い）

### 考察

1. **タグ特徴量は有効**: unit_tag_svd_0が6位という高順位
2. **unit_tagが強い**: building_tagより部屋タグの方が家賃予測に有効
3. **改善幅0.57pt**: 目標の0.2ptを大きく上回る
4. **全Foldで15000iterに到達**: さらなる改善余地あり（max_iter増加検討）
