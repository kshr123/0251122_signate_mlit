# exp006_route_features

## 概要

exp005をベースに、交通アクセス特徴量と位置情報の次元圧縮を追加する実験。

- **ベース**: exp005_log_transform (CV MAPE: 14.74%)
- **目標**: CV MAPE < 14.0%（0.7pt以上の改善）

## 変更点

### 1. 交通アクセス特徴量の追加

| 特徴量 | 処理方法 | 説明 |
|--------|----------|------|
| rosen_name1 | Target Encoding | 路線名1のTE（508種類） |
| rosen_name1/2 + eki_name1/2 + bus_stop1/2 | TF-IDF | 6カラムを1テキストに結合してTF-IDF |
| rosen_name1, rosen_name2 | Label Encoding | 路線名のLE |
| eki_name1, eki_name2 | Label Encoding | 駅名のLE（5191種類） |
| rosen_name1, rosen_name2 | Count Encoding | 路線名のCE |
| eki_name1, eki_name2 | Count Encoding | 駅名のCE |
| walk_time1, walk_time2 | 数値変換 | walk_distance / 80（分） |
| total_access_time1, total_access_time2 | 数値計算 | walk_time + bus_time |

### 2. 位置情報の次元圧縮

| 特徴量 | 処理方法 | 説明 |
|--------|----------|------|
| geo_pca_0, geo_pca_1 | PCA (n=2) | lon, lat, nl, el の4次元を2次元に圧縮 |

### 3. 元データのカラム情報

| カラム | ユニーク数 | 欠損率 | 備考 |
|--------|-----------|--------|------|
| rosen_name1 | 508 | 2.1% | 路線名1 |
| eki_name1 | 5,191 | 2.3% | 駅名1 |
| bus_stop1 | - | - | バス停名1 |
| bus_time1 | 77 | 86.8% | バス時間1（分） |
| walk_distance1 | 2,026 | 2.1% | 徒歩距離1（m）、平均1017m |
| rosen_name2 | 493 | 41.7% | 路線名2（2路線目） |
| eki_name2 | 4,456 | 41.8% | 駅名2 |
| bus_stop2 | - | 92.0% | バス停名2 |
| bus_time2 | 81 | 92.0% | バス時間2（分） |
| walk_distance2 | 1,535 | 41.7% | 徒歩距離2（m） |

**注**: 2路線目（rosen_name2等）の欠損41.7%はLightGBMがnullとして自然に学習するため、存在フラグは不要。

## 特徴量数

- exp005: 88個
- 追加特徴量:
  - rosen_name1_te: 1個
  - TF-IDF: N個（max_features設定による、初期値20）
  - LE: 4個（rosen_name1/2, eki_name1/2）
  - CE: 4個（rosen_name1/2, eki_name1/2）
  - walk_time: 2個
  - total_access_time: 2個
  - geo_pca: 2個
- **合計**: 88 + 1 + 20 + 4 + 4 + 2 + 2 + 2 = 123個（TF-IDF 20次元の場合）

## 実装詳細

### TF-IDF処理

```python
# 6カラムを結合してテキスト化
text = f"{rosen_name1} {rosen_name2} {eki_name1} {eki_name2} {bus_stop1} {bus_stop2}"
# nullは空文字として扱う

# TfidfVectorizer設定
vectorizer = TfidfVectorizer(
    max_features=20,  # 次元数制限
    token_pattern=r'(?u)\b\w+\b',  # 日本語対応
)
```

### PCA処理

```python
# 4つの緯度経度カラムを標準化してPCA
scaler = StandardScaler()
pca = PCA(n_components=2)

geo_cols = ['lon', 'lat', 'nl', 'el']
geo_scaled = scaler.fit_transform(df[geo_cols])
geo_pca = pca.fit_transform(geo_scaled)
```

## ファイル構成

```
exp006_route_features/
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

1. **路線名TE**: 路線による家賃水準の違いを捉える（山手線沿線は高い等）
2. **TF-IDF**: 複数路線・駅の組み合わせパターンを捉える
3. **LE/CE**: 路線・駅の出現頻度情報
4. **アクセス時間**: 駅からの時間は家賃に直接影響
5. **geo_pca**: 緯度経度の冗長性を削減しつつ位置情報を保持

## 注意事項

- TF-IDFとPCAはtrain/testでfitをtrainのみで行い、testはtransformのみ
- Target Encodingは既存のCV-aware実装を使用（リーク防止）
