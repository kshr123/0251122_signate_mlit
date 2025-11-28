# exp010 コードディレクトリ

---

## ファイル構成

```
code/
├── train.py            # エントリーポイント（学習実行、MLflow記録）
├── preprocessing.py    # 前処理（基本処理 + パイプライン呼び出し）
├── pipeline.py         # 特徴量パイプライン（Blockの組み合わせ）
├── constants.py        # パス定義・カラムリスト（ハイパーパラメータは含まない）
└── exp010_features.py  # exp010固有の特徴量（Block、関数）
```

---

## 設定の分離方針

| ファイル | 内容 | 変更頻度 |
|----------|------|----------|
| `../configs/experiment.yaml` | ハイパーパラメータ（学習設定・モデル・特徴量） | 実験毎に調整 |
| `constants.py` | パス定義、カラムリスト | ほぼ固定 |

### experiment.yaml で管理する項目

- 学習設定: `seed`, `n_splits`, `early_stopping_rounds`
- モデルパラメータ: `learning_rate`, `max_depth`, `num_leaves` 等
- 特徴量パラメータ: TF-IDF/PCA/SVDの次元数
- exp010固有閾値: `area_age_category` の閾値

### constants.py で管理する項目

- データパス: `LANDPRICE_BASE_PATH`
- カラムリスト: `NUMERIC_COLUMNS`, `CATEGORICAL_COLUMNS` 等
- 特徴量名リスト: `LP_RATIO_COLUMNS` 等

---

## 依存関係

```
train.py
├── configs/experiment.yaml     # ハイパーパラメータ読み込み
└── preprocessing.py
    ├── constants.py            # パス・カラム定義
    ├── preprocess_base()       # 基本前処理
    └── pipeline.fit_transform()
        ├── 04_src/features/    # 共通Block（TfidfBlock, PCABlock等）
        └── exp010_features.py  # exp010固有Block・関数
```

---

## 各ファイルの役割

| ファイル | 役割 |
|----------|------|
| `train.py` | 学習実行、MLflow記録、サブミッション作成 |
| `preprocessing.py` | 基本前処理（築年数計算等）+ パイプライン呼び出し |
| `pipeline.py` | 特徴量変換の定義（どのBlockをどの順で適用するか） |
| `constants.py` | パス定義・カラムリスト（ハイパーパラメータは含まない） |
| `exp010_features.py` | exp010固有Block・関数 |

---

## 使用する04_src共通Block

| Block | 用途 |
|-------|------|
| `TfidfBlock` | 交通テキストのTF-IDF |
| `PCABlock` | 緯度経度PCA |
| `MultiHotSVDBlock` | タグSVD |
| `MultiColumnMultiHotSVDBlock` | リフォームSVD |
| `TargetEncodingBlock` | ターゲットエンコーディング |
| `CountEncodingBlock` | カウントエンコーディング |
| `LabelEncodingBlock` | ラベルエンコーディング |
| `TopNCategoryLEBlock` | 上位Nカテゴリのラベルエンコーディング |
| `GroupByAggBlock` | カテゴリ別集計（mean, ratio） |
| `MultiKeyTEBlock` | 複合キーOOFターゲットエンコーディング |
| `RenameBlock` | カラム名リネーム |

---

## exp010_features.py 内容

### Blockクラス

| クラス | 継承元 | 用途 | 出力次元 |
|--------|--------|------|----------|
| `PostalCodeTEBlock` | TargetEncodingBlock | 郵便番号TE（階層フォールバック） | 2 |
| `AreaRegionalRatioBlock` | BaseBlock | 面積地域平均比率 | 6 |
| `AreaAgeCategoryTEBlock` | MultiKeyTEBlock | 面積×築年数カテゴリTE | 2 |

### 関数

| 関数 | 用途 |
|------|------|
| `add_lp_area_value` | 土地価値目安（地価×面積） |
| `add_area_age_category` | 面積×築年数カテゴリ付与（閾値はexperiment.yamlで設定可能） |

### 定数

| 定数 | 用途 |
|------|------|
| `CURRENT_USE_TOP_CATEGORIES` | 利用現況の上位カテゴリリスト |
| `ROAD_CATEGORY_COLUMNS` | 道路関連カテゴリカラム |
| `LP_RATIO_COLUMNS` | 地価比率計算対象カラム |
| `DEFAULT_*` | 各種デフォルト値（experiment.yamlで上書き可能） |
