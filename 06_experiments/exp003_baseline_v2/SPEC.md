# exp003_baseline_v2 仕様書

## 概要
特徴量を厳選し、エンコーディング手法を適用した新ベースラインモデル。

## 目的
- 重要な特徴量に絞ってモデルを構築
- ターゲットエンコーディング、ラベル/カウントエンコーディングの効果検証

---

## 特徴量設計

### 入力カラム（元データから使用: 68個）

```python
BASE_COLUMNS = [
    # 物件基本情報
    'building_status', 'building_type', 'unit_count', 'lon', 'lat',
    'building_structure', 'total_floor_area', 'floor_count',
    'basement_floor_count', 'year_built', 'building_land_area', 'land_area_all',

    # 土地情報
    'building_land_chimoku', 'land_youto', 'land_toshi', 'land_chisei',
    'land_kenpei', 'land_youseki', 'land_road_cond',

    # 管理情報
    'management_form', 'management_association_flg',

    # 部屋情報
    'room_floor', 'balcony_area', 'dwelling_unit_window_angle',
    'room_count', 'unit_area', 'floor_plan_code',

    # 物件詳細
    'bukken_type', 'flg_investment', 'empty_number',
    'post1', 'post2', 'addr1_1', 'addr1_2',

    # 位置情報
    'nl', 'el', 'snapshot_land_area', 'snapshot_land_shidou',

    # 物件属性
    'house_area', 'flg_new', 'house_kanrinin', 'room_kaisuu',
    'snapshot_window_angle', 'madori_number_all', 'madori_kind_all',

    # 費用情報
    'money_kyoueki', 'money_shuuzen', 'money_shuuzenkikin',
    'money_sonota1', 'money_sonota2', 'money_sonota3',  # → 合計に集約

    # 駐車場
    'parking_money', 'parking_kubun', 'parking_keiyaku',

    # 物件状態
    'genkyo_code', 'usable_status',

    # 周辺施設距離
    'convenience_distance', 'super_distance', 'hospital_distance',
    'park_distance', 'drugstore_distance', 'bank_distance',
    'shopping_street_distance', 'est_other_distance',
]
```

### 除外カラム（2個）
| カラム | 理由 |
|--------|------|
| building_area | 欠損率 98.07% |
| money_hoshou_company | 欠損率 100.00% |

---

## 特徴量変換

### 1. year_built 変換
- 入力: YYYYMM形式（例: 199211）
- 出力: YYYY形式（例: 1992）
- 処理: `year_built // 100`
- カラム名: そのまま `year_built`

### 2. money_sonota 集約
- 入力: money_sonota1, money_sonota2, money_sonota3
- 出力: `money_sonota_sum`（3カラムの合計、NULLは0扱い）
- 元カラム: 削除

### 3. ターゲットエンコーディング（6個）
高カーディナリティのカテゴリカル変数に適用。

| カラム | 意味 | 出力カラム名 |
|--------|------|-------------|
| addr1_1 | 都道府県コード | addr1_1_te |
| addr1_2 | 市区町村コード | addr1_2_te |
| building_land_chimoku | 地目 | building_land_chimoku_te |
| bukken_type | 物件タイプ | bukken_type_te |
| land_youto | 用途地域 | land_youto_te |
| land_toshi | 都市計画 | land_toshi_te |

**実装方針:**
- CVリーク防止のため、fold内でfit/transformを行う
- smoothing（平滑化）を適用
- 元カラムは削除

### 4. ラベルエンコーディング + カウントエンコーディング（10個）
低カーディナリティのカテゴリカル変数に両方を適用。

| カラム | 出力（ラベル） | 出力（カウント） |
|--------|---------------|-----------------|
| building_status | building_status_label | building_status_count |
| building_type | building_type_label | building_type_count |
| building_structure | building_structure_label | building_structure_count |
| land_chisei | land_chisei_label | land_chisei_count |
| management_form | management_form_label | management_form_count |
| flg_investment | flg_investment_label | flg_investment_count |
| flg_new | flg_new_label | flg_new_count |
| genkyo_code | genkyo_code_label | genkyo_code_count |
| usable_status | usable_status_label | usable_status_count |
| parking_kubun | parking_kubun_label | parking_kubun_count |

**実装方針:**
- ラベルエンコーディング: trainでfit、testにtransform
- カウントエンコーディング: trainの出現回数で置換
- 元カラムは削除

---

## 最終特徴量一覧

### 数値特徴量（そのまま使用）
```
unit_count, lon, lat, total_floor_area, floor_count, basement_floor_count,
year_built (変換後), building_land_area, land_area_all,
land_kenpei, land_youseki, land_road_cond,
management_association_flg, room_floor, balcony_area, dwelling_unit_window_angle,
room_count, unit_area, floor_plan_code, empty_number,
post1, post2, nl, el, snapshot_land_area, snapshot_land_shidou,
house_area, house_kanrinin, room_kaisuu, snapshot_window_angle,
madori_number_all, madori_kind_all,
money_kyoueki, money_shuuzen, money_shuuzenkikin, money_sonota_sum (集約),
parking_money, parking_keiyaku,
convenience_distance, super_distance, hospital_distance,
park_distance, drugstore_distance, bank_distance,
shopping_street_distance, est_other_distance
```
→ 約46個

### ターゲットエンコーディング
```
addr1_1_te, addr1_2_te, building_land_chimoku_te,
bukken_type_te, land_youto_te, land_toshi_te
```
→ 6個

### ラベル + カウントエンコーディング
```
building_status_label, building_status_count,
building_type_label, building_type_count,
building_structure_label, building_structure_count,
land_chisei_label, land_chisei_count,
management_form_label, management_form_count,
flg_investment_label, flg_investment_count,
flg_new_label, flg_new_count,
genkyo_code_label, genkyo_code_count,
usable_status_label, usable_status_count,
parking_kubun_label, parking_kubun_count
```
→ 20個

### 合計
約 **72個** の特徴量

---

## モデル設定

- モデル: LightGBM
- CV: 3-Fold
- 評価指標: MAPE
- その他パラメータ: exp002と同様

---

## ファイル構成

```
exp003_baseline_v2/
├── SPEC.md              # 本仕様書
├── code/
│   ├── preprocessing.py # 前処理・特徴量生成
│   └── train.py         # 学習スクリプト
├── configs/
│   └── feature_config.yaml
├── outputs/
└── notebooks/
```

---

## 期待される改善

- ターゲットエンコーディングによる地域性の反映
- ラベル+カウントの組み合わせによるカテゴリ情報の活用
- 不要カラム除去によるノイズ低減
