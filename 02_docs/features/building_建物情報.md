# 建物情報

**カテゴリID**: `building`  
**特徴量数**: 16件

---

## 特徴量一覧

| No. | 特徴量名 | 意味 | データ型 | 補足 |
|-----|----------|------|---------|------|
| 3 | `building_id` | 棟ID | category | AUTO_INCREMENT、UNSIGNED　確認事項参照 |
| 4 | `building_status` | 状態 | category | 1: 棟が存在する、9: 棟が存在しない |
| 5 | `building_create_date` | 作成日時 | datetime | データ作成日時 |
| 6 | `building_modify_date` | 修正日時 | datetime | データ修正日時 |
| 7 | `building_type` | 建物種別 | unknown | 1: マンション, 3: アパート, その他: 欠損 |
| 8 | `building_name` | 建物名 | text | - |
| 9 | `building_name_ruby` | 建物名フリガナ | text | - |
| 16 | `building_structure` | 建物構造 | text | 1:木造 2:ブロック 3:鉄骨造 4:RC 5:SRC 6:PC 7:HPC 9:その他 10:軽量鉄骨 11:ALC 12:鉄筋ブロック 13:CFT(コンクリート充填鋼管) |
| 18 | `building_area` | 建築面積 | numeric | 建築面積 |
| 22 | `building_land_area` | 土地面積 | numeric | - |
| 26 | `building_land_chimoku` | 地目 | unknown | 地目 　1 :宅地 2: 田 3:畑  4:山林 5 : 雑種地 9 : その他 10:原野 11:田･畑 その他: 欠損 |
| 37 | `building_area_kind` | 建物面積計測方式 | numeric | 1:壁芯 2:内法　その他: 欠損 |
| 45 | `building_tag_id` | タグ情報 | category | スラッシュ(/) 区切り形式 タグマスタシート参照 |
| 51 | `dwelling_unit_window_angle` | 主要採光面 | unknown | - |
| 103 | `house_area` | 建物面積/専有面積(代表) | numeric | 単位：平米 |
| 105 | `house_kanrinin` | 管理人 | unknown | 売買：マンションのみ必須 1:常駐 2:日勤 3:巡回 4:無 (5:非常駐 V3互換用) その他: 欠損 |

---

## 詳細説明

### `building_id`

**意味**: 棟ID

**補足**:

- AUTO_INCREMENT、UNSIGNED　確認事項参照

### `building_status`

**意味**: 状態

**補足**:

- 1: 棟が存在する、9: 棟が存在しない

### `building_create_date`

**意味**: 作成日時

**補足**:

- データ作成日時

### `building_modify_date`

**意味**: 修正日時

**補足**:

- データ修正日時

### `building_type`

**意味**: 建物種別

**補足**:

- 1: マンション, 3: アパート, その他: 欠損

### `building_name`

**意味**: 建物名

### `building_name_ruby`

**意味**: 建物名フリガナ

### `building_structure`

**意味**: 建物構造

**補足**:

- 1:木造 2:ブロック 3:鉄骨造 4:RC 5:SRC 6:PC 7:HPC 9:その他 10:軽量鉄骨 11:ALC 12:鉄筋ブロック 13:CFT(コンクリート充填鋼管)

### `building_area`

**意味**: 建築面積

**補足**:

- 建築面積

### `building_land_area`

**意味**: 土地面積

### `building_land_chimoku`

**意味**: 地目

**補足**:

- 地目 　1 :宅地 2: 田 3:畑  4:山林 5 : 雑種地 9 : その他 10:原野 11:田･畑 その他: 欠損

### `building_area_kind`

**意味**: 建物面積計測方式

**補足**:

- 1:壁芯 2:内法　その他: 欠損

### `building_tag_id`

**意味**: タグ情報

**補足**:

- スラッシュ(/) 区切り形式 タグマスタシート参照

### `dwelling_unit_window_angle`

**意味**: 主要採光面

### `house_area`

**意味**: 建物面積/専有面積(代表)

**補足**:

- 単位：平米

### `house_kanrinin`

**意味**: 管理人

**補足**:

- 売買：マンションのみ必須 1:常駐 2:日勤 3:巡回 4:無 (5:非常駐 V3互換用) その他: 欠損
