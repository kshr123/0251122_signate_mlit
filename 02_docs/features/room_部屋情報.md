# 部屋情報

**カテゴリID**: `room`  
**特徴量数**: 8件

---

## 特徴量一覧

| No. | 特徴量名 | 意味 | データ型 | 補足 |
|-----|----------|------|---------|------|
| 19 | `floor_count` | 建物階数(地上) | numeric | - |
| 49 | `room_floor` | 所在階数 | unknown | - |
| 50 | `balcony_area` | バルコニー面積 | numeric | - |
| 52 | `room_count` | 間取部屋数 | numeric | - |
| 54 | `floor_plan_code` | 間取り種類コード | category | 部屋数+間取種類 (Sは丸める) 間取り種類 10:R 20:K,SK 30:DK,SDK 40:LK,SLK 50:LDK,SLDK |
| 106 | `room_kaisuu` | 部屋階数 | unknown | 部屋の所在階数 (マイナスの場合は地下) |
| 108 | `madori_number_all` | 間取部屋数(代表) | unknown | 部屋の数 |
| 109 | `madori_kind_all` | 間取部屋種類(代表) | unknown | 10:R 20:K 25:SK 30:DK 35:SDK 40:LK 45:SLK 50:LDK 55:SLDK その他: 欠損 |

---

## 詳細説明

### `floor_count`

**意味**: 建物階数(地上)

### `room_floor`

**意味**: 所在階数

### `balcony_area`

**意味**: バルコニー面積

### `room_count`

**意味**: 間取部屋数

### `floor_plan_code`

**意味**: 間取り種類コード

**補足**:

- 部屋数+間取種類 (Sは丸める) 間取り種類 10:R 20:K,SK 30:DK,SDK 40:LK,SLK 50:LDK,SLDK

### `room_kaisuu`

**意味**: 部屋階数

**補足**:

- 部屋の所在階数 (マイナスの場合は地下)

### `madori_number_all`

**意味**: 間取部屋数(代表)

**補足**:

- 部屋の数

### `madori_kind_all`

**意味**: 間取部屋種類(代表)

**補足**:

- 10:R 20:K 25:SK 30:DK 35:SDK 40:LK 45:SLK 50:LDK 55:SLDK その他: 欠損
