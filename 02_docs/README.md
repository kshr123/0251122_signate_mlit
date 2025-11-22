# ドキュメント一覧

このディレクトリには、コンペのデータ定義とドキュメントが含まれています。

## 📚 ドキュメント構成

### 1. データ定義書

**メインドキュメント**:
- [data_definition.md](./data_definition.md) - 全特徴量の完全なリファレンス

**カテゴリ別詳細**:

| カテゴリ | ドキュメント | 件数 | 概要 |
|---------|-------------|------|------|
| 目的変数 | [target_目的変数.md](./features/target_目的変数.md) | 1件 | 売買価格 |
| 時系列情報 | [time_時系列情報.md](./features/time_時系列情報.md) | 1件 | 対象年月 |
| 金額関連 | [price_金額関連.md](./features/price_金額関連.md) | 13件 | 価格、共益費、管理費、利回り |
| 建物情報 | [building_建物情報.md](./features/building_建物情報.md) | 16件 | 建物種別、構造、面積、築年月 |
| 部屋情報 | [room_部屋情報.md](./features/room_部屋情報.md) | 8件 | 間取り、階数、バルコニー |
| 土地情報 | [land_土地情報.md](./features/land_土地情報.md) | 15件 | 敷地面積、用途地域、建ぺい率 |
| リフォーム | [reform_リフォーム情報.md](./features/reform_リフォーム情報.md) | 17件 | 外装、内装、水回りリフォーム |
| 駐車場 | [parking_駐車場情報.md](./features/parking_駐車場情報.md) | 7件 | 駐車場料金、空き台数 |
| 位置情報 | [location_位置情報.md](./features/location_位置情報.md) | 2件 | 緯度、経度 |
| 交通アクセス | [access_交通アクセス.md](./features/access_交通アクセス.md) | 12件 | 路線、駅、徒歩距離、バス |
| 周辺施設 | [facility_周辺施設.md](./features/facility_周辺施設.md) | 13件 | 学校、病院、商業施設 |

### 2. 補足情報

- **タグマスタ情報**: 257種類のタグ（設備、性能、条件など）
- **設備情報**: 301種類の設備タグ
- **エリア情報**: 1,953エリア（都道府県・市区町村）

詳細は [data_definition.md](./data_definition.md) を参照してください。

## 🎯 クイックリファレンス

### 主要な特徴量

```python
# 目的変数
target = 'money_room'  # 売買価格

# 時系列
time_col = 'target_ym'  # 対象年月 (yyyymm)

# 建物基本情報
building_cols = [
    'building_type',      # 建物種別 (1:マンション, 3:アパート)
    'building_structure', # 建物構造 (1:木造, 3:鉄骨, 4:RC, 5:SRC)
    'year_built',         # 築年月 (yyyymm)
    'building_area',      # 建築面積 (㎡)
]

# 部屋情報
room_cols = [
    'room_floor',         # 所在階数
    'room_count',         # 間取部屋数
    'house_area',         # 専有面積 (㎡)
]

# 位置情報
location_cols = ['lat', 'lon']  # 緯度、経度 (世界測地系)

# 交通アクセス
access_cols = [
    'rosen_name1', 'eki_name1',  # 路線名1、駅名1
    'walk_distance1',             # 徒歩距離1 (m)
]
```

### データ型の注意点

| 形式 | 例 | 説明 |
|------|-----|------|
| yyyymm | 202310 | 年月（2023年10月） |
| スラッシュ区切り | "210101/210201" | 複数タグ（公営水道/都市ガス） |
| カテゴリコード | 1, 2, 3... | 数値だが実際はカテゴリ |
| 欠損値 | NaN, その他 | 補足欄に"その他: 欠損"の記載あり |

## 📖 使い方

### 特徴量を探す

1. **カテゴリがわかっている場合**
   - `docs/features/` 配下のカテゴリ別ドキュメントを参照

2. **全体を俯瞰したい場合**
   - [data_definition.md](./data_definition.md) の特徴量サマリーを参照

3. **特定の単語で検索したい場合**
   ```bash
   # 例: "駐車場" を含む特徴量を検索
   grep -r "駐車場" docs/
   ```

### タグ・設備情報を確認

- タグマスタ: [data_definition.md](./data_definition.md) の「タグマスタ情報」セクション
- 設備情報: [data_definition.md](./data_definition.md) の「設備情報」セクション

タグは6桁のコードで、以下のように分類されます：
- `11xxxx`: 入居条件
- `21xxxx`: インフラ・電力
- `22xxxx`: 水回り設備
- `31xxxx`: セキュリティ
- など

## 🔄 更新履歴

- **2025-11-22**: 初版作成
  - データ定義書のMarkdown化
  - カテゴリ別ドキュメント作成
  - タグ・設備情報の整理

---

**関連ドキュメント**:
- [プロジェクトルール](../.claude/project_rules.md)
- [進捗記録](../progress/progress_log.md)
