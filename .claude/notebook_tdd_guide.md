# Jupyter Notebook × TDD ガイド

> **このファイルは汎用ガイド**: データ分析プロジェクト（Kaggle、SIGNATE、研究等）で再利用可能

---

## 🎯 Notebookの位置づけ

**基本原則**: プロジェクトの主体は`.py`ファイル、Notebookは補助的ツール

### ✅ Notebookを使うべき場面（限定的）

1. **EDA（探索的データ分析）**
   - データの可視化
   - 分布確認
   - 相関分析
   - 外れ値検出

2. **プロトタイプ作成**
   - アイデアの試作
   - 仮説検証
   - クイックな動作確認

3. **結果の可視化・レポート**
   - モデル評価結果のグラフ化
   - 実験結果の記録
   - ドキュメント作成

### ❌ Notebookを使うべきでない場面

1. **本実装**
   - データパイプライン
   - 特徴量生成
   - モデル定義
   - 学習ループ

2. **テストが必要なコード**
   - エッジケースの処理
   - エラーハンドリング
   - 複雑なロジック

---

## 🚨 Notebook実装時の問題

### 問題1: エラー検知が困難

```python
# Notebookのセル実行
def calculate_distance(lat1, lon1, lat2, lon2):
    # 実装...
    return distance

# 見た目は完成しているが...
# → 実際に実行するとエラーが出る
# → Claude Codeは実行確認できないため気づかない
```

### 問題2: テストが書けない

```python
# Notebookではpytestが実行できない
# → 品質担保が困難
# → リファクタリングが怖い
```

### 問題3: 再利用性が低い

```python
# Notebookの関数を別のNotebookで使いたい
# → コピペしかない
# → 修正が発生すると全て修正が必要
```

---

## ✅ 解決策: .py → Notebook 移植パターン

### 基本フロー

```
Step 1: .pyファイルでTDD（品質担保）
   ↓
Step 2: pytest実行（エラー検知）
   ↓
Step 3: Notebookに移植（import使用）
   ↓
Step 4: 実データで検証・可視化
```

---

## 📝 具体的な実装例

### Step 1: 仕様作成（specs/）

```markdown
# specs/features.md

## Haversine距離計算

### 要件
- 目的: 2地点間の距離を計算
- 入力: lat1, lon1, lat2, lon2 (float)
- 出力: 距離 (km, float)

### 仕様
- 計算方法: Haversine公式
- 地球半径: 6371 km
- 欠損値: 入力がNaNの場合はNaNを返す

### テストケース
1. 同一地点: distance = 0.0
2. 東京-新宿: 6 < distance < 8
3. 負の座標: 正常動作（南半球・西半球対応）
```

### Step 2: テスト作成（tests/）

```python
# tests/test_features/test_location.py
import pytest
import numpy as np
from src.features.location import haversine_distance

def test_haversine_same_location():
    """同一地点の距離は0"""
    distance = haversine_distance(35.681236, 139.767125, 35.681236, 139.767125)
    assert distance == 0.0

def test_haversine_tokyo_shinjuku():
    """東京駅-新宿駅の距離（約7km）"""
    distance = haversine_distance(35.681236, 139.767125, 35.689592, 139.700464)
    assert 6 < distance < 8

def test_haversine_with_nan():
    """NaN入力はNaNを返す"""
    distance = haversine_distance(np.nan, 139.767125, 35.689592, 139.700464)
    assert np.isnan(distance)

def test_haversine_southern_hemisphere():
    """南半球でも動作"""
    # Sydney - Melbourne
    distance = haversine_distance(-33.8688, 151.2093, -37.8136, 144.9631)
    assert 700 < distance < 750
```

**実行**: `pytest tests/test_features/test_location.py -v`

### Step 3: 実装（src/）

```python
# src/features/location.py
import numpy as np

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    2点間のHaversine距離（km）を計算

    Parameters
    ----------
    lat1, lon1 : float
        地点1の緯度・経度
    lat2, lon2 : float
        地点2の緯度・経度

    Returns
    -------
    float
        距離（km）。入力にNaNが含まれる場合はNaNを返す

    Examples
    --------
    >>> haversine_distance(35.681236, 139.767125, 35.689592, 139.700464)
    7.123456789
    """
    # NaNチェック
    if any(np.isnan([lat1, lon1, lat2, lon2])):
        return np.nan

    R = 6371  # 地球の半径 (km)

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)

    a = (np.sin(delta_lat/2)**2 +
         np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c
```

**検証**: `pytest tests/test_features/test_location.py -v`

```
test_location.py::test_haversine_same_location PASSED
test_location.py::test_haversine_tokyo_shinjuku PASSED
test_location.py::test_haversine_with_nan PASSED
test_location.py::test_haversine_southern_hemisphere PASSED

✅ 4 passed in 0.12s
```

### Step 4: Notebookに移植

```python
# notebooks/eda/location_features.ipynb

# セル1: セットアップ
import sys
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt

# プロジェクトルートをパスに追加
project_root = Path.cwd().parent.parent
sys.path.append(str(project_root))

# テスト済みの関数をimport
from src.features.location import haversine_distance

print("✅ セットアップ完了")

# セル2: 実データで特徴量生成
train = pl.read_csv(project_root / "data" / "raw" / "train.csv")

# 東京駅までの距離を計算
tokyo_station = (35.681236, 139.767125)
train = train.with_columns(
    pl.struct(['lat', 'lon'])
    .map_elements(
        lambda row: haversine_distance(
            row['lat'], row['lon'],
            tokyo_station[0], tokyo_station[1]
        )
    )
    .alias('distance_to_tokyo')
)

print(f"✅ 特徴量生成完了: {len(train)} 件")
print(train.select(['lat', 'lon', 'distance_to_tokyo']).head())

# セル3: 可視化（Notebookならでは）
plt.figure(figsize=(10, 6))
plt.scatter(
    train['distance_to_tokyo'],
    train['money_room'],
    alpha=0.5,
    s=10
)
plt.xlabel('東京駅までの距離 (km)')
plt.ylabel('物件価格')
plt.title('距離と価格の関係')
plt.grid(True, alpha=0.3)
plt.show()

# 統計
correlation = train.select([
    pl.corr('distance_to_tokyo', 'money_room').alias('correlation')
])
print(f"\n相関係数: {correlation['correlation'][0]:.3f}")
```

**ポイント**:
- ✅ `import`で既にテスト済みの関数を使用
- ✅ Notebookはデータ確認・可視化のみ
- ✅ エラーが出ない（.pyで既に品質担保済み）

---

## 🔄 2つのワークフロー

### パターンA: .py先行型（推奨）

**用途**: 本実装（特徴量生成、前処理、モデリング）

```
仕様作成 → テスト作成 → .py実装 → pytest → Notebook移植
```

**メリット**:
- 品質担保
- エラー防止
- 再利用性

### パターンB: Notebook先行型（限定的）

**用途**: EDA初期探索のみ

```
Notebookでプロトタイプ → 有用な関数を発見 → .pyに移動 → TDD
```

**注意**:
- プロトタイプが完成したら必ず.pyに移動
- Notebookに残すのは可視化コードのみ

---

## 📋 チェックリスト

### ✅ .pyファイル実装時

- [ ] 仕様書を作成したか？
- [ ] テストケースを網羅したか？（正常系・異常系・境界値）
- [ ] pytestが全てパスしたか？
- [ ] docstringを書いたか？
- [ ] 型ヒントを付けたか？

### ✅ Notebook作成時

- [ ] .pyから`import`しているか？（直接実装していないか？）
- [ ] パス表示は相対パスか？（個人情報保護）
- [ ] 全セルがエラーなく実行できるか？
- [ ] 日本語フォント設定は適切か？

---

## 🎯 再利用性のポイント

### このガイドが使えるプロジェクト

- ✅ Kaggleコンペ
- ✅ SIGNATEコンペ
- ✅ 研究プロジェクト（機械学習）
- ✅ データ分析業務

### プロジェクト固有で調整が必要な部分

1. **ディレクトリ構造**
   ```
   # プロジェクトごとに構造が異なる
   src/ features/  # または lib/, modules/, etc.
   tests/          # または test/, __tests__/, etc.
   notebooks/      # または analysis/, exploration/, etc.
   ```

2. **import文のパス**
   ```python
   # プロジェクト構造に応じて調整
   from src.features.location import haversine_distance
   # または
   from lib.utils.geo import haversine_distance
   ```

3. **データパス**
   ```python
   # プロジェクトごとに異なる
   data_dir = project_root / "data" / "raw"
   # または
   data_dir = project_root / "datasets" / "input"
   ```

### 汎用的に使える部分

- ✅ 開発フロー（仕様→テスト→実装→Notebook）
- ✅ TDDサイクル（Red→Green→Refactor）
- ✅ Notebookのエラー対策（.pyから import）
- ✅ 品質担保の考え方

---

## 📚 関連ドキュメント

- `general_rules.md`: 汎用的な開発ルール（SDD+TDD）
- `project_rules.md`: プロジェクト固有のルール
- `eda_guide.md`: EDA実践ガイド

---

**最終更新**: 2025-11-23
