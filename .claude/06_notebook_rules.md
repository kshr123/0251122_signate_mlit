# Notebook ルール

> Notebookは**エラーなく実行され、出力付きでcommit**する

---

## 📁 配置場所

| 用途 | 配置場所 | 例 |
|------|----------|-----|
| 汎用EDA | `05_notebooks/01_eda/` | 初期EDA、相関分析 |
| 汎用特徴量検証 | `05_notebooks/02_feature/` | 特徴量の効果検証 |
| 汎用モデリング | `05_notebooks/03_modeling/` | モデル比較 |
| 汎用評価 | `05_notebooks/04_evaluation/` | 評価手法の検証 |
| **実験固有分析** | `06_experiments/expXXX/notebooks/` | 実験結果の分析 |

**原則**: 実験に紐づく分析は `06_experiments/expXXX/notebooks/` に配置

---

## 🚨 必須ルール

### 1. エラーなし・出力付きでcommit
```bash
# commit前に必ず実行
Kernel > Restart & Run All
# 全セルがエラーなく完了 → git add → git commit
```

### 2. 図の日本語表示
```python
# セットアップセルに必ず含める
import japanize_matplotlib
```

### 3. PNG保存不要
- **図はNotebook内で表示すれば十分**（別途pngファイル保存は不要）
- `plt.show()` で表示し、出力付きでcommitすれば記録される

### 4. パス設定（標準パターン）

**05_notebooks/01_eda/ から（3階層上）**:
```python
import sys
from pathlib import Path
project_root = Path().resolve().parents[2]  # 05_notebooks/01_eda → project_root
sys.path.insert(0, str(project_root / "04_src"))
import os
os.chdir(project_root)
```

**06_experiments/expXXX/notebooks/ から（3階層上）**:
```python
import sys
from pathlib import Path
project_root = Path().resolve().parents[2]  # expXXX/notebooks → project_root
sys.path.insert(0, str(project_root / "04_src"))
import os
os.chdir(project_root)
```

### 5. 絶対パスの非表示（個人情報保護）
- **絶対パスは出力に表示しない**（ユーザー名等の個人情報を含むため）
- パスを表示する場合は相対パスを使用

```python
# ❌ NG: 絶対パスを表示
print(f"Loading: {file_path}")  # /Users/kotaro/... が表示される

# ✅ OK: 相対パスに変換して表示
print(f"Loading: {file_path.relative_to(project_root)}")

# ✅ OK: ファイル名のみ表示
print(f"Loading: {file_path.name}")
```

### 6. polars優先
- **DataFrameはpolarsを優先使用**（pandasより高速）
- pandasが必要な場合のみ `.to_pandas()` で変換

```python
# ✅ 推奨: polars
import polars as pl
df = pl.read_csv("data.csv")

# pandas必要時のみ変換
df_pd = df.to_pandas()
```

---

## 🎯 Notebookの使い分け

### ✅ 使うべき場面
- EDA（可視化・分布確認・相関分析）
- 実験結果のレポート
- プロトタイプ検証

### ❌ 使うべきでない場面
- 本実装（特徴量生成、モデル定義、学習ループ）→ `.py`で実装

### 基本フロー
```
.pyでTDD実装 → pytest通過 → Notebookでimportして可視化
```

---

## 📋 チェックリスト

- [ ] `Restart & Run All` でエラーなし
- [ ] 図に日本語が正しく表示されている
- [ ] 出力付き状態でcommit
- [ ] `.py`からimportしている（Notebook内で本実装しない）
- [ ] 絶対パスが出力に含まれていない
- [ ] DataFrameはpolarsを使用（pandas必要時のみ変換）

---

**最終更新**: 2025-11-28

---

**関連ルール**: [03_experiment_management_rules.md](./03_experiment_management_rules.md)
