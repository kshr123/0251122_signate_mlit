# プロジェクト固有ルール

> データ分析コンペ固有のルール

---

## 📁 ディレクトリ構造

```
├── 01_specs/           # 仕様書（SDD）
├── 02_docs/            # ドキュメント
├── 03_configs/         # 設定ファイル（YAML）
├── 04_src/             # ソースコード（TDD対象）
├── 05_notebooks/       # Jupyter Notebook（探索用）
├── 06_experiments/     # 実験管理（MLflow）
├── 07_tests/           # テストコード
├── 08_scripts/         # 実行スクリプト
├── 09_submissions/     # 提出ファイル
└── data/               # データセット（.gitignore）
```

---

## 🎯 開発方針

| 方針 | 内容 |
|------|------|
| SDD | 仕様を明確にしてから実装 |
| TDD | Red → Green → Refactor |
| Polars | pandasではなくPolarsを使用 |

---

## 📓 Notebook vs 04_src

| 場所 | 用途 | テスト |
|------|------|--------|
| 05_notebooks/ | 探索・プロトタイプ・可視化 | 不要 |
| 04_src/ | 本実装・再利用コード | 必須 |

**フロー**: Notebookでプロトタイプ → 仕様策定 → テスト作成 → 04_srcで実装

---

## ⚠️ 必須ルール

### 1. Polars使用
```python
# ❌ pandas
import pandas as pd

# ✅ polars
import polars as pl
df = df.with_columns((pl.col("a") + pl.col("b")).alias("c"))
```

### 2. 設定ファイル活用
```python
# ❌ ハードコーディング
RANDOM_SEED = 42
DATA_DIR = "data/processed"

# ✅ 設定ファイルから読み込み
config = load_config("data")
RANDOM_SEED = config["random_seed"]
```

### 3. Transformerパターン（fit/transform分離）
```python
# ✅ 正しい呼び出し
train_result = transformer.fit_transform(train_df)  # trainはfit_transform
test_result = transformer.transform(test_df)        # testはtransformのみ

# ❌ データリーク
all_result = transformer.fit_transform(concat([train_df, test_df]))
```

### 4. パス表記
```python
# ❌ 絶対パス
print("/Users/kotaro/Desktop/project/data")

# ✅ 相対パス
print("data/raw")
```

---

## 🧪 実験管理（06_experiments/）

```bash
# MLflow UI起動
mlflow ui --backend-store-uri file:./06_experiments
# http://localhost:5000
```

**.gitignore**:
```
mlruns/
mlflow.db
models/
```

---

## 🔄 開発フロー

```
1. Notebook探索 → 2. 仕様策定 → 3. テスト作成(Red) → 4. 実装(Green) → 5. Refactor
```

---

**最終更新**: 2025-11-27
