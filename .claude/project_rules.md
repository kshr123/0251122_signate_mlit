# データ分析コンペティション - プロジェクト固有ルール

> このファイルは**データ分析コンペ固有**のルールです。汎用的な開発ルールは [general_rules.md](./general_rules.md) を参照してください。

---

## 🎯 開発方針

このプロジェクトでは以下の3つの柱で開発を進めます：

1. **仕様駆動開発（SDD）**: まず仕様を明確にしてから実装
2. **テスト駆動開発（TDD）**: Red → Green → Refactorのサイクル
3. **モダンツール活用**: Polars、uv、最新のベストプラクティス

---

## 📁 ディレクトリ構造とルール

### ディレクトリの役割（作業フロー順）

```
.
├── .claude/                    # Claude Code設定
│   ├── claude.md              # プロジェクト概要
│   ├── general_rules.md       # 汎用開発ルール
│   ├── project_rules.md       # このファイル
│   └── eda_guide.md          # EDA実践ガイド
│
├── 01_specs/                   # 📋 仕様書（SDD）
│   ├── data.md               # データ仕様
│   ├── eda.md                # EDA方針
│   ├── preprocessing.md      # 前処理仕様
│   └── features.md           # 特徴量仕様
│
├── 02_docs/                    # 📚 ドキュメント
│   ├── data_definition.md    # データ定義書
│   └── table_definition.txt  # テーブル定義
│
├── 03_configs/                 # ⚙️ 設定ファイル
│   ├── base.yaml
│   ├── data.yaml
│   ├── preprocessing.yaml
│   ├── features.yaml
│   ├── model.yaml
│   └── training.yaml
│
├── 04_src/                     # 💻 ソースコード（TDD対象）
│   ├── data/                 # データ処理
│   ├── eda/                  # EDAユーティリティ
│   ├── preprocessing/        # 前処理
│   ├── features/             # 特徴量生成
│   ├── models/               # モデル定義
│   ├── training/             # 学習・CV
│   ├── evaluation/           # 評価
│   └── utils/                # ユーティリティ
│
├── 05_notebooks/               # 🔬 Jupyter Notebook（探索用）
│   ├── 01_eda/               # 探索的データ分析
│   ├── 02_feature/           # 特徴量検証
│   ├── 03_modeling/          # モデリング実験
│   └── 04_evaluation/        # 評価・検証
│
├── 06_experiments/             # 🧪 実験管理
│   ├── mlruns/               # MLflow実験記録
│   ├── mlflow.db             # MLflowメタDB
│   ├── models/               # 学習済みモデル
│   ├── experiment_notes/     # 実験メモ（Git管理）
│   └── configs/              # 実験設定（Git管理）
│
├── 07_tests/                   # ✅ テストコード（TDD）
│   ├── test_data/
│   ├── test_eda/
│   ├── test_preprocessing/
│   ├── test_features/
│   └── test_models/
│
├── 08_scripts/                 # 🚀 実行スクリプト
│   ├── train.py
│   └── predict.py
│
├── 09_submissions/             # 📤 提出ファイル
│
└── data/                       # データセット（.gitignore）
    ├── raw/
    ├── processed/
    └── external/
```

### 重要なルール

#### 1. `05_notebooks/` vs `04_src/` の使い分け

**05_notebooks/（探索・プロトタイプ）**:
- 目的: データ理解、仮説検証、可視化
- 品質: 使い捨てOK、試行錯誤の記録
- テスト: 不要
- **EDA実行時は `.claude/eda_guide.md` を参照**

**04_src/（本実装）**:
- 目的: 再利用可能なコード、本番投入
- 品質: **仕様駆動 + テスト駆動で厳格に開発**
- テスト: **必須**（07_tests/ に対応するテストコード）

**ワークフロー**:
```
1. 05_notebooks/ でプロトタイプ（eda_guide.mdに従う）
   ↓
2. 01_specs/ で仕様を明確化
   ↓
3. 07_tests/ でテスト作成（Red）
   ↓
4. 04_src/ で実装（Green）
   ↓
5. Refactor
```

#### 2. `01_specs/` で仕様を明確化（SDD）

実装前に必ず仕様書を作成：

**例: `01_specs/features.md`**
```markdown
# 特徴量仕様書

## 位置情報特徴量

### 要件
- 目的: 物件の立地を数値化
- 入力: lat (緯度), lon (経度)
- 出力: 主要駅までの距離（km）

### 仕様
- 主要駅リスト: 東京駅、新宿駅、渋谷駅（coords定義済み）
- 計算方法: Haversine距離
- 出力列名: `distance_to_tokyo`, `distance_to_shinjuku`, `distance_to_shibuya`
- データ型: float64
- 欠損値: 座標が欠損している場合はNaNを返す

### テストケース
1. 正常系: 東京駅の座標 → distance_to_tokyo = 0.0
2. 異常系: lat/lonがNaN → 全てNaN
3. 境界値: 緯度/経度の範囲外 → ValueError
```

#### 3. `tests/` でテスト駆動開発（TDD）

**TDDサイクル**:
1. **Red**: テストを書く（失敗する）
2. **Green**: 最小限の実装でテストを通す
3. **Refactor**: コードをきれいにする

**例: `tests/test_features/test_location.py`**
```python
import polars as pl
import pytest
from src.features.location import calculate_station_distance


def test_distance_to_tokyo_from_tokyo():
    """東京駅の座標からの距離は0"""
    df = pl.DataFrame({
        "lat": [35.681236],
        "lon": [139.767125]
    })
    result = calculate_station_distance(df, "tokyo")
    assert result["distance_to_tokyo"][0] == pytest.approx(0.0, abs=0.01)


def test_distance_with_missing_coords():
    """座標欠損時はNaNを返す"""
    df = pl.DataFrame({
        "lat": [None],
        "lon": [None]
    })
    result = calculate_station_distance(df, "tokyo")
    assert result["distance_to_tokyo"][0] is None
```

#### 4. Polarsを使用（高速・省メモリ）

**pandas → polars 移行ガイド**:

```python
# ❌ pandas
import pandas as pd
df = pd.read_csv("data.csv")
df["new_col"] = df["col1"] + df["col2"]

# ✅ polars
import polars as pl
df = pl.read_csv("data.csv")
df = df.with_columns(
    (pl.col("col1") + pl.col("col2")).alias("new_col")
)
```

**理由**:
- 高速（pandasの10-100倍）
- 省メモリ（lazy evaluation）
- 型安全
- 表現力が高い（SQL的な操作）

#### 5. テスト結果の管理（07_tests/）

**このプロジェクトでの実行方法**:
```bash
# 全テスト実行
./08_scripts/run_tests.sh

# 特定モジュールのみ
./08_scripts/run_tests.sh test_data

# カバレッジ測定付き
./08_scripts/run_tests.sh test_data --coverage

# 最新結果の確認
cat 07_tests/test_data/test_results/latest_result.txt
```

**詳細ルール**: テスト結果の管理方法、ディレクトリ構造、Git管理ルールについては、汎用的な開発ルールとして [general_rules.md](./general_rules.md) の「テスト結果の管理」セクションを参照してください。

**プロジェクト固有の情報**:
- テストディレクトリ: `07_tests/`
- テスト実行スクリプト: `08_scripts/run_tests.sh`
- 詳細ガイド: [07_tests/README.md](../07_tests/README.md)、[07_tests/QUICK_REFERENCE.md](../07_tests/QUICK_REFERENCE.md)

---

## 💻 コーディング規約（コンペ固有）

### Python

- **バージョン**: Python 3.13
- **パッケージマネージャ**: uv
- **データフレーム**: Polars（pandasは使用しない）
- **コメント**: 日本語で記載

#### 必須ライブラリ

```bash
# データ処理（Polars使用）
uv pip install polars pyarrow

# 機械学習
uv pip install scikit-learn lightgbm xgboost

# 可視化
uv pip install matplotlib seaborn plotly

# Notebook
uv pip install jupyter notebook ipykernel

# テスト
uv pip install pytest pytest-cov

# コード品質
uv pip install black ruff mypy

# 実験管理（オプション）
uv pip install mlflow wandb optuna
```

### ファイル構成

#### 必須ファイル

- `README.md`: プロジェクト説明、セットアップ、実行方法
- `pyproject.toml`: 依存関係とプロジェクトメタデータ
- `specs/`: 仕様書（実装前に作成）
- `tests/`: テストコード（実装と同時に作成）
- `.gitignore`: データファイルとモデルを除外

#### オプションファイル

- `Makefile`: よく使うコマンドのショートカット
- `.env.example`: 環境変数のサンプル
- `pytest.ini`: pytest設定

---

## 🔄 開発フロー（SDD + TDD）

### フェーズ1: 探索（Notebook）

1. `notebooks/01_eda/` でデータを理解
2. 仮説を立て、検証
3. プロトタイプコードを書く

### フェーズ2: 仕様策定（SDD）

1. `specs/` に仕様書を作成
   - 要件
   - 入出力仕様
   - エッジケース
   - テストケース

### フェーズ3: テスト作成（TDD - Red）

1. `tests/` にテストを書く
2. テストを実行 → **失敗する**（Red）

### フェーズ4: 実装（TDD - Green）

1. `src/` に最小限の実装
2. テストを実行 → **成功する**（Green）

### フェーズ5: リファクタリング（TDD - Refactor）

1. コードをきれいにする
2. テストを実行 → **成功を維持**

### フェーズ6: 統合

1. Notebookで実装をテスト
2. パイプライン全体で動作確認

---

## 📊 実験管理

### 実験記録のフォーマット

`docs/experiments.md` に以下を記録：

```markdown
## 実験 #{N}: {実験名}

**日付**: YYYY-MM-DD
**目的**: この実験で検証したいこと

### 仕様
- 仕様書: [specs/features.md](../specs/features.md)
- 変更内容: {...}

### テスト結果
- テストカバレッジ: XX%
- 全テスト: Pass/Fail

### モデル設定
- アルゴリズム: LightGBM
- ハイパーパラメータ: {...}
- 特徴量: {...}

### 結果
- CVスコア: 0.XXXX
- LBスコア: 0.YYYY

### 考察
- うまくいった点
- うまくいかなかった点
- 次のアクション
```

---

## 🤝 Claude Codeとの協働ルール（コンペ固有）

### Claude Codeに期待すること

1. **仕様策定支援**: 要件からSPECを作成
2. **テスト設計**: エッジケースの洗い出し
3. **TDDガイド**: Red→Green→Refactorのサポート
4. **Polarsコード**: 効率的なPolars表現の提案
5. **コードレビュー**: 実装の改善提案
6. **ドキュメント作成**: 仕様書、実験記録の整備

### Claude Codeが守るべきこと

1. **仕様駆動開発**
   - 実装前に必ず仕様書を作成
   - 仕様が曖昧なまま実装しない
2. **テスト駆動開発**
   - テストを先に書く
   - Red → Green → Refactor を厳守
3. **Polars使用**
   - pandasは使わない
   - Polarsの表現力を活かす
4. **段階的アプローチ**
   - まずシンプルなベースライン
   - 徐々に複雑化
   - 各ステップでテストを通す

### 指示例

#### 良い指示:

**フェーズ1: 理解**
- "データを読み込んで基本統計を確認して（Notebook）"
- "目的変数の分布を可視化して、仮説を立てて"

**フェーズ2: 仕様策定**
- "位置情報特徴量の仕様書を作成して（specs/features.md）"
- "入出力、エッジケース、テストケースを明確にして"

**フェーズ3: テスト作成**
- "仕様書に基づいてテストを作成して（tests/test_features/test_location.py）"
- "まず失敗するテストから（Red）"

**フェーズ4: 実装**
- "テストを通すための最小限の実装をして（src/features/location.py）"
- "Polarsで効率的に実装して（Green）"

**フェーズ5: リファクタリング**
- "実装をレビューして、改善点を提案して（Refactor）"
- "テストが通ることを確認して"

#### 悪い指示:

- "仕様なしでいきなり実装して" ← SDD違反
- "テストは後で書く" ← TDD違反
- "pandasで実装して" ← Polars使用ルール違反

---

## 🎓 学習の優先順位

### 推奨学習順序

1. **Phase 1: データ理解**（1-2日）← 今ここ
   - データ定義書の確認 ✅
   - 基本統計量の確認
   - 欠損値・外れ値の確認

2. **Phase 2: EDA**（2-3日）
   - 可視化による傾向分析
   - 相関分析
   - 仮説立案

3. **Phase 3: 特徴量エンジニアリング**（3-5日）
   - 仕様書作成（specs/features.md）
   - テスト作成（tests/test_features/）
   - 実装（src/features/）

4. **Phase 4: モデリング**（5-10日）
   - 仕様書作成（specs/models.md）
   - テスト作成（tests/test_models/）
   - 実装（src/models/）

5. **Phase 5: 評価と提出**（継続的）
   - クロスバリデーション
   - エラー分析
   - 提出

---

## 🧪 実験管理（06_experiments/）

### MLflowによる実験記録

**ディレクトリ構成**:
```
06_experiments/
├── mlruns/                 # MLflow実験記録（.gitignore）
├── mlflow.db              # メタデータDB（.gitignore）
├── models/                # 学習済みモデル（.gitignore）
├── experiment_notes/      # 実験メモ（Git管理）
│   ├── 2025-11-23_baseline.md
│   └── experiment_summary.md
└── configs/               # 実験設定（Git管理）
    ├── exp001_baseline.yaml
    └── best_config.yaml
```

### MLflow UI の起動

```bash
# プロジェクトルートから
mlflow ui --backend-store-uri file:./06_experiments

# ブラウザで http://localhost:5000 を開く
```

### 実験の記録方法

**学習スクリプトでの記録**:
```python
import mlflow

# トラッキングURI設定
mlflow.set_tracking_uri("file:./06_experiments")
mlflow.set_experiment("baseline_models")

with mlflow.start_run(run_name="lgbm_v1"):
    # パラメータ記録
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.1)

    # モデル学習
    model.fit(X_train, y_train)

    # メトリクス記録
    rmse = calculate_rmse(y_val, y_pred)
    mlflow.log_metric("rmse", rmse)

    # モデル保存
    mlflow.sklearn.log_model(model, "model")
```

### 実験結果の確認

**Notebookでの比較**:
```python
import mlflow
mlflow.set_tracking_uri("file:./06_experiments")

# 全実験の比較
runs = mlflow.search_runs(experiment_ids=["1"])
print(runs[["tags.mlflow.runName", "metrics.rmse", "params.max_depth"]])
```

### .gitignoreルール

```gitignore
# 06_experiments/.gitignore
mlruns/          # 実験データ（大容量）
mlflow.db        # メタDB
models/          # 学習済みモデル

# Git管理する
!experiment_notes/
!configs/
```

---

## 📊 EDA実践ガイド

EDAを実施する際は、**必ず** `.claude/eda_guide.md` を参照してください。

### ガイドの内容

1. **EDA標準フロー（4フェーズ）**
   - Phase 1: データ概要把握（10分）
   - Phase 2: 目的変数の分析（15分）
   - Phase 3: 特徴量の分析（30-60分）
   - Phase 4: データ品質チェック（15分）

2. **必須の可視化チェックリスト**
   - 目的変数: ヒストグラム、Box plot、Q-Qプロット
   - 数値変数: 分布、相関行列、Scatter plot
   - カテゴリ変数: 頻度、Box plot
   - データ品質: 欠損パターン、Train/Test重複

3. **即実行可能なコードテンプレート**
   - 基本統計
   - 分布の可視化
   - 相関分析
   - データ品質チェック

**重要**: Claude CodeにEDAを依頼する際は、「eda_guide.mdに従ってEDAを実施してください」と指示すること。

---

## 🔄 更新履歴

- **2025-11-22**: 不動産価格予測コンペ用に作成
  - SDD + TDD + Polarsの開発方針を明確化
  - ディレクトリ構造の定義
  - データ定義書の整備

- **2025-11-23**: ディレクトリ整理・実験管理・EDA追加
  - ディレクトリを作業フロー順に連番化（01-09）
  - MLflow実験管理ガイド追加（06_experiments/）
  - EDA実践ガイドの参照を追加

---

**最終更新**: 2025-11-23
