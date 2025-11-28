# データ分析コンペティション プロジェクト

> **このプロジェクトについて**: データ分析コンペ（Kaggle、SIGNATE等）で再利用可能なテンプレートプロジェクトです。

---

## 📋 プロジェクト概要

このプロジェクトは以下の3つの柱で構成されています：

1. **仕様駆動開発（SDD）** - まず分析方針を明確にしてから実装
2. **テスト駆動開発（TDD）** - Red→Green→Refactorのサイクル
3. **実験管理** - MLflowやW&Bでの実験記録

### プロジェクトの目的

- データ分析コンペで高速に分析・モデリングを進める
- 再利用可能なコードベースとテンプレートを構築
- 実験管理と再現性を確保
- 知見を体系的に蓄積

---

## 📁 ディレクトリ構造（作業フロー順）

```
.
├── .claude/                    # Claude Code設定
│   ├── claude.md              # プロジェクト概要（このファイル）
│   ├── 01_general_rules.md    # 汎用開発ルール（SDD/TDD/Git等）
│   ├── 02_project_rules.md    # プロジェクト固有ルール
│   ├── 03_experiment_management_rules.md  # 実験管理
│   ├── 04_feature_engineering_rules.md    # 特徴量エンジニアリング
│   ├── 05_eda_guide.md        # EDA実践ガイド
│   └── 06_notebook_rules.md   # Notebookルール
│
├── 01_specs/                   # 📋 仕様書（SDD）
│   ├── data.md               # データ仕様
│   ├── eda.md                # EDA方針
│   ├── preprocessing.md      # 前処理仕様
│   └── features.md           # 特徴量仕様
│
├── 02_docs/                    # 📚 ドキュメント
│   ├── README.md             # ドキュメント案内
│   ├── data_definition.md    # データ定義書（149特徴量）
│   ├── features/             # 特徴量カテゴリ別詳細
│   └── guides/               # ガイド・Tips集（プロセス別）
│       ├── README.md                    # ガイド索引
│       ├── feature_engineering/         # 特徴量エンジニアリング
│       ├── eda/                        # EDA
│       ├── modeling/                   # モデリング
│       ├── validation/                 # バリデーション
│       └── tips/                       # 汎用Tips
│
├── 03_configs/                 # ⚙️ 設定ファイル（YAML）
│   ├── base.yaml             # プロジェクト基本情報
│   ├── data.yaml             # データパス・分割設定
│   ├── preprocessing.yaml
│   ├── features.yaml
│   ├── model.yaml
│   └── training.yaml
│
├── 04_src/                     # 💻 ソースコード（TDD/SDD対象）
│   ├── data/                 # データ読み込み・分割
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
│   ├── 02_feature/           # 特徴量エンジニアリング検証
│   ├── 03_modeling/          # モデリング実験
│   └── 04_evaluation/        # 評価・検証
│
├── 06_experiments/             # 🧪 実験管理
│   └── expXXX_name/          # 各実験ディレクトリ
│       ├── README.md         # 実験サマリー
│       ├── SPEC.md           # 実験仕様書
│       ├── configs/
│       │   └── experiment.yaml  # 全ハイパーパラメータ
│       ├── code/             # 実験コード
│       │   ├── train.py      # エントリーポイント
│       │   ├── preprocessing.py
│       │   ├── pipeline.py
│       │   ├── constants.py  # パス・カラムリスト
│       │   └── expXXX_features.py
│       ├── outputs/          # 実行結果（.gitignore）
│       └── mlruns/           # MLflow記録（.gitignore）
│
├── 07_tests/                   # ✅ テストコード（TDD）
│   ├── test_data/
│   ├── test_eda/
│   ├── test_preprocessing/
│   ├── test_features/
│   ├── test_models/
│   ├── test_training/
│   └── test_evaluation/
│
├── 08_scripts/                 # 🚀 実行スクリプト
│   ├── train.py              # 学習実行
│   ├── predict.py            # 推論実行
│   └── export_markdown.sh    # Markdownエクスポート
│
├── 09_submissions/             # 📤 提出ファイル
│   └── submission_*.csv
│
├── 10_progress/                # 📊 進捗管理
│   ├── progress_log.md       # 進捗ログ
│   └── implementation_plan.md # 実装計画
│
├── data/                       # データセット（.gitignore）
│   ├── raw/                  # 生データ
│   ├── processed/            # 前処理済みデータ（enriched含む）
│   ├── master/               # マスターデータ（CSV）
│   └── external/             # 外部データ
│
└── models/                     # 学習済みモデル（.gitignore、06と統合予定）
```

---

## 🚀 クイックスタート

### 新しいコンペを始める場合

```bash
# 1. プロジェクトディレクトリ作成
mkdir -p data/{raw,processed,external}
mkdir -p notebooks/{01_eda,02_feature,03_modeling,04_evaluation}
mkdir -p src/{data,features,models,utils}
mkdir -p models submissions configs tests docs scripts

# 2. 仮想環境セットアップ
echo "3.13" > .python-version
uv venv
source .venv/bin/activate

# 3. 基本ライブラリインストール
uv pip install pandas numpy scikit-learn matplotlib seaborn
uv pip install jupyter notebook ipykernel

# 4. 開発ツールインストール
uv pip install pytest pytest-cov black ruff mypy

# 5. 実験管理ツール（オプション）
uv pip install mlflow wandb optuna
```

### 分析の流れ

```
データ理解 → EDA → 特徴量設計 → モデリング → 評価 → 提出 → 振り返り
```

詳細は [02_project_rules.md](./02_project_rules.md) を参照してください。

---

## 📚 ルールファイル構成

全体ルール → プロジェクト固有 → 作業別の順で構成：

### 1. [01_general_rules.md](./01_general_rules.md) 🌐 汎用
**どのプロジェクトでも使える開発ルール**
- 仕様駆動開発（SDD）/ テスト駆動開発（TDD）
- コーディング規約 / Git管理
- Claude Codeとの協働ルール

### 2. [02_project_rules.md](./02_project_rules.md) 📋 プロジェクト固有
**このコンペ固有のルール**
- ディレクトリ構造詳細
- データ分析コンペの進め方
- 提出ファイルの管理

### 3. [03_experiment_management_rules.md](./03_experiment_management_rules.md) 🧪 実験
**実験管理ルール（exp010ベース）**
- code/, outputs/, mlruns/ の構造
- 実験実行フロー / Git管理

### 4. [04_feature_engineering_rules.md](./04_feature_engineering_rules.md) ⚙️ 特徴量
**BaseBlock / FeaturePipeline パターン**
- fit/transform分離
- データリーク防止

### 5. [05_eda_guide.md](./05_eda_guide.md) 📊 EDA
**EDA標準フロー（4フェーズ）**
- 可視化チェックリスト
- コードテンプレート

### 6. [06_notebook_rules.md](./06_notebook_rules.md) 📓 Notebook
**Notebookルール**
- エラーなし・出力付きでcommit
- 図の日本語表示
- PNG保存不要（Notebook内表示で十分）

**再利用可能**: 1, 5, 6 は他プロジェクトでもそのまま使用可

---

## 🎓 推奨ワークフロー

### Phase 1: データ理解（1-2日）
- データセットのダウンロードと確認
- 基本統計量の確認
- データ型、欠損値、外れ値の確認

### Phase 2: EDA（2-3日）
- 可視化による傾向分析
- 特徴量間の相関分析
- ターゲット変数の分布確認

### Phase 3: 特徴量エンジニアリング（3-5日）
- 基本的な特徴量生成
- カテゴリ変数のエンコーディング
- 数値変数のスケーリング

### Phase 4: モデリング（5-10日）
- ベースラインモデルの構築
- 複数モデルの実験
- ハイパーパラメータチューニング
- アンサンブル

### Phase 5: 評価と提出（継続的）
- クロスバリデーション
- リーダーボードスコア確認
- 提出ファイル作成

ただし、コンペの特性に応じて柔軟に調整してください。

---

## 🔌 外部ツール連携

このプロジェクトでは、以下の外部ツールをMCP (Model Context Protocol) 経由で利用できます：

- **GitHub MCP**: リポジトリ操作、Issue/PR管理
- **Notion MCP**: 実験記録の管理
- **Serena MCP**: 高度なコード分析・編集
- **Context7 MCP**: 最新ライブラリドキュメントの参照

詳細な設定方法は [MCP_SETUP.md](../MCP_SETUP.md) を参照してください。

---

## 📖 参考情報

- **汎用開発ルール**: [01_general_rules.md](./01_general_rules.md)
- **プロジェクト固有ルール**: [02_project_rules.md](./02_project_rules.md)
- **EDA実践ガイド**: [05_eda_guide.md](./05_eda_guide.md)
- **データ定義書**: [02_docs/data_definition.md](../02_docs/data_definition.md)
- **進捗記録**: [10_progress/progress_log.md](../10_progress/progress_log.md)
