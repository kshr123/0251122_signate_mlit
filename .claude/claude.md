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
│   ├── claude.md              # プロジェクト概要
│   ├── general_rules.md       # 汎用開発ルール（SDD/TDD/Git等）
│   ├── project_rules.md       # プロジェクト固有ルール
│   └── eda_guide.md          # EDA実践ガイド ⭐NEW
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
│   ├── mlruns/               # MLflow実験記録（.gitignore）
│   ├── mlflow.db             # MLflowメタDB（.gitignore）
│   ├── models/               # 学習済みモデル（.gitignore）
│   ├── experiment_notes/     # 実験メモ（Git管理）
│   └── configs/              # 実験設定（Git管理）
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
├── data/                       # データセット（.gitignore）
│   ├── raw/                  # 生データ
│   ├── processed/            # 前処理済みデータ
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

詳細は [project_rules.md](./project_rules.md) を参照してください。

---

## 📚 ルールファイル構成

このプロジェクトのルールは4つのファイルに分かれています：

### 1. このファイル (claude.md)
- プロジェクト概要
- ディレクトリ構造
- クイックリファレンス

### 2. [project_rules.md](./project_rules.md)
**このプロジェクト固有のルール**
- ディレクトリ構造とルール（詳細）
- データ分析コンペの進め方
- 実験管理ルール（MLflow）
- 提出ファイルの管理

### 3. [eda_guide.md](./eda_guide.md) ⭐NEW
**EDA実践ガイド（即実行可能）**
- EDA標準フロー（4フェーズ）
- 必須の可視化チェックリスト
- コードテンプレート集
- データ品質チェック手順

### 4. [general_rules.md](./general_rules.md)
**どんなプロジェクトでも使える汎用的なルール**
- 仕様駆動開発（SDD）
- テスト駆動開発（TDD）
- コーディング規約
- セキュリティ・パフォーマンス
- Git管理
- Claude Codeとの協働ルール

**重要**: `general_rules.md`と`eda_guide.md`は他のプロジェクト（Kaggle、SIGNATEなど）でもそのまま使えます。

---

## 🎯 現在の進捗

- **コンペ名**: 不動産価格予測（SIGNATE）
- **開始日**: 2025-11-22
- **締切**: 未設定
- **現在のフェーズ**: Phase 2 - ベースラインモデル構築

### 完了タスク

**Phase 0: プロジェクトセットアップ**
- ✅ プロジェクトセットアップ
- ✅ 仮想環境構築（Python 3.13 + uv）
- ✅ データ定義書の作成とMarkdown化
  - 全特徴量（149件）の定義書
  - カテゴリ別詳細ドキュメント（11カテゴリ）
  - タグマスタ情報（257種類）、設備情報（301種類）、エリア情報（1,953エリア）

**Phase 1: テンプレート化基盤構築**
- ✅ ディレクトリ構造の整備（framework モジュール構造）
- ✅ 設定ファイルシステム導入（configs/*.yaml）
  - base.yaml, data.yaml, preprocessing.yaml, features.yaml, model.yaml, training.yaml
- ✅ Config Loader実装（TDD完了、6 tests passed）
- ✅ Data Loader実装（SDD + TDD完了、5 tests passed）
  - specs/data.md（仕様書）
  - src/data/loader.py（Polars使用）
- ✅ EDA utilities 実装（src/eda/）
  - src/eda/profiler.py（データプロファイリング）
  - src/eda/visualizer.py（可視化ユーティリティ）
- ✅ EDA notebook template 作成（notebooks/01_eda/）
  - 01_initial_eda.ipynb（初期EDA）
  - 02_target_analysis.ipynb（ターゲット変数分析）

**Phase 2: ベースラインモデル構築（方針転換）**
- ✅ ベースライン仕様書作成（01_specs/baseline_model.md）
  - MAPE指標、LightGBM、3-Fold CV、再現性確保
- ✅ SeedManager実装（TDD完了、4 tests passed）
  - features/base.py - 乱数シード固定
- ✅ MAPE計算実装（TDD完了、5 tests passed）
  - evaluation/metrics.py - sklearn.metrics wrapper
- ✅ MLflow補助関数実装
  - training/utils/mlflow_helper.py
- ✅ 実験管理構成策定
  - 06_experiments/の構成決定（実験ごとにディレクトリ分割）
- ✅ 特徴量コンポーネント仕様書作成（01_specs/features_components.md）
  - Blockベース設計、fit/transformパターン
  - FeaturePipelineは実験固有のロジックとして扱う
- ❌ SimplePreprocessor実装 → **削除**（抽象化されすぎ）

### 次のステップ（優先度順）

**Phase 2続き: 特徴量Blockシステム構築（最優先）**
1. [ ] BaseBlock実装（TDD）- 04_src/features/base.py拡張
2. [ ] NumericBlock実装（TDD）- 04_src/features/blocks/numeric.py
3. [ ] TargetYmBlock実装（TDD）- 04_src/features/blocks/temporal.py
4. [ ] LabelEncodingBlock実装（TDD）- 04_src/features/blocks/encoding.py
5. [ ] exp001をゼロベースで再構築（新Block使用、明示的な前処理）
6. [ ] 初回SIGNATE提出

**Phase 3: 評価・改善（後回し）**
- [ ] 評価モジュール実装（feature_importance, error_analysis, visualizer）
- [ ] 追加Block実装（CountEncoding, TargetEncoding等）
- [ ] ハイパーパラメータチューニング
- [ ] アンサンブル

詳細は [docs/](../docs/) と [specs/](../specs/) を参照してください。

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

- **プロジェクト固有ルール**: [project_rules.md](./project_rules.md)
- **EDA実践ガイド**: [eda_guide.md](./eda_guide.md) ⭐NEW
- **汎用開発ルール**: [general_rules.md](./general_rules.md)
- **データ定義書**: [02_docs/data_definition.md](../02_docs/data_definition.md)
- **進捗記録**: [progress/progress_log.md](../progress/progress_log.md)

---

## 🔄 更新履歴

**2025-11-22 (初期化)**:
- データ分析コンペ用プロジェクトテンプレートとして初期化
- ディレクトリ構造をコンペ用に最適化
- 推奨ワークフローを追加

**2025-11-22 (テンプレート化基盤)**:
- framework モジュール構造に再構成
- 設定ファイルシステム導入（YAML）
- Config Loader実装（TDD）
- Data Loader実装（SDD + TDD）

**2025-11-22 (EDA基盤構築)**:
- EDA utilities実装（profiler.py, visualizer.py）
- EDA notebook templates作成（初期EDA、ターゲット変数分析）
- テンプレート化可能なEDAワークフロー確立

**2025-11-23 (ディレクトリ整理・EDAガイド追加)**:
- ディレクトリを作業フロー順に連番化（01-09）
- EDA実践ガイド追加（eda_guide.md）- 即実行可能なコードテンプレート集
- ドキュメント構成を4ファイルに整理

**2025-11-24 (ベースラインモデル構築・方針転換)**:
- ベースライン仕様書作成（MAPE、LightGBM、3-Fold CV）
- 基本コンポーネント実装（SeedManager、MAPE、MLflow補助関数）
- 実験管理構成策定（06_experiments/構造決定）
- **方針転換**: SimplePreprocessor削除、Blockベース設計へ移行
- 特徴量コンポーネント仕様書作成（features_components.md）
  - BaseBlock + 個別Block（Numeric, TargetYm, LabelEncoding）
  - FeaturePipelineは作らない（実験固有のロジック）

---

**最終更新**: 2025-11-24
