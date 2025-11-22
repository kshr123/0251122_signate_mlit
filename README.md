# Tutorial Learning Template

このディレクトリは、書籍やチュートリアルを学習するための**再利用可能なプロジェクトテンプレート**です。

## 🎯 目的

GitHubの技術書やチュートリアルを読んで学習する際に：
- 参考コードを分析
- AI駆動開発（Claude Code）でゼロから実装
- 学習記録をGitHubにプッシュ

このワークフローを**標準化・再利用可能**にするためのテンプレートです。

## 📦 含まれるもの

```
tutorial-learning-template/
├── .gitignore                    # Python/ML汎用gitignore
├── .mcp.json.example             # MCP設定サンプル
├── README.template.md            # プロジェクト説明テンプレート
├── README.md                     # このファイル（使い方説明）
├── .claude/
│   └── CLAUDE.template.md        # 汎用化したプロジェクトルール
├── 02_templates/                 # コードテンプレート
│   ├── SPECIFICATION.template.md
│   ├── pyproject.toml.template
│   ├── test_unit.template.py
│   ├── test_integration.template.py
│   ├── test_e2e.template.py
│   └── mcp_settings.json.template
└── 05_progress/
    └── learning_log.template.md  # 進捗管理テンプレート
```

## 🚀 使い方

### Step 1: 新しいプロジェクトを作成

```bash
# 1. このテンプレートをコピー
cp -r ~/Desktop/dev/tutorial-learning-template ~/Desktop/dev/my-new-learning-project
cd ~/Desktop/dev/my-new-learning-project

# 2. テンプレートファイルをリネーム
mv README.template.md README.md
mv .claude/CLAUDE.template.md .claude/CLAUDE.md
mv 05_progress/learning_log.template.md 05_progress/learning_log.md

# 3. 必要なディレクトリを作成
mkdir -p 01_reference 03_my_implementations 04_notes 06_docs
```

### Step 2: プレースホルダーを置換

以下のファイル内のプレースホルダーを実際の値に置き換えてください：

**README.md**:
- `{プロジェクト名}`: 例: "Deep Learning Book Learning"
- `{教材名}`: 例: "ゼロから作るDeep Learning"
- `{参考リポジトリURL}`: 例: "https://github.com/oreilly-japan/deep-learning-from-scratch"
- `{開始日}`: 例: "2025-11-05"

**.claude/CLAUDE.md**:
- `{プロジェクト名}`
- `{書籍名またはチュートリアル名}`
- `{参考リポジトリURL}`

**05_progress/learning_log.md**:
- `{プロジェクト名}`
- `{開始日}`
- `{総トピック数}`
- `{現在のフェーズ}`
- `{次の目標}`

### Step 3: 参考リポジトリをクローン

```bash
cd 01_reference
git clone {参考リポジトリURL}
```

### Step 4: Git初期化とプッシュ

```bash
# Git初期化
cd ~/Desktop/dev/my-new-learning-project
git init
git add .
git commit -m "Initial commit: Setup learning project from template"

# GitHubリポジトリを作成してプッシュ
# （GitHubでリポジトリを作成後）
git remote add origin https://github.com/your-username/your-repo.git
git push -u origin main
```

### Step 5: 学習開始！

Claude Codeと一緒に、仕様駆動開発（SDD）+ テスト駆動開発（TDD）で進めましょう！

```bash
# 最初のトピックを実装
mkdir -p 03_my_implementations/01_first_topic
cd 03_my_implementations/01_first_topic

# テンプレートから開始
cp ../../02_templates/pyproject.toml.template pyproject.toml
cp ../../02_templates/SPECIFICATION.template.md SPECIFICATION.md

# 仮想環境作成
uv venv
source .venv/bin/activate

# 開発開始！
```

## 📚 主要な開発プロセス

このテンプレートは以下の開発プロセスをサポートしています：

1. **理解フェーズ** - 参考コードの分析
2. **仕様策定フェーズ** - SPECIFICATION.md作成
3. **テスト設計フェーズ** - テスト作成（Red）
4. **実装フェーズ** - コード実装（Green）
5. **リファクタリングフェーズ** - コード改善（Refactor）
6. **検証フェーズ** - 動作確認
7. **振り返りフェーズ** - ドキュメント更新

詳細は `.claude/CLAUDE.template.md` を参照してください。

## 🔧 前提条件

- **Python 3.13以上**
- **uv** (Pythonパッケージマネージャー)
- **Claude Code** (AI駆動開発ツール)
- **Git & GitHub**

## 💡 このテンプレートの特徴

### ✅ 含まれているもの

- 仕様駆動開発（SDD）+ テスト駆動開発（TDD）のワークフロー
- Pythonプロジェクトの標準的なディレクトリ構造
- コードテンプレート（仕様書、テスト、pyproject.toml）
- 汎用的なGitルールとコーディング規約
- セキュリティとデータ管理のベストプラクティス

### ❌ 含まれていないもの（プロジェクトごとにカスタマイズ）

- 具体的な教材やトピック名
- 参考リポジトリ（01_reference/は空）
- 実装コード（03_my_implementations/は空）
- 学習ノート（04_notes/は空）

## 📖 参考

このテンプレートは以下のプロジェクトから作成されました：
- [ML_designpattern](https://github.com/kshr123/ML_designpattern) - 機械学習システムデザインパターンの学習プロジェクト

## 📝 License

MIT License

---

**作成日**: 2025-11-05
**元プロジェクト**: ML_designpattern
