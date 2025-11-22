# 汎用開発ルール - Claude Code協働ガイド

> このファイルは**どんなプロジェクトでも使える汎用的な開発ルール**です。プロジェクト固有のルールは別ファイルで定義してください。

---

## 📝 開発プロセス（仕様駆動 + テスト駆動）

このガイドでは、実践に近い形で**仕様駆動開発（SDD）**と**テスト駆動開発（TDD）**を組み合わせた開発プロセスを推奨します。

### 開発フロー（全8フェーズ）

#### 1. **理解フェーズ** 📖

- 既存コードや参考資料を読む
- アーキテクチャ、データフロー、依存関係を理解する
- ビジネス要件と技術的課題を把握する
- メモを作成

#### 2. **仕様策定フェーズ** 📋 ← **仕様駆動開発**

- `SPECIFICATION.md` を作成する
- 以下を明確に定義：
  - **要件定義**: 機能要件と非機能要件
  - **アーキテクチャ設計**: システム構成、コンポーネント設計
  - **API仕様**: エンドポイント、入出力形式、エラーレスポンス（APIを持つ場合）
  - **データモデル**: データ構造、スキーマ
  - **成功基準**: 何を持って完成とするか
- 仕様をClaude Codeとレビュー・議論する

#### 3. **テスト設計フェーズ** 🧪 ← **テスト駆動開発（Red）**

- 仕様に基づいてテストケースを作成
- テストの種類：
  - **ユニットテスト**: 個々の関数・クラスのテスト
  - **統合テスト**: コンポーネント間の連携テスト
  - **E2Eテスト**: システム全体の動作テスト
- まず失敗するテストを書く（Red）
- テストツール（pytest, Jest, JUnitなど）で実行してテストが失敗することを確認
- **テスト結果を保存**: 後から見返せるようにRedフェーズの結果を記録

#### 4. **実装フェーズ** 💻 ← **テスト駆動開発（Green）**

- テストを通すための最小限の実装から始める
- 段階的に機能を拡張
- 各ステップでテストを実行してGreenにする
- コードはシンプルから始めて段階的に拡張
- **テスト結果を保存**: Greenフェーズの結果を記録
- **Red/Green比較**: 進捗を確認

#### 5. **リファクタリングフェーズ** ♻️ ← **テスト駆動開発（Refactor）**

- テストが通った状態でコードを改善
- 以下を意識：
  - コードの可読性向上
  - 重複の削除（DRY原則）
  - 適切な抽象化
  - パフォーマンス最適化
- リファクタリング後もテストがGreenであることを確認

#### 6. **検証フェーズ** ✅

- 実装したコードを実際に動かして確認
- ログやメトリクスを確認
- 仕様書の成功基準を満たしているか検証
- エッジケースやエラーハンドリングの確認
- 負荷テスト（必要に応じて）

#### 7. **振り返りフェーズ** 📊

- **ドキュメント更新（必須）**
  - README.mdを更新（実装の説明、セットアップ手順、実行方法）
  - 進捗管理ファイルに完了日とステータスを記録
- **学習内容の整理**
  - 学んだこと、疑問点、改善点をまとめる
  - 仕様と実装のギャップを分析
- **コードレビュー**（自己レビューまたはペアレビュー）を実施
- **Git操作**
  - 変更をコミット
  - プッシュ（確認を取る）

#### 8. **クリーンアップフェーズ** 🧹

パターン実装が完了したら、次のパターンに進む前に環境をクリーンアップする。

- **バックグラウンドプロセスの停止**
  - `KillShell`でバックグラウンドのBashプロセスを停止
  - uvicorn、gunicorn等のWebサーバーを停止
  - Workerプロセスを停止

- **Dockerコンテナの停止と削除**
  - 開発用に起動したコンテナを停止: `docker stop <container>`
  - コンテナを削除: `docker rm <container>`
  - Docker Composeの場合: `docker-compose down`

- **ポートの解放確認**
  - 使用したポートが解放されているか確認: `lsof -ti:<port>`
  - 必要に応じてプロセスをkill: `lsof -ti:<port> | xargs kill -9`

- **Kubernetesリソース（該当する場合）**
  - 不要なnamespaceは削除: `kubectl delete namespace <namespace>`
  - ただし、後で使う可能性があるリソースは残してもOK（minikubeなど）

**クリーンアップの確認コマンド**:
```bash
# バックグラウンドプロセス確認
jobs

# Docker確認
docker ps -a

# ポート確認
lsof -ti:8000  # 空ならOK
```

### TDDサイクル（Red-Green-Refactor）

```
┌─────────────────────────────────────┐
│  1. 仕様を書く（SPECIFICATION.md） │
└──────────────┬──────────────────────┘
               ↓
┌──────────────────────────────────────┐
│  2. テストを書く（失敗するテスト）  │ ← Red
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────┐
│  3. 実装する（テストを通す）    │ ← Green
└──────────────┬───────────────────┘
               ↓
┌──────────────────────────────┐
│  4. リファクタリングする    │ ← Refactor
└──────────────┬───────────────┘
               ↓
      テストが通っているか？
               ↓
          [次の機能へ]
```

---

## 💻 コーディング規約（汎用）

### 基本原則

- **可読性**: コードは書く時間より読む時間の方が長い
- **一貫性**: チーム/プロジェクト内で統一されたスタイルを使用
- **シンプルさ**: 複雑さを避け、明確で理解しやすいコードを書く
- **ドキュメント**: 重要な関数・クラスにはdocstringやコメントを付ける

### Python

- **スタイル**: PEP 8に準拠
- **型ヒント**: 可能な限り使用する（Python 3.5+）
- **docstring**: 重要な関数・クラスには必須
- **命名規則**:
  - 関数/変数: `snake_case`
  - クラス: `PascalCase`
  - 定数: `UPPER_CASE`
  - プライベート: `_leading_underscore`

### JavaScript/TypeScript

- **スタイル**: ESLint推奨設定に準拠
- **型**: TypeScriptを使用する場合は型定義を明確に
- **命名規則**:
  - 関数/変数: `camelCase`
  - クラス/コンポーネント: `PascalCase`
  - 定数: `UPPER_SNAKE_CASE`
  - プライベート: `#privateField` (ES2022+)

### Docker

- **ベースイメージ**: 公式イメージを使用
- **マルチステージビルド**: 可能な限り使用してイメージサイズを削減
- **環境変数**: センシティブな情報は`.env`ファイルで管理（gitignore必須）
- **.dockerignore**: 不要なファイルをコンテナに含めない

---

## 📄 SPECIFICATION.md テンプレート（汎用）

```markdown
# {プロジェクト/機能名} 仕様書

## 1. 要件定義

### 1.1 機能要件

- [ ] 要件1
- [ ] 要件2

### 1.2 非機能要件

- **パフォーマンス**: レスポンスタイム < 100ms
- **スケーラビリティ**: 同時接続数 > 100
- **可用性**: 99.9%以上
- **セキュリティ**: [セキュリティ要件]

## 2. アーキテクチャ設計

### 2.1 システム構成

[構成図またはテキストで説明]

### 2.2 コンポーネント設計

- **コンポーネントA**: 役割と責務
- **コンポーネントB**: 役割と責務

### 2.3 技術スタック

- [言語・フレームワーク]
- [データベース]
- [使用するライブラリ]

## 3. API仕様（該当する場合）

### 3.1 エンドポイント

| Method | Path      | Description |
| ------ | --------- | ----------- |
| POST   | /resource | リソース作成  |

### 3.2 リクエスト/レスポンス形式

[詳細なスキーマ]

### 3.3 エラーレスポンス

[エラーコードとメッセージ]

## 4. データモデル

### 4.1 入力データ

[スキーマ定義]

### 4.2 出力データ

[スキーマ定義]

## 5. 成功基準

- [ ] 全テストケースがパス
- [ ] パフォーマンス要件を満たす
- [ ] エラーハンドリングが適切
- [ ] ログが適切に出力される
```

---

## 🔧 実装時の注意事項

### セキュリティ

- **機密情報**: API キー、パスワード等は環境変数で管理
- **`.gitignore`**: `.env`, `*.pem`, `*.key` 等を必ず追加
- **依存関係**: セキュリティ脆弱性のあるパッケージを避ける
- **入力検証**: 外部からの入力は必ず検証・サニタイズ
- **OWASP Top 10**: 主要な脆弱性（SQLインジェクション、XSS、CSRF等）を理解し対策

### パフォーマンス

- **リソース管理**: メモリリーク、ファイルディスクリプタのクローズ漏れに注意
- **非同期処理**: I/Oバウンドな処理は非同期化を検討
- **バッチサイズ**: メモリを考慮した適切なバッチサイズを設定
- **キャッシュ**: 計算コストの高い処理は適切にキャッシュ
- **データベース**: インデックスの最適化、N+1問題の回避

### エラーハンドリング

- **例外処理**: 適切なtry-catch/except、エラーメッセージの明確化
- **ロギング**: エラー時の状況を記録（スタックトレース、入力値）
- **フェイルセーフ**: システムが停止しない仕組み
- **ユーザーフィードバック**: ユーザーに分かりやすいエラーメッセージ

---

## 🧪 テスト結果の管理

### テスト結果とコード品質チェック結果の取り扱い

**重要**: テスト結果とコード品質チェック結果は **Gitで管理しません**。

**理由**:

- 実行環境やタイミングで結果が変わる
- ローカル環境固有の情報を含む可能性がある
- CI/CDで自動生成されるべき
- 容量が大きくなる可能性がある

### ディレクトリ構造

テスト結果は各テストモジュール配下に保存します：

```
tests/
├── test_module_a/
│   ├── test_*.py
│   └── test_results/              # テスト結果
│       ├── test_result_YYYYMMDD_HHMMSS.txt
│       ├── coverage_YYYYMMDD_HHMMSS.txt
│       ├── latest_result.txt      # 最新へのシンボリックリンク
│       ├── latest_coverage.txt
│       └── .gitkeep               # ディレクトリ保持用
├── test_module_b/
│   └── test_results/
├── README.md                       # テスト結果の見方ガイド
└── QUICK_REFERENCE.md              # よく使うコマンド集
```

### Git管理しないファイル ❌

**.gitignore に追加すべきパターン**:

```gitignore
# テスト結果（すべて除外）
**/test_results/

# Pytestキャッシュ
__pycache__/
.pytest_cache/
*.pyc

# カバレッジ
.coverage
htmlcov/
coverage.xml

# ただし、test_resultsディレクトリ自体は保持
!**/test_results/
!**/test_results/.gitkeep

# コード品質チェック結果
quality_checks/
```

### テスト結果ファイルの形式

**推奨フォーマット（構造化された結果）**:

タイムスタンプ付きファイル名で履歴管理し、誰が見てもわかるようにコメント・サマリを追加：

```
================================================================================
テスト実行結果
================================================================================

実行日時: 2025-11-23 14:30:52
対象: tests/test_module
実行コマンド: pytest tests/test_module -v

================================================================================
テスト結果
================================================================================

test_module/test_file.py::test_case_1 PASSED     [ 20%]
test_module/test_file.py::test_case_2 PASSED     [ 40%]
...

============================== 5 passed in 0.12s ===============================

================================================================================
サマリ
================================================================================

テスト状態: ✓ PASSED (or ✗ FAILED)
実行日時: 2025-11-23 14:30:52
結果ファイル: tests/test_module/test_results/test_result_20251123_143052.txt
```

**カバレッジレポート形式**:

```
================================================================================
カバレッジ測定結果
================================================================================

実行日時: 2025-11-23 14:35:20
対象: tests/test_module
実行コマンド: pytest tests/test_module --cov=src --cov-report=term-missing

================================================================================
カバレッジレポート
================================================================================

---------- coverage: platform darwin, python 3.13.9-final-0 ----------
Name                           Stmts   Miss  Cover   Missing
------------------------------------------------------------
src/__init__.py                    0      0   100%
src/module.py                     45      2    96%   78-79
------------------------------------------------------------
TOTAL                             45      2    96%

================================================================================
サマリ
================================================================================

全体カバレッジ: 96%
目標: 80%以上
状態: ✓ 目標達成

未カバー箇所:
- src/module.py: 78-79行目
  → エラーハンドリングのエッジケース
  → 追加テストケースを検討
```

### ローカルで結果を確認する方法

#### 基本的なテスト実行

```bash
# Python: pytest
mkdir -p tests/test_module/test_results
pytest tests/test_module -v > tests/test_module/test_results/test_result_$(date +%Y%m%d_%H%M%S).txt 2>&1

# JavaScript: Jest
mkdir -p tests/test_module/test_results
npm test > tests/test_module/test_results/test_result_$(date +%Y%m%d_%H%M%S).txt 2>&1
```

#### カバレッジ測定

```bash
# Python
pytest tests/ --cov=src --cov-report=term-missing > tests/test_results/coverage_$(date +%Y%m%d_%H%M%S).txt 2>&1

# JavaScript
npm test -- --coverage > tests/test_results/coverage_$(date +%Y%m%d_%H%M%S).txt 2>&1
```

#### 自動化スクリプト例

**`scripts/run_tests.sh`** (Python):

```bash
#!/bin/bash

# 引数チェック
if [ $# -eq 0 ]; then
    TEST_TARGET="tests"
else
    TEST_TARGET="tests/$1"
fi

# test_resultsディレクトリ作成
RESULTS_DIR="$TEST_TARGET/test_results"
mkdir -p "$RESULTS_DIR"

# タイムスタンプ
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# カバレッジオプション
if [[ "$*" == *"--coverage"* ]]; then
    RESULT_FILE="$RESULTS_DIR/coverage_$TIMESTAMP.txt"
    pytest "$TEST_TARGET" --cov=src --cov-report=term-missing > "$RESULT_FILE" 2>&1
    ln -sf "coverage_$TIMESTAMP.txt" "$RESULTS_DIR/latest_coverage.txt"
else
    RESULT_FILE="$RESULTS_DIR/test_result_$TIMESTAMP.txt"
    pytest "$TEST_TARGET" -v > "$RESULT_FILE" 2>&1
    ln -sf "test_result_$TIMESTAMP.txt" "$RESULTS_DIR/latest_result.txt"
fi

echo "結果: $RESULT_FILE"
```

### README.mdは Git管理する ✅

以下のドキュメントファイルは**Git管理します**（結果ファイル自体は管理しない）：

- `tests/README.md` - テスト結果の見方、実行方法ガイド
- `tests/QUICK_REFERENCE.md` - よく使うコマンド集
- `tests/test_results/.gitkeep` - ディレクトリ保持用

**理由**:

- 初見の人がテスト結果を理解するために必要
- 環境に依存しない一般的な情報
- プロジェクトのドキュメントとして価値がある

### CI/CD での利用

テスト結果は CI/CD で自動生成し、アーティファクトとして保存：

```yaml
# GitHub Actions 例
- name: Run tests
  run: |
    pytest tests/ -v --cov=src --cov-report=xml --cov-report=term

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

---

## 📝 一時ファイルの管理ルール

### セッション関連の一時ファイル

プロジェクト進行中に作成される一時的なファイルは、セッション終了後に削除するかgitignoreに追加してください。

#### 一時ファイルの種類

1. **セッション記録ファイル**
   - `SESSION_*.txt` - セッション固有の作業記録
   - `PROJECT_STATUS.md` - その時点のプロジェクト状態
   - `QUICKSTART.md` - セッション固有のクイックスタートガイド
   - `RESTART_*.md` - セッション固有の再開手順
2. **一時的なガイドファイル**
   - セッション固有の手順書
   - 特定の問題解決のための一時ドキュメント

#### 管理方針

- **セッション終了時**: 一時ファイルを削除または整理
- **永続的な情報**: プロジェクトのドキュメントディレクトリに統合
- **.gitignore**: 一時ファイルのパターンを追加

```gitignore
# Session files (may contain tokens)
SESSION_*.txt
PROJECT_STATUS.md
QUICKSTART.md
RESTART_*.md
```

#### 判断基準

| ファイル               | 保持 | 理由                     |
| ---------------------- | ---- | ------------------------ |
| 学習ガイド（汎用的）   | ✅   | 今後も参照価値がある     |
| セッション手順書       | ❌   | その時点の状況に特化     |
| 技術ドキュメント       | ✅   | 永続的な知識             |
| ステータスファイル     | ❌   | 時間経過で陳腐化         |

---

## 🔒 個人情報とセキュリティルール

### 個人情報の取り扱い

公開リポジトリの場合、個人情報の管理を徹底してください。

#### 禁止事項

以下の個人情報をコード、ドキュメント、コミットメッセージに含めないでください：

1. **実名**
   - フルネーム
   - 名前の一部（ファーストネーム、ラストネーム）
2. **連絡先情報**
   - メールアドレス（プライベート）
   - 電話番号
   - 住所
3. **システム固有情報**
   - ローカルマシンのユーザー名
   - ホスト名
   - 絶対パスに含まれる個人名

#### 推奨事項

**GitHubユーザー名を使用**:

```bash
git config --global user.name "your_github_username"
git config --global user.email "your_username@users.noreply.github.com"
```

**パス表記**:

```bash
# ❌ 個人情報を含む
/Users/tanaka_taro/project

# ✅ プレースホルダーを使用
/Users/username/project
# または
/path/to/project
# または
~/project
```

### 機密情報の取り扱い

#### 管理が必要な情報

1. **APIキー・トークン**
   - GitHub Personal Access Token
   - その他のAPI認証情報
2. **環境変数**
   - データベース接続情報
   - サービスの認証情報
3. **秘密鍵**
   - SSH秘密鍵
   - 暗号化キー
   - 証明書

#### 安全な管理方法

**環境変数ファイル**:

```bash
# .env ファイルを使用（gitignore必須）
API_KEY=your_key_here
DATABASE_URL=postgresql://user:pass@localhost/db
```

**サンプルファイルの提供**:

```bash
# .env.example をリポジトリにコミット
API_KEY=your_api_key_here
DATABASE_URL=postgresql://localhost/postgres
```

**.gitignoreの設定**:

```gitignore
# Secrets
*.pem
*.key
credentials.json
secrets.yaml

# Environment variables
.env
.env.local
.env.*.local
```

### Git履歴のクリーンアップ

もし個人情報や機密情報をコミットしてしまった場合：

**最新コミットのみの場合**:

```bash
# ファイルを削除してコミットを修正
git rm <sensitive-file>
git commit --amend --no-edit
git push --force
```

**履歴全体から削除する場合**:

```bash
# git-filter-repoを使用（推奨）
pip install git-filter-repo
git filter-repo --path <sensitive-file> --invert-paths

# または BFG Repo-Cleaner
java -jar bfg.jar --delete-files <sensitive-file>

# 強制プッシュ
git push --force
```

**注意**: 強制プッシュは共同作業者に影響を与えるため、個人プロジェクトでのみ実施してください。

### GitHub Push Protectionについて

GitHubは自動的にトークンやシークレットの検出を行います：

- プッシュ時に機密情報を検出すると自動的にブロック
- 検出されたら該当ファイルを削除してから再プッシュ
- `.gitignore`を適切に設定して予防

---

## 🧹 Git Push前のクリーンアップルール

### 必須: Push前に不要ファイルを削除

**重要**: `git push` 前に必ず以下の不要ファイルを削除してください。

#### 削除対象ファイル

**1. 一時ファイル**
- `*_tmp.*` - 一時作業ファイル
- `*.tmp` - 一時ファイル
- `temp_*` - 一時ファイル
- `test_*.txt` (テスト用のメモなど)
- `.DS_Store` - macOSのシステムファイル
- `Thumbs.db` - Windowsのシステムファイル

**2. サンプル・テストファイル**
- `example_*.py`, `example_*.js` など
- `sample_*.py`, `sample_*.js` など
- `*_example.txt`, `*_sample.txt` など
- プロジェクトで使わない検証用ファイル

**3. 個人メモ・作業ファイル**
- `TODO.txt`, `MEMO.txt` (プロジェクトのTODO管理以外)
- `notes.md`, `scratch.md`
- セッション固有のメモファイル

**4. 生成されたファイル（.gitignoreで管理）**
- ログファイル (`*.log`)
- キャッシュ (`__pycache__/`, `.pytest_cache/`)
- ビルド成果物 (`dist/`, `build/`)
- 環境固有の設定 (`.env`, `*.local`)

#### クリーンアップコマンド

**手動確認（推奨）**:
```bash
# 不要ファイルを検索
find . -name "*_tmp.*" -o -name "*.tmp" -o -name "temp_*" -o -name "example_*" -o -name "sample_*"

# macOSシステムファイル
find . -name ".DS_Store"

# 確認してから削除
rm -i <file>
```

**自動削除（慎重に）**:
```bash
# .DS_Storeの削除
find . -name ".DS_Store" -delete

# 一時ファイルの削除（プロジェクトルートで実行）
find . -name "*_tmp.*" -delete
find . -name "*.tmp" -delete
```

#### .gitignore の確認

Push前に `.gitignore` が適切に設定されているか確認：

```bash
# 追跡されているファイルを確認
git status

# 誤って追加されたファイルを削除
git rm --cached <file>
```

**.gitignore必須パターン**:
```gitignore
# 一時ファイル
*_tmp.*
*.tmp
temp_*
.DS_Store
Thumbs.db

# IDE/エディタ
.vscode/
.idea/
*.swp
*.swo

# ログ・キャッシュ
*.log
__pycache__/
.pytest_cache/

# ビルド成果物
dist/
build/
*.egg-info/

# 環境変数・秘密情報
.env
.env.local
*.pem
*.key
credentials.json
```

#### Push前チェックリスト

```bash
# 1. 不要ファイルの検索と削除
find . -name "*_tmp.*" -o -name "*.tmp" -o -name "example_*" -o -name ".DS_Store"

# 2. .gitignore確認
cat .gitignore

# 3. 追跡ファイル確認
git status

# 4. 差分確認
git diff

# 5. コミット
git add .
git commit -m "コミットメッセージ"

# 6. Push（ユーザー確認後）
git push
```

---

## 🤝 Claude Codeとの協働ルール（汎用）

### Claude Code に期待すること

1. **コード分析**: 既存コードの詳細な説明
2. **設計支援**: アーキテクチャの提案と議論
3. **実装サポート**: ゼロからのコーディング支援
4. **レビュー**: 実装したコードの改善提案
5. **トラブルシューティング**: エラーの解決支援
6. **ドキュメント作成**: README等の作成補助

### Claude Code が守るべきこと（汎用）

1. **返答言語**: **必ず日本語で返答すること**（絶対厳守）
   - すべての説明、コメント、ドキュメントは日本語で記載
   - 技術用語は適宜英語を併記してもよい（例: テスト駆動開発（TDD））
   - コード内のコメント・docstringも日本語で記載

2. **仕様駆動**: まず仕様を明確にし、それに基づいて開発を進める
   - **SPECIFICATION.mdは最優先事項**: 実装前に必ず確認する
   - **仕様と異なる実装をする場合は必ずユーザーに確認**: 勝手に仕様を変更しない
     - 例: 「仕様ではMySQL 5.7ですが、SQLiteでテストしてもいいですか？」
   - **完了時の仕様チェック（必須）**: タスク完了後、以下を確認する
     - ✅ SPECIFICATION.mdの要件が全て満たされているか
     - ✅ アーキテクチャ設計通りに実装されているか
     - ✅ 技術スタック（データベース、ライブラリ等）が仕様通りか
     - ✅ 成功基準を満たしているか
     - ❌ 差異がある場合は必ずユーザーに報告し、修正する

3. **テスト駆動**: テストを先に書き、Red→Green→Refactorのサイクルを守る
4. **段階的実装**: 一度に全てを実装せず、小さなステップで進める
5. **説明**: コードだけでなく、なぜそう実装するのか説明する
6. **選択肢の提示**: 複数のアプローチがある場合は選択肢を示す
7. **学習重視**: 単にコードを書くだけでなく、理解を深める支援をする
8. **ベストプラクティス**: 本番環境を想定した品質のコードを書く
9. **実践的**: エラーハンドリング、ログ、モニタリングを含める
10. **Git操作**: `git push` 前に以下を必ず実施
    - **不要ファイルの削除**: 上記クリーンアップルールに従う
    - **ユーザーの確認を取る**: Push前に必ず確認

---

## 🔌 外部ツール連携（MCP）

### Model Context Protocol (MCP) について

MCPを使用すると、Claude CodeからGitHub、Notion、データベースなどの外部ツールにアクセスできます。

主なMCPサーバー：

- **GitHub MCP**: リポジトリ操作、Issue/PR管理
- **Notion MCP**: ドキュメント管理
- **Serena MCP**: 高度なコード分析・編集
- **Context7 MCP**: 最新ライブラリドキュメントの参照
- **PostgreSQL MCP**: データベース操作

### セキュリティ注意事項

- MCPサーバーの設定ファイル（`.mcp.json`）には認証トークンが含まれるため、**必ず`.gitignore`に追加**
- トークンは環境変数で管理し、設定ファイルにはハードコードしない
- `.env.example`でサンプル設定を提供

---

## 📚 推奨リソース

### 開発原則

- [The Twelve-Factor App](https://12factor.net/) - モダンアプリケーション開発の12の原則
- [Clean Code](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882) - Robert C. Martin著
- [The Pragmatic Programmer](https://pragprog.com/titles/tpp20/the-pragmatic-programmer-20th-anniversary-edition/) - 実践的プログラマー

### テスト

- [Test Driven Development](https://www.amazon.com/Test-Driven-Development-Kent-Beck/dp/0321146530) - Kent Beck著
- [pytest Documentation](https://docs.pytest.org/) - Pythonテストフレームワーク
- [Jest Documentation](https://jestjs.io/) - JavaScriptテストフレームワーク

### セキュリティ

- [OWASP Top 10](https://owasp.org/www-project-top-ten/) - Webアプリケーションのセキュリティリスク
- [CWE Top 25](https://cwe.mitre.org/top25/) - 最も危険なソフトウェアの脆弱性

---

## 🔄 更新履歴

**2025-11-13**:
- クリーンアップフェーズを追加（フェーズ8）
- 開発フローを7フェーズから8フェーズに更新
- バックグラウンドプロセス、Dockerコンテナ、ポートのクリーンアップ手順を明記

---

**このファイルは汎用的な開発ルールです。プロジェクト固有のルールは別途定義してください。**

**最終更新**: 2025-11-13
