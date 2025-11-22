# MCP (Model Context Protocol) セットアップガイド

## 📋 概要

このプロジェクトではMCPを使用して、Claude Codeから各種サービス（GitHub、Notion、PostgreSQLなど）にアクセスできます。

## 🔧 セットアップ手順

### 1. `.mcp.json` ファイルの作成

テンプレートをコピーして設定ファイルを作成します：

```bash
# プロジェクトルートで実行
cp .mcp.json.template .mcp.json
```

### 2. APIキーの設定

`.mcp.json` を編集して、各サービスのAPIキーを設定します。

#### GitHub Personal Access Token

1. GitHub設定: https://github.com/settings/tokens
2. "Generate new token (classic)" をクリック
3. 必要なスコープを選択：
   - `repo` - リポジトリへのフルアクセス
   - `read:org` - 組織情報の読み取り
4. トークンをコピーして `.mcp.json` の `GITHUB_PERSONAL_ACCESS_TOKEN` に貼り付け

```json
"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_あなたのトークン"
```

#### Notion Integration Token

1. Notion Integrations: https://www.notion.so/my-integrations
2. "New integration" をクリック
3. Integration名を入力して作成
4. "Internal Integration Token" をコピー
5. `.mcp.json` の `NOTION_TOKEN` に貼り付け

```json
"NOTION_TOKEN": "ntn_あなたのトークン"
```

6. Notionで使用するページを開き、右上の "..." → "Connections" → 作成したIntegrationを追加

#### Context7 API Key (オプション)

1. Context7: https://context7.ai
2. アカウント作成してAPIキーを取得
3. `.mcp.json` の `UPSTREAM_API_KEY` に貼り付け

```json
"UPSTREAM_API_KEY": "up_あなたのAPIキー"
```

#### Serena プロジェクトパス

Serenaの設定でプロジェクトパスを更新します：

```json
"args": [
  "--from",
  "git+https://github.com/oraios/serena",
  "serena",
  "start-mcp-server",
  "--context",
  "ide-assistant",
  "--project",
  "/Users/あなたのユーザー名/path/to/your/project"  // ← 実際のパスに変更
]
```

#### PostgreSQL (オプション)

ローカルにPostgreSQLがインストールされている場合、接続文字列を更新します：

```json
"postgresql://username:password@localhost/dbname"
```

### 3. `.gitignore` の確認

**重要**: `.mcp.json` は機密情報を含むため、必ず `.gitignore` に追加してください。

```gitignore
# MCP Configuration (contains API tokens)
.mcp.json
```

### 4. Claude Codeの再起動

設定ファイルを作成・編集したら、Claude Codeを再起動してMCPサーバーを読み込みます。

## 🔍 利用可能なMCPサーバー

### GitHub MCP
- リポジトリの作成・更新
- Issue/Pull Requestの管理
- ファイルの読み書き
- コードの検索

### Notion MCP
- ページの作成・更新
- データベースのクエリ
- ブロックの追加・編集
- コメントの作成

### Serena MCP
- 高度なコード分析
- シンボル検索
- リファクタリング
- コード編集

### Context7 MCP
- 最新ライブラリドキュメントの参照
- コード例の検索
- API仕様の確認

### PostgreSQL MCP
- SQLクエリの実行
- データベーススキーマの確認
- データの取得・分析

## 🔒 セキュリティのベストプラクティス

1. **APIキーの管理**
   - `.mcp.json` は必ず `.gitignore` に追加
   - APIキーは定期的にローテーション
   - 最小限の権限のみを付与

2. **バージョン管理**
   - `.mcp.json.template` または `.mcp.json.example` をリポジトリにコミット
   - 実際の `.mcp.json` はローカルのみに保持

3. **トークンの取り扱い**
   - スクリーンショットや画面共有に注意
   - ログファイルに出力しない
   - 不要になったトークンは削除

## 🐛 トラブルシューティング

### MCPサーバーが起動しない

1. Claude Codeを再起動
2. `.mcp.json` の構文エラーをチェック（JSONパーサーで検証）
3. APIキーが正しいか確認

### Notion接続ができない

1. Integration TokenがNotionページに接続されているか確認
2. 必要な権限が付与されているか確認
3. トークンが正しくコピーされているか確認

### Serenaが動作しない

1. プロジェクトパスが正しいか確認
2. `uvx` がインストールされているか確認
3. Python環境が正しくセットアップされているか確認

## 📚 参考リンク

- [MCP公式ドキュメント](https://modelcontextprotocol.io/)
- [GitHub MCP Server](https://github.com/modelcontextprotocol/servers/tree/main/src/github)
- [Notion MCP Server](https://github.com/notionhq/notion-mcp-server)
- [Serena](https://github.com/oraios/serena)
- [Context7](https://context7.ai)

## 🆘 サポート

問題が解決しない場合は、以下を確認してください：
1. Claude Codeのバージョンが最新か
2. Node.js/npm/uvxがインストールされているか
3. ネットワーク接続が正常か

---

最終更新: 2025-11-05
