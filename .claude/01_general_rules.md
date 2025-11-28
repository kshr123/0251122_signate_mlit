# 汎用開発ルール

> どのプロジェクトでも使える汎用的な開発ルール

---

## 📝 開発フロー（SDD + TDD）

```
1. 理解     → 既存コード・要件を把握
2. 仕様策定 → SPECIFICATION.md作成（SDD）
3. テスト   → 失敗するテストを書く（Red）
4. 実装     → テストを通す最小限のコード（Green）
5. リファクタ → コード改善（Refactor）
6. 検証     → 動作確認・成功基準チェック
7. 振り返り → ドキュメント更新・Git操作
8. クリーンアップ → バックグラウンドプロセス・Docker停止
```

---

## 💻 コーディング規約

### 基本原則
- **可読性**: 読みやすさ優先
- **シンプルさ**: 複雑さを避ける
- **ライブラリ優先**: 自作より既存ライブラリを使う

### Python
- PEP 8準拠、型ヒント使用
- 命名: `snake_case`（関数/変数）、`PascalCase`（クラス）、`UPPER_CASE`（定数）

---

## 🔒 セキュリティ

### 禁止事項
- 個人情報（実名、メールアドレス、電話番号）をコードに含めない
- APIキー、パスワードをハードコードしない
- 絶対パスに個人名を含めない（`/Users/username/`は`~/`で表記）

### 必須対応
```gitignore
# .gitignoreに必ず追加
.env
*.pem
*.key
credentials.json
.DS_Store
__pycache__/
```

---

## 🧹 Git Push前チェック

```bash
# 1. 不要ファイル削除
find . -name ".DS_Store" -delete
find . -name "*_tmp.*" -delete

# 2. 状態確認
git status

# 3. コミット・プッシュ
git add .
git commit -m "メッセージ"
git push  # ユーザー確認後
```

---

## 🧪 テスト結果の管理

- **Git管理しない**: `**/test_results/`、`__pycache__/`、`.coverage`
- **Git管理する**: `tests/README.md`、`.gitkeep`

```bash
# テスト実行（結果をファイルに保存）
pytest tests/ -v > tests/test_results/result_$(date +%Y%m%d_%H%M%S).txt 2>&1
```

---

## 🤝 Claude Codeルール

1. **日本語で返答**（絶対厳守）
2. **仕様駆動**: 実装前に仕様を確認
3. **テスト駆動**: Red → Green → Refactor
4. **段階的実装**: 小さなステップで進める
5. **Git push前**: 不要ファイル削除、ユーザー確認

---

## 🔌 MCP連携

- **GitHub MCP**: リポジトリ操作
- **Notion MCP**: ドキュメント管理
- **Serena MCP**: コード分析
- **Context7 MCP**: ライブラリドキュメント参照

設定ファイル（`.mcp.json`）は必ず`.gitignore`に追加。

---

**最終更新**: 2025-11-27
