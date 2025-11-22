# テスト実行クイックリファレンス

## ⚡ よく使うコマンド

```bash
# 全テスト実行
./08_scripts/run_tests.sh

# 特定モジュールのテスト
./08_scripts/run_tests.sh test_data
./08_scripts/run_tests.sh test_utils

# カバレッジ付き
./08_scripts/run_tests.sh test_data --coverage

# 最新結果を確認
cat 07_tests/test_data/test_results/latest_result.txt
cat 07_tests/test_data/test_results/latest_coverage.txt
```

## 📂 結果ファイルの場所

| テストモジュール | 結果ディレクトリ |
|----------------|----------------|
| test_data | `07_tests/test_data/test_results/` |
| test_utils | `07_tests/test_utils/test_results/` |
| test_eda | `07_tests/test_eda/test_results/` |
| test_preprocessing | `07_tests/test_preprocessing/test_results/` |
| test_features | `07_tests/test_features/test_results/` |
| test_models | `07_tests/test_models/test_results/` |
| test_training | `07_tests/test_training/test_results/` |
| test_evaluation | `07_tests/test_evaluation/test_results/` |

## 📝 ファイル命名規則

- **テスト結果**: `test_result_YYYYMMDD_HHMMSS.txt`
- **カバレッジ**: `coverage_YYYYMMDD_HHMMSS.txt`
- **最新リンク**: `latest_result.txt`, `latest_coverage.txt`

## 🎯 結果の読み方

### ✅ 成功時
```
テスト状態: ✓ PASSED
```
→ 問題なし。開発続行。

### ❌ 失敗時
```
テスト状態: ✗ FAILED
```
→ 結果ファイルを開いてエラー内容を確認。

### 📊 カバレッジ
```
TOTAL    83    2    98%
```
→ 目標: 80%以上

## 🔧 トラブルシューティング

### テストが実行できない
```bash
# 仮想環境が有効化されているか確認
which python
# → /Users/.../20251122_signamte_mlit/.venv/bin/python

# 有効化されていない場合
source .venv/bin/activate
```

### pytest-covが見つからない
```bash
uv pip install pytest-cov
```

### 結果ファイルが見つからない
```bash
# test_resultsディレクトリを確認
ls 07_tests/test_data/test_results/

# 最新のタイムスタンプファイルを探す
ls -lt 07_tests/test_data/test_results/ | head
```

---

**詳細**: [07_tests/README.md](./README.md)
