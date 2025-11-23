# Jupyter Notebook 品質保証ルール

> **目的**: Notebookのエラーを防ぎ、常に実行可能な状態を保つ

---

## 🚨 最重要ルール

### ルール1: 作成・修正後は必ず "Restart & Run All"

**原則**: Notebookを作成・修正したら、**必ず全セルを最初から実行して確認**

```bash
# Jupyter Notebookで
Kernel > Restart & Run All

# 全セルが上から順にエラーなく実行されることを確認
```

**適用タイミング**:
- [ ] Notebook新規作成後
- [ ] セルの追加・修正後
- [ ] インポート文の変更後
- [ ] パス設定の変更後
- [ ] Git commit前（必須）

**理由**:
- セルの実行順序が狂っていてもエラーに気づかない
- 変数が前のセルに依存していても気づかない
- パス設定が不足していても気づかない

---

### ルール2: Notebookは出力付きでGit管理

**原則**: Notebookは**セル実行済み・出力表示状態**でcommit

```bash
# ❌ 出力をクリアしてcommit（禁止）
Kernel > Restart & Clear Output
git add notebook.ipynb
git commit

# ✅ 出力を残してcommit（推奨）
Kernel > Restart & Run All  # 全セル実行
# エラーがないことを確認
git add notebook.ipynb
git commit
```

**メリット**:
1. **即座にエラー検出**: GitHubで見たときにエラーが一目瞭然
2. **実行結果の記録**: どんな出力が出るか事前に分かる
3. **レビュー容易**: コードと結果を同時に確認可能

**例外**:
- 出力が巨大すぎる場合（画像多数、大量ログ）
- 個人情報が含まれる場合

---

### ルール3: パス設定は統一パターンを使用

**原則**: すべてのNotebookで同じパス設定パターンを使う

```python
# ✅ 標準パターン（必ずこれを使う）
import sys
from pathlib import Path

# プロジェクトルートを特定
project_root = Path().resolve().parent.parent

# 必要なパスをすべて追加
sys.path.insert(0, str(project_root / "04_src"))  # EDA関数用
sys.path.insert(0, str(project_root / "src"))     # DataLoader等用

# 作業ディレクトリをプロジェクトルートに変更
import os
os.chdir(project_root)
```

**禁止パターン**:
```python
# ❌ パスを一部だけ追加
sys.path.insert(0, str(project_root / "04_src"))
# src/ を追加し忘れ → from src.data.loader でエラー

# ❌ ハードコーディング
sys.path.insert(0, "/Users/kotaro/Desktop/ML/project/src")

# ❌ 相対パス（実行場所に依存）
sys.path.insert(0, "../../src")
```

**理由**:
- パス不足でインポートエラー
- 環境依存で他の人が実行できない

---

### ルール4: NotebookEditツール使用後は必ず確認

**原則**: NotebookEditで修正した後、**必ず実際のNotebookで確認**

```bash
# 1. NotebookEditツールで修正
NotebookEdit(notebook_path, cell_id, new_source)

# 2. Jupyter Notebookで開いて確認（必須）
jupyter notebook

# 3. 該当セルを実行してエラーがないか確認
# 4. 問題なければ全セル実行
Kernel > Restart & Run All
```

**理由**:
- NotebookEditは正しく動作しても、セル内容が意図通りか分からない
- セルIDが間違っていても気づかない
- 複数セルの依存関係が壊れていても気づかない

---

## 📝 Notebook作成チェックリスト

### 新規Notebook作成時

- [ ] **1. セットアップセル作成**
  ```python
  # パス設定（標準パターン）
  import sys
  from pathlib import Path
  project_root = Path().resolve().parent.parent
  sys.path.insert(0, str(project_root / "04_src"))
  sys.path.insert(0, str(project_root / "src"))
  import os
  os.chdir(project_root)
  ```

- [ ] **2. インポート確認**
  ```python
  # 各セルを実行してImportErrorがないか確認
  from eda.correlation import calculate_correlations  # OK?
  from data.loader import DataLoader  # OK?
  ```

- [ ] **3. データ読み込み確認**
  ```python
  # DataLoaderパターンが動くか確認
  from data.loader import DataLoader
  from utils.config import load_config
  loader = DataLoader(load_config("data"))
  train = loader.load_train()
  print(train.shape)  # 出力を確認
  ```

- [ ] **4. 全セル実行**
  ```
  Kernel > Restart & Run All
  ```

- [ ] **5. エラーチェック**
  - すべてのセルが緑チェックマーク ✅
  - エラー出力がない
  - 警告は許容範囲内

- [ ] **6. Git commit前の最終確認**
  ```bash
  # もう一度全セル実行
  Kernel > Restart & Run All
  # 問題なし → commit
  git add notebook.ipynb
  git commit
  ```

---

## 🔍 エラー発生時の対処

### よくあるエラーと原因

#### 1. ModuleNotFoundError: No module named 'src'

**原因**: `src/`パスを追加していない

**修正**:
```python
# cell-2 (セットアップ)に追加
sys.path.insert(0, str(project_root / "src"))
```

#### 2. ModuleNotFoundError: No module named 'eda'

**原因**: `04_src/`パスを追加していない

**修正**:
```python
# cell-2 (セットアップ)に追加
sys.path.insert(0, str(project_root / "04_src"))
```

#### 3. NameError: name 'project_root' is not defined

**原因**: セットアップセルを実行していない

**修正**:
```
Kernel > Restart & Run All
# 最初から順に実行
```

#### 4. FileNotFoundError: data/processed/train.parquet

**原因**: 作業ディレクトリが間違っている

**修正**:
```python
# cell-2 (セットアップ)に追加
import os
os.chdir(project_root)
```

---

## 🤝 Claude Codeとの協働ルール

### Claude Codeが守ること

1. **Notebook作成後は必ず動作確認**
   - Restart & Run All を実行
   - エラーがあれば修正
   - 修正後も再度確認

2. **NotebookEditツール使用後は再確認**
   - 実際のNotebookで該当セルを実行
   - 全体への影響を確認

3. **Todoリストの"動作確認"を無視しない**
   - 「Notebookの動作確認」がPendingなら必ず実施
   - 完了してからcommit/push

4. **出力付きでcommit**
   - セル実行済み状態でGit管理
   - エラーが見える状態で記録

### ユーザーに確認を求めるタイミング

- [ ] Notebook作成完了時: "Restart & Run Allで確認しましたが、ブラウザで確認していただけますか？"
- [ ] エラー修正後: "修正しました。もう一度実行して確認していただけますか？"
- [ ] commit前: "すべてのNotebookが正常に実行できることを確認しました。commitしてよろしいですか？"

---

## 📊 品質保証プロセス

### Phase 1: .pyファイル実装（TDD）

```
1. 仕様書作成 (01_specs/)
2. テスト作成 (07_tests/)
3. 実装 (04_src/)
4. pytest実行 → Pass確認
```

### Phase 2: Notebook作成

```
1. Notebookファイル作成
2. セットアップセル作成（標準パターン）
3. インポート確認セル追加
4. 各セルを順次実行して確認
5. Restart & Run All
```

### Phase 3: 動作確認（必須）

```
1. Kernel > Restart & Run All
2. すべてのセルが✅
3. エラー出力なし
4. 出力内容が妥当
```

### Phase 4: Git commit

```
1. 最終確認: Restart & Run All
2. git add (出力付き状態)
3. git commit
4. git push
```

---

## ❌ 今回の失敗ケース分析

### 何が起きたか

1. ✅ TDD完了（pytest Pass）
2. ✅ Notebook作成
3. ❌ **Restart & Run All せずcommit**
4. ❌ **Todoの"動作確認"を無視**
5. ❌ **NotebookEditで修正したつもりが不完全**
6. ❌ **修正後も実行確認せず**

### なぜ起きたか

- **ルールが不明確**: "Notebookは必ず実行確認"が明文化されていなかった
- **Todoを無視**: "新しいNotebookの動作確認"がPendingだったのに放置
- **ツールへの過信**: NotebookEditツールで修正=完了と勘違い

### 学び

- **ルール明文化**: このドキュメントを作成
- **チェックリスト化**: 確認項目を明示
- **出力付きcommit**: GitHubで即座にエラー検出可能に

---

## 📚 関連ドキュメント

- [project_rules.md](./.claude/project_rules.md) - プロジェクト固有ルール
- [general_rules.md](./.claude/general_rules.md) - TDD/SDD等の汎用ルール
- [notebook_tdd_guide.md](./.claude/notebook_tdd_guide.md) - Notebook TDD実践ガイド

---

**最終更新**: 2025-11-23
