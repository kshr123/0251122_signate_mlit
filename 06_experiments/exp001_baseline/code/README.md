# 実験コード - exp001_baseline

このディレクトリには、ベースライン実験の実行コードが含まれています。

## ファイル構成

```
code/
├── README.md              # このファイル
├── train.py               # 訓練スクリプト
├── predict.py             # 推論スクリプト
├── preprocessing.py       # 前処理（実験固有・明示的）
└── feature_selection.py   # 特徴量選択ロジック
```

## 🎯 重要: 実験固有の前処理について

この実験では、**特徴量エンジニアリングの内容を明示的に記述**しています。

- `preprocessing.py`: 106個の特徴量を明示的にリスト化し、前処理ロジックを記述
- `feature_selection.py`: 149個の元特徴量から106個を選択した過程を記録

`04_src/preprocessing/simple.py`のような抽象化されたクラスは使用せず、
**このディレクトリを見れば実験内容が完全に理解できる**ように設計しています。

### preprocessing.py の特徴

```python
# 数値特徴量（96個）を明示的にリスト化
NUMERIC_FEATURES = [
    "building_id", "building_status", "building_type", ...
]

# カテゴリカル特徴量（8個）を明示的にリスト化
CATEGORICAL_FEATURES = [
    "building_name_ruby", "reform_exterior", ...
]

# 前処理の各ステップを明示的に記述
def preprocess_for_training(train, test):
    # 1. target_ym 分解
    # 2. 特徴量選択
    # 3. データ型統一（traffic_car問題対処）
    # 4. カテゴリカルエンコーディング
    # 5. NumPy配列変換
    ...
```

---

## 🚀 使い方

### 訓練

```bash
# プロジェクトルートから実行
cd 06_experiments/exp001_baseline/code
python train.py
```

**注意**: `train.py`内でプロジェクトルートを参照しているため、
必ず`06_experiments/exp001_baseline/code/`ディレクトリから実行してください。

### 推論

```bash
cd 06_experiments/exp001_baseline/code
python predict.py --model ../models/final_model.txt --output ../outputs/submission_new.csv
```

### 特徴量選択の確認（オプション）

```bash
cd 06_experiments/exp001_baseline/code
python feature_selection.py
```

このコマンドで、149個の元特徴量から106個を選択した過程を確認できます。

---

## 📦 依存関係

- Python 3.13+
- lightgbm
- polars
- scikit-learn
- mlflow
- numpy

プロジェクトルートの仮想環境を使用してください。

---

## 🔄 再現方法

### 完全再現

1. プロジェクトルートで仮想環境を有効化
   ```bash
   cd /Users/kotaro/Desktop/ML/20251122_signamte_mlit
   source .venv/bin/activate
   ```

2. 実験ディレクトリに移動
   ```bash
   cd 06_experiments/exp001_baseline/code
   ```

3. 訓練を実行
   ```bash
   python train.py
   ```

4. 結果確認
   - `outputs/cv_scores.json` - CV結果
   - `outputs/metrics.json` - メトリクス
   - `outputs/submission_*.csv` - 提出ファイル

### 推論のみ実行

学習済みモデルがある場合:

```bash
cd 06_experiments/exp001_baseline/code
python predict.py --model ../models/final_model.txt
```

---

## 💡 設計方針

### なぜpreprocessing.pyを作ったか？

元々は`04_src/preprocessing/simple.py`の`SimplePreprocessor`を使用していましたが、
これには以下の問題がありました:

1. **抽象化されすぎて何をやっているか不明**
   - 後から実験ディレクトリを見ても、どの特徴量を使ったか分からない
   - どんな前処理をしたか追跡が困難

2. **実験の再現性が低い**
   - `SimplePreprocessor`の実装が変わると過去の実験が再現できない

### 解決策

`preprocessing.py`に**この実験で使った特徴量と前処理を全て明示的に記述**:

- 96個の数値特徴量を全てリスト化
- 8個のカテゴリカル特徴量を全てリスト化
- 前処理の各ステップを明示的に記述
- データ型問題（traffic_car）の対処も明記

→ **このファイルを見れば実験内容が完全に理解できる**

### 共通コンポーネントとの使い分け

- **04_src/**（共通コンポーネント）を使うもの:
  - `DataLoader` - データ読み込み
  - `set_seed` - シード固定
  - `calculate_mape` - メトリクス計算
  - `mlflow_helper` - MLflow補助関数

- **code/**（実験固有）に記述するもの:
  - **特徴量エンジニアリング** ← 最重要
  - 実験固有のパラメータ
  - 訓練・推論ロジック

---

## 🔧 トラブルシューティング

### ModuleNotFoundError

```
ModuleNotFoundError: No module named 'src'
```

→ `06_experiments/exp001_baseline/code/`から実行してください。

### データ型エラー

```
TypeError: ... Int64 vs String
```

→ `preprocessing.py`で train/test の型不一致を自動処理しています。
   問題が発生した場合は`preprocessing.py`の型変換ロジックを確認してください。

---

## 📚 参考

- 実験の詳細: `../README.md`
- 特徴量の詳細: `../features/feature_engineering.md`
- 設定ファイル: `../configs/*.yaml`

---

**作成日**: 2025-11-24
**実験**: exp001_baseline
