# 実験管理ルール

> 実験の再現性を保証するためのディレクトリ構造とファイル管理ルール

---

## 📁 実験ディレクトリ構造

```
06_experiments/expXXX_name/
├── README.md               # 実験サマリー（必須）
├── SPEC.md                 # 実験仕様書
├── configs/
│   └── experiment.yaml     # 全ハイパーパラメータ（必須）
├── code/                   # フラット構造（サブディレクトリ禁止）
│   ├── train.py            # エントリーポイント
│   ├── preprocessing.py    # 前処理
│   ├── pipeline.py         # 特徴量パイプライン
│   ├── constants.py        # パス定義・カラムリスト（ハイパラ含まない）
│   ├── expXXX_features.py  # 実験固有Block・関数
│   └── objectives.py       # カスタム損失関数（必要時のみ）
├── notebooks/              # 実験固有の分析ノートブック
│   └── analysis.ipynb      # 結果分析・可視化
├── outputs/                # Git管理外
│   └── run_{objective}_{YYYYMMDD_HHMMSS}/
│       ├── submission.csv
│       ├── oof_predictions.csv
│       ├── test_predictions.csv  # アンサンブル用（生の予測値）
│       ├── feature_importance.json
│       ├── X_train.parquet       # 前処理済み特徴量（再利用用）
│       ├── X_test.parquet
│       └── y_train.parquet
└── mlruns/                 # Git管理外
```

---

## ⚙️ 設定ファイル構成

### experiment.yaml（ハイパーパラメータ）

```yaml
experiment:     # 実験メタ情報
  id: "expXXX"
  name: "experiment_name"
  base: "expYYY"  # ベース実験

training:       # 学習設定
  seed: 42
  n_splits: 3
  early_stopping_rounds: 200
  target_transform: "log1p"

model:          # モデルパラメータ
  type: "lightgbm"
  params:           # 本番用
    learning_rate: 0.05
    n_estimators: 50000
    # ...
  params_test:      # テスト用（--test フラグ）
    learning_rate: 0.5
    n_estimators: 10
    # ...

features:       # 特徴量パラメータ
  tfidf:
    max_features: 20
  geo_pca:
    n_components: 2
  # ...

loss:           # 損失関数設定（複数目的関数を扱う場合）
  objectives: ["mse", "huber", "quantile"]  # 実行する目的関数リスト
  huber:
    alpha: 1.0
  quantile:
    alpha: 0.5
  sample_weight:
    transform: "none"  # none, inverse, sqrt_inverse, log_inverse, threshold

expXXX:         # 実験固有設定
  thresholds:
    # ...
```

### constants.py（固定値のみ）

```python
# パス定義
LANDPRICE_BASE_PATH = Path("data/external/landprice")

# カラムリスト
NUMERIC_COLUMNS = ["house_area", "unit_area", ...]
CATEGORICAL_COLUMNS = ["addr1_1", "addr1_2", ...]
```

**原則**: ハイパーパラメータは全て `experiment.yaml` に集約。`constants.py` にはパスとカラムリストのみ。

---

## 🔧 インポートパターン

```python
import sys
from pathlib import Path
import yaml

# パス設定
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "04_src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# インポート
from constants import LANDPRICE_BASE_PATH       # パス・カラムリスト
from exp010_features import PostalCodeTEBlock   # 実験固有Block
from features.blocks.encoding import TargetEncodingBlock  # 04_src

# 設定読み込み
def load_config(test_mode: bool = False) -> dict:
    config_path = Path(__file__).parent.parent / "configs" / "experiment.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if test_mode:
        config["model"]["params"] = config["model"]["params_test"]
    return config
```

**禁止**:
- `from .config import ...`（相対インポート）
- ファイル名 `features.py`（04_src/features/ と衝突）
- `constants.py` にハイパーパラメータを書く

---

## 📋 04_src との使い分け

| 場所 | 用途 | 例 |
|------|------|-----|
| 04_src/features/ | 汎用Block | TfidfBlock, PCABlock |
| code/pipeline.py | Blockの組み合わせ | 04_srcのBlockを直接使用 |
| code/expXXX_features.py | 実験固有Block | PostalCodeTEBlock |

**原則**:
- 04_srcのBlockをそのまま使う（推奨）
- 対応できない場合のみ実験固有Blockを作成

---

## 🔄 実験実行フロー

> ⚠️ **必須**:
> - 学習実行時は必ず `caffeinate -i` を付けること（macOSスリープ防止）
> - 環境変数は `env` コマンドで設定すること（`caffeinate` との組み合わせで必須）

```bash
# 1. 前の実験をコピー
cp -r 06_experiments/exp009_name 06_experiments/exp010_name

# 2. outputs/, mlruns/ を削除
rm -rf 06_experiments/exp010_name/outputs/* 06_experiments/exp010_name/mlruns/*

# 3. experiment.yaml の experiment.id, experiment.name を更新

# 4. テスト実行（軽量: params_test を使用）
cd 06_experiments/exp010_name
caffeinate -i env PYTHONPATH=../../04_src:code python code/train.py --test

# 5. 本番実行
caffeinate -i env PYTHONPATH=../../04_src:code python code/train.py
```

**コマンド構文**:
```bash
# ✅ 正しい: env を使う
caffeinate -i env PYTHONPATH=../../04_src:code python code/train.py

# ❌ 間違い: env なし（"No such file or directory" エラー）
caffeinate -i PYTHONPATH=../../04_src:code python code/train.py
```

**caffeinate オプション**:
- `-i`: システムアイドルスリープを防止（必須）
- `-d`: ディスプレイスリープも防止（オプション）
- `-s`: システムスリープを防止（AC電源時のみ）

### CLI オプション（train.py）

```bash
# 基本オプション
python code/train.py --test              # テストモード（params_test使用）
python code/train.py --objective huber   # 目的関数指定

# 特徴量キャッシュ（複数目的関数実行時に有効）
python code/train.py --objective mse                           # 1回目: 特徴量計算＋保存
python code/train.py --objective huber --features-dir outputs/run_mse_xxx  # 2回目以降: 再利用

# Quantile回帰
python code/train.py --objective quantile --alpha 0.3
```

**特徴量キャッシュの利点**:
- 前処理時間を大幅に削減（数分 → 数秒）
- 同一特徴量で異なる損失関数を比較可能
- outputs/run_*/に保存されたparquetを再利用

---

## 📦 Git管理

**含める**:
- `README.md`, `SPEC.md`
- `configs/experiment.yaml`
- `code/` 全体
- `notebooks/` 全体（出力付きでcommit）

**含めない**:
- `outputs/`
- `mlruns/`
- `__pycache__/`

**code/README.md 記載内容**:
- ファイル構成
- 設定の分離方針（experiment.yaml vs constants.py）
- 依存関係
- 使用Block一覧
- expXXX_features.py 内容

（参考: `exp010_low_price_improvement/code/README.md`）

---

## 📝 README テンプレート

### 実験ルート README.md

```markdown
# expXXX_name

説明文

## 概要
| 項目 | 内容 |
|------|------|
| 実験ID | expXXX |
| ベース | expYYY (CV MAPE: XX.XX%) |
| 目的 | ... |

## クイックスタート
（実行コマンド）

## ディレクトリ構成
（configs/, code/ を含む）

## 設定ファイル構成
（experiment.yaml と constants.py の役割）

## 実験結果
（CV MAPE、特徴量重要度）
```

---

**最終更新**: 2025-11-29
