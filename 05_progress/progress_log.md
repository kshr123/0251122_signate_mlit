# プロジェクト進捗ログ

> **このドキュメントについて**: プロジェクトの進捗を時系列で記録します。

---

## プロジェクト情報

- **コンペ名**: SIGNATE - 賃貸物件の家賃予測
- **コンペURL**: https://signate.jp/competitions/1365
- **開始日**: 2025-11-22
- **締切**: 未確認
- **現在のフェーズ**: Phase 2完了 - ベースラインモデル構築完了（CV MAPE: 28.34%）

---

## 進捗サマリー

### ✅ 完了したタスク

1. **プロジェクトセットアップ** (2025-11-22)
   - ディレクトリ構造作成
   - Python 3.13環境構築
   - 基本ライブラリインストール（Polars、scikit-learn等）

2. **データ理解・EDA** (2025-11-23)
   - データ読み込み機能実装（DataLoader）
   - 基本統計量確認
   - 都道府県×年別分析ノートブック作成

3. **特徴量エンジニアリング基盤** (2025-11-24)
   - コーディングルール策定（`.claude/feature_engineering_rules.md`）
   - ベースライン戦略決定（数値データ + 低カーディナリティ）
   - アーキテクチャ設計（BaseBlock、FeaturePipeline、SeedManager）

4. **MLflow実験管理基盤** (2025-11-24)
   - MLflow仕様書作成（`01_specs/mlflow_experiment.md`）
   - ヘルパー関数実装（`04_src/training/utils/mlflow_helper.py`）
   - テストコード作成（7/7通過）
   - 初心者向け使い方ガイド作成（`02_docs/mlflow_usage_guide.md`）
   - 依存関係追加（mlflow 3.6.0、lightgbm 4.6.0）

5. **評価モジュール実装** (2025-11-24)
   - 仕様書作成（feature_importance.md、error_analysis.md、evaluation_visualizer.md）
   - feature_importance.py実装（TDD、8/8テスト通過）
   - error_analysis.py実装（TDD、9/9テスト通過）

6. **ベースラインモデル構築** (2025-11-24) ✅
   - SimplePreprocessor実装（TDD、8/8テスト通過）
   - ベースライン訓練スクリプト作成（`04_src/training/train_baseline.py`）
   - **CV MAPE: 28.3432% ± 0.0883%**
   - 提出ファイル生成（`06_experiments/exp001_baseline/submission_20251124_122920.csv`）
   - 実験ノート作成（`06_experiments/exp001_baseline/README.md`）
   - MLflow Run ID: b1541b503505448d8567f82d22166a1d

### 🔄 進行中のタスク

なし

### 📋 予定タスク

1. **初回提出**
   - 提出ファイルの妥当性確認
   - SIGNATEへ提出
   - リーダーボードスコア確認

2. **モデル改善**
   - 特徴量追加
   - ハイパーパラメータチューニング
   - モデルアンサンブル

---

## 詳細ログ

### 2025-11-24: MLflow実験管理基盤の構築

#### 実施内容

**1. 特徴量エンジニアリングルール策定**
- ファイル: `.claude/feature_engineering_rules.md`
- 内容:
  - 基本方針（Polarsファースト、不変性、データリーク防止、再現性）
  - アーキテクチャ設計（BaseBlock、FeaturePipeline、SeedManager）
  - ベースライン戦略（数値データそのまま、低カーディナリティラベルエンコーディング）
  - 命名規則、テスト方針

**2. MLflow仕様書作成**
- ファイル: `01_specs/mlflow_experiment.md`
- 内容:
  - 記録するもの（パラメータ、メトリクス、アーティファクト、タグ）
  - 実装パターン（ベースライン訓練スクリプト、ヘルパー関数）
  - MLflow UI使用方法
  - 成功基準
- 設計方針: ローカルファイルベース（SQLite・REST API不使用）

**3. MLflowヘルパー関数実装**
- ファイル: `04_src/training/utils/mlflow_helper.py`
- 実装した関数:
  - `log_dataset_info()`: データセット基本情報・欠損値情報を記録
  - `log_cv_results()`: CV統計量・Fold別スコアを記録
  - `log_feature_list()`: 使用特徴量リストをアーティファクトとして保存
  - `log_model_params()`: モデルパラメータを記録（ネスト対応）

**4. TDDによるテスト実装**
- ファイル: `07_tests/test_training/test_mlflow_helper.py`
- テスト内容（7件、全通過）:
  - データセット基本メトリクス記録
  - 欠損値情報記録
  - CV統計量記録
  - Fold別スコア記録
  - 特徴量リストアーティファクト作成
  - 単純パラメータ記録
  - ネストパラメータ記録

**5. MLflow使い方ガイド作成**
- ファイル: `02_docs/mlflow_usage_guide.md`
- 内容:
  - MLflowの概要と目的
  - 基本的な使い方（実験実行、UI起動）
  - MLflow UI操作方法
  - 実験結果の見方（単一実験、比較、ベストモデル特定）
  - FAQ（6項目）

#### 技術的な決定事項

- **MLflowストレージ**: ローカルファイルベース（`mlruns/`）
  - SQLite等のDBは不要
  - チーム開発時はリモートトラッキングサーバーを検討

- **ベースライン戦略**:
  - 数値データ: そのまま使用
  - カテゴリカルデータ: 低カーディナリティ（<50）のみラベルエンコーディング
  - 時系列データ: target_ymを年・月に分解
  - モデル: LightGBM（デフォルトパラメータ）
  - 検証: Time-Series Split

- **再現性確保**:
  - SeedManagerで以下を管理:
    - Python標準ライブラリ（`random`）
    - NumPy
    - Polars
    - PyTorch
    - PYTHONHASHSEED

#### 依存関係追加

```bash
uv pip install mlflow lightgbm
```

- mlflow 3.6.0
- lightgbm 4.6.0
- その他48個の依存パッケージ

#### 成果物

- ドキュメント: 3ファイル
  - `.claude/feature_engineering_rules.md`（更新）
  - `01_specs/mlflow_experiment.md`（新規）
  - `02_docs/mlflow_usage_guide.md`（新規）

- ソースコード: 1ファイル
  - `04_src/training/utils/mlflow_helper.py`（新規）

- テストコード: 1ファイル
  - `07_tests/test_training/test_mlflow_helper.py`（新規）

#### 次のステップ

---

### 2025-11-24 (午後): ベースラインモデル構築と初回提出準備

#### 実施内容

**1. 評価モジュール実装（TDD）**

**feature_importance.py**:
- ファイル: `04_src/evaluation/feature_importance.py`
- テスト: `07_tests/test_evaluation/test_feature_importance.py`（8/8通過）
- 実装機能:
  - `FeatureImportanceAnalyzer` クラス
  - `calculate_importance()`: gain/split重要度計算
  - `calculate_permutation_importance()`: Permutation Importance計算
  - `get_top_features()`: 上位N件取得
  - `compare_importance_types()`: 複数タイプ比較
- **技術的工夫**: LightGBM BoosterをLGBMWrapperでラップしてsklearn互換化

**error_analysis.py**:
- ファイル: `04_src/evaluation/error_analysis.py`
- テスト: `07_tests/test_evaluation/test_error_analysis.py`（9/9通過）
- 実装機能:
  - `ErrorAnalyzer` クラス
  - `calculate_metrics()`: MAPE/RMSE/MAE計算
  - `get_residual_stats()`: 残差統計量取得
  - `analyze_by_segment()`: セグメント別分析
  - `find_outliers()`: 外れ値検出（標準偏差・パーセンタイル）
  - `get_outlier_details()`: 外れ値詳細
  - `analyze_by_feature_bins()`: 特徴量ビニング分析
- **技術的工夫**: Polars qcutでのビニング時、`allow_duplicates=True` + `to_physical()` で数値化

**2. ベースライン訓練スクリプト作成**

- ファイル: `04_src/training/train_baseline.py`
- 実装内容:
  - LightGBM + 3-Fold CV
  - SimplePreprocessor使用（数値 + 低カーディナリティ）
  - MLflowへの自動記録
  - 提出ファイル生成

**3. 訓練実行と問題解決**

**遭遇した問題**:
```
ValueError: pandas dtypes must be int, float or bool.
Fields with bad pandas dtypes: traffic_car: object
```

**原因**:
- Train: `traffic_car` が `Int64` 型
- Test: `traffic_car` が `String` 型
- 元データの型が異なるため、SimplePreprocessorで異なる扱いを受ける

**解決策**:
trainとtestの両方で文字列型カラムを検出し、すべてをCategorical → ordinalに変換:

```python
# trainとtestで型が異なる可能性があるため、両方で文字列型を検出
string_cols_train = [col for col in X_train.columns if X_train[col].dtype == pl.Utf8]
string_cols_test = [col for col in X_test.columns if X_test[col].dtype == pl.Utf8]
string_cols = list(set(string_cols_train + string_cols_test))

# すべての文字列型カラムを数値に変換
for col in string_cols:
    if col in X_train.columns and X_train[col].dtype == pl.Utf8:
        X_train = X_train.with_columns(
            pl.col(col).cast(pl.Categorical).to_physical().alias(col)
        )
    if col in X_test.columns and X_test[col].dtype == pl.Utf8:
        X_test = X_test.with_columns(
            pl.col(col).cast(pl.Categorical).to_physical().alias(col)
        )
```

#### 実験結果（exp001_baseline）

**CV結果**:
- **MAPE**: **28.3432% ± 0.0883%**
- Min MAPE: 28.2762%
- Max MAPE: 28.4680%

**Fold別スコア**:
| Fold | MAPE (%) | Best Iteration |
|------|----------|----------------|
| 1    | 28.4680  | 100            |
| 2    | 28.2762  | 100            |
| 3    | 28.2854  | 100            |

**データ**:
- Train: 363,924 samples × 149 features
- Test: 112,437 samples × 149 features
- 使用特徴量: 106（数値96 + カテゴリカル8 + target_year/month）

**ハイパーパラメータ**:
- learning_rate: 0.05
- num_leaves: 31
- subsample: 0.8
- colsample_bytree: 0.8
- num_boost_round: 100

**観察事項**:
- すべてのFoldで `best_iteration=100` → Early Stopping未発動
- CV標準偏差が小さい（0.0883%）→ モデルが安定している
- `num_boost_round` を増やす余地あり

#### 成果物

- ソースコード: 3ファイル
  - `04_src/evaluation/feature_importance.py`（新規）
  - `04_src/evaluation/error_analysis.py`（新規）
  - `04_src/training/train_baseline.py`（新規）

- テストコード: 2ファイル
  - `07_tests/test_evaluation/test_feature_importance.py`（新規）
  - `07_tests/test_evaluation/test_error_analysis.py`（新規）

- 実験成果物:
  - 提出ファイル: `06_experiments/exp001_baseline/submission_20251124_122920.csv`
  - 実験ノート: `06_experiments/exp001_baseline/README.md`
  - MLflow Run ID: `b1541b503505448d8567f82d22166a1d`

#### 技術的な決定事項

- **データ型の不整合対応**: train/testで型が異なるカラムの自動検出・変換
- **ベースライン完成**: これ以降の実験との比較基準を確立
- **Early Stopping未発動**: より多くの反復が有効な可能性 → 次回実験で調整

#### 次のステップ

1. **初回提出**: 提出ファイルの妥当性確認 → SIGNATEへ提出
2. **特徴量追加**: 住所情報（都道府県・市区町村名）
3. **ハイパーパラメータチューニング**: Optunaで最適化
4. **モデル改善**: num_boost_round増加、アンサンブル
5. **リファクタリング**: DataLoaderでのデータ型統一
3. MLflow記録の動作確認
4. 初回提出

---

### 2025-11-23: データ理解・EDA

#### 実施内容

**1. DataLoader実装**
- ファイル: `04_src/data/loader.py`
- 機能:
  - train.csv / test.csvの読み込み
  - prefecture_code / city_codeから都道府県名・市区町村名の追加（オプション）
  - Polarsベース

**2. 都道府県×年別分析ノートブック作成**
- ファイル: `05_notebooks/01_eda/prefecture_year_analysis.ipynb`
- 内容:
  - 都道府県別の物件数分析
  - 年別の物件数推移
  - 都道府県×年のクロス集計
  - 可視化（棒グラフ、ヒートマップ）

#### 成果物

- ノートブック: 1ファイル
- ソースコード: 1ファイル
- テストコード: 1ファイル

---

### 2025-11-22: プロジェクトセットアップ

#### 実施内容

**1. ディレクトリ構造作成**
```
data/{raw,processed,external}
05_notebooks/{01_eda,02_feature,03_modeling,04_evaluation}
04_src/{data,eda,preprocessing,features,models,training,evaluation,utils}
07_tests/
01_specs/
02_docs/
03_configs/
06_submissions/
.claude/
```

**2. Python環境構築**
- Python 3.13
- uv仮想環境

**3. 基本ライブラリインストール**
- Polars
- NumPy
- scikit-learn
- matplotlib
- seaborn
- pytest
- black
- ruff
- mypy

**4. プロジェクトルール策定**
- `.claude/claude.md`（プロジェクト概要）
- `.claude/general_rules.md`（汎用開発ルール）
- `.claude/project_rules.md`（プロジェクト固有ルール）

#### 成果物

- ドキュメント: 3ファイル
- ディレクトリ構造完成

---

## メトリクス

### コードメトリクス

- **ソースコード**: 2ファイル
  - `04_src/data/loader.py`
  - `04_src/training/utils/mlflow_helper.py`

- **テストコード**: 2ファイル
  - `07_tests/test_data/test_loader.py`
  - `07_tests/test_training/test_mlflow_helper.py`

- **テストカバレッジ**: 未計測

### ドキュメントメトリクス

- **仕様書**: 2ファイル
  - `01_specs/features.md`
  - `01_specs/mlflow_experiment.md`

- **ドキュメント**: 1ファイル
  - `02_docs/mlflow_usage_guide.md`

- **ノートブック**: 1ファイル
  - `05_notebooks/01_eda/prefecture_year_analysis.ipynb`

### 依存関係

- **Python**: 3.13
- **主要ライブラリ**:
  - polars
  - numpy
  - scikit-learn
  - mlflow 3.6.0
  - lightgbm 4.6.0
  - pytest
  - black
  - ruff
  - mypy

---

## 学んだこと・知見

### MLflow関連

1. **ローカルファイルベースで十分**
   - 個人開発ではSQLite/PostgreSQL不要
   - `mlruns/`ディレクトリに自動保存
   - チーム開発時にリモートトラッキングサーバーを検討

2. **ヘルパー関数の重要性**
   - 共通処理を関数化することで記録漏れを防ぐ
   - テストで動作保証

3. **アーティファクト記録**
   - `mlflow.tracking.MlflowClient()`でアーティファクト一覧取得
   - 一時ファイル作成→記録→削除のパターン

### 特徴量エンジニアリング関連

1. **ドキュメントの分離**
   - ルール（`.claude/`）: 原則・概念のみ
   - 仕様書（`01_specs/`）: 実装詳細
   - ソースコード（`04_src/`）: 再利用可能な実装

2. **ベースライン戦略**
   - シンプル・高速・再現性を最優先
   - 複雑な特徴量は後回し

3. **再現性の確保**
   - すべての乱数生成をSeedManagerで管理
   - PYTHONHASHSEED設定も重要

---

## 問題・課題

### 現在の問題

なし

### 今後の課題

1. **特徴量ブロックの実装**
   - BaseBlock、FeaturePipeline、SeedManagerの実装
   - 各種特徴量ブロックの実装（NumericBlock、SimpleImputeBlock等）

2. **訓練スクリプトの実装**
   - MLflowヘルパー関数の統合
   - Time-Series Split実装
   - 提出ファイル生成

3. **リーダーボード提出**
   - 初回ベースラインスコアの確認

---

## リンク・参考資料

### プロジェクト内ドキュメント

- [プロジェクト概要](./.claude/claude.md)
- [汎用開発ルール](./.claude/general_rules.md)
- [プロジェクト固有ルール](./.claude/project_rules.md)
- [特徴量エンジニアリングルール](./.claude/feature_engineering_rules.md)
- [特徴量仕様書](./01_specs/features.md)
- [MLflow実験記録仕様書](./01_specs/mlflow_experiment.md)
- [MLflow使い方ガイド](./02_docs/mlflow_usage_guide.md)

### 外部リンク

- [SIGNATE コンペページ](https://signate.jp/competitions/1365)
- [MLflow公式ドキュメント](https://mlflow.org/docs/latest/index.html)
- [Polars公式ドキュメント](https://pola-rs.github.io/polars/)
- [LightGBM公式ドキュメント](https://lightgbm.readthedocs.io/)

---

**最終更新**: 2025-11-24
