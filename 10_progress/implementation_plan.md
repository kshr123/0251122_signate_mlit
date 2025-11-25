# 実装方針と進捗管理（2025-11-24更新）

## 🎯 現在の方針

### 設計原則

1. **Blockベース設計**
   - 各特徴量処理を独立したBlockクラスとして実装
   - BaseBlockを継承してfit/transformパターン
   - データリーク防止（fitはtrainのみ、transformはtrain/test両方）

2. **実験ディレクトリの自己完結性**
   - 実験ディレクトリを見れば完全に理解できる
   - FeaturePipelineのような抽象化は作らない
   - 各実験で明示的にBlockを組み合わせる

3. **TDD（テスト駆動開発）**
   - Red（テスト作成） → Green（実装） → Refactor
   - 各Blockは独立してテスト可能

---

## 📋 実装タスク（優先度順）

### Phase 1: Block基盤実装（TDD）

#### 1. BaseBlock実装
- **ファイル**: `04_src/features/base.py`（既存のset_seed()と共存）
- **仕様書**: `01_specs/features_components.md`
- **ステータス**: ⏳ 未着手

**実装内容**:
```python
class BaseBlock:
    def __init__(self):
        self._fitted = False

    def fit(self, input_df: pl.DataFrame, y: pl.Series = None) -> pl.DataFrame:
        self._fitted = True
        return self.transform(input_df)

    def transform(self, input_df: pl.DataFrame) -> pl.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"{self.__class__.__name__}: fit()を先に実行してください")
        raise NotImplementedError()
```

**テスト**:
- `test_base_block_fit_transform()` - fit→transform正常動作
- `test_base_block_not_fitted_error()` - fit前のtransformでエラー

---

#### 2. NumericBlock実装
- **ファイル**: `04_src/features/blocks/numeric.py`
- **テストファイル**: `07_tests/test_features/test_blocks_numeric.py`
- **ステータス**: ⏳ 未着手

**目的**: 数値特徴量をそのまま返す（前処理なし）

**テスト**:
- `test_numeric_block_normal()` - 正常系
- `test_numeric_block_not_fitted_error()` - fit前エラー
- `test_numeric_block_immutability()` - 不変性

---

#### 3. TargetYmBlock実装
- **ファイル**: `04_src/features/blocks/temporal.py`
- **テストファイル**: `07_tests/test_features/test_blocks_temporal.py`
- **ステータス**: ⏳ 未着手

**目的**: YYYYMMフォーマットを年・月に分解

**テスト**:
- `test_target_ym_block_normal()` - 正常系
- `test_target_ym_block_custom_column()` - カスタムカラム名
- `test_target_ym_block_not_fitted_error()` - fit前エラー
- `test_target_ym_block_immutability()` - 不変性

---

#### 4. LabelEncodingBlock実装
- **ファイル**: `04_src/features/blocks/encoding.py`
- **テストファイル**: `07_tests/test_features/test_blocks_encoding.py`
- **ステータス**: ⏳ 未着手

**目的**: カテゴリカル変数を数値に変換

**テスト**:
- `test_label_encoding_categorical()` - Categorical型
- `test_label_encoding_utf8()` - Utf8型（文字列）
- `test_label_encoding_numeric_skip()` - 数値型はスキップ
- `test_label_encoding_not_fitted_error()` - fit前エラー
- `test_label_encoding_immutability()` - 不変性

---

### Phase 2: exp001再構築

#### 5. exp001_baseline再構築
- **ディレクトリ**: `06_experiments/exp001_baseline/`
- **ステータス**: ⏳ 未着手

**構成**:
```
06_experiments/exp001_baseline/
├── README.md              # 実験概要（CV結果、使用Block等）
├── code/
│   ├── preprocessing.py   # Block組み合わせロジック（明示的）
│   ├── train.py          # 訓練スクリプト
│   └── predict.py        # 推論スクリプト
├── configs/              # 実験設定（YAML）
├── outputs/              # 提出ファイル、CV結果
└── models/               # 学習済みモデル
```

**preprocessing.pyの構成**:
```python
from src.features.blocks.numeric import NumericBlock
from src.features.blocks.temporal import TargetYmBlock
from src.features.blocks.encoding import LabelEncodingBlock

# 特徴量リスト明示
NUMERIC_FEATURES = [
    "building_id", "building_status", "lon", "lat", ...
]

CATEGORICAL_FEATURES = [
    "building_name_ruby", "reform_exterior", ...
]

def preprocess_for_training(train, test):
    # Blockリスト作成
    blocks = [
        NumericBlock(columns=NUMERIC_FEATURES),
        TargetYmBlock(source_col="target_ym"),
        LabelEncodingBlock(columns=CATEGORICAL_FEATURES),
    ]

    # 訓練データ処理
    feature_dfs = []
    for block in blocks:
        feature_dfs.append(block.fit(train, y=train["money_room"]))
    X_train = pl.concat(feature_dfs, how="horizontal")

    # テストデータ処理
    feature_dfs = []
    for block in blocks:
        feature_dfs.append(block.transform(test))
    X_test = pl.concat(feature_dfs, how="horizontal")

    return X_train.to_numpy(), X_test.to_numpy(), train["money_room"].to_numpy()
```

**受け入れ基準**:
- [ ] preprocessing.pyで特徴量リストが明示的
- [ ] どのBlockを使ったか一目瞭然
- [ ] CV MAPE: 28.34% ± 0.09% を再現
- [ ] 提出ファイル生成可能

---

### Phase 3: 初回提出

#### 6. SIGNATE初回提出
- **ステータス**: ⏳ 未着手

**手順**:
1. exp001で提出ファイル生成
2. SIGNATEへアップロード
3. リーダーボードスコア記録
4. README.mdに記録

---

## ✅ 完了済み

### Phase 0: プロジェクトセットアップ
- ✅ 仮想環境構築（Python 3.13 + uv）
- ✅ データ定義書作成（149特徴量）

### Phase 1: テンプレート化基盤
- ✅ ディレクトリ構造整備
- ✅ Config Loader実装（TDD完了）
- ✅ Data Loader実装（TDD完了）
- ✅ EDA utilities実装
- ✅ EDA notebook templates作成

### Phase 2: ベースライン準備
- ✅ ベースライン仕様書作成
- ✅ SeedManager実装（features/base.py）
- ✅ MAPE計算実装（evaluation/metrics.py）
- ✅ MLflow補助関数実装
- ✅ 特徴量コンポーネント仕様書作成
  - BaseBlock、NumericBlock、TargetYmBlock、LabelEncodingBlock
  - FeaturePipelineは実験固有のロジックとして扱う

---

## ❌ 削除済み

- ❌ SimplePreprocessor（04_src/preprocessing/simple.py）
  - 理由: 抽象化されすぎて実験内容が不明瞭
- ❌ test_simple.py（07_tests/test_preprocessing/test_simple.py）

---

## 🔄 後回し（Phase 4以降）

### 評価モジュール
- [ ] feature_importance.py実装
- [ ] error_analysis.py実装
- [ ] visualizer.py実装

### 追加Block実装（Priority 2）
- [ ] CountEncodingBlock - 頻度エンコーディング
- [ ] TargetEncodingBlock - ターゲットエンコーディング（CV付き）
- [ ] CategoryNumBlock - カテゴリ×数値集約

### モデル改善
- [ ] ハイパーパラメータチューニング
- [ ] アンサンブル
- [ ] 特徴量追加（住所情報など）

---

## 📊 進捗トラッキング

| タスク | ステータス | 所要時間（予想） | 完了日 |
|--------|-----------|----------------|--------|
| BaseBlock実装 | ⏳ 未着手 | 30分 | - |
| NumericBlock実装 | ⏳ 未着手 | 30分 | - |
| TargetYmBlock実装 | ⏳ 未着手 | 30分 | - |
| LabelEncodingBlock実装 | ⏳ 未着手 | 45分 | - |
| exp001再構築 | ⏳ 未着手 | 1-2時間 | - |
| 初回SIGNATE提出 | ⏳ 未着手 | 30分 | - |

**合計予想時間**: 4-5時間

---

## 📝 参考資料

- **仕様書**: `01_specs/features_components.md`
- **ルール**: `.claude/feature_engineering_rules.md`
- **プロジェクト概要**: `.claude/CLAUDE.md`
- **実験管理**: `.claude/experiment_management_rules.md`

---

**作成日**: 2025-11-24
**最終更新**: 2025-11-24
**次回アクション**: BaseBlock実装（TDD）
