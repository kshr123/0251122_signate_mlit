# データモジュール仕様書

## 1. 目的

データの読み込み、分割、保存を担当するモジュールの仕様を定義する。

## 2. 要件

### 2.1 機能要件

1. **データ読み込み**
   - CSV形式のファイルを読み込める
   - 設定ファイル（data.yaml）からパスを取得
   - Polarsを使用して高速に読み込む

2. **データ分割**
   - 時系列ベースの分割に対応
   - ランダム分割にも対応（将来）
   - 学習/検証データの分割

3. **データ保存**
   - 前処理済みデータをParquet形式で保存
   - メタデータ（shape、dtypes）も保存

### 2.2 非機能要件

- Polarsを使用（pandasは使用しない）
- メモリ効率の良い処理
- 設定ファイルベースの柔軟な構成

## 3. モジュール構成

```
src/data/
├── __init__.py
├── loader.py       # データ読み込み
├── splitter.py     # データ分割
└── saver.py        # データ保存
```

## 4. loader.py の仕様

### 4.1 クラス: DataLoader

**責務**: データの読み込み

**インターフェース**:
```python
class DataLoader:
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: data.yaml の内容
        """
        pass

    def load_train(self) -> pl.DataFrame:
        """訓練データを読み込む"""
        pass

    def load_test(self) -> pl.DataFrame:
        """テストデータを読み込む"""
        pass

    def load_sample_submit(self) -> pl.DataFrame:
        """サンプル提出ファイルを読み込む"""
        pass
```

### 4.2 テストケース

| No | テスト名 | 入力 | 期待する出力 | 備考 |
|----|---------|------|-------------|------|
| 1 | 正常系: 訓練データ読み込み | 03_configs/data.yaml | pl.DataFrame | shape確認 |
| 2 | 正常系: テストデータ読み込み | 03_configs/data.yaml | pl.DataFrame | shape確認、idカラム存在確認 |
| 3 | 異常系: ファイルが存在しない | 存在しないパス | FileNotFoundError | エラーメッセージ確認 |
| 4 | 境界値: 空のデータフレーム | 空のCSV | pl.DataFrame (0行) | エラーにしない |

### 4.3 実装の詳細

**入力**:
- `config["data"]["train_path"]`: 訓練データのパス（例: "data/raw/train.csv"）
- `config["data"]["test_path"]`: テストデータのパス
- `config["data"]["sample_submit_path"]`: サンプル提出ファイルのパス

**出力**:
- `pl.DataFrame`: Polarsデータフレーム

**処理フロー**:
1. 設定ファイルからパスを取得
2. ファイルの存在確認
3. pl.read_csv()でデータ読み込み
4. 基本的な検証（shape、dtypes）
5. データフレームを返す

**エラーハンドリング**:
- ファイルが存在しない場合: FileNotFoundError
- 読み込みエラー: CSVParseError（Polarsの例外）

## 5. splitter.py の仕様

### 5.1 クラス: DataSplitter

**責務**: データの分割（学習/検証）

**インターフェース**:
```python
class DataSplitter:
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: data.yaml の split セクション
        """
        pass

    def split_time_series(
        self,
        df: pl.DataFrame,
        time_column: str,
        train_end: int,
        val_start: int
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        時系列ベースでデータ分割

        Args:
            df: 全データ
            time_column: 時系列カラム名（例: "target_ym"）
            train_end: 訓練データの終了時点（例: 202207）
            val_start: 検証データの開始時点（例: 202301）

        Returns:
            (train_df, val_df): 訓練データと検証データ
        """
        pass

    def split_random(
        self,
        df: pl.DataFrame,
        val_size: float,
        random_seed: int
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        ランダムにデータ分割

        Args:
            df: 全データ
            val_size: 検証データの割合（0.0-1.0）
            random_seed: ランダムシード

        Returns:
            (train_df, val_df): 訓練データと検証データ
        """
        pass
```

### 5.2 テストケース

| No | テスト名 | 入力 | 期待する出力 | 備考 |
|----|---------|------|-------------|------|
| 1 | 正常系: 時系列分割 | df, time_col="target_ym", train_end=202207, val_start=202301 | (train_df, val_df) | 期間で正しく分割 |
| 2 | 正常系: ランダム分割 | df, val_size=0.2, seed=42 | (train_df, val_df) | 割合が正しい |
| 3 | 境界値: val_size=0 | df, val_size=0 | (df, empty_df) | 検証データが空 |
| 4 | 境界値: val_size=1 | df, val_size=1 | (empty_df, df) | 訓練データが空 |
| 5 | 異常系: 時系列カラムがない | df, time_col="nonexistent" | ColumnNotFoundError | エラーメッセージ |

### 5.3 実装の詳細

**時系列分割の処理フロー**:
1. 時系列カラムの存在確認
2. train_end 以下のデータを訓練データとして抽出
3. val_start 以上のデータを検証データとして抽出
4. (train_df, val_df) を返す

**ランダム分割の処理フロー**:
1. データのインデックスをシャッフル
2. val_size の割合で分割
3. (train_df, val_df) を返す

## 6. saver.py の仕様

### 6.1 クラス: DataSaver

**責務**: 前処理済みデータの保存

**インターフェース**:
```python
class DataSaver:
    def __init__(self, output_dir: str = "data/processed"):
        """
        Args:
            output_dir: 保存先ディレクトリ
        """
        pass

    def save(
        self,
        df: pl.DataFrame,
        name: str,
        format: str = "parquet"
    ) -> Path:
        """
        データフレームを保存

        Args:
            df: 保存するデータフレーム
            name: ファイル名（拡張子なし）
            format: ファイル形式（"parquet" or "csv"）

        Returns:
            保存先のPath
        """
        pass

    def save_metadata(
        self,
        df: pl.DataFrame,
        name: str
    ) -> Path:
        """
        メタデータ（shape、dtypes）をJSON形式で保存

        Args:
            df: データフレーム
            name: ファイル名（拡張子なし）

        Returns:
            保存先のPath
        """
        pass
```

### 6.2 テストケース

| No | テスト名 | 入力 | 期待する出力 | 備考 |
|----|---------|------|-------------|------|
| 1 | 正常系: Parquet保存 | df, name="train_processed" | Pathオブジェクト | ファイル存在確認 |
| 2 | 正常系: CSV保存 | df, name="train", format="csv" | Pathオブジェクト | ファイル存在確認 |
| 3 | 正常系: メタデータ保存 | df, name="train" | Pathオブジェクト | JSONファイル存在・内容確認 |
| 4 | 異常系: 無効なフォーマット | df, format="invalid" | ValueError | エラーメッセージ |

## 7. 使用例

```python
from src.data.loader import DataLoader
from src.data.splitter import DataSplitter
from src.data.saver import DataSaver
from src.utils.config import load_config

# 設定読み込み
config = load_config()

# データ読み込み
loader = DataLoader(config)
train = loader.load_train()
test = loader.load_test()

# データ分割
splitter = DataSplitter(config)
train_df, val_df = splitter.split_time_series(
    train,
    time_column=config["data"]["time_column"],
    train_end=config["data"]["split"]["time_based"]["train_end"],
    val_start=config["data"]["split"]["time_based"]["val_start"]
)

# 保存
saver = DataSaver()
saver.save(train_df, "train_split")
saver.save(val_df, "val_split")
saver.save_metadata(train_df, "train_split")
```

## 8. 依存関係

- polars: データフレーム操作
- pathlib: ファイルパス操作
- typing: 型ヒント
- json: メタデータ保存

## 9. 今後の拡張

- [ ] Parquet以外の形式対応（Feather、Arrow等）
- [ ] 複数ファイルの一括読み込み
- [ ] データバリデーション機能
- [ ] ストリーミング処理（大規模データ対応）

---

**作成日**: 2025-11-22
**更新日**: 2025-11-22
