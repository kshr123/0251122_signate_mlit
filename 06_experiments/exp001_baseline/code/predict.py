"""
推論スクリプト（exp001_baseline）

学習済みモデルで予測を生成します。
この実験の前処理を明示的に適用します。
"""

import sys
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import polars as pl
import lightgbm as lgb
from datetime import datetime

# 共通コンポーネント
from src.data.loader import DataLoader

# 実験固有の前処理
from preprocessing import preprocess_for_prediction


def predict(
    model_path: str,
    output_path: str = None,
):
    """
    学習済みモデルで予測を生成

    Args:
        model_path: 学習済みモデルのパス
        output_path: 提出ファイルの出力パス（Noneの場合は自動生成）
    """
    print("🔮 推論開始")

    # データ読み込み
    print("📂 データ読み込み中...")
    loader = DataLoader(add_address_columns=False)
    _, test = loader.load_train_test()

    print(f"  - Test: {test.shape}")

    # 前処理（実験固有の処理）
    print("🔧 前処理中...")
    print("  - preprocessing.py で明示的な前処理を実行")
    X_test = preprocess_for_prediction(test)

    # モデル読み込み
    print(f"📦 モデル読み込み: {model_path}")
    model = lgb.Booster(model_file=model_path)

    # 予測
    print("🔮 予測中...")
    predictions = model.predict(X_test)

    # 提出ファイル生成
    submission = test.select("id").with_columns(
        pl.Series("money_room", predictions)
    )

    # 出力パス決定
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"../submission_{timestamp}.csv"

    # 保存
    submission.write_csv(output_path, include_header=False)

    print(f"✅ 推論完了!")
    print(f"  - 提出ファイル: {output_path}")
    print(f"  - 予測値範囲: {predictions.min():.2f} - {predictions.max():.2f}")
    print(f"  - 予測値平均: {predictions.mean():.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="推論スクリプト")
    parser.add_argument(
        "--model",
        type=str,
        default="../models/final_model.txt",
        help="学習済みモデルのパス",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="提出ファイルの出力パス",
    )

    args = parser.parse_args()

    predict(
        model_path=args.model,
        output_path=args.output,
    )
