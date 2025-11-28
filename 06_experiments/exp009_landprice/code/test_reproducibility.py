"""再現性確認用テストスクリプト（小サンプル・高速ハイパラ）"""

import sys
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "04_src"))

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import KFold

# exp009のコードをインポート
code_dir = Path(__file__).parent
sys.path.insert(0, str(code_dir))

from join_landprice import join_landprice_by_year
from landprice_features import (
    create_landprice_features,
    LandpriceCategoryRatioTransformer,
    get_landprice_feature_columns,
)

# 定数
RANDOM_SEED = 42
SAMPLE_SIZE = 2000  # 小サンプル
N_SPLITS = 3  # 高速CV
LANDPRICE_BASE_PATH = project_root / "data/external/landprice"

# 高速LightGBMパラメータ
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mape",
    "boosting_type": "gbdt",
    "num_leaves": 16,  # 小さめ
    "max_depth": 4,  # 浅め
    "learning_rate": 0.1,
    "n_estimators": 50,  # 少なめ
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}


def load_sample_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """小サンプルデータを読み込み"""
    train = pl.read_csv(
        project_root / "data/raw/train.csv",
        infer_schema_length=50000,
    )
    test = pl.read_csv(
        project_root / "data/raw/test.csv",
        infer_schema_length=50000,
    )

    # ランダムサンプリング（シード固定）
    np.random.seed(RANDOM_SEED)
    train_indices = np.random.choice(len(train), min(SAMPLE_SIZE, len(train)), replace=False)
    test_indices = np.random.choice(len(test), min(SAMPLE_SIZE // 2, len(test)), replace=False)

    train_sample = train[sorted(train_indices)]
    test_sample = test[sorted(test_indices)]

    print(f"Train sample: {len(train_sample):,} rows")
    print(f"Test sample: {len(test_sample):,} rows")

    return train_sample, test_sample


def create_features(train: pl.DataFrame, test: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """地価公示特徴量を作成"""
    # 目的変数（money_room）を先に保存
    y_train_series = train["money_room"]

    print("\n=== 地価公示データ結合 ===")

    # 地価公示データ結合
    train_with_lp = join_landprice_by_year(train, base_path=LANDPRICE_BASE_PATH)
    test_with_lp = join_landprice_by_year(test, base_path=LANDPRICE_BASE_PATH)

    # post_fullカラムを作成（post1 + post2）- カテゴリ別地価比率用
    train_with_lp = train_with_lp.with_columns(
        (pl.col("post1").cast(pl.Utf8) + pl.col("post2").cast(pl.Utf8)).alias("post_full")
    )
    test_with_lp = test_with_lp.with_columns(
        (pl.col("post1").cast(pl.Utf8) + pl.col("post2").cast(pl.Utf8)).alias("post_full")
    )

    # 基本特徴量作成（tuple[DataFrame, dict]を返す）
    print("\n=== 地価公示特徴量作成 ===")
    train_lp_df, train_transformers = create_landprice_features(train_with_lp, is_train=True)
    test_lp_df, _ = create_landprice_features(
        test_with_lp,
        shape_pca=train_transformers["shape_pca"],
        road_svd=train_transformers["road_svd"],
        current_use_le=train_transformers["current_use_le"],
        is_train=False,
    )

    # 基本特徴量のみ抽出（lp_プレフィックスのカラム）
    base_cols = [c for c in train_lp_df.columns if c.startswith("lp_")]
    train_lp_base = train_lp_df.select(base_cols)
    test_lp_base = test_lp_df.select(base_cols)

    # カテゴリ別地価比率特徴量
    print("\n=== カテゴリ別地価比率特徴量 ===")
    lp_ratio_transformer = LandpriceCategoryRatioTransformer()
    lp_ratio_transformer.fit()
    train_lp_ratio = lp_ratio_transformer.transform(train_with_lp)
    test_lp_ratio = lp_ratio_transformer.transform(test_with_lp)

    # 結合
    train_lp = pl.concat([train_lp_base, train_lp_ratio], how="horizontal")
    test_lp = pl.concat([test_lp_base, test_lp_ratio], how="horizontal")

    # 正しい特徴量カラムのみ選択（get_landprice_feature_columnsで定義されたもの）
    feature_cols = get_landprice_feature_columns(include_ratio=True)
    # building_age_binがないため関連カラムを除外
    feature_cols = [c for c in feature_cols if "building_age_bin" not in c]
    # 存在するカラムのみ選択
    feature_cols = [c for c in feature_cols if c in train_lp.columns]
    train_lp = train_lp.select(feature_cols)
    test_lp = test_lp.select(feature_cols)

    print(f"地価公示特徴量数: {train_lp.shape[1]}")
    print(f"特徴量カラム: {feature_cols[:10]}... (+ {len(feature_cols) - 10} more)")

    # NumPy配列に変換
    X_train = train_lp.to_numpy().astype(np.float32)
    X_test = test_lp.to_numpy().astype(np.float32)
    # yは関数冒頭で保存したものを使用
    y_train = y_train_series.to_numpy().astype(np.float32)

    # NaNを-999に置換（LightGBM用）
    X_train = np.nan_to_num(X_train, nan=-999)
    X_test = np.nan_to_num(X_test, nan=-999)

    return X_train, X_test, y_train


def train_and_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, float]:
    """CVで学習し、テスト予測とCV MAPEを返す"""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(X_tr, y_tr)

        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / N_SPLITS

    # CV MAPE計算
    mape = np.mean(np.abs((y_train - oof_preds) / y_train)) * 100

    return test_preds, mape


def run_experiment() -> tuple[np.ndarray, float]:
    """実験を1回実行"""
    train, test = load_sample_data()
    X_train, X_test, y_train = create_features(train, test)
    test_preds, mape = train_and_predict(X_train, y_train, X_test)
    return test_preds, mape


def main():
    print("=" * 60)
    print("再現性確認テスト（小サンプル・高速ハイパラ）")
    print("=" * 60)
    print(f"サンプルサイズ: {SAMPLE_SIZE}")
    print(f"CV分割数: {N_SPLITS}")
    print(f"LightGBM: n_estimators={LGBM_PARAMS['n_estimators']}, max_depth={LGBM_PARAMS['max_depth']}")
    print(f"ランダムシード: {RANDOM_SEED}")

    # 1回目の実行
    print("\n" + "=" * 60)
    print("【1回目の実行】")
    print("=" * 60)
    preds1, mape1 = run_experiment()
    print(f"\n1回目 CV MAPE: {mape1:.4f}%")
    print(f"1回目 テスト予測: mean={preds1.mean():.2f}, std={preds1.std():.2f}")

    # 2回目の実行
    print("\n" + "=" * 60)
    print("【2回目の実行】")
    print("=" * 60)
    preds2, mape2 = run_experiment()
    print(f"\n2回目 CV MAPE: {mape2:.4f}%")
    print(f"2回目 テスト予測: mean={preds2.mean():.2f}, std={preds2.std():.2f}")

    # 再現性チェック
    print("\n" + "=" * 60)
    print("【再現性チェック】")
    print("=" * 60)

    mape_diff = abs(mape1 - mape2)
    preds_diff = np.abs(preds1 - preds2).max()

    print(f"CV MAPE差分: {mape_diff:.6f}%")
    print(f"テスト予測最大差分: {preds_diff:.6f}")

    if mape_diff < 1e-6 and preds_diff < 1e-6:
        print("\n✅ 再現性OK: 2回の実行結果が完全に一致")
        return True
    else:
        print("\n❌ 再現性NG: 2回の実行結果に差異あり")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
