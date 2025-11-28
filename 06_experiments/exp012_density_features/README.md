# exp012_density_features

「近傍物件数」を需要・人口密度の代理変数として導入し、低価格帯の予測精度改善を目指す実験。

## 概要

| 項目 | 内容 |
|------|------|
| 実験ID | exp012 |
| ベース | exp011 (CV MAPE: 12.17%) |
| 目的 | 密度特徴量の追加による低価格帯予測改善 |

## 背景

exp011の分析で以下が判明：
- 物件数が少ない地域ほど予測誤差が大きい
  - 郵便番号内物件数 1-5件: MAPE 15.4% vs 101-500件: 10.1%
  - 駅物件数 1-50件: MAPE 14.3% vs 1000件+: 11.9%
- 郵便番号物件数 vs 駅物件数の相関は0.19と低く、両方入れる価値あり

## 追加特徴量

| # | 特徴量名 | 説明 |
|---|----------|------|
| ① | post_full_count | フル郵便番号の物件数 |
| ② | post_full_count_bin | ①のビン分け（5カテゴリ） |
| ③ | post_full_bin_TE | ②でグループしてTarget Encoding |
| ③' | post_full_bin_agg | ②でグループして集計 |
| ④ | area_age_category | 面積×築年数カテゴリ（3→4カテゴリに拡張） |
| ⑤ | rosen_count_bin | 駅物件数のビン分け（5カテゴリ） |
| ⑥ | rosen_bin_TE | ⑤でグループしてTarget Encoding |
| ⑥' | rosen_bin_agg | ⑤でグループして集計 |

## クイックスタート

```bash
cd 06_experiments/exp012_density_features
source ../../.venv/bin/activate

# テスト実行
caffeinate -i env PYTHONPATH=../../04_src:code python code/train.py --test

# 本番実行（MSE）
caffeinate -i env PYTHONPATH=../../04_src:code python code/train.py --objective mse

# 本番実行（Huber）
caffeinate -i env PYTHONPATH=../../04_src:code python code/train.py --objective huber

# 特徴量キャッシュを使った2回目以降の実行
caffeinate -i env PYTHONPATH=../../04_src:code python code/train.py --objective huber --features-dir outputs/run_mse_xxx
```

## ディレクトリ構成

```
exp012_density_features/
├── README.md            # この文書
├── SPEC.md              # 実験仕様書
├── configs/
│   └── experiment.yaml  # 全ハイパーパラメータ
├── code/
│   ├── train.py         # エントリーポイント
│   ├── preprocessing.py # 前処理
│   ├── pipeline.py      # 特徴量パイプライン
│   ├── constants.py     # パス・カラムリスト
│   └── exp012_features.py  # 密度特徴量Block
├── outputs/             # 実行結果（Git管理外）
├── notebooks/           # 分析ノートブック
└── mlruns/              # MLflow記録（Git管理外）
```

## 設定ファイル構成

| ファイル | 役割 |
|----------|------|
| `configs/experiment.yaml` | 全ハイパーパラメータ（モデル、特徴量、損失関数） |
| `code/constants.py` | パス定義・カラムリスト（ハイパラ含まない） |

## 期待効果

- 全体MAPE: 12.17% → 11.8%程度（0.3-0.5pt改善）
- cat3セグメント（200㎡+ AND 45年+）: 20.5% → 18%程度
- 大外れ率: 2.2% → 1.8%程度

## 実験結果

_実行後に更新_

---

**最終更新**: 2025-11-29
