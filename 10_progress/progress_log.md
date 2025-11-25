# 進捗ログ

## 現在のステータス

- **現在のフェーズ**: Phase 4 - モデリング・改善
- **最終更新日**: 2025-11-25
- **進捗率**: 約60%（ベースラインモデル構築完了、改善フェーズへ）

## ベストスコア

| 日付 | モデル | CV Score | LB Score | 備考 |
|------|--------|----------|----------|------|
| 2025-11-25 | LightGBM (baseline) | MAPE 28.29% | - | 3-Fold CV, 106特徴量 |

## フェーズ別進捗

### Phase 0: プロジェクトセットアップ ✅
- [x] プロジェクトディレクトリ作成（2025-11-22）
- [x] 環境構築（Python 3.13 + uv + Polars）
- [x] データセット取得（train.csv, test.csv, data_definition.xlsx）
- [x] 基盤構築
  - [x] Config管理システム（YAML + 自動バックアップ）
  - [x] Data Loader実装（Polars + TDD完了）
  - [x] 実験管理システム（train.py, train_with_mlflow.py）

### Phase 1: データ理解 ✅
- [x] データセットのダウンロード
- [x] データ定義書の作成（149特徴量の詳細ドキュメント化）
- [x] 基本統計量の確認
- [x] データ型、欠損値、外れ値の確認

### Phase 2: EDA（探索的データ分析）✅ **完了！**
**完了**:
- [x] **01_initial_eda.ipynb**（19/20セル実行済み）
  - データ読み込みと基本情報確認
  - プロファイルサマリー
  - 欠損値分析
  - 数値・カテゴリカラムの分布確認
  - 重複行の確認
  - Train/Testデータの比較とベン図分析

- [x] **02_target_analysis.ipynb**（14/15セル実行済み）
  - ターゲット変数（money_room）の基本統計量
  - 分布の可視化（ヒストグラム、Box plot、Q-Qプロット）
  - 外れ値の検出と分析
  - 時系列トレンドの確認
  - 主要特徴量との関係性分析
  - カテゴリ別のターゲット分布

- [x] **03_feature_correlation.ipynb**（ノートブック作成完了）
  - 数値特徴量間の相関行列
  - ターゲット変数との相関（TOP 20）
  - 多重共線性のチェック
  - 散布図マトリクス

- [x] **04_categorical_analysis.ipynb**（ノートブック作成完了）
  - カーディナリティ別の分類（低・中・高）
  - ターゲット変数との関係性
  - エンコーディング戦略の決定
  - Target Encoding優先度の算出

- [x] **05_geospatial_analysis.ipynb**（ノートブック作成完了）
  - 緯度経度の分布確認
  - 地理的な価格分布
  - ヒートマップとクラスタリング
  - 距離ベース特徴量のアイデア

- [x] **06_prefecture_year_analysis.ipynb**（✅ 実行完了）
  - **住所カラムの追加**（AddressParser実装 + TDD）
    - prefecture_name（都道府県名）自動追加
    - city_name（市区町村名）自動追加
    - DataLoaderに統合（add_address_columns=True）
  - **target_ym（予測対象年月）による時系列分析**
    - 2019/01 〜 2022/07の8期間分析
    - 都道府県×年月のデータ数分布（ヒートマップ）
    - 都道府県×年月の平均賃料分布
    - 主要都道府県の時系列トレンド
    - 季節性分析（1月 vs 7月）
    - 年次トレンド分析
  - **モデリングへの示唆**
    - target_ymから年・月を抽出
    - 都道府県×年月の交互作用特徴量
    - 時系列分割CV戦略
    - Test予測時の注意点（2023年1月/7月）

- [x] **EDA結果まとめ**（02_docs/eda_summary.md）
  - 主な発見と知見の整理
  - 特徴量エンジニアリング方針
  - 実装の優先順位
  - 次のアクションプラン

### Phase 3: 特徴量エンジニアリング ✅
- [x] 仕様書作成（01_specs/features_components.md）
- [x] Blockベース特徴量システム実装
  - [x] BaseBlock（04_src/features/base.py）
  - [x] NumericBlock（数値特徴量選択）
  - [x] TargetYmBlock（年月分解）
  - [x] LabelEncodingBlock（カテゴリエンコーディング）
- [x] 欠損値の処理（NumericBlock内で対応）
- [x] カテゴリ変数のエンコーディング（LabelEncoding）
- [ ] 新規特徴量の作成（今後の改善で対応）
  - [ ] CountEncoding / TargetEncoding
  - [ ] 位置情報特徴量（主要駅までの距離等）
  - [ ] 集約特徴量

### Phase 4: モデリング 🔄 **進行中**
- [x] 仕様書作成（01_specs/baseline_model.md）
- [x] ベースラインモデルの構築（LightGBM）
  - CV MAPE: 28.29%
  - 3-Fold CV, 106特徴量
- [x] OOF予測・Feature Importance保存
- [x] エラー分析ノートブック作成（01_error_analysis.ipynb）
- [ ] クロスバリデーションの設定（Time-series split）
- [ ] 複数モデルの実験（XGBoost, CatBoost）
- [ ] ハイパーパラメータチューニング（Optuna）
- [ ] アンサンブル

### Phase 5: 評価と提出
- [x] 提出ファイル作成（submission_*.csv）
- [ ] SIGNATE提出・リーダーボード確認
- [ ] 最終モデルの評価

## 完了した主なタスク

**2025-11-25（Day 4）**:
- **ベースラインモデル構築完了**
  - exp001_baseline: LightGBM, 3-Fold CV
  - CV MAPE: 28.29%（±0.06%）
  - 106特徴量（NumericBlock + TargetYmBlock + LabelEncodingBlock）
- **エラー分析**
  - OOF予測・Feature Importance保存
  - 01_error_analysis.ipynb作成
  - 低価格帯（Q1, Q2）でMAPE高い（82.9%, 39.2%）
  - Top特徴量: house_area, post1, year_built, money_kyoueki
- **マスターデータ整備**
  - data_definition.xlsx からマスター切り出し
    - area_master.csv（都道府県名・市区町村名）
    - tag_master.csv（タグID→タグ内容）
    - equipment_master.csv（設備情報）
    - feature_definition.csv（特徴量定義）
  - train/test enrichedファイル作成（都道府県名・市区町村名結合済み）
  - DataLoader更新（enrichedファイルをデフォルトに）
  - 変換スクリプト保存（08_scripts/create_enriched_data.py）

**2025-11-24（Day 3）**:
- Blockベース特徴量システム設計・実装
  - 01_specs/features_components.md（仕様書）
  - BaseBlock, NumericBlock, TargetYmBlock, LabelEncodingBlock
- 評価モジュール実装（TDD）
  - evaluation/metrics.py（MAPE計算）
- MLflow補助関数実装
  - training/utils/mlflow_helper.py

**2025-11-22（Day 1）**:
- プロジェクト初期化
- ディレクトリ構造整備
- データ定義書の作成（149特徴量の詳細ドキュメント）
- MLIT DATA PLATFORM MCPサーバーセットアップ
- Config管理システム実装（YAML + 自動バックアップ）
- Data Loader実装（SDD + TDD）

**2025-11-23（Day 2）**:
- 実験管理システム実装（train.py, train_with_mlflow.py）
- EDA完全実施（6つのノートブック作成）:
  - 01_initial_eda.ipynb（基本確認）
  - 02_target_analysis.ipynb（ターゲット分析）
  - 03_feature_correlation.ipynb（相関分析）
  - 04_categorical_analysis.ipynb（カテゴリ分析）
  - 05_geospatial_analysis.ipynb（地理空間分析）
  - **06_prefecture_year_analysis.ipynb（都道府県×年月分析）** ← New!
- **AddressParser実装（SDD + TDD）**:
  - PREFECTURE_MASTER（47都道府県マスター）
  - city_name抽出（正規表現による自動抽出）
  - DataLoaderに統合（add_address_columns=True）
  - テスト18件パス（AddressParser: 11件、DataLoader: 7件）
- EDA結果まとめドキュメント作成（02_docs/eda_summary.md）

## 学んだこと・気づき

### データ特性
1. **ターゲット変数（money_room）**
   - 右に歪んだ分布（対数変換が有効かも）
   - 外れ値が存在（高額物件）
   - 時系列でのトレンドあり

2. **欠損値**
   - 特定の特徴量で欠損が多い
   - 欠損パターンに意味がある可能性（例：築年数不明の新築物件等）

3. **Train/Test分布**
   - 時系列分割（2019-2022 vs 2023）
   - **Train**: target_ym = 201901, 201907, 202001, 202007, 202101, 202107, 202201, 202207（8期間）
   - **Test**: 2023年1月または7月（target_ym不明）
   - データシフトの可能性を考慮する必要あり

4. **地域性（New!）**
   - 都道府県別にデータ数・賃料が大きく異なる
   - 上位都道府県（東京、神奈川、大阪、愛知、兵庫等）にデータが集中
   - 時系列トレンドが都道府県ごとに異なる
   - 季節性（1月 vs 7月）が存在する可能性

### 技術的な知見
1. **Polarsの効果**
   - 大規模データ（330MB+）でもメモリ効率的
   - pandas比で処理が高速
   - `replace_strict()` で型安全なマッピング

2. **設定管理**
   - YAML + 自動バックアップで実験管理が容易
   - 過去の実験を簡単に復元可能

3. **住所データの扱い（New!）**
   - 正規表現で市区町村名を抽出可能（`r'^([^0-9]+?[市区町村])'`）
   - DataLoaderで遅延ロード・キャッシュにより効率化
   - 662市区町村を自動抽出成功

## 次回セッションでやること

1. **SIGNATE提出**
   - 現在のベースラインをSIGNATEに提出
   - リーダーボードスコア確認

2. **モデル改善**
   - 低価格帯（Q1, Q2）の予測精度改善
   - num_boost_round増加（Early Stopping導入）
   - Time-series CV（2019-2021年Train / 2022年Validation）

3. **特徴量追加**
   - CountEncoding / TargetEncoding
   - 都道府県名・市区町村名を使った集約特徴量
   - 位置情報特徴量（緯度経度からの距離等）

4. **ハイパーパラメータチューニング**
   - Optunaによる自動チューニング

---

**最終更新**: 2025-11-25
