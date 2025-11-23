# ガイド・Tips 集

> データ分析コンペで使える実践的な知識・ノウハウ

---

## 📁 ディレクトリ構成

```
guides/
├── README.md                    # このファイル
├── feature_engineering/         # 特徴量エンジニアリング
│   ├── cardinality_guide.md            # カーディナリティ完全ガイド
│   ├── target_encoding_guide.md        # Target Encoding完全ガイド
│   ├── hashing_trick_guide.md          # Hashing Trick完全ガイド
│   └── (今後追加予定)
├── eda/                         # EDA（探索的データ分析）
│   └── (今後追加予定)
├── modeling/                    # モデリング
│   └── (今後追加予定)
├── validation/                  # バリデーション戦略
│   └── (今後追加予定)
└── tips/                        # 汎用Tips・メモ
    └── (今後追加予定)
```

---

## 📚 ガイド一覧

### 特徴量エンジニアリング

#### [カーディナリティ完全ガイド](./feature_engineering/cardinality_guide.md)
**内容**:
- カーディナリティの定義と分類（低・中・高）
- 各分類に適したエンコーディング手法
- このプロジェクトでの実例（126変数の分類）
- エンコーディング選択フローチャート
- 実装コード（Polars対応）

**こんな時に読む**:
- カテゴリ変数をどうエンコードすべきか迷った時
- One-Hotが使えるか判断したい時
- 高カーディナリティ変数の扱いに困った時

---

#### [Target Encoding 完全ガイド](./feature_engineering/target_encoding_guide.md)
**内容**:
- Target Encodingの基本原理
- 正しい実装方法（クロスバリデーション版）
- データリーク対策とスムージング
- Target Encoding効果の予測方法
- このプロジェクトの分析結果
- 実践的なTips（複数統計量、組み合わせ等）

**こんな時に読む**:
- 中〜高カーディナリティ変数をエンコードしたい時
- Target Encodingの効果を事前に見積もりたい時
- 過学習を防ぎたい時
- どの変数にTarget Encodingすべきか迷った時

---

#### [Hashing Trick 完全ガイド](./feature_engineering/hashing_trick_guide.md)
**内容**:
- Hashing Trickの基本原理
- scikit-learn / カスタム実装
- メリット・デメリット詳細
- 次元数の選び方（衝突確率表付き）
- 使いどころガイドライン
- 実践Tips（複数ハッシュ、併用戦略等）

**こんな時に読む**:
- 超高カーディナリティ変数（数万〜数百万種類）を扱う時
- メモリ不足で困った時
- 未知カテゴリへの対応が必要な時
- One-Hotが使えない時

---

## 🔍 用途別ガイド検索

### カテゴリ変数のエンコーディング
1. まず [カーディナリティ完全ガイド](./feature_engineering/cardinality_guide.md) でユニーク数を確認
2. 低（<10）→ One-Hot Encoding
3. 中（10〜50）→ [Target Encoding完全ガイド](./feature_engineering/target_encoding_guide.md)
4. 高（>50）→ Target Encoding or [Hashing Trick完全ガイド](./feature_engineering/hashing_trick_guide.md)

### 過学習対策
- [Target Encoding完全ガイド](./feature_engineering/target_encoding_guide.md) の「データリーク対策」セクション
- クロスバリデーション実装
- スムージング手法

### メモリ効率化
- [Hashing Trick完全ガイド](./feature_engineering/hashing_trick_guide.md)
- 次元削減手法

### 未知カテゴリ対策
- [Hashing Trick完全ガイド](./feature_engineering/hashing_trick_guide.md) の「未知カテゴリに自動対応」
- [Target Encoding完全ガイド](./feature_engineering/target_encoding_guide.md) の「未知カテゴリ」セクション

---

## 📝 今後追加予定のガイド

### EDA
- [ ] 欠損値処理パターン集
- [ ] 外れ値検出・処理ガイド
- [ ] 相関分析ベストプラクティス
- [ ] 時系列データEDAガイド

### 特徴量エンジニアリング
- [ ] 数値変数の変換テクニック（log, sqrt, box-cox等）
- [ ] 特徴量選択手法ガイド
- [ ] 集約特徴量作成パターン
- [ ] 時系列特徴量エンジニアリング

### モデリング
- [ ] LightGBMチューニングガイド
- [ ] アンサンブル手法パターン集
- [ ] ハイパーパラメータ探索戦略

### バリデーション
- [ ] クロスバリデーション戦略ガイド
- [ ] 時系列データのCV設計
- [ ] リーク検出チェックリスト

### Tips
- [ ] Polars vs Pandas使い分けガイド
- [ ] メモリ効率化Tips
- [ ] 高速化テクニック集

---

## 🔄 更新履歴

**2025-11-23**:
- ディレクトリ構成を整理
- 特徴量エンジニアリングガイド3本を追加
  - カーディナリティ完全ガイド
  - Target Encoding完全ガイド
  - Hashing Trick完全ガイド

---

**最終更新**: 2025-11-23
