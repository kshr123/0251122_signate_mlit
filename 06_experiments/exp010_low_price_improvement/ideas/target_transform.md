# ターゲット変換の見直しメモ

exp010以降で試す価値のあるターゲット変換の代替案。

---

## 現状の設定

- **変換**: `log1p(price)`
- **逆変換**: `expm1(pred)`

log変換は右裾が重い分布（高価格帯に外れ値が多い）を正規分布に近づける効果がある。

---

## 問題点

価格分布によっては最適でない場合がある：
- log変換が強すぎる/弱すぎる
- 低価格帯が圧縮されすぎて情報が失われる

---

## 代替案

### Box-Cox変換

データに最適なλ（変換強度）を自動推定。

- λ=0 → log変換と同じ
- λ=1 → 変換なし
- λ=0.5 → √変換

**特徴**: データに合わせて最適な変換を選ぶ。ただし正の値のみ対応。

**実装**: `sklearn.preprocessing.PowerTransformer(method='box-cox')`

---

### Yeo-Johnson変換

Box-Coxの拡張版。負の値も対応可能。

**特徴**: Box-Coxより汎用的。価格データは正なのでBox-Coxでも可。

**実装**: `sklearn.preprocessing.PowerTransformer(method='yeo-johnson')`

---

### QuantileTransformer

値を一様分布または正規分布に強制的に変換。

**特徴**: どんな分布でも正規分布にできる。ただし元のスケール情報が失われる。

**実装**: `sklearn.preprocessing.QuantileTransformer(output_distribution='normal')`

---

### 変換なし（元スケール）

高価格帯の絶対誤差を重視したい場合に有効。

**特徴**: アンサンブルの多様性確保に使える。

---

## 比較

| 変換 | 特徴 | 優先度 |
|------|------|--------|
| log1p（現状） | 汎用的、定番 | - |
| Box-Cox | データ適応型、最適λ自動推定 | 高 |
| Yeo-Johnson | 負値対応、Box-Coxより汎用 | 中 |
| Quantile | 強制正規化、情報損失大 | 低 |
| なし | 高価格帯重視、アンサンブル用 | 中 |

---

## 試す優先度

1. **Box-Cox**: log1pより最適なλが見つかる可能性
2. **変換なし**: アンサンブル用の多様性確保
3. QuantileTransformerは情報損失が大きいので優先度低め

---

## 実装時の注意

- fit時にλを学習 → transform/inverse_transformで使用
- trainで学習したλをtestにも適用（データリーク防止）

---

**作成日**: 2025-11-27
