# EDA実践ガイド

> **目的**: データ分析コンペでClaude Codeが即座に実行できる標準EDAフローを提供

---

## 📊 EDA標準フロー（4フェーズ）

### Phase 1: データ概要把握（10分）

```python
import pandas as pd
import numpy as np

# データ読み込み
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

# 基本情報
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nColumns: {train.columns.tolist()}")
print(f"\nData types:\n{train.dtypes}")

# 欠損値確認
missing = train.isnull().sum()
print(f"\n欠損値:\n{missing[missing > 0]}")

# 基本統計量
train.describe()
```

**チェックリスト**:
- [ ] データ形状（行数・列数）を確認
- [ ] カラム名・データ型を確認
- [ ] 欠損値の有無を確認
- [ ] 数値変数の基本統計量を確認

---

### Phase 2: 目的変数の分析（15分）

```python
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

target_col = 'target'  # 目的変数名に応じて変更

# 1. 分布の可視化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# ヒストグラム + KDE
axes[0].hist(train[target_col], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_title('目的変数の分布')
axes[0].set_xlabel(target_col)
axes[0].set_ylabel('頻度')

# Box plot
axes[1].boxplot(train[target_col])
axes[1].set_title('外れ値の確認')
axes[1].set_ylabel(target_col)

# Q-Q plot（正規性の確認）
from scipy import stats
stats.probplot(train[target_col], dist="norm", plot=axes[2])
axes[2].set_title('Q-Qプロット（正規性）')

plt.tight_layout()
plt.savefig('05_notebooks/01_eda/figures/target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# 2. 統計量
print(f"平均: {train[target_col].mean():.2f}")
print(f"中央値: {train[target_col].median():.2f}")
print(f"標準偏差: {train[target_col].std():.2f}")
print(f"最小値: {train[target_col].min():.2f}")
print(f"最大値: {train[target_col].max():.2f}")
print(f"歪度: {train[target_col].skew():.2f}")
print(f"尖度: {train[target_col].kurtosis():.2f}")

# 3. Train/Test分布比較（可能な場合）
# testに目的変数がある場合のみ
if target_col in test.columns:
    plt.figure(figsize=(10, 6))
    plt.hist(train[target_col], bins=50, alpha=0.5, label='Train', edgecolor='black')
    plt.hist(test[target_col], bins=50, alpha=0.5, label='Test', edgecolor='black')
    plt.legend()
    plt.title('Train vs Test 目的変数分布')
    plt.xlabel(target_col)
    plt.ylabel('頻度')
    plt.savefig('05_notebooks/01_eda/figures/train_test_target.png', dpi=150, bbox_inches='tight')
    plt.show()
```

**チェックリスト**:
- [ ] ヒストグラムで分布を確認
- [ ] Box plotで外れ値を確認
- [ ] 正規性を確認（必要に応じて対数変換など検討）
- [ ] 歪度・尖度を確認
- [ ] Train/Testで分布に差がないか確認

---

### Phase 3: 特徴量の分析（30-60分）

#### 3.1 数値変数の分析

```python
# 数値変数の抽出
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

# 1. 分布の可視化（全変数）
n_cols = 4
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    axes[i].hist(train[col].dropna(), bins=30, edgecolor='black')
    axes[i].set_title(f'{col}')
    axes[i].set_xlabel('')

# 余ったサブプロットを非表示
for i in range(len(numeric_cols), len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('05_notebooks/01_eda/figures/numeric_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# 2. 相関行列
plt.figure(figsize=(12, 10))
corr = train[numeric_cols + [target_col]].corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('相関行列ヒートマップ')
plt.savefig('05_notebooks/01_eda/figures/correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# 3. 目的変数との相関が高い変数トップ10
target_corr = corr[target_col].abs().sort_values(ascending=False)
print("目的変数との相関（絶対値）トップ10:")
print(target_corr.head(11))  # 目的変数自身を除いて10個

# 4. 重要変数のScatter plot
top_features = target_corr.index[1:5]  # 上位4変数
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, col in enumerate(top_features):
    axes[i].scatter(train[col], train[target_col], alpha=0.3, s=10)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel(target_col)
    axes[i].set_title(f'{col} vs {target_col} (相関: {corr.loc[col, target_col]:.3f})')

plt.tight_layout()
plt.savefig('05_notebooks/01_eda/figures/top_features_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
```

#### 3.2 カテゴリ変数の分析

```python
# カテゴリ変数の抽出
categorical_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()

# 1. 各カテゴリ変数の頻度
for col in categorical_cols[:5]:  # 最初の5個のみ表示
    print(f"\n{col} の値分布:")
    print(train[col].value_counts().head(10))

# 2. カテゴリ変数の可視化
n_cols = 3
n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    value_counts = train[col].value_counts().head(10)
    axes[i].barh(range(len(value_counts)), value_counts.values)
    axes[i].set_yticks(range(len(value_counts)))
    axes[i].set_yticklabels(value_counts.index)
    axes[i].set_title(f'{col}')
    axes[i].set_xlabel('頻度')

for i in range(len(categorical_cols), len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('05_notebooks/01_eda/figures/categorical_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# 3. カテゴリ別の目的変数分布（重要なカテゴリ変数のみ）
for col in categorical_cols[:3]:
    plt.figure(figsize=(12, 6))
    train.boxplot(column=target_col, by=col, figsize=(12, 6))
    plt.suptitle('')
    plt.title(f'{col} ごとの {target_col} 分布')
    plt.xlabel(col)
    plt.ylabel(target_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'05_notebooks/01_eda/figures/{col}_vs_target.png', dpi=150, bbox_inches='tight')
    plt.show()
```

**チェックリスト**:
- [ ] 数値変数の分布を確認
- [ ] 相関行列で多重共線性をチェック
- [ ] 目的変数と相関が高い変数を特定
- [ ] カテゴリ変数の頻度を確認
- [ ] カテゴリごとの目的変数分布を確認

---

### Phase 4: データ品質チェック（15分）

```python
# 1. 欠損値パターンの可視化
import missingno as msno

msno.matrix(train, figsize=(12, 6))
plt.savefig('05_notebooks/01_eda/figures/missing_pattern.png', dpi=150, bbox_inches='tight')
plt.show()

# 2. 欠損値の割合
missing_pct = (train.isnull().sum() / len(train) * 100).sort_values(ascending=False)
print("欠損値の割合（%）:")
print(missing_pct[missing_pct > 0])

# 3. 重複レコードの確認
duplicates = train.duplicated().sum()
print(f"\n重複レコード数: {duplicates}")

# 4. Train/Test重複の確認（Venn図）
from matplotlib_venn import venn2

# ID列がある場合
if 'id' in train.columns and 'id' in test.columns:
    train_ids = set(train['id'])
    test_ids = set(test['id'])

    plt.figure(figsize=(8, 6))
    venn2([train_ids, test_ids], set_labels=('Train', 'Test'))
    plt.title('Train/Test データセット重複')
    plt.savefig('05_notebooks/01_eda/figures/train_test_overlap.png', dpi=150, bbox_inches='tight')
    plt.show()

    overlap = len(train_ids & test_ids)
    print(f"Train/Test重複: {overlap}件")

# 5. データリーク検出（簡易版）
# 目的変数と完全に相関する変数がないかチェック
perfect_corr = corr[target_col][corr[target_col].abs() > 0.99]
if len(perfect_corr) > 1:  # 目的変数自身を除く
    print("\n⚠️ 警告: 目的変数と完全に相関する変数:")
    print(perfect_corr)
```

**チェックリスト**:
- [ ] 欠損パターンを可視化
- [ ] 欠損率が高い変数を特定
- [ ] 重複レコードをチェック
- [ ] Train/Testの重複を確認
- [ ] データリークの可能性をチェック

---

## 🎨 必須の可視化チェックリスト

EDA完了の判断基準として、以下の図をすべて作成する：

### 目的変数
- [ ] ヒストグラム + KDE
- [ ] Box plot（外れ値確認）
- [ ] Q-Qプロット（正規性確認）
- [ ] Train vs Test分布比較（可能な場合）

### 数値変数
- [ ] 全変数のヒストグラム（一覧）
- [ ] 相関行列ヒートマップ
- [ ] 重要変数のScatter plot（vs 目的変数）

### カテゴリ変数
- [ ] 頻度分布（棒グラフ）
- [ ] カテゴリ別の目的変数分布（Box plot）

### データ品質
- [ ] 欠損値パターン（missingnoのmatrix）
- [ ] Train/Test重複（Venn図）

---

## 💡 EDA完了後のアクション

EDAが完了したら、以下を明確にする：

1. **データの特性理解**
   - データのドメイン（不動産、金融、etc.）
   - データ生成プロセス
   - 重要な特徴量の傾向

2. **前処理の方針**
   - 欠損値の補完方法
   - 外れ値の対処
   - スケーリングの必要性

3. **特徴量エンジニアリングの方針**
   - 新規特徴量のアイデア
   - カテゴリ変数のエンコーディング方法
   - 交互作用項の候補

4. **モデリングの方針**
   - ベースラインモデルの選択
   - 評価指標の確認
   - クロスバリデーション戦略

5. **リスク要因**
   - データリークの可能性
   - Train/Test分布の差異
   - クラス不均衡（分類の場合）

---

## 📝 EDAサマリーテンプレート

EDA完了後、以下のサマリーを`05_notebooks/01_eda/eda_summary.md`に記録：

```markdown
# EDAサマリー

**実施日**: YYYY-MM-DD
**データセット**: [コンペ名]

## データ概要
- Train: X行 × Y列
- Test: X行 × Y列
- 目的変数: [変数名]（[タスク種別: 回帰/分類]）

## 主な発見
1. [発見1]
2. [発見2]
3. [発見3]

## 重要な特徴量
1. [特徴量1]: [理由]
2. [特徴量2]: [理由]
3. [特徴量3]: [理由]

## データ品質
- 欠損値: [状況]
- 外れ値: [状況]
- 重複: [状況]
- Train/Test差異: [有無と内容]

## 次のアクション
- [ ] [前処理1]
- [ ] [特徴量生成1]
- [ ] [ベースラインモデル構築]
```

---

## 🔧 ユーティリティ関数

頻繁に使う関数は`04_src/eda/eda_utils.py`に集約：

```python
def plot_distribution(df, col, target=None, figsize=(12, 4)):
    """変数の分布を可視化"""
    pass

def plot_correlation_matrix(df, figsize=(12, 10)):
    """相関行列を可視化"""
    pass

def check_missing(df):
    """欠損値を確認"""
    pass

def compare_train_test(train, test, col):
    """Train/Testの分布を比較"""
    pass
```

---

**最終更新**: 2025-11-23
