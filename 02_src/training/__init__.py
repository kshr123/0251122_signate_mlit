"""Training module - CV学習・ターゲット変換・CV分割

Components:
    Core:
        - CVRunner: フレームワーク非依存のCVループ実行
        - CVResult: CV実行結果のデータクラス
        - PrunedException: プルーニング時の例外

    FoldTrainers (Strategy Pattern):
        - FoldTrainer: 1 fold学習の抽象クラス
        - FoldResult: 1 fold学習結果
        - LightGBMFoldTrainer: LightGBM用
        - SklearnFoldTrainer: sklearn互換モデル用

    Callbacks:
        - Callback: コールバック基底クラス
        - VerboseCallback: 詳細ログ出力
        - OptunaPruningCallback: Optunaプルーニング対応

    Transforms:
        - TargetTransform: ターゲット変換基底クラス
        - Log1pTransform: log1p変換
        - IdentityTransform: 変換なし

    CV Splitters:
        - CVSplitter: CV分割基底クラス
        - KFoldSplitter: 標準KFold分割
        - StratifiedKFoldSplitter: 層化KFold分割（分類用）

    High-level API:
        - Trainer: 高レベルCV学習API（後方互換）
"""

from training.transforms import TargetTransform, Log1pTransform, IdentityTransform
from training.cv import CVSplitter, KFoldSplitter, StratifiedKFoldSplitter
from training.trainer import Trainer
from training.core import CVRunner, CVResult, PrunedException
from training.fold_trainers import (
    FoldResult,
    FoldTrainer,
    LightGBMFoldTrainer,
    LightGBMClassifierFoldTrainer,
    SklearnFoldTrainer,
    CatBoostFoldTrainer,
)
from training.objectives import (
    asymmetric_mse_objective,
    asymmetric_mse_metric,
    huber_objective,
    huber_metric,
)
from training.callbacks import Callback, VerboseCallback, OptunaPruningCallback

__all__ = [
    # Core
    "CVRunner",
    "CVResult",
    "PrunedException",
    # FoldTrainers
    "FoldResult",
    "FoldTrainer",
    "LightGBMFoldTrainer",
    "LightGBMClassifierFoldTrainer",
    "SklearnFoldTrainer",
    "CatBoostFoldTrainer",
    # Callbacks
    "Callback",
    "VerboseCallback",
    "OptunaPruningCallback",
    # Transforms
    "TargetTransform",
    "Log1pTransform",
    "IdentityTransform",
    # CV Splitters
    "CVSplitter",
    "KFoldSplitter",
    "StratifiedKFoldSplitter",
    # Objectives
    "asymmetric_mse_objective",
    "asymmetric_mse_metric",
    "huber_objective",
    "huber_metric",
    # High-level API
    "Trainer",
]
