"""PyTorch NN Training Module

CVRunner/FoldTrainerアーキテクチャに統合されたNNモジュール。
既存のGBDT系モデルと同じインターフェースで使用可能。

Usage:
    # MLP
    from training.nn import NNFoldTrainer, MLP

    fold_trainer = NNFoldTrainer(
        model_class=MLP,
        model_params={"hidden_dims": [512, 256, 128], "dropout": 0.3},
        target_transform=Log1pTransform(),
        epochs=100,
        early_stopping_rounds=20,
    )

    # TabNet
    from training.nn import TabNetFoldTrainer

    fold_trainer = TabNetFoldTrainer(
        tabnet_params={"n_d": 8, "n_a": 8, "n_steps": 3},
        target_transform=Log1pTransform(),
        max_epochs=100,
        patience=20,
    )

    # CVRunnerで使用
    result = cv_runner.run(X, y, cv_splits, fold_trainer)
"""

from .dataset import RegressionDataset
from .early_stopping import EarlyStopping
from .fold_trainer import NNFoldTrainer, NNModelWrapper
from .models import MLP
from .tabnet_trainer import TabNetFoldTrainer, TabNetModelWrapper
from .transforms import FeatureScaler

__all__ = [
    # MLP
    "NNFoldTrainer",
    "NNModelWrapper",
    "MLP",
    # TabNet
    "TabNetFoldTrainer",
    "TabNetModelWrapper",
    # Common
    "FeatureScaler",
    "RegressionDataset",
    "EarlyStopping",
]
