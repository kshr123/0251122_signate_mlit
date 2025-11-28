"""
Custom objective functions for LightGBM

汎用的なカスタム目的関数モジュール。

使用例:
    from objectives import MAPEObjective, FocalObjective, BaseObjective

    # MAPE目的関数
    objective = MAPEObjective()
    model = LGBMRegressor(objective=objective, metric="none")
    model.fit(X, y, eval_metric=objective.eval_metric)

    # Focal目的関数（パラメータ付き）
    objective = FocalObjective(gamma=2.0)

    # 独自目的関数の作成
    class MyObjective(BaseObjective):
        name = "my_loss"
        def gradient(self, y_true, y_pred): ...
        def hessian(self, y_true, y_pred): ...
        def loss(self, y_true, y_pred): ...
"""

from objectives.base import BaseObjective, MAPEObjective, FocalObjective

__all__ = [
    "BaseObjective",
    "MAPEObjective",
    "FocalObjective",
]
