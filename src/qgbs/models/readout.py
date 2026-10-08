"""读出模型抽象与 Ridge / GBM 实现。"""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


class ReadoutModel(Protocol):
    """监督读出模型抽象：特征表 → 截面 alpha 预测。

    fit 仅在训练窗口调用；超参通过验证窗口在 selection 模块中确定。
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "ReadoutModel":
        """训练模型。X 索引为 (date, symbol)，y 为中性化 alpha 标签。"""
        ...

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """输出截面预测，索引与 X 一致。"""
        ...


class RidgeReadout:
    """岭回归读出（线性基线，C2 / Q1-Q3 默认）。"""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._model: Ridge | None = None
        self._features: list[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RidgeReadout":
        y = y_train.reindex(X_train.index)
        mask = y.notna() & X_train.notna().all(axis=1)
        self._features = list(X_train.columns)
        self._model = Ridge(alpha=self.alpha)
        self._model.fit(X_train[mask].to_numpy(dtype=float), y[mask].to_numpy(dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("RidgeReadout 未 fit")
        X2 = X.reindex(columns=self._features).fillna(0.0)
        return pd.Series(self._model.predict(X2.to_numpy(dtype=float)), index=X.index, name="alpha_pred")

    @property
    def feature_names_(self) -> list[str]:
        """训练时记录的特征名。"""
        return list(self._features)


class GBMReadout:
    """梯度提升树读出（经典非线性基线 C3 与完整配置 Q4）。

    默认后端为 sklearn ``HistGradientBoostingRegressor``（不引入
    lightgbm 依赖）；``backend="lightgbm"`` 预留切换位（需可选依赖 ``gbm``）。
    """

    def __init__(self, params: dict[str, Any] | None = None, backend: str = "sklearn") -> None:
        self.params = params or {}
        self.backend = backend
        self._model: Any = None
        self._features: list[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "GBMReadout":
        if self.backend == "lightgbm":
            try:
                import lightgbm as lgb
            except ImportError as exc:
                raise ImportError("backend='lightgbm' 需要可选依赖：pip install qgbs[gbm]") from exc
            self._model = lgb.LGBMRegressor(**self.params)
        elif self.backend == "sklearn":
            self._model = HistGradientBoostingRegressor(**self.params)
        else:
            raise ValueError(f"未知 GBM 后端: {self.backend!r}")
        y = y_train.reindex(X_train.index)
        mask = y.notna() & X_train.notna().all(axis=1)
        self._features = list(X_train.columns)
        self._model.fit(X_train[mask].to_numpy(dtype=float), y[mask].to_numpy(dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("GBMReadout 未 fit")
        X2 = X.reindex(columns=self._features).fillna(0.0)
        return pd.Series(self._model.predict(X2.to_numpy(dtype=float)), index=X.index, name="alpha_pred")

    @property
    def feature_names_(self) -> list[str]:
        return list(self._features)
