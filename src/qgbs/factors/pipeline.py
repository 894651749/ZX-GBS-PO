"""特征预处理流水线：缩尾/标准化/缺失处理。

防泄漏约束：缩尾分位点只允许在训练窗口上 fit，验证/测试期仅可
transform；截面统计量（均值/中位数/标准差）逐日横截面计算，天然无泄漏。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PipelineConfig:
    """预处理超参数。"""

    winsorize_quantile: float = 0.01  # 缩尾分位（双侧）
    fill_strategy: str = "cross_section_median"  # 缺失填充策略
    standardize: bool = True  # 是否做截面 z-score 标准化


class FactorPipeline:
    """特征预处理流水线。

    状态机：未 fit 不可 transform（抛 RuntimeError）；fit 记录训练期
    各特征缩尾分位点后冻结使用。transform 逐期截面处理：

    1. 按训练期分位点缩尾；
    2. 截面中位数填充缺失；
    3. 截面 z-score 标准化（可选）。
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._fitted: bool = False
        self._lower: pd.Series | None = None
        self._upper: pd.Series | None = None

    def fit(self, X_train: pd.DataFrame) -> "FactorPipeline":
        """在训练窗口上估计各特征的缩尾分位点。"""
        q = self.config.winsorize_quantile
        self._lower = X_train.quantile(q)
        self._upper = X_train.quantile(1.0 - q)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """应用冻结的预处理规则；未 fit 时抛 RuntimeError。"""
        if not self._fitted or self._lower is None or self._upper is None:
            raise RuntimeError("FactorPipeline 未 fit：请在训练窗口上先调用 fit()")
        out = X.clip(lower=self._lower, upper=self._upper, axis=1)
        if self.config.fill_strategy == "cross_section_median":
            med = out.groupby(level="date").transform("median")
            out = out.fillna(med)
        elif self.config.fill_strategy != "none":
            raise ValueError(f"未知 fill_strategy: {self.config.fill_strategy!r}")
        if self.config.standardize:
            g = out.groupby(level="date")
            mean = g.transform("mean")
            std = g.transform("std").replace(0.0, 1.0)
            out = (out - mean) / std
        return out

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """训练期便捷入口：fit 后立即 transform。"""
        return self.fit(X_train).transform(X_train)

    @property
    def is_fitted(self) -> bool:
        return self._fitted
