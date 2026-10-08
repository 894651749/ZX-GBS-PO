"""交易成本模型：佣金 / 印花税 / 滑点 / 冲击成本。

P0 实现：线性费率（佣金双边 + 印花税卖出单边 + 滑点双边），
提供成交额时附加平方冲击成本项 impact_coef · trades²/volume。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """交易成本参数（基点为万分之一）。"""

    commission_bps: float = 2.5  # 佣金（双边）
    stamp_tax_bps: float = 10.0  # 印花税（卖出单边）
    slippage_bps: float = 5.0  # 滑点（双边）
    impact_coef: float = 0.1  # 冲击成本系数（相对成交额）

    def trade_cost(self, trades: pd.Series, volumes: pd.Series | None = None) -> pd.Series:
        """计算各资产交易成本。

        Args:
            trades: 各资产交易金额（正买负卖）。
            volumes: 对应市场成交额（用于冲击成本）；为空则忽略冲击项。

        Returns:
            各资产成本金额（非负）。
        """
        trades = trades.fillna(0.0)
        linear_bps = (self.commission_bps + self.slippage_bps) / 1e4
        cost = trades.abs() * linear_bps
        sells = (-trades).clip(lower=0.0)
        cost = cost + sells * (self.stamp_tax_bps / 1e4)
        if volumes is not None:
            v = volumes.reindex(trades.index).fillna(0.0)
            # 冲击项：coef · |t| · min(参与率, 1)，参与率 = |t|/v（成交额占比口径）
            # 归一化单位下 self-consistent；参与率封顶 1 防止极端值爆炸
            participation = (trades.abs() / v.where(v > 0)).clip(upper=1.0).fillna(0.0)
            cost = cost + self.impact_coef * trades.abs() * participation
        return cost.clip(lower=0.0)

    def total_cost(self, trades: pd.Series, volumes: pd.Series | None = None) -> float:
        """本期总交易成本。"""
        return float(self.trade_cost(trades, volumes).sum())
