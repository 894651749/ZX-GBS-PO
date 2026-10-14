"""交易日历与再平衡时点计算。

P0 实现：以工作日（Mon-Fri）近似交易日历；真实日历接入后替换数据源。
"""

from __future__ import annotations

import pandas as pd


def trading_days(start: str, end: str, calendar: str = "cnx") -> pd.DatetimeIndex:
    """返回 [start, end] 区间内的交易日序列（P0：工作日近似）。"""
    return pd.bdate_range(start, end)


def rebalance_dates(dates: pd.DatetimeIndex, freq: str = "1m") -> pd.DatetimeIndex:
    """在交易日序列中选出再平衡时点（P0：每月首个交易日，freq="1m"）。"""
    if freq != "1m":
        raise NotImplementedError(f"P0 仅支持月度调仓 freq='1m'，收到 {freq!r}")
    s = pd.Series(dates, index=dates)
    first_of_month = s.groupby(s.index.to_period("M")).min()
    return pd.DatetimeIndex(first_of_month.to_numpy())


def is_trading_day(date: str, calendar: str = "cnx") -> bool:
    """判断某日是否为交易日（P0：工作日近似）。"""
    return bool(pd.Timestamp(date).dayofweek < 5)
