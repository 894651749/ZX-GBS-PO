"""资产池历史成分管理（point-in-time 查询）。"""

from __future__ import annotations

import pandas as pd

from qgbs.data.adapters.base import AssetPoolHistory


class AssetPool:
    """资产池（如沪深300）的历史成分表。

    只提供 point-in-time 查询：给定日期返回当时真实的成分列表，
    避免幸存者偏差与前视偏差。由 :class:`AssetPoolHistory` 区间表构造。
    """

    def __init__(self, pool_id: str, membership: pd.DataFrame) -> None:
        """membership 为成分区间表（列结构见 AssetPoolRecord schema）。"""
        self.pool_id = pool_id
        self.membership = membership

    @classmethod
    def from_history(cls, history: AssetPoolHistory) -> "AssetPool":
        """从适配器返回的 :class:`AssetPoolHistory` 构造。"""
        return cls(pool_id=history.pool_id, membership=history.records)

    def members_at(self, date: str, eligible_only: bool = True) -> list[str]:
        """返回 ``date`` 当日的成分列表（默认仅含允许入组合的成员）。"""
        d = pd.Timestamp(date).date()
        rec = self.membership
        eff = pd.to_datetime(rec["effective_date"]).dt.date
        exp = pd.to_datetime(rec["expiry_date"]).dt.date
        mask = (eff <= d) & (exp.isna() | (exp > d))
        if eligible_only and "eligible" in rec.columns:
            mask &= rec["eligible"].astype(bool)
        return rec.loc[mask, "symbol"].tolist()

    def history(self, start: str, end: str) -> pd.DataFrame:
        """返回区间内有效的成分区间记录，用于审计与复现。"""
        s, e = pd.Timestamp(start).date(), pd.Timestamp(end).date()
        rec = self.membership
        eff = pd.to_datetime(rec["effective_date"]).dt.date
        exp = pd.to_datetime(rec["expiry_date"]).dt.date
        mask = (eff <= e) & (exp.isna() | (exp > s))
        return rec[mask].reset_index(drop=True)
