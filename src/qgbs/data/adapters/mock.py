"""Mock 数据适配器：开发调试用，生成可复现的合成数据（覆盖全部 7 类交付数据）。

合成模型：带行业板块结构的多股票几何布朗运动
（市场因子 + 行业因子 + 特质噪声），给定 seed 完全可复现。

口径约定：

- 价格为原始未复权价；``adj_factor`` 在公司行动除权日阶跃；
- 停牌日：价格冻结在上一可交易收盘价、成交量/额为 0、日收益为 0，
  复牌后价格一次性补涨/补跌（catch-up 到真实路径）；
- 上市前/退市后无行情行；涨跌停日价格截断在 ±10%。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from qgbs.data.adapters.base import (
    AssetPoolHistory,
    BaselineResults,
    ConstraintsCosts,
    CorporateActions,
    DailyMarket,
    MarketSectorFactor,
    TradingStatus,
)


@dataclass(frozen=True)
class MockDataConfig:
    """合成数据参数，对应 ``configs/data/mock.yaml``。"""

    n_assets: int = 100  # 合成股票数（pool100 规模）
    start: str = "2018-01-01"
    end: str = "2025-12-31"
    n_industries: int = 10  # 中信一级行业数（合成）
    n_sub_industries: int = 30  # 中信二级行业数（合成）
    n_indices: int = 3  # 宽基指数数（合成）
    n_style_factors: int = 5  # 风格因子数（合成）
    n_cash_assets: int = 2  # 现金资产数
    pool29_size: int = 29  # 29 池规模
    suspension_prob: float = 0.005  # 每日停牌概率
    limit_pct: float = 0.10  # 涨跌停幅度
    rf_annual: float = 0.02  # 无风险年化利率（常数）
    cash_yield_annual: float = 0.015  # 现金管理年化收益率（常数）
    # 约束与成本（load_constraints_costs 输出）
    rebalance_freq: str = "1m"
    commission_bps: float = 2.5
    stamp_tax_bps: float = 10.0
    slippage_bps: float = 5.0
    impact_coef: float = 0.0  # 冲击成本系数（默认 0 关闭保持既有口径；链路已接通可测）
    weight_upper: float = 0.10  # 个股上限（全池统一）
    cash_range: tuple[float, float] = (0.0, 0.05)
    max_turnover: float | None = None
    liquidity_limit: float | None = None
    with_corporate_actions: bool = True  # 是否生成分红送转事件
    with_suspension: bool = True  # 是否生成停牌/涨跌停状态
    seed: int = 42  # 合成数据随机种子


class MockAdapter:
    """基于 :class:`MockDataConfig` 生成确定性合成数据的适配器。

    实现 :class:`~qgbs.data.adapters.base.DataAdapter` 全部 7 个方法；
    生成数据满足 base.py 中行级 schema 的字段/缺失约定。
    """

    def __init__(self, config: MockDataConfig | None = None) -> None:
        self.config = config or MockDataConfig()
        self._universe: dict[str, object] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MockAdapter":
        """从 ``configs/data/mock.yaml`` 读取参数构造。"""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {f.name for f in fields(MockDataConfig)}
        kwargs = {k: v for k, v in raw.items() if k in known}
        if "cash_range" in kwargs:
            kwargs["cash_range"] = tuple(kwargs["cash_range"])
        return cls(MockDataConfig(**kwargs))

    # ------------------------------------------------------------------
    # DataAdapter 接口
    # ------------------------------------------------------------------

    def load_asset_pool_history(self, pool_id: str) -> AssetPoolHistory:
        u = self._ensure()
        if pool_id == "pool29":
            symbols = list(u["pool29"])
        elif pool_id == "pool100":
            symbols = list(u["symbols"])
        else:
            raise KeyError(f"未知 pool_id: {pool_id!r}（可用: pool29 / pool100）")
        rec = u["pool_records"]
        rows = rec[rec["symbol"].isin(symbols + list(u["cash_symbols"]))].copy()
        return AssetPoolHistory(pool_id=pool_id, records=rows.reset_index(drop=True))

    def load_daily_market(self, symbols: list[str], start: str, end: str) -> DailyMarket:
        u = self._ensure()
        rows = u["market_rows"]
        mask = (
            rows["symbol"].isin(symbols)
            & (rows["trade_date"] >= pd.Timestamp(start))
            & (rows["trade_date"] <= pd.Timestamp(end))
        )
        return DailyMarket(rows=rows[mask].reset_index(drop=True))

    def load_trading_status(self, symbols: list[str], start: str, end: str) -> TradingStatus:
        u = self._ensure()
        rows = u["status_rows"]
        mask = (
            rows["symbol"].isin(symbols)
            & (rows["trade_date"] >= pd.Timestamp(start))
            & (rows["trade_date"] <= pd.Timestamp(end))
        )
        return TradingStatus(rows=rows[mask].reset_index(drop=True))

    def load_corporate_actions(self, symbols: list[str], start: str, end: str) -> CorporateActions:
        u = self._ensure()
        rec = u["corp_actions"]
        mask = (
            rec["symbol"].isin(symbols)
            & (rec["ex_date"] >= pd.Timestamp(start))
            & (rec["ex_date"] <= pd.Timestamp(end))
        )
        return CorporateActions(records=rec[mask].reset_index(drop=True))

    def load_market_sector_factor(self, start: str, end: str) -> MarketSectorFactor:
        u = self._ensure()
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        idx_ret = u["index_returns"]
        ind_ret = u["industry_returns"]
        rf = u["risk_free"]
        cm = u["cash_yield"]
        sel = (idx_ret.index >= s) & (idx_ret.index <= e)
        return MarketSectorFactor(
            index_returns=idx_ret[sel],
            industry_index_returns=ind_ret[sel],
            risk_free_rate=rf[sel],
            cash_mgmt_yield=cm[sel],
        )

    def load_constraints_costs(self, config_id: str) -> ConstraintsCosts:
        if config_id not in ("default", "mock"):
            raise KeyError(f"未知 config_id: {config_id!r}（mock 仅提供 'default'/'mock'）")
        c = self.config
        u = self._ensure()
        return ConstraintsCosts(
            config_id=config_id,
            rebalance_freq=c.rebalance_freq,
            commission_bps=c.commission_bps,
            stamp_tax_bps=c.stamp_tax_bps,
            slippage_bps=c.slippage_bps,
            impact_coef=c.impact_coef,
            weight_upper={s: c.weight_upper for s in u["symbols"]},
            cash_range=tuple(c.cash_range),
            max_turnover=c.max_turnover,
            liquidity_limit=c.liquidity_limit,
        )

    def load_baseline_results(self) -> BaselineResults:
        """基线交付物：暂返回全 None 的空容器（允许部分缺失）。

        预留接口：mock 不合成基线交付物，待中信 baseline_results 交付后
        再实现对应占位数据。
        """
        return BaselineResults()

    # ------------------------------------------------------------------
    # 内部：一次性生成全部合成数据并缓存
    # ------------------------------------------------------------------

    def _ensure(self) -> dict[str, object]:
        if self._universe is None:
            self._universe = self._generate()
        return self._universe

    def _generate(self) -> dict[str, object]:
        c = self.config
        rng = np.random.default_rng(c.seed)
        dates = pd.bdate_range(c.start, c.end)
        t_n = len(dates)
        n = c.n_assets
        symbols = [f"MOCK{i + 1:04d}" for i in range(n)]
        cash_symbols = [f"CASH{i + 1:02d}" for i in range(c.n_cash_assets)]
        ind_codes = [f"IND{k + 1:02d}" for k in range(c.n_industries)]
        industry = np.array([ind_codes[i % c.n_industries] for i in range(n)])

        # --- 因子结构收益：市场 + 行业 + 特质 ---
        beta_m = rng.uniform(0.7, 1.3, n)
        beta_i = rng.uniform(0.5, 1.5, n)
        idio_vol = rng.uniform(0.010, 0.020, n)
        mkt_ret = rng.normal(0.0004, 0.010, t_n)
        ind_ret = rng.normal(0.0, 0.006, (t_n, c.n_industries))
        eps = rng.normal(0.0, 1.0, (t_n, n)) * idio_vol
        ind_idx = np.arange(n) % c.n_industries
        true_ret = beta_m * mkt_ret[:, None] + ind_ret[:, ind_idx] * beta_i + eps

        # --- 上市/退市 ---
        list_idx = np.zeros(n, dtype=int)
        late = rng.random(n) < 0.1
        list_idx[late] = rng.integers(60, max(61, t_n // 3), late.sum())
        delist_idx = np.full(n, t_n, dtype=int)  # t_n 表示不退市
        n_delist = min(2, n // 20)
        if n_delist:
            dj = rng.choice(n, n_delist, replace=False)
            delist_idx[dj] = rng.integers(int(t_n * 0.7), int(t_n * 0.9), n_delist)

        # --- 涨跌停截断 ---
        lp = c.limit_pct
        sealed_up = true_ret >= lp - 0.005
        sealed_down = true_ret <= -(lp - 0.005)
        exec_ret = np.clip(true_ret, -lp, lp)

        # --- 真实价格路径（GBM） ---
        p0 = rng.uniform(10.0, 100.0, n)
        true_close = p0 * np.exp(np.cumsum(exec_ret, axis=0))

        # --- 停牌掩码（上市区间内的 warm-up 之后） ---
        susp = rng.random((t_n, n)) < c.suspension_prob if c.with_suspension else np.zeros((t_n, n), bool)
        susp[:120] = False
        for j in range(n):
            susp[: list_idx[j], j] = False
            susp[delist_idx[j]:, j] = False

        # --- 报告价格：停牌冻结，复牌补涨 ---
        reported = np.where(susp, np.nan, true_close)
        rep_close = pd.DataFrame(reported).ffill().to_numpy()
        prev_close = np.vstack([rep_close[0] / np.exp(exec_ret[0]), rep_close[:-1]])
        # 上市首日 prev_close 无前值：用当日执行收益反推，保持收益口径自洽
        first_listed = np.isnan(prev_close) & ~np.isnan(rep_close)
        prev_close[first_listed] = rep_close[first_listed] / np.exp(exec_ret[first_listed])

        # --- OHLC 自洽 ---
        open_ = prev_close * np.exp(rng.normal(0.0, 0.002, (t_n, n)))
        open_ = np.where(susp, rep_close, open_)
        hi = np.maximum(open_, rep_close) * (1.0 + np.abs(rng.normal(0.0, 0.003, (t_n, n))))
        lo = np.minimum(open_, rep_close) * (1.0 - np.abs(rng.normal(0.0, 0.003, (t_n, n))))

        # --- 量额/市值/换手 ---
        base_vol = rng.lognormal(mean=np.log(2e6), sigma=0.5, size=n)
        volume = base_vol * (1.0 + 30.0 * np.abs(exec_ret)) * rng.lognormal(0.0, 0.2, (t_n, n))
        volume = np.where(susp, 0.0, volume)
        amount = volume * (open_ + rep_close) / 2.0
        float_shares = rng.uniform(5e7, 5e8, n)
        total_shares = float_shares / rng.uniform(0.6, 0.9, n)
        turnover_rate = volume / float_shares * 100.0
        total_mv = rep_close * total_shares
        float_mv = rep_close * float_shares
        daily_return = np.where(prev_close > 0, rep_close / prev_close - 1.0, 0.0)

        # --- 公司行动与复权因子 ---
        adj_factor = np.ones((t_n, n))
        ca_records: list[dict] = []
        if c.with_corporate_actions:
            n_events = min(8, n // 5)
            for j in rng.choice(n, n_events, replace=False):
                lo_i, hi_i = max(list_idx[j] + 150, 1), min(delist_idx[j] - 20, t_n - 20)
                if lo_i >= hi_i:
                    continue
                ex = int(rng.integers(lo_i, hi_i))
                is_split = bool(rng.random() < 0.4)
                cash_div = float(round(rng.uniform(0.1, 0.5), 4))
                split_ratio = float(round(rng.uniform(1.1, 1.3), 4)) if is_split else 1.0
                mult = (1.0 + cash_div / prev_close[ex, j]) * split_ratio
                adj_factor[ex:, j] *= mult
                ca_records.append(
                    {
                        "symbol": symbols[j],
                        "action_type": "split" if is_split else "dividend",
                        "ex_date": dates[ex],
                        "effective_date": dates[ex],
                        "expiry_date": None,
                        "cash_dividend": cash_div,
                        "split_ratio": split_ratio,
                        "allotment_ratio": 0.0,
                        "allotment_price": None,
                        "adj_factor": float(mult),
                        "adj_factor_effective_date": dates[ex],
                    }
                )
        corp_actions = pd.DataFrame(
            ca_records,
            columns=[
                "symbol", "action_type", "ex_date", "effective_date", "expiry_date",
                "cash_dividend", "split_ratio", "allotment_ratio", "allotment_price",
                "adj_factor", "adj_factor_effective_date",
            ],
        )

        # --- 行情/状态长表 ---
        market_rows, status_rows = self._build_long_tables(
            dates, symbols, list_idx, delist_idx, susp, sealed_up, sealed_down,
            open_, hi, lo, rep_close, prev_close, volume, amount,
            turnover_rate, total_mv, float_mv, adj_factor, daily_return,
        )

        # --- 资产池区间表 ---
        mean_mv = np.nanmean(total_mv, axis=0)
        rank_order = np.argsort(-mean_mv)
        pool29 = tuple(symbols[j] for j in rank_order[: c.pool29_size])
        st_mask = np.zeros(n, bool)
        st_mask[rng.choice(n, 1, replace=False)] = True
        pool_records = self._build_pool_records(
            dates, symbols, cash_symbols, industry, list_idx, delist_idx, st_mask
        )

        # --- 市场/行业因子 ---
        half = c.n_industries // 2
        index_returns = pd.DataFrame(
            {
                "IDX_MKT": mkt_ret,
                "IDX_GROWTH": ind_ret[:, :half].mean(axis=1) + rng.normal(0, 0.002, t_n),
                "IDX_VALUE": ind_ret[:, half:].mean(axis=1) + rng.normal(0, 0.002, t_n),
            },
            index=dates,
        )
        industry_returns = pd.DataFrame(ind_ret, index=dates, columns=ind_codes)
        risk_free = pd.Series(c.rf_annual / 252.0, index=dates, name="risk_free_rate")
        cash_yield = pd.Series(c.cash_yield_annual / 252.0, index=dates, name="cash_mgmt_yield")

        return {
            "dates": dates,
            "symbols": symbols,
            "cash_symbols": cash_symbols,
            "pool29": pool29,
            "market_rows": market_rows,
            "status_rows": status_rows,
            "corp_actions": corp_actions,
            "pool_records": pool_records,
            "index_returns": index_returns,
            "industry_returns": industry_returns,
            "risk_free": risk_free,
            "cash_yield": cash_yield,
        }

    def _build_long_tables(
        self, dates, symbols, list_idx, delist_idx, susp, sealed_up, sealed_down,
        open_, hi, lo, rep_close, prev_close, volume, amount,
        turnover_rate, total_mv, float_mv, adj_factor, daily_return,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        t_n, n = len(dates), len(symbols)
        mkt_parts: list[pd.DataFrame] = []
        sts_parts: list[pd.DataFrame] = []
        for j, sym in enumerate(symbols):
            lo_i, hi_i = list_idx[j], min(delist_idx[j], t_n)
            sl = slice(lo_i, hi_i)
            d = dates[sl]
            susp_j = susp[sl, j]
            # 复牌日：停牌段之后的首个非停牌交易日
            resume = pd.Series(pd.NaT, index=d, dtype="datetime64[ns]")
            if susp_j.any():
                non_susp_dates = d[~susp_j]
                for i in np.where(susp_j)[0]:
                    later = non_susp_dates[non_susp_dates > d[i]]
                    if len(later):
                        resume.iloc[i] = later[0]
            mkt_parts.append(
                pd.DataFrame(
                    {
                        "symbol": sym,
                        "trade_date": d,
                        "open": open_[sl, j],
                        "high": hi[sl, j],
                        "low": lo[sl, j],
                        "close": rep_close[sl, j],
                        "prev_close": prev_close[sl, j],
                        "volume": volume[sl, j],
                        "amount": amount[sl, j],
                        "turnover_rate": turnover_rate[sl, j],
                        "total_mv": total_mv[sl, j],
                        "float_mv": float_mv[sl, j],
                        "adj_factor": adj_factor[sl, j],
                        "daily_return": daily_return[sl, j],
                        "return_basis": "raw",
                    }
                )
            )
            limit_up_px = np.where(susp_j, np.nan, np.round(prev_close[sl, j] * 1.1, 2))
            limit_dn_px = np.where(susp_j, np.nan, np.round(prev_close[sl, j] * 0.9, 2))
            sts_parts.append(
                pd.DataFrame(
                    {
                        "symbol": sym,
                        "trade_date": d,
                        "is_suspended": susp_j,
                        "resume_date": resume.to_numpy(),
                        "limit_up_price": limit_up_px,
                        "limit_down_price": limit_dn_px,
                        "is_limit_up_sealed": sealed_up[sl, j] & ~susp_j,
                        "is_limit_down_sealed": sealed_down[sl, j] & ~susp_j,
                        "is_st": False,
                        "list_date": dates[list_idx[j]],
                        "delist_date": pd.NaT if delist_idx[j] >= t_n else dates[delist_idx[j]],
                        "buyable": ~susp_j & ~sealed_up[sl, j],
                        "sellable": ~susp_j & ~sealed_down[sl, j],
                        "lot_size": 100,
                    }
                )
            )
        return pd.concat(mkt_parts, ignore_index=True), pd.concat(sts_parts, ignore_index=True)

    def _build_pool_records(
        self, dates, symbols, cash_symbols, industry, list_idx, delist_idx, st_mask
    ) -> pd.DataFrame:
        t_n = len(dates)
        rows: list[dict] = []
        for j, sym in enumerate(symbols):
            l1 = industry[j]
            l2 = f"{l1}_S{j % 3 + 1:02d}"
            rows.append(
                {
                    "symbol": sym,
                    "exchange": "SH" if j % 2 == 0 else "SZ",
                    "effective_date": dates[list_idx[j]].date(),
                    "expiry_date": None if delist_idx[j] >= t_n else dates[delist_idx[j]].date(),
                    "citics_industry_l1": l1,
                    "citics_industry_l2": l2,
                    "industry_change_date": None,
                    "industry_rank_rule": "total_mv_desc",
                    "industry_rank_date": dates[0].date(),
                    "is_cash_asset": False,
                    "cash_category": None,
                    "eligible": not st_mask[j],
                }
            )
        for k, csym in enumerate(cash_symbols):
            rows.append(
                {
                    "symbol": csym,
                    "exchange": "OTC",
                    "effective_date": dates[0].date(),
                    "expiry_date": None,
                    "citics_industry_l1": "CASH",
                    "citics_industry_l2": "CASH",
                    "industry_change_date": None,
                    "industry_rank_rule": "",
                    "industry_rank_date": None,
                    "is_cash_asset": True,
                    "cash_category": "money_fund" if k == 0 else "reverse_repo",
                    "eligible": True,
                }
            )
        return pd.DataFrame(rows)
