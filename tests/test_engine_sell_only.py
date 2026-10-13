"""引擎 sell-only 语义：buyable=False 且 sellable=True 的持仓只减不增（派生业务池口径）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qgbs.backtest.engine import BacktestEngine
from qgbs.backtest.rolling import WindowSpec
from qgbs.data.adapters.base import DailyMarket, TradingStatus
from qgbs.portfolio.costs import CostModel
from qgbs.portfolio.optimizer import OptimizerParams, PortfolioConstraints

SPEC = WindowSpec(train="1y", valid="3m", test="1m", purge="5d", embargo="2d")
SYMS = ["S0", "S1", "S2"]


def _mk_inputs(n_days: int = 60, sell_only_days: set | None = None) -> tuple[DailyMarket, TradingStatus]:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rows = []
    for s_i, s in enumerate(SYMS):
        rets = rng.normal(0.0002 * (s_i + 1), 0.01, n_days)
        close = 100.0 * np.cumprod(1.0 + rets)
        for d, c, r in zip(dates, close, rets):
            rows.append({"symbol": s, "trade_date": d, "close": c, "daily_return": r, "amount": 1e8})
    market = DailyMarket(rows=pd.DataFrame(rows))
    sell_only_days = sell_only_days or set()
    srows = []
    for s in SYMS:
        for d in dates:
            sell_only = d in sell_only_days and s == "S0"
            srows.append({"symbol": s, "trade_date": d, "buyable": not sell_only, "sellable": True})
    status = TradingStatus(rows=pd.DataFrame(srows))
    return market, status


def _engine() -> BacktestEngine:
    return BacktestEngine(
        window_spec=SPEC,
        constraints=PortfolioConstraints(
            weight_cap={s: 0.9 for s in SYMS}, cash_range=(0.0, 0.05), exposure_bounds={}
        ),
        optimizer_params=OptimizerParams(risk_aversion=1.0, turnover_penalty=0.0, l2_regularization=1e-6),
        cost_model=CostModel(commission_bps=0, stamp_tax_bps=0, slippage_bps=0, impact_coef=0),
        cov_lookback=20,
        min_history=0,
    )


def test_sell_only_stock_not_added_and_capped_at_prev() -> None:
    """S0 在第二调仓日 buyable=False：目标权重不得超过漂移后权重（只减不增）。"""
    dates = pd.bdate_range("2024-01-01", periods=60)
    d1, d2 = dates[25], dates[45]
    market, status = _mk_inputs(60, sell_only_days={d2})
    alpha = pd.DataFrame(
        {"S0": [3.0, 3.0], "S1": [1.0, 0.5], "S2": [-1.0, -0.5]}, index=[d1, d2]
    )
    bt = _engine().run(market, status, risk_free_daily=0.0, alpha_panel=alpha)
    w1 = bt.weights.loc[d1]
    assert w1["S0"] > 0.5  # 第一期 S0 alpha 最高，正常重仓买入
    # 两日间收益漂移后的权重（ret>0 → 略高于 w1）
    drift = float(w1["S0"] * 1.02)
    w2 = bt.weights.loc[d2]
    assert w2["S0"] <= drift  # sell-only：不得加仓（alpha 仍最高也不加）
    assert w2["S1"] + w2["S2"] > 0  # 其余正常建仓


def test_sell_only_never_buys_unheld() -> None:
    """未持仓股票在 buyable=False 的调仓日目标权重为 0。"""
    dates = pd.bdate_range("2024-01-01", periods=60)
    d1, d2 = dates[25], dates[45]
    market, status = _mk_inputs(60, sell_only_days={d1, d2})
    # 两期 S0 均 buyable=False；第二期 S0 alpha 变最高也不建仓
    alpha = pd.DataFrame(
        {"S0": [-3.0, 3.0], "S1": [1.0, 0.5], "S2": [-1.0, -0.5]}, index=[d1, d2]
    )
    bt = _engine().run(market, status, risk_free_daily=0.0, alpha_panel=alpha)
    assert bt.weights.loc[d1, "S0"] == 0.0
    assert bt.weights.loc[d2, "S0"] == 0.0  # 未持仓 + 不可买 → 不建仓


def test_unsellable_position_stays_frozen() -> None:
    """sellable=False（停牌口径）持仓仍冻结：目标权重 = 漂移后权重。"""
    dates = pd.bdate_range("2024-01-01", periods=60)
    d1, d2 = dates[25], dates[45]
    market, status = _mk_inputs(60)
    # d2 日 S0 完全不可交易
    m = (status.rows["symbol"] == "S0") & (status.rows["trade_date"] == d2)
    status.rows.loc[m, ["buyable", "sellable"]] = False
    alpha = pd.DataFrame(
        {"S0": [3.0, -3.0], "S1": [1.0, 2.0], "S2": [-1.0, 1.0]}, index=[d1, d2]
    )
    bt = _engine().run(market, status, risk_free_daily=0.0, alpha_panel=alpha)
    assert bt.weights.loc[d1, "S0"] > 0.5
    # 冻结：d2 目标权重等于漂移权重（成本为零 → 换手不含 S0）
    ledger_d2 = bt.ledger[bt.ledger["date"] == d2].iloc[0]
    assert ledger_d2["status"] == "optimal"
    assert abs(bt.weights.loc[d2, "S0"] - bt.weights.loc[d1, "S0"]) < 0.05  # 仅漂移
