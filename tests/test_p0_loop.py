"""P0 端到端冒烟：mock 数据 → 占位 alpha → LW 风险 → cvxpy 优化 → 回测 → 指标。

可直接运行查看关键输出：``.venv/bin/python -m tests.test_p0_loop``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qgbs.backtest.engine import BacktestEngine, BacktestResult
from qgbs.backtest.metrics import (
    annual_return,
    annual_volatility,
    max_drawdown,
    sharpe,
    sortino,
    total_return,
    turnover,
)
from qgbs.backtest.rolling import WindowSpec
from qgbs.data.adapters.mock import MockAdapter, MockDataConfig
from qgbs.portfolio.costs import CostModel
from qgbs.portfolio.optimizer import OptimizerParams, PortfolioConstraints

P0_CONFIG = MockDataConfig(
    n_assets=40,
    start="2020-01-01",
    end="2022-06-30",
    n_industries=5,
    n_sub_industries=10,
    seed=42,
)


def run_p0_loop(config: MockDataConfig = P0_CONFIG) -> BacktestResult:
    adapter = MockAdapter(config)
    symbols = [f"MOCK{i + 1:04d}" for i in range(config.n_assets)]
    market = adapter.load_daily_market(symbols, config.start, config.end)
    status = adapter.load_trading_status(symbols, config.start, config.end)
    cc = adapter.load_constraints_costs("default")
    engine = BacktestEngine(
        window_spec=WindowSpec(train="1y", valid="3m", test="1m", purge="5d", embargo="2d"),
        constraints=PortfolioConstraints(
            weight_cap={s: 0.20 for s in symbols},
            cash_range=tuple(cc.cash_range),
            exposure_bounds={},
        ),
        optimizer_params=OptimizerParams(
            risk_aversion=5.0, turnover_penalty=1e-3, l2_regularization=1e-4
        ),
        cost_model=CostModel(
            commission_bps=cc.commission_bps,
            stamp_tax_bps=cc.stamp_tax_bps,
            slippage_bps=cc.slippage_bps,
            impact_coef=cc.impact_coef,
        ),
    )
    return engine.run(market, status, risk_free_daily=config.rf_annual / 252.0)


@pytest.fixture(scope="module")
def result() -> BacktestResult:
    return run_p0_loop()


def test_nav_length_and_monotone_index(result: BacktestResult) -> None:
    assert len(result.nav) > 300  # 调仓启动后覆盖至样本末端
    assert result.nav.index.is_monotonic_increasing
    assert (result.nav > 0).all()


def test_weights_budget_and_counts(result: BacktestResult) -> None:
    assert len(result.weights) >= 15  # 月度调仓期数
    cash = result.ledger.set_index("date")["cash"]
    total = result.weights.sum(axis=1) + cash.reindex(result.weights.index)
    assert ((total - 1.0).abs() < 1e-4).all()
    assert (result.weights >= -1e-8).values.all()
    assert (result.weights <= 0.20 + 1e-5).values.all()


def test_metrics_computable(result: BacktestResult) -> None:
    rets = result.returns
    assert np.isfinite(sharpe(rets))
    assert np.isfinite(sortino(rets))
    assert np.isfinite(annual_return(rets))
    assert np.isfinite(annual_volatility(rets))
    mdd = max_drawdown(result.nav)
    assert 0.0 <= mdd <= 1.0
    to = turnover(result.weights)
    assert (to >= 0).all() and (to <= 1.0 + 1e-9).all()
    assert (result.costs >= 0).all()


def test_reproducible_same_seed() -> None:
    r1 = run_p0_loop()
    r2 = run_p0_loop()
    pd.testing.assert_series_equal(r1.nav, r2.nav)
    pd.testing.assert_frame_equal(r1.weights, r2.weights)


if __name__ == "__main__":
    res = run_p0_loop()
    rets = res.returns
    print("=== P0 最小闭环 demo ===")
    print(f"调仓期数        : {len(res.weights)}")
    print(f"NAV 序列长度    : {len(res.nav)}（{res.nav.index[0].date()} → {res.nav.index[-1].date()}）")
    print(f"NAV 末值        : {res.nav.iloc[-1]:.4f}")
    print(f"总收益          : {total_return(res.nav):.2%}")
    print(f"年化收益        : {annual_return(rets):.2%}")
    print(f"年化波动        : {annual_volatility(rets):.2%}")
    print(f"Sharpe          : {sharpe(rets):.3f}")
    print(f"Sortino         : {sortino(rets):.3f}")
    print(f"最大回撤        : {max_drawdown(res.nav):.2%}")
    print(f"平均单边换手    : {res.turnover.mean():.2%}")
    print(f"累计成本(净值比): {res.costs.sum():.4%}")
    print(f"求解状态分布    : {res.ledger['status'].value_counts().to_dict()}")
