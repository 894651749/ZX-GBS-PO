"""修正 3 测试：暴露约束生效、流动性约束实现/降级、冲击成本链路。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qgbs.portfolio.costs import CostModel
from qgbs.portfolio.optimizer import (
    OptimizerParams,
    PortfolioConstraints,
    solve_portfolio,
)

SYMBOLS = [f"S{i:02d}" for i in range(10)]
PARAMS = OptimizerParams(risk_aversion=5.0, turnover_penalty=1e-3, l2_regularization=1e-4)


@pytest.fixture()
def base_inputs() -> dict:
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.05, (200, len(SYMBOLS)))
    return {
        "alpha": pd.Series(rng.normal(0, 0.01, len(SYMBOLS)), index=SYMBOLS),
        "cov": a.T @ a / 200,
        "w_prev": pd.Series(0.0, index=SYMBOLS),
        "params": PARAMS,
    }


def _expo() -> pd.DataFrame:
    # 前 5 只属 IND1，后 5 只属 IND2
    return pd.DataFrame(
        {"IND1": [1.0] * 5 + [0.0] * 5, "IND2": [0.0] * 5 + [1.0] * 5},
        index=SYMBOLS,
    )


def test_exposure_bounds_bind(base_inputs: dict) -> None:
    alpha = base_inputs["alpha"].copy()
    alpha.iloc[:5] = 1.0  # IND1 全是最优 alpha
    cons_free = PortfolioConstraints(
        weight_cap={s: 0.3 for s in SYMBOLS}, cash_range=(0.0, 0.05), exposure_bounds={}
    )
    cons_tight = PortfolioConstraints(
        weight_cap={s: 0.3 for s in SYMBOLS}, cash_range=(0.0, 0.05),
        exposure_bounds={"IND1": (0.0, 0.2)},  # 收紧 IND1 暴露
    )
    expo = _expo()
    res_free = solve_portfolio(**{**base_inputs, "alpha": alpha}, constraints=cons_free, exposure_matrix=expo)
    res_tight = solve_portfolio(**{**base_inputs, "alpha": alpha}, constraints=cons_tight, exposure_matrix=expo)
    assert res_free.status == "optimal" and res_tight.status == "optimal"
    # 收紧区间改变权重
    assert not np.allclose(res_free.weights.to_numpy(), res_tight.weights.to_numpy())
    # 暴露约束真实生效
    ind1_expo = float(res_tight.weights.iloc[:5].sum())
    assert ind1_expo <= 0.2 + 1e-3  # 求解器数值容差


def test_liquidity_limit_with_amounts(base_inputs: dict) -> None:
    amounts = pd.Series(np.linspace(1.0, 2.0, 10), index=SYMBOLS) * 1e8
    cons = PortfolioConstraints(
        weight_cap={s: 0.5 for s in SYMBOLS}, cash_range=(0.0, 0.9),
        exposure_bounds={}, liquidity_limit=0.10,
    )
    res = solve_portfolio(**base_inputs, constraints=cons, day_amount=amounts)
    assert res.status == "optimal"
    lim = 0.10 * amounts / amounts.sum()
    assert ((res.weights - base_inputs["w_prev"]).abs() <= lim + 1e-3).all()  # 求解器数值容差


def test_liquidity_limit_degrades_gracefully(base_inputs: dict) -> None:
    cons = PortfolioConstraints(
        weight_cap={s: 0.5 for s in SYMBOLS}, cash_range=(0.0, 0.9),
        exposure_bounds={}, liquidity_limit=0.10,
    )
    res = solve_portfolio(**base_inputs, constraints=cons)  # 无 day_amount
    assert res.status == "optimal"
    assert "degraded" in res.diagnostics.get("liquidity_limit", "")


def test_impact_cost_link() -> None:
    trades = pd.Series({"a": 0.10, "b": -0.05})
    volumes = pd.Series({"a": 0.2, "b": 0.1})  # 成交额占比口径
    c0 = CostModel(impact_coef=0.0)
    c1 = CostModel(impact_coef=0.5)
    base = c0.trade_cost(trades, volumes)
    with_impact = c1.trade_cost(trades, volumes)
    assert (with_impact > base).all()
    # 冲击项 = coef · |t| · min(|t|/v, 1)（参与率封顶 1）
    expected_impact_a = 0.5 * 0.10 * min(0.10 / 0.2, 1.0)
    assert (with_impact["a"] - base["a"]) == pytest.approx(expected_impact_a)
    # 无 volumes 时冲击项不生效
    assert c1.trade_cost(trades).equals(c1.trade_cost(trades, None))
