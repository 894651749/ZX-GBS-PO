"""组合优化器测试：约束满足、PSD 修正、极端输入行为。"""

from __future__ import annotations

from dataclasses import replace

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from qgbs.portfolio.optimizer import (
    SOLVER_PREFERENCE,
    OptimizerParams,
    PortfolioConstraints,
    solve_portfolio,
)
from qgbs.portfolio.risk import check_psd, ledoit_wolf_cov, nearest_psd

SYMBOLS = [f"S{i:02d}" for i in range(10)]


@pytest.fixture()
def basic_inputs() -> dict:
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.05, (200, len(SYMBOLS)))
    cov = a.T @ a / 200  # PSD
    return {
        "alpha": pd.Series(rng.normal(0, 0.01, len(SYMBOLS)), index=SYMBOLS),
        "cov": cov,
        "w_prev": pd.Series(0.0, index=SYMBOLS),
        "constraints": PortfolioConstraints(
            weight_cap={s: 0.3 for s in SYMBOLS},
            cash_range=(0.0, 0.05),
            exposure_bounds={},
        ),
        "params": OptimizerParams(risk_aversion=5.0, turnover_penalty=1e-3, l2_regularization=1e-4),
    }


def test_solution_satisfies_constraints(basic_inputs: dict) -> None:
    res = solve_portfolio(**basic_inputs)
    assert res.status in ("optimal", "optimal_inaccurate")
    w = res.weights
    cash = res.diagnostics["cash"]
    assert w.sum() + cash == pytest.approx(1.0, abs=1e-5)
    assert (w >= -1e-8).all()  # 无卖空
    assert (w <= 0.3 + 1e-6).all()  # 个股上限
    assert 0.0 - 1e-8 <= cash <= 0.05 + 1e-8  # 现金区间
    assert np.isfinite(res.objective_value)


def test_zero_alpha_reasonable(basic_inputs: dict) -> None:
    res = solve_portfolio(
        alpha=pd.Series(0.0, index=SYMBOLS),
        cov=basic_inputs["cov"],
        w_prev=basic_inputs["w_prev"],
        constraints=basic_inputs["constraints"],
        params=basic_inputs["params"],
    )
    assert res.status in ("optimal", "optimal_inaccurate")
    w = res.weights
    # alpha=0 时风险/正则主导：权重应分散（不超过上限）且满足预算
    assert w.sum() + res.diagnostics["cash"] == pytest.approx(1.0, abs=1e-5)
    assert (w <= 0.3 + 1e-6).all()


def test_non_psd_cov_fixed(basic_inputs: dict) -> None:
    cov = basic_inputs["cov"].copy()
    cov[0, 1] += 5.0  # 破坏 PSD
    assert not check_psd(cov)
    res = solve_portfolio(
        alpha=basic_inputs["alpha"],
        cov=cov,
        w_prev=basic_inputs["w_prev"],
        constraints=basic_inputs["constraints"],
        params=basic_inputs["params"],
    )
    assert res.diagnostics["psd_fixed"] is True
    assert res.status in ("optimal", "optimal_inaccurate")


def test_nearest_psd_properties() -> None:
    rng = np.random.default_rng(1)
    m = rng.normal(0, 1, (6, 6))
    sym = (m + m.T) / 2
    assert not check_psd(sym)  # 随机对称矩阵一般有负特征值
    fixed = nearest_psd(sym)
    assert check_psd(fixed)
    assert np.allclose(fixed, fixed.T)  # 对称化


def test_singular_cov_solvable(basic_inputs: dict) -> None:
    f = np.random.default_rng(2).normal(0, 0.02, 200)
    rets = pd.DataFrame({s: f * (1 + 0.01 * i) for i, s in enumerate(SYMBOLS)})
    cov = ledoit_wolf_cov(rets)
    assert check_psd(cov)
    res = solve_portfolio(
        alpha=basic_inputs["alpha"],
        cov=cov,
        w_prev=basic_inputs["w_prev"],
        constraints=basic_inputs["constraints"],
        params=basic_inputs["params"],
    )
    assert res.status in ("optimal", "optimal_inaccurate")


def test_infeasible_returns_fallback_not_raise(basic_inputs: dict) -> None:
    cons = PortfolioConstraints(
        weight_cap={s: 0.01 for s in SYMBOLS},  # 10×1% + 现金5% < 1 → 不可解
        cash_range=(0.0, 0.05),
        exposure_bounds={},
    )
    res = solve_portfolio(
        alpha=basic_inputs["alpha"],
        cov=basic_inputs["cov"],
        w_prev=basic_inputs["w_prev"],
        constraints=cons,
        params=basic_inputs["params"],
    )
    assert res.status not in ("optimal", "optimal_inaccurate")
    pd.testing.assert_series_equal(res.weights, basic_inputs["w_prev"], check_names=False)  # 回退 w_prev
    assert np.isnan(res.objective_value)


def test_max_turnover_constraint(basic_inputs: dict) -> None:
    w_prev = pd.Series(0.0, index=SYMBOLS)
    w_prev.iloc[:5] = 0.19  # 持仓分散在上限内，问题可行
    cons = PortfolioConstraints(
        weight_cap={s: 0.3 for s in SYMBOLS},
        cash_range=(0.0, 0.05),
        exposure_bounds={},
        max_turnover=0.05,
    )
    res = solve_portfolio(
        alpha=basic_inputs["alpha"],
        cov=basic_inputs["cov"],
        w_prev=w_prev,
        constraints=cons,
        params=basic_inputs["params"],
    )
    assert res.status in ("optimal", "optimal_inaccurate")
    single_side = 0.5 * float((res.weights - w_prev).abs().sum())
    assert single_side <= 0.05 + 1e-5


def test_alpha_uncertainty_shrinks(basic_inputs: dict) -> None:
    unc = pd.Series(0.0, index=SYMBOLS)
    unc.iloc[0] = 1.0  # 第一只股票不确定性极大 → 保守口径下其 alpha 被打到深负
    res = solve_portfolio(
        alpha=basic_inputs["alpha"],
        cov=basic_inputs["cov"],
        w_prev=basic_inputs["w_prev"],
        constraints=basic_inputs["constraints"],
        params=basic_inputs["params"],
        alpha_uncertainty=unc,
    )
    assert res.status in ("optimal", "optimal_inaccurate")
    assert res.weights.iloc[0] <= 1e-6


def test_infeasible_fallback_preserves_budget(basic_inputs: dict) -> None:
    """回退路径账目守恒：weights 回退 w_prev，cash 回退漂移后现金，合计恒为 1。"""
    w_prev = pd.Series(0.0, index=SYMBOLS)
    w_prev.iloc[:4] = 0.2  # 漂移后股票仓位 0.8 → 现金 0.2
    cons = PortfolioConstraints(
        weight_cap={s: 0.01 for s in SYMBOLS},  # 10×1% + 现金上限5% < 1 → 不可解
        cash_range=(0.0, 0.05),
        exposure_bounds={},
    )
    res = solve_portfolio(
        alpha=basic_inputs["alpha"],
        cov=basic_inputs["cov"],
        w_prev=w_prev,
        constraints=cons,
        params=basic_inputs["params"],
    )
    assert res.status not in ("optimal", "optimal_inaccurate")
    pd.testing.assert_series_equal(res.weights, w_prev, check_names=False)  # 回退 w_prev
    assert np.isnan(res.objective_value)
    # 失败路径也携带现金分量（漂移后现金权重），股票权重和 + 现金 = 1
    assert res.diagnostics["cash"] == pytest.approx(0.2)
    assert float(res.weights.sum()) + res.diagnostics["cash"] == pytest.approx(1.0)


def test_uncertainty_kappa_parametrized(basic_inputs: dict) -> None:
    """κ 由 OptimizerParams 传入：κ=0 时不确定性不打折（等价于不传 uncertainty）。"""
    unc = pd.Series(1.0, index=SYMBOLS)
    params0 = replace(basic_inputs["params"], uncertainty_kappa=0.0)
    res_k0 = solve_portfolio(**{**basic_inputs, "params": params0}, alpha_uncertainty=unc)
    res_ref = solve_portfolio(**basic_inputs)  # 不传 uncertainty
    assert res_k0.status in ("optimal", "optimal_inaccurate")
    assert np.allclose(res_k0.weights.to_numpy(), res_ref.weights.to_numpy(), atol=1e-6)


def test_solver_preference_recorded(basic_inputs: dict) -> None:
    """按 SOLVER_PREFERENCE 回退链选定求解器并记入 diagnostics（优先 CLARABEL）。"""
    res = solve_portfolio(**basic_inputs)
    expected = next(s for s in SOLVER_PREFERENCE if s in cp.installed_solvers())
    assert res.diagnostics["solver"] == expected
