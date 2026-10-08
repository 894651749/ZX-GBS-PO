"""cvxpy 稳健均值-方差组合优化（方案书 6.2 节）。

目标函数：max αᵀw − λwᵀΣw − τ‖w−w_prev‖₁ − γ‖w‖₂²
约束：权重和为 1（含现金变量 c）、无卖空、个股权重上下限、现金比例区间、
（可选）行业/风格暴露区间、（可选）最大换手、禁止投资名单。

不可解 / 求解器失败时返回带状态的 :class:`OptimizeResult`（权重回退为
w_prev、现金回退为漂移后现金权重 1 − Σw_prev），不抛异常；成功与失败
两条路径的 diagnostics 均携带 ``cash`` 键，保证 股票权重和 + 现金 = 1。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from qgbs.portfolio.risk import check_psd, nearest_psd

#: alpha_uncertainty 保守口径的折扣系数 κ（alpha − κ·uncertainty）默认值，
#: 实际取值以 :class:`OptimizerParams`.uncertainty_kappa 为准
DEFAULT_UNCERTAINTY_KAPPA = 1.0

#: cvxpy 求解器回退链：优先 CLARABEL（内点法，QP 精度稳定），依次退到 SCS、OSQP；
#: 全部不可用/报错时以 solver_error 状态返回
SOLVER_PREFERENCE: tuple[str, ...] = ("CLARABEL", "SCS", "OSQP")


@dataclass(frozen=True)
class PortfolioConstraints:
    """组合约束（方案书 §8.2(6)）。"""

    weight_cap: dict[str, float]  # 个股上限（symbol → 权重上限）
    cash_range: tuple[float, float]  # 现金比例区间
    exposure_bounds: dict[str, tuple[float, float]]  # 行业/风格暴露区间
    weight_floor: dict[str, float] = field(default_factory=dict)  # 个股下限
    max_turnover: float | None = None  # 单期最大换手（单边）
    liquidity_limit: float | None = None  # 流动性限制（如单日成交额占比上限）
    blacklist: tuple[str, ...] = ()  # 禁止投资名单


@dataclass(frozen=True)
class OptimizerParams:
    """目标函数系数。"""

    risk_aversion: float  # λ
    turnover_penalty: float  # τ
    l2_regularization: float  # γ
    uncertainty_kappa: float = DEFAULT_UNCERTAINTY_KAPPA  # κ：alpha − κ·uncertainty 折扣


@dataclass(frozen=True)
class OptimizeResult:
    """优化结果。"""

    weights: pd.Series  # 最优股票权重（索引为 symbol；不含现金）
    status: str  # 求解器状态（optimal / infeasible / solver_error 等）
    objective_value: float  # 目标函数值（失败时为 NaN）
    diagnostics: dict[str, Any] = field(default_factory=dict)  # 求解诊断（现金权重、PSD 修正等）


def solve_portfolio(
    alpha: pd.Series,
    cov: np.ndarray,
    w_prev: pd.Series,
    constraints: PortfolioConstraints,
    params: OptimizerParams,
    exposure_matrix: pd.DataFrame | None = None,
    alpha_uncertainty: pd.Series | None = None,
    day_amount: pd.Series | None = None,
) -> OptimizeResult:
    """求解稳健均值-方差组合。

    Args:
        alpha: 截面预期收益预测（索引为 symbol）。
        cov: 资产协方差矩阵（非 PSD 时先做特征值截断投影）。
        w_prev: 上一期权重（换手惩罚基准，索引与 alpha 一致）。
        constraints: 组合约束。
        params: 目标函数系数。
        exposure_matrix: 行业/风格暴露矩阵（symbol × factor），
            用于暴露区间约束；为空则跳过暴露约束。
        alpha_uncertainty: 各资产 alpha 预测不确定性（方案书 §5.4 链路），
            提供时以 alpha − κ·uncertainty 的保守口径进入目标函数
            （κ 取 ``params.uncertainty_kappa``）；为空退化为点估计。
        day_amount: 调仓日各资产成交额。``liquidity_limit`` 设置时：
            提供则实现逐名交易额上限 |Δw_s| ≤ liquidity_limit × amount_s/Σamount；
            缺失则显式降级（跳过并在 diagnostics 标注）。

    Returns:
        求解成功：status="optimal"（或 solver 实际状态），weights 为最优权重，
        diagnostics["cash"] 为现金权重，diagnostics["solver"] 为实际使用的
        求解器（按 SOLVER_PREFERENCE 回退链选择）；失败：status 记录原因，
        weights 回退为 w_prev，diagnostics["cash"] 回退为漂移后现金权重
        1 − Σw_prev，objective_value 为 NaN。
    """
    symbols = list(alpha.index)
    n = len(symbols)
    a = alpha.to_numpy(dtype=float)
    w0 = w_prev.reindex(symbols).fillna(0.0).to_numpy(dtype=float)
    if alpha_uncertainty is not None:
        a = a - params.uncertainty_kappa * alpha_uncertainty.reindex(symbols).fillna(0.0).to_numpy()

    cov = np.asarray(cov, dtype=float)
    psd_fixed = False
    if not check_psd(cov):
        cov = nearest_psd(cov)
        psd_fixed = True
    # 上面已完成稠密 PSD 校验/修复；psd_wrap 跳过 cvxpy 基于 ARPACK 的 PSD 认证，
    # 该认证在大规模矩阵（数百资产）上存在数值不收敛问题（ARPACK NoConvergence）
    cov_expr = cp.psd_wrap(cov)

    upper = np.array([constraints.weight_cap.get(s, 1.0) for s in symbols])
    lower = np.array([constraints.weight_floor.get(s, 0.0) for s in symbols])
    blacklist_mask = np.array([s in constraints.blacklist for s in symbols])
    upper = np.where(blacklist_mask, 0.0, upper)
    lower = np.where(blacklist_mask, 0.0, lower)
    upper = np.maximum(upper, lower)

    w = cp.Variable(n)
    cash = cp.Variable()
    objective = (
        a @ w
        - params.risk_aversion * cp.quad_form(w, cov_expr)
        - params.turnover_penalty * cp.norm1(w - w0)
        - params.l2_regularization * cp.sum_squares(w)
    )
    cons: list[cp.Constraint] = [
        cp.sum(w) + cash == 1.0,
        w >= lower,
        w <= upper,
        cash >= constraints.cash_range[0],
        cash <= constraints.cash_range[1],
    ]
    if constraints.max_turnover is not None:
        cons.append(0.5 * cp.norm1(w - w0) <= constraints.max_turnover)
    if exposure_matrix is not None and constraints.exposure_bounds:
        expo = exposure_matrix.reindex(symbols).fillna(0.0)
        for factor, (lo, hi) in constraints.exposure_bounds.items():
            if factor in expo.columns:
                cons.append(expo[factor].to_numpy() @ w >= lo)
                cons.append(expo[factor].to_numpy() @ w <= hi)
    liquidity_note: str | None = None
    if constraints.liquidity_limit is not None:
        if day_amount is not None and float(day_amount.sum()) > 0:
            am = day_amount.reindex(symbols).fillna(0.0).to_numpy(dtype=float)
            lim = constraints.liquidity_limit * am / am.sum()  # 逐名交易额占当日成交比上限
            cons.append(cp.abs(w - w0) <= lim)
        else:
            liquidity_note = "degraded: liquidity_limit 已设置但无成交额数据，约束跳过"

    problem = cp.Problem(cp.Maximize(objective), cons)
    diagnostics: dict[str, Any] = {"psd_fixed": psd_fixed}
    if liquidity_note:
        diagnostics["liquidity_limit"] = liquidity_note
    fallback = pd.Series(w0, index=symbols, name="weight")
    # 回退路径的现金分量：与 w_prev 漂移状态守恒（Σw_prev + cash = 1），
    # 使回退结果与成功路径一样满足 股票权重和 + 现金 = 1
    fallback_cash = 1.0 - float(w0.sum())

    solved = False
    errors: list[str] = []
    for solver in SOLVER_PREFERENCE:
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=solver)
        except cp.SolverError as exc:
            errors.append(f"{solver}: {exc}")
            continue
        diagnostics["solver"] = solver
        solved = True
        break
    if not solved:
        reason = "; ".join(errors) if errors else f"无可用求解器（回退链 {SOLVER_PREFERENCE} 均未安装）"
        return OptimizeResult(
            weights=fallback, status="solver_error",
            objective_value=float("nan"),
            diagnostics={**diagnostics, "cash": fallback_cash, "error": reason},
        )

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or w.value is None:
        return OptimizeResult(
            weights=fallback, status=problem.status or "failed",
            objective_value=float("nan"),
            diagnostics={**diagnostics, "cash": fallback_cash},
        )

    w_opt = np.clip(np.asarray(w.value).ravel(), lower, upper)
    diagnostics["cash"] = float(cash.value)
    return OptimizeResult(
        weights=pd.Series(w_opt, index=symbols, name="weight"),
        status=problem.status,
        objective_value=float(problem.value),
        diagnostics=diagnostics,
    )
