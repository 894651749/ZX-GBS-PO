"""回测引擎：逐期执行 预测 → 优化 → 扣费 → 记账。

接口（P1）：

- ``alpha_panel``：再平衡日 × symbol 的 alpha 预测表（由 runner 按滚动
  协议产出）。提供时引擎仅在这些日期调仓，alpha 为当日横截面 z-score
  后的预测值（单调变换，不改变截面排序）；
- 不提供时走内置基线路径（C1 口径）：每月首个交易日调仓，alpha 为
  回看窗历史均值收益的横截面 z-score，协方差用 Ledoit-Wolf 收缩；
- C0 口径：``cov_method="sample"``（样本均值 + 样本协方差经典
  Markowitz 占位；真正的 C0 复现需中信 baseline_results）；
- 交易日：在调仓日收盘价成交（简化），成本直接从净值扣减；
- 不可交易规则：调仓日停牌/涨跌停封板的持仓股票权重**冻结**
  （保持漂移后权重，不参与优化），未持仓的不可交易股票目标权重为 0；
- 现金：由优化器的现金变量承载，按 ``risk_free_daily`` 计息。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from qgbs.backtest.rolling import WindowSpec
from qgbs.data.adapters.base import DailyMarket, TradingStatus
from qgbs.data.calendar import rebalance_dates
from qgbs.portfolio.costs import CostModel
from qgbs.portfolio.optimizer import (
    OptimizerParams,
    PortfolioConstraints,
    solve_portfolio,
)
from qgbs.portfolio.risk import FactorRiskConfig, FactorRiskModel, ledoit_wolf_cov


@dataclass
class BacktestResult:
    """回测输出：净值、持仓、换手与成本流水。"""

    nav: pd.Series  # 扣费后净值序列（日频）
    weights: pd.DataFrame  # 各再平衡期目标权重 (date × symbol)
    turnover: pd.Series  # 各期换手率（单边）
    costs: pd.Series  # 各期交易成本（占净值比例）
    returns: pd.Series = field(default_factory=pd.Series)  # 日收益（含成本影响）
    ledger: pd.DataFrame = field(default_factory=pd.DataFrame)  # 逐期记账明细


class BacktestEngine:
    """滚动回测引擎。

    每个调仓日：取 alpha（外部预测表或内置基线信号）→ 协方差估计 →
    solve_portfolio → 成本模型扣费 → 用下一期实际收益记账。
    所有实验共用同一引擎，保证对照实验口径一致。
    """

    def __init__(
        self,
        window_spec: WindowSpec,
        constraints: PortfolioConstraints,
        optimizer_params: OptimizerParams,
        cost_model: CostModel,
        cov_lookback: int = 120,
        min_history: int = 126,
        cov_method: Literal["ledoit_wolf", "sample", "factor"] = "ledoit_wolf",
        factor_risk_config: FactorRiskConfig | None = None,
    ) -> None:
        self.window_spec = window_spec
        self.constraints = constraints
        self.optimizer_params = optimizer_params
        self.cost_model = cost_model
        self.cov_lookback = cov_lookback
        self.min_history = min_history
        self.cov_method = cov_method
        # 因子风险模型配置（cov_method="factor" 时生效；None = 类默认值）
        self.factor_risk_config = factor_risk_config

    def run(
        self,
        market: DailyMarket,
        status: TradingStatus,
        risk_free_daily: float = 0.0,
        alpha_panel: pd.DataFrame | None = None,
        rebalance_override: pd.DatetimeIndex | None = None,
        exposure_matrix: pd.DataFrame | None = None,
        alpha_uncertainty_panel: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """执行完整滚动回测。

        Args:
            market: 日频行情长表（close/daily_return/amount 列）。
            status: 日频交易状态长表（buyable=False 禁止加仓、sellable=False
                冻结持仓；buyable=False 且 sellable=True 的持仓只减不增——
                派生业务池非成分期即按此标记）。
            risk_free_daily: 现金部分的日化利率。
            alpha_panel: 再平衡日 × symbol 的 alpha 预测表；提供时仅在这些
                日期调仓；为空走内置基线信号（历史均值 z-score）。
            rebalance_override: 显式指定调仓时点（用于无监督基线实验与
                监督实验共用同一样本外调仓日程）。
            exposure_matrix: symbol × factor 暴露矩阵（如行业哑变量），
                调仓时传入优化器使 exposure_bounds 约束真实生效。
            alpha_uncertainty_panel: 再平衡日 × symbol 的 alpha 不确定性
                （方案书 §5.4 链路，逐日截面 z-score 后送入优化器）。
        """
        close = market.rows.pivot(index="trade_date", columns="symbol", values="close").sort_index()
        ret = market.rows.pivot(index="trade_date", columns="symbol", values="daily_return").sort_index()
        amount = market.rows.pivot(index="trade_date", columns="symbol", values="amount").sort_index()
        dates = close.index
        buyable, sellable = self._trade_status_masks(status, close)

        if alpha_panel is not None:
            rebal = [d for d in alpha_panel.index if d in dates]
        elif rebalance_override is not None:
            rebal = [d for d in rebalance_override if d in dates]
        else:
            rebal = [
                d for d in rebalance_dates(dates, self.window_spec.test)
                if dates.get_loc(d) >= self.min_history
            ]

        symbols = list(close.columns)
        w = pd.Series(0.0, index=symbols)  # 当前漂移权重
        cash = 1.0
        nav = 1.0
        nav_idx: list[pd.Timestamp] = []
        nav_val: list[float] = []
        ret_idx: list[pd.Timestamp] = []
        ret_val: list[float] = []
        weight_rows: dict[pd.Timestamp, pd.Series] = {}
        turnover_map: dict[pd.Timestamp, float] = {}
        cost_map: dict[pd.Timestamp, float] = {}
        ledger_rows: list[dict] = []

        started = False
        for d in dates:
            day_ret = ret.loc[d].fillna(0.0)
            port_ret = float((w * day_ret).sum() + cash * risk_free_daily)
            nav_before = nav
            nav *= 1.0 + port_ret
            # 权重漂移
            gross = w * (1.0 + day_ret)
            total_gross = float(gross.sum() + cash * (1.0 + risk_free_daily))
            if total_gross > 0:
                w = gross / total_gross
                cash = cash * (1.0 + risk_free_daily) / total_gross

            if d in rebal:
                w, cash, cost_frac, to, info = self._rebalance(
                    d, close, ret, buyable, sellable, w, cash, alpha_panel,
                    amount.loc[d] if d in amount.index else None,
                    exposure_matrix,
                    alpha_uncertainty_panel.loc[d] if (
                        alpha_uncertainty_panel is not None and d in alpha_uncertainty_panel.index
                    ) else None,
                )
                nav *= 1.0 - cost_frac
                weight_rows[d] = w.copy()
                turnover_map[d] = to
                cost_map[d] = cost_frac
                ledger_rows.append({"date": d, "turnover": to, "cost": cost_frac, **info})
                started = True

            if started:
                nav_idx.append(d)
                nav_val.append(nav)
                ret_idx.append(d)
                ret_val.append(nav / nav_before - 1.0)

        weights_df = pd.DataFrame(weight_rows).T if weight_rows else pd.DataFrame()
        return BacktestResult(
            nav=pd.Series(nav_val, index=pd.DatetimeIndex(nav_idx), name="nav"),
            weights=weights_df,
            turnover=pd.Series(turnover_map, name="turnover"),
            costs=pd.Series(cost_map, name="cost"),
            returns=pd.Series(ret_val, index=pd.DatetimeIndex(ret_idx), name="return"),
            ledger=pd.DataFrame(ledger_rows),
        )

    @staticmethod
    def _align_mask(df: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
        return df.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)

    def _trade_status_masks(
        self, status: TradingStatus, close: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """buyable / sellable 两个布尔矩阵（交易日 × symbol）。

        分开返回以支持"只卖不买"语义：buyable=False 且 sellable=True 的
        股票（如派生业务池的非成分期）允许减仓但不允许加仓。
        """
        buy = status.rows.pivot(index="trade_date", columns="symbol", values="buyable")
        sell = status.rows.pivot(index="trade_date", columns="symbol", values="sellable")
        return self._align_mask(buy, close), self._align_mask(sell, close)

    def _alpha_at(
        self,
        d: pd.Timestamp,
        sleeve: list[str],
        hist: pd.DataFrame,
        alpha_panel: pd.DataFrame | None,
    ) -> pd.Series:
        """取调仓日 alpha：外部预测表优先，否则内置基线信号。"""
        if alpha_panel is not None:
            raw = alpha_panel.loc[d].reindex(sleeve).astype(float).fillna(0.0)
        else:
            # 内置基线（C1 口径）：回看窗历史均值收益
            raw = hist[sleeve].mean()
        std = float(raw.std())
        return (raw - float(raw.mean())) / (std if std > 0 else 1.0)

    def _rebalance(
        self,
        d: pd.Timestamp,
        close: pd.DataFrame,
        ret: pd.DataFrame,
        buyable: pd.DataFrame,
        sellable: pd.DataFrame,
        w: pd.Series,
        cash: float,
        alpha_panel: pd.DataFrame | None,
        day_amount: pd.Series | None = None,
        exposure_matrix: pd.DataFrame | None = None,
        alpha_uncertainty: pd.Series | None = None,
    ) -> tuple[pd.Series, float, float, float, dict]:
        """在调仓日 d（收盘后）计算目标权重、换手与成本。"""
        hist = ret.loc[:d].tail(self.cov_lookback)
        valid_cols = hist.columns[hist.notna().all()]
        sellable_today = sellable.loc[d]
        buyable_today = buyable.loc[d]

        # 冻结：持仓且当日不可卖的股票保持漂移后权重（可卖不可买的持仓不冻结，
        # 进 sleeve 以"只减不增"上限参与优化）
        held = w[w > 1e-12]
        frozen = held[~sellable_today[held.index].astype(bool)]
        frozen_sum = float(frozen.sum())

        sleeve = [
            s for s in valid_cols
            if sellable_today.get(s, False) and s not in frozen.index
        ]
        budget = 1.0 - frozen_sum
        if not sleeve or budget <= 1e-9:
            return w, cash, 0.0, 0.0, {"status": "skipped", "cash": cash, "n_holdings": int((w > 1e-12).sum())}

        alpha = self._alpha_at(d, sleeve, hist, alpha_panel)

        cov_note = None
        if self.cov_method == "sample":
            cov = hist[sleeve].cov().to_numpy()  # C0 口径：样本协方差（非 PSD 由优化器修正）
        elif self.cov_method == "factor":
            # 因子风险模型（方案书 §6.1）：B = 行业哑变量/风格暴露
            if exposure_matrix is None:
                cov = ledoit_wolf_cov(hist[sleeve])
                cov_note = "degraded: cov_method='factor' 但无暴露矩阵，回退 Ledoit-Wolf"
            else:
                frm = FactorRiskModel(self.factor_risk_config).fit(
                    hist[sleeve], exposure_matrix.reindex(index=sleeve).fillna(0.0)
                )
                cov = frm.covariance()
        else:
            cov = ledoit_wolf_cov(hist[sleeve])

        # alpha 不确定性：与 alpha 同口径截面 z-score（单位一致）
        unc_sleeve = None
        if alpha_uncertainty is not None:
            u = alpha_uncertainty.reindex(sleeve).astype(float)
            u = u.fillna(u.median() if u.notna().any() else 0.0)
            ustd = float(u.std())
            unc_sleeve = (u - float(u.mean())) / (ustd if ustd > 0 else 1.0)

        w_prev_sleeve = w.reindex(sleeve).fillna(0.0) / budget
        cons = PortfolioConstraints(
            weight_cap={
                # 只可减仓：buyable=False（如派生业务池非成分期）的持仓上限为当前权重
                s: min(self.constraints.weight_cap.get(s, 1.0) / budget, 1.0)
                if buyable_today.get(s, False)
                else min(float(w_prev_sleeve.get(s, 0.0)), 1.0)
                for s in sleeve
            },
            cash_range=(
                self.constraints.cash_range[0] / budget,
                min(self.constraints.cash_range[1] / budget, 1.0),
            ),
            exposure_bounds={
                k: (lo / budget, hi / budget)
                for k, (lo, hi) in self.constraints.exposure_bounds.items()
            },
            weight_floor={s: self.constraints.weight_floor.get(s, 0.0) / budget for s in sleeve},
            max_turnover=(
                None if self.constraints.max_turnover is None
                else self.constraints.max_turnover / budget
            ),
            liquidity_limit=self.constraints.liquidity_limit,
            blacklist=tuple(s for s in self.constraints.blacklist if s in sleeve),
        )
        expo_sleeve = (
            exposure_matrix.reindex(index=sleeve).fillna(0.0) if exposure_matrix is not None else None
        )
        result = solve_portfolio(
            alpha, cov, w_prev_sleeve, cons, self.optimizer_params,
            exposure_matrix=expo_sleeve,
            alpha_uncertainty=unc_sleeve,
            day_amount=day_amount.reindex(sleeve) if day_amount is not None else None,
        )

        w_target = pd.Series(0.0, index=w.index)
        w_target[result.weights.index] = result.weights * budget
        w_target[frozen.index] = frozen
        # 优化器成功/失败两条路径均保证 diagnostics["cash"]（失败时为漂移后现金权重）
        cash_target = float(result.diagnostics["cash"]) * budget

        trades = w_target - w
        turnover = 0.5 * float(trades.abs().sum())
        # 成本：volumes 传入当日成交额占比（与 trades 的净值占比同口径），冲击项生效
        vol_share = None
        if day_amount is not None:
            am = day_amount.reindex(trades.index).fillna(0.0)
            total_am = float(am.sum())
            if total_am > 0:
                vol_share = am / total_am
        cost_frac = self.cost_model.total_cost(trades, volumes=vol_share)
        info = {
            "status": result.status,
            "cash": cash_target,
            "n_holdings": int((w_target > 1e-12).sum()),
        }
        if cov_note:
            info["cov_note"] = cov_note
        if "liquidity_limit" in result.diagnostics:
            info["liquidity_limit"] = result.diagnostics["liquidity_limit"]
        return w_target, cash_target, cost_frac, turnover, info
