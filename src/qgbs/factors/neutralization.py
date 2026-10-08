"""市场+行业联合因子模型与 alpha 标签构建。

口径约定：现金资产单独走无风险利率/现金管理收益率口径，
不进入股票 alpha 模型（接口分离，见 :func:`cash_return_label`）。

因子模型（方案书 §5.1）：训练窗内对每只股票做含截距的时间序列 OLS——

    r_i,t = a_i + β_m,i·MKT_t + β_ind,i·IND_{k(i),t} + Σ_s β_s,i·f_s,t + ε_i,t

回归必须含截距：因子收益均值非零时，无截距回归的 β̂ 有偏。
截距 a_i 是个股专属 alpha——标签只扣因子部分
（label_i,t = 未来 h 日收益 − β̂_iᵀ·未来 h 日因子收益），
a_i 留在标签中由读出模型学习。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MARKET_COL = "MKT"


@dataclass(frozen=True)
class NeutralizationResult:
    """中性化回归结果：残差（alpha）、暴露与拟合统计量。"""

    alpha: pd.DataFrame  # (date × symbol) 完整残差 r − a_i − β̂ᵀf；缺因子收益的日期为 NaN
    exposures: pd.DataFrame  # 因子暴露估计（symbol × factor），不含截距
    r_squared: pd.Series  # 各股票训练窗回归 R²（索引为 symbol）


def _factor_projection(
    factor_panel: pd.DataFrame,
    beta: pd.DataFrame,
    index: pd.Index,
    columns: pd.Index,
) -> tuple[np.ndarray, np.ndarray]:
    """计算 β̂ᵀf 投影矩阵，并标记缺所需因子收益的 (行, 列) 位置。

    factor_panel 对齐到 (index, beta.columns)；因子缺失值在投影中按 0
    参与计算，由返回的掩码标记“该股票暴露非零的任一因子当日缺失”的位置，
    调用方须将这些位置置 NaN（不得以未扣因子收益的值冒充 alpha）。

    Returns:
        (projection, invalid)：projection 为 (index × columns) 的 β̂ᵀf
        矩阵；invalid 为同形状布尔掩码，True 表示因子数据不足。
    """
    beta_arr = beta.T.reindex(columns=columns, fill_value=0.0).to_numpy()  # factor × column
    F = factor_panel.reindex(index=index, columns=beta.columns).to_numpy()
    invalid = np.isnan(F) @ (beta_arr != 0.0)
    projection = np.nan_to_num(F, nan=0.0) @ beta_arr
    return projection, invalid


class JointFactorModel:
    """市场+行业（+风格）联合因子模型（含个股截距 a_i）。

    fit 仅允许在训练期调用；现金资产不进入本模型。
    因子收益约定：``factor_returns`` 须含 "MKT" 列与各行业代码列
    （如 "IND01"），风格因子列名由 ``style_factors`` 指定。
    截距 a_i 单独保存，不进入 :attr:`exposures_`。
    """

    def __init__(self, style_factors: list[str] | None = None) -> None:
        self.style_factors = style_factors or []
        self._fitted: bool = False
        self._beta: pd.DataFrame | None = None  # symbol × factor
        self._intercept: pd.Series | None = None  # symbol → a_i
        self._r_squared: pd.Series | None = None

    def fit(
        self,
        returns: pd.DataFrame,
        industry: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> "JointFactorModel":
        """在训练窗内估计各股票的截距 a_i 与因子暴露 β̂（含截距 OLS）。

        每只股票只对实际用到的因子列（MKT + 所属行业列 + 风格因子列）
        要求非缺失；其他因子列的 NaN 不会导致整天被丢弃。

        Args:
            returns: 股票收益（date × symbol），不含现金资产。
            industry: 行业分类（索引 symbol，值为行业代码）。
            factor_returns: 因子收益（date × factor），须含 MKT 与行业列。
        """
        if MARKET_COL not in factor_returns.columns:
            raise ValueError(f"factor_returns 缺少市场因子列 {MARKET_COL!r}")
        common = returns.index.intersection(factor_returns.index)
        R = returns.loc[common]
        F = factor_returns.loc[common]

        betas: dict[str, pd.Series] = {}
        intercepts: dict[str, float] = {}
        r2: dict[str, float] = {}
        for sym in R.columns:
            ind_code = industry.get(sym)
            cols = [MARKET_COL]
            if ind_code is not None and ind_code in F.columns:
                cols.append(ind_code)
            cols += [s for s in self.style_factors if s in F.columns]
            ri = R[sym].dropna()
            Fi = F[cols].reindex(ri.index)
            valid = Fi.notna().all(axis=1)  # 只要求该股票实际使用的因子列非缺失
            ri = ri[valid]
            Fi = Fi.loc[ri.index]
            if len(ri) < len(cols) + 11:  # 截距多占 1 个自由度，沿用 10 个富余观测
                betas[sym] = pd.Series(0.0, index=cols)
                intercepts[sym] = float(ri.mean()) if len(ri) else 0.0
                r2[sym] = float("nan")
                continue
            X = np.column_stack([np.ones(len(ri)), Fi.to_numpy(dtype=float)])
            y = ri.to_numpy(dtype=float)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            intercepts[sym] = float(coef[0])
            resid = y - X @ coef
            var = float(y.var())
            r2[sym] = 1.0 - float(resid.var()) / var if var > 0 else float("nan")
            betas[sym] = pd.Series(coef[1:], index=cols)

        self._beta = pd.DataFrame(betas).T.fillna(0.0)
        self._beta.index.name = "symbol"
        self._intercept = pd.Series(intercepts, dtype=float).rename("intercept")
        self._r_squared = pd.Series(r2, name="r_squared")
        self._fitted = True
        return self

    def transform(
        self,
        returns: pd.DataFrame,
        industry: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> NeutralizationResult:
        """对新区间做中性化，返回完整残差 alpha = r − a_i − β̂_iᵀ·f_t。

        缺因子收益的 (date, symbol) 不输出 alpha（置 NaN），不会用
        原始收益冒充；未参与训练的股票暴露与截距为 0，alpha 即原始收益。
        """
        if not self._fitted or self._beta is None:
            raise RuntimeError("JointFactorModel 未 fit：请先在训练窗口上调用 fit()")
        beta = self._beta
        intercept = (
            self._intercept.reindex(returns.columns).fillna(0.0).to_numpy()
            if self._intercept is not None
            else np.zeros(returns.shape[1])
        )
        projection, invalid = _factor_projection(
            factor_returns, beta, returns.index, returns.columns
        )
        alpha_arr = returns.to_numpy(dtype=float) - projection - intercept
        alpha_arr[invalid] = np.nan
        alpha = pd.DataFrame(alpha_arr, index=returns.index, columns=returns.columns)
        return NeutralizationResult(
            alpha=alpha,
            exposures=beta,
            r_squared=self._r_squared if self._r_squared is not None else pd.Series(dtype=float),
        )

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def exposures_(self) -> pd.DataFrame:
        """各股票因子暴露（symbol × factor），不含截距。"""
        if self._beta is None:
            raise RuntimeError("JointFactorModel 未 fit")
        return self._beta


def build_alpha_label(
    prices: pd.DataFrame,
    industry: pd.Series,
    horizon: int,
    factor_returns: pd.DataFrame,
    model: JointFactorModel | None = None,
) -> pd.Series:
    """构建监督学习标签：未来 ``horizon`` 日、扣除因子收益后的中性化超额收益。

    标签（方案书 §5.1）：label_i,t = (P_{t+h}/P_t − 1) − β̂_iᵀ·FWD_factor_t，
    其中 FWD_factor 为同期因子累计收益（复利）。回归截距 a_i 不从标签中
    扣除——截距是个股专属 alpha，留给读出模型学习。``model`` 为空时在
    ``prices`` 覆盖的窗口内现场拟合（调用方须保证该窗口为训练窗，
    防泄漏由调用方与滚动协议的 purge 间隔共同保证）。

    前视窗 (t, t+h] 内任一实际使用的因子收益缺失时，该 (date, symbol)
    标签不输出（不得以原始收益冒充 alpha）。

    Returns:
        索引为 (date, symbol) 的 Series（名 "alpha_label"），末尾 horizon
        日无未来数据，不出现在输出中。
    """
    returns = prices.pct_change()
    if model is None:
        model = JointFactorModel().fit(returns, industry, factor_returns)
    if not model.is_fitted:
        raise RuntimeError("传入的 JointFactorModel 未 fit")

    fwd_ret = prices.shift(-horizon) / prices - 1.0
    fwd_factor = (
        (1.0 + factor_returns).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1.0
    )
    beta = model.exposures_
    projection, invalid = _factor_projection(
        fwd_factor, beta, fwd_ret.index, fwd_ret.columns
    )
    label = fwd_ret.to_numpy(dtype=float) - projection
    label[invalid] = np.nan
    out = pd.DataFrame(label, index=fwd_ret.index, columns=fwd_ret.columns).stack().dropna()
    out.index.names = ["date", "symbol"]
    return out.rename("alpha_label")


def cash_return_label(risk_free_rate: pd.Series, horizon: int) -> pd.Series:
    """现金资产标签：未来 ``horizon`` 日无风险利率累计收益（(t, t+h] 区间）。

    现金资产不进入股票 alpha 模型，其收益标签单独按无风险利率口径构建。
    与股票标签同为复利口径：label_t = Π_{s=t+1}^{t+h} (1 + rf_s) − 1，
    用 log1p/rolling-sum/expm1 向量化实现；窗口内含 NaN 或末尾不足
    ``horizon`` 日时输出 NaN。
    """
    log_ret = np.log1p(risk_free_rate)
    compounded = np.expm1(log_ret.rolling(horizon).sum().shift(-horizon))
    return compounded.rename("cash_label")
