"""绩效指标：IC/ICIR/分层收益/Sharpe/Sortino/回撤/CVaR/换手率等。

IC 类指标约定：``pred``/``realized`` 为单截面 Series（索引 symbol）
或 (date, symbol) MultiIndex 面板（此时用 :func:`ic_series` 逐期计算）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def total_return(nav: pd.Series) -> float:
    """区间总收益：nav 末值/初值 − 1。"""
    if len(nav) < 2:
        return 0.0
    return float(nav.iloc[-1] / nav.iloc[0] - 1.0)


def annual_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """年化收益（几何口径）。"""
    returns = returns.dropna()
    if len(returns) == 0:
        return float("nan")
    total = float((1.0 + returns).prod())
    return total ** (periods_per_year / len(returns)) - 1.0


def annual_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """年化波动率。"""
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")
    return float(returns.std() * np.sqrt(periods_per_year))


def rank_ic(pred: pd.Series, realized: pd.Series) -> float:
    """Spearman rank IC（单截面）。"""
    df = pd.DataFrame({"p": pred, "r": realized}).dropna()
    if len(df) < 3 or df["p"].nunique() < 2 or df["r"].nunique() < 2:
        return float("nan")
    return float(df["p"].corr(df["r"], method="spearman"))


def pearson_ic(pred: pd.Series, realized: pd.Series) -> float:
    """Pearson IC（单截面）。"""
    df = pd.DataFrame({"p": pred, "r": realized}).dropna()
    if len(df) < 3 or df["p"].std() == 0 or df["r"].std() == 0:
        return float("nan")
    return float(df["p"].corr(df["r"], method="pearson"))


def ic_series(pred: pd.Series, realized: pd.Series, method: str = "spearman") -> pd.Series:
    """逐期 IC 序列：pred/realized 均为 (date, symbol) 面板。"""
    df = pd.DataFrame({"p": pred, "r": realized}).dropna()

    def _ic(g: pd.DataFrame) -> float:
        if len(g) < 3 or g["p"].nunique() < 2 or g["r"].nunique() < 2:
            return float("nan")
        return float(g["p"].corr(g["r"], method=method))

    return df.groupby(level="date").apply(_ic).dropna()


def icir(ic_series: pd.Series) -> float:
    """ICIR = IC 均值 / IC 标准差。"""
    ic_series = ic_series.dropna()
    if len(ic_series) < 2:
        return float("nan")
    std = float(ic_series.std())
    if std == 0.0:
        return float("nan")
    return float(ic_series.mean() / std)


def layered_returns(pred: pd.Series, realized: pd.Series, n_layers: int = 10) -> pd.Series:
    """横截面分层收益：按预测分层后的各层平均实现收益（跨期平均）。

    pred/realized 为 (date, symbol) 面板；返回索引为层号（0=最低预测层）
    的 Series。
    """
    df = pd.DataFrame({"p": pred, "r": realized}).dropna()
    layer = df.groupby(level="date")["p"].transform(
        lambda s: pd.qcut(s.rank(method="first"), min(n_layers, len(s)), labels=False)
    )
    df["layer"] = layer
    out = df.groupby("layer")["r"].mean()
    out.index.name = "layer"
    return out


def direction_accuracy(pred: pd.Series, realized: pd.Series) -> float:
    """方向准确率：预测符号与实现收益符号一致的比例（单截面或面板）。"""
    df = pd.DataFrame({"p": pred, "r": realized}).dropna()
    if len(df) == 0:
        return float("nan")
    return float((np.sign(df["p"]) == np.sign(df["r"])).mean())


def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """年化 Sharpe 比率（零无风险利率口径）：均值/标准差 × √N。"""
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")
    std = float(returns.std())
    if std == 0.0:
        return float("nan")
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: int = 252, mar: float = 0.0) -> float:
    """年化 Sortino 比率（下行偏差，MAR 为最低可接受收益）。"""
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")
    downside = returns[returns < mar] - mar
    if len(downside) == 0:
        return float("nan")
    dd = float(np.sqrt((downside**2).mean()))
    if dd == 0.0:
        return float("nan")
    return float((returns.mean() - mar) / dd * np.sqrt(periods_per_year))


def max_drawdown(nav: pd.Series) -> float:
    """最大回撤（净值序列口径，返回非负值）。"""
    if len(nav) < 2:
        return 0.0
    running_max = nav.cummax()
    drawdown = 1.0 - nav / running_max
    return float(drawdown.max())


def cvar(returns: pd.Series, level: float = 0.95) -> float:
    """条件在险价值（Expected Shortfall）：左尾 (1−level) 分位以下的均值（负值口径）。"""
    returns = returns.dropna()
    if len(returns) == 0:
        return float("nan")
    threshold = returns.quantile(1.0 - level)
    tail = returns[returns <= threshold]
    return float(tail.mean()) if len(tail) else float("nan")


def turnover(weights: pd.DataFrame) -> pd.Series:
    """各期单边换手率：首期 0.5·Σ|w₀|，其后 0.5·Σ|Δw|。"""
    if weights.empty:
        return pd.Series(dtype=float, name="turnover")
    first = 0.5 * weights.iloc[0].abs().sum()
    rest = 0.5 * weights.diff().abs().sum(axis=1).iloc[1:]
    return pd.concat([pd.Series([first], index=[weights.index[0]]), rest])


def weight_concentration(weights: pd.DataFrame) -> pd.Series:
    """权重集中度 HHI（Herfindahl-Hirschman 指数）：各期 Σw²。"""
    hhi = (weights**2).sum(axis=1)
    hhi.name = "hhi"
    return hhi


def net_of_cost_utility(
    returns_net: pd.Series,
    risk_aversion: float,
    periods_per_year: int = 252,
) -> float:
    """扣费后效用：年化收益 − (λ/2)·年化方差（均值-方差效用口径）。"""
    returns_net = returns_net.dropna()
    if len(returns_net) < 2:
        return float("nan")
    ann_ret = annual_return(returns_net, periods_per_year)
    ann_var = float(returns_net.var() * periods_per_year)
    return ann_ret - 0.5 * risk_aversion * ann_var


def exposure_report(weights: pd.DataFrame, exposure_matrix: pd.DataFrame) -> pd.DataFrame:
    """暴露报告：各期组合的行业/风格暴露（用于约束合规检查）。

    Args:
        weights: date × symbol 权重表。
        exposure_matrix: symbol × factor 暴露矩阵。

    Returns:
        date × factor 的暴露表。
    """
    expo = exposure_matrix.reindex(index=weights.columns).fillna(0.0)
    return weights.dot(expo)
