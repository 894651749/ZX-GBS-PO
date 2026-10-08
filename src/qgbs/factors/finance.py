"""经典金融特征库：趋势/反转/波动/流动性 + 扩展风险与市场状态特征。

约定：所有特征在 t 日只使用 ≤t 的数据（rolling 窗口右端点为 t，无未来函数）。
输入既可以是宽表（date × symbol），也可以用
:func:`build_finance_features` 直接消费 mock 的日频长表
:class:`~qgbs.data.adapters.base.DailyMarket`。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qgbs.data.adapters.base import DailyMarket

DEFAULT_WINDOWS: dict[str, int] = {
    "momentum_short": 5,
    "momentum_mid": 20,
    "momentum_long": 60,
    "momentum_xlong": 120,
    "reversal": 5,
    "volatility_short": 20,
    "volatility_long": 60,
    "shape": 20,
    "liquidity": 20,
    # extended（feature_set="extended" 才启用）
    "downside": 20,
    "drawdown_short": 20,
    "drawdown_long": 60,
    "turnover_volatility": 20,
    "market_state": 20,
}


def momentum(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """动量/趋势特征：过去 ``window`` 个交易日的累计收益（close_t/close_{t-w} − 1）。"""
    return prices / prices.shift(window) - 1.0


def reversal(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """短期反转特征：近 ``window`` 日收益的反向。"""
    return -(prices / prices.shift(window) - 1.0)


def volatility(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """已实现波动率特征：日收益的 ``window`` 日滚动标准差。"""
    return prices.pct_change().rolling(window).std()


def rolling_skew(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """日收益的 ``window`` 日滚动偏度。"""
    return prices.pct_change().rolling(window).skew()


def rolling_kurtosis(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """日收益的 ``window`` 日滚动峰度。"""
    return prices.pct_change().rolling(window).kurt()


def liquidity(volumes: pd.DataFrame, prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """流动性特征：``window`` 日平均成交额（量额代理）。

    成交额以 volume × close 近似；Amihud 非流动性由
    :func:`amihud_illiquidity` 单独提供。
    """
    amount = volumes * prices
    return amount.rolling(window).mean()


def amihud_illiquidity(prices: pd.DataFrame, amount: pd.DataFrame, window: int) -> pd.DataFrame:
    """Amihud 非流动性：|r|/成交额 的 ``window`` 日平均。"""
    daily_abs_ret = prices.pct_change().abs()
    ratio = daily_abs_ret / amount.where(amount != 0.0)
    return ratio.rolling(window).mean()


def factor_exposure(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """对经典风险因子（市场/规模/价值等）的滚动暴露估计。"""
    raise NotImplementedError  # P1：需要外部因子收益序列，P0 闭环不依赖


# ---------------------------------------------------------------------------
# extended 特征（方案书 §5.2：下行风险/回撤/量价异常/市场状态）
# ---------------------------------------------------------------------------


def downside_volatility(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """下行波动：负日收益平方的 ``window`` 日均值开根（半波动口径）。"""
    downside = prices.pct_change().clip(upper=0.0)
    return np.sqrt((downside**2).rolling(window).mean())


def rolling_max_drawdown(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """过去 ``window`` 日内最大回撤：1 − min(p / cummax(p))（非负）。"""
    def _mdd(p: np.ndarray) -> float:
        return 1.0 - float(np.min(p / np.maximum.accumulate(p)))

    return prices.rolling(window).apply(_mdd, raw=True)


def volume_abnormal(volumes: pd.DataFrame, window: int) -> pd.DataFrame:
    """成交量异常：当日成交量 / 过去 ``window`` 日均量。"""
    return volumes / volumes.rolling(window).mean()


def turnover_volatility(turnover: pd.DataFrame, window: int) -> pd.DataFrame:
    """换手率波动：换手率的 ``window`` 日滚动标准差。"""
    return turnover.rolling(window).std()


def _rolling_avg_pairwise_corr(returns_wide: pd.DataFrame, window: int) -> pd.Series:
    """窗口内截面平均两两 Pearson 相关（逐日，仅用 ≤t 数据）。

    利用恒等式 Var(Σzᵢ) = m + 2·Σᵢ<ⱼρᵢⱼ（z 为窗口内标准化的收益，
    m 为当日可用股票数）向量化计算，避免显式构造 m×m 相关矩阵。
    """
    values = returns_wide.to_numpy(dtype=float)
    n_dates = values.shape[0]
    out = np.full(n_dates, np.nan)
    for end in range(window - 1, n_dates):
        w = values[end - window + 1 : end + 1]
        w = w[:, ~np.isnan(w).any(axis=0)]
        sd = w.std(axis=0, ddof=1) if w.shape[1] else np.array([])
        w = w[:, sd > 0]
        m = w.shape[1]
        if m < 2:
            continue
        z = (w - w.mean(axis=0)) / w.std(axis=0, ddof=1)
        out[end] = (float(z.sum(axis=1).var(ddof=1)) - m) / (m * (m - 1))
    return pd.Series(out, index=returns_wide.index)


def market_state_features(
    returns_wide: pd.DataFrame,
    window: int,
    market_ret: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """市场状态特征（逐日单值，由调用方按 date 广播到所有股票）。

    - ``mkt_vol_{window}``：市场收益 ``window`` 日滚动 std；``market_ret``
      缺省时用候选池等权平均收益作市场代理（真实场景应传入宽基指数收益）；
    - ``mkt_corr_{window}``：窗口内截面平均两两相关；
    - ``mkt_disp_{window}``：截面离散度（当日收益横截面 std 的 ``window`` 日均值）；
    - ``mkt_breadth``：当日上涨股票占比（0–1）。
    """
    mkt = market_ret.reindex(returns_wide.index) if market_ret is not None else returns_wide.mean(axis=1)
    up = (returns_wide > 0).sum(axis=1)
    n_valid = returns_wide.notna().sum(axis=1)
    breadth = up / n_valid.where(n_valid > 0)
    return {
        f"mkt_vol_{window}": mkt.rolling(window).std(),
        f"mkt_corr_{window}": _rolling_avg_pairwise_corr(returns_wide, window),
        f"mkt_disp_{window}": returns_wide.std(axis=1).rolling(window).mean(),
        "mkt_breadth": breadth,
    }


def _broadcast_to_universe(series: pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    """把逐日市场状态序列按 date 广播到所有股票（宽表，列与 ``like`` 一致）。"""
    values = np.repeat(series.reindex(like.index).to_numpy()[:, None], like.shape[1], axis=1)
    return pd.DataFrame(values, index=like.index, columns=like.columns)


def build_finance_features(
    market: DailyMarket,
    windows: dict[str, int] | None = None,
    feature_set: str = "base",
    market_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """从日频行情长表汇总经典金融特征面板。

    Args:
        market: mock/真实适配器返回的日频行情长表。
        windows: 窗口参数覆盖，默认见 :data:`DEFAULT_WINDOWS`。
        feature_set: "base"（默认，11 个基础特征，向后兼容）或
            "extended"（基础特征 + 下行波动/回撤/量价异常/市场状态，
            对齐方案书 §5.2）。
        market_returns: 可选的宽基指数日收益（索引为交易日），仅 extended
            的市场波动特征使用；缺省时用候选池等权平均收益作市场代理。

    Returns:
        索引为 (date, symbol) 的 MultiIndex DataFrame，列为特征名。
        base：mom_5/mom_20/mom_60/mom_120、rev_5、vol_20/vol_60、
        skew_20、kurt_20、amount_mean_20、amihud_20。
        extended 追加：downside_vol_20、mdd_20/mdd_60、volume_abnormal_20、
        turnover_vol_20、mkt_vol_20、mkt_corr_20、mkt_disp_20、mkt_breadth
        （市场状态特征按 date 广播到所有股票）。
        窗口不足的特征为 NaN（由下游 pipeline 处理）。
    """
    if feature_set not in ("base", "extended"):
        raise ValueError(f"未知 feature_set: {feature_set!r}（支持 'base'/'extended'）")
    w = dict(DEFAULT_WINDOWS)
    if windows:
        w.update(windows)

    rows = market.rows
    close = rows.pivot(index="trade_date", columns="symbol", values="close").sort_index()
    volume = rows.pivot(index="trade_date", columns="symbol", values="volume").sort_index()

    feats: dict[str, pd.DataFrame] = {
        "mom_5": momentum(close, w["momentum_short"]),
        "mom_20": momentum(close, w["momentum_mid"]),
        "mom_60": momentum(close, w["momentum_long"]),
        "mom_120": momentum(close, w["momentum_xlong"]),
        "rev_5": reversal(close, w["reversal"]),
        "vol_20": volatility(close, w["volatility_short"]),
        "vol_60": volatility(close, w["volatility_long"]),
        "skew_20": rolling_skew(close, w["shape"]),
        "kurt_20": rolling_kurtosis(close, w["shape"]),
        "amount_mean_20": liquidity(volume, close, w["liquidity"]),
    }
    amount = volume * close
    feats["amihud_20"] = amihud_illiquidity(close, amount, w["liquidity"])

    if feature_set == "extended":
        turnover = rows.pivot(index="trade_date", columns="symbol", values="turnover_rate").sort_index()
        ret_wide = close.pct_change()
        feats[f"downside_vol_{w['downside']}"] = downside_volatility(close, w["downside"])
        feats[f"mdd_{w['drawdown_short']}"] = rolling_max_drawdown(close, w["drawdown_short"])
        feats[f"mdd_{w['drawdown_long']}"] = rolling_max_drawdown(close, w["drawdown_long"])
        feats[f"volume_abnormal_{w['liquidity']}"] = volume_abnormal(volume, w["liquidity"])
        feats[f"turnover_vol_{w['turnover_volatility']}"] = turnover_volatility(turnover, w["turnover_volatility"])
        for name, series in market_state_features(ret_wide, w["market_state"], market_ret=market_returns).items():
            feats[name] = _broadcast_to_universe(series, close)

    panel = pd.DataFrame({name: f.stack() for name, f in feats.items()})
    panel.index.names = ["date", "symbol"]
    return panel.sort_index()
