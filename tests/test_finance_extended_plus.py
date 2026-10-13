"""extended_plus 新特征族（E4.4）：手算对照、因果性、NaN 模式与回归测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qgbs.data.adapters.base import DailyMarket, TradingStatus
from qgbs.factors.finance import build_finance_features

SYMS = ["A", "B", "C"]
IND_MAP = pd.Series({"A": "IND1", "B": "IND1", "C": "IND2"})  # A/B 同行业
N_DAYS = 40


def _mk_inputs(n_days: int = N_DAYS, seed: int = 0, st_symbol: str | None = None):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rows = []
    for i, s in enumerate(SYMS):
        rets = rng.normal(0.001 * (i + 1), 0.01, n_days)
        close = 100.0 * np.cumprod(1 + rets)
        for j, d in enumerate(dates):
            rows.append({"symbol": s, "trade_date": d, "close": close[j],
                         "volume": 1e6, "turnover_rate": 0.01,
                         "float_mv": 1e10 * (i + 1) * (1 + 0.001 * j)})
    market = DailyMarket(rows=pd.DataFrame(rows))
    srows = []
    for s in SYMS:
        for d in dates:
            srows.append({"symbol": s, "trade_date": d,
                          "is_st": s == st_symbol, "list_date": "2020-01-01"})
    status = TradingStatus(rows=pd.DataFrame(srows))
    # 行业指数日收益：IND1 常数 0.001、IND2 常数 -0.002（手算友好）
    ind_ret = pd.DataFrame(
        {"IND1": np.full(n_days, 0.001), "IND2": np.full(n_days, -0.002)},
        index=dates,
    )
    return market, status, ind_ret


def _build(market, status, ind_ret, **kw):
    return build_finance_features(
        market, feature_set="extended_plus", industry_index_returns=ind_ret,
        industry_map=IND_MAP, status=status, **kw,
    )


def test_new_family_hand_computed() -> None:
    """手算对照：mv_log / ind_rel_* / is_st / list_age。"""
    market, status, ind_ret = _mk_inputs()
    panel = _build(market, status, ind_ret)
    close = market.rows.pivot(index="trade_date", columns="symbol", values="close").sort_index()

    # mv_log = log(float_mv)；A 第 0 日 float_mv=1e10
    assert panel.loc[(close.index[0], "A"), "mv_log"] == pytest.approx(np.log(1e10))
    assert panel.loc[(close.index[10], "C"), "mv_log"] == pytest.approx(np.log(3e10 * 1.01))

    # ind_rel_mom_20（第 20 日起有值）：个股 mom_20 − [(1.001)^20 − 1]（A/B 属 IND1）
    t = 25
    mom_stock = close["A"].iloc[t] / close["A"].iloc[t - 20] - 1.0
    expect = mom_stock - (1.001 ** 20 - 1.0)
    assert panel.loc[(close.index[t], "A"), "ind_rel_mom_20"] == pytest.approx(expect, rel=1e-9)
    # ind_rel_rev_5 = 个股 rev_5 − 行业 rev_5；行业 rev_5 = -(1.001^5-1)
    mom5_stock = close["B"].iloc[t] / close["B"].iloc[t - 5] - 1.0
    expect_rev = -mom5_stock + (1.001 ** 5 - 1.0)
    assert panel.loc[(close.index[t], "B"), "ind_rel_rev_5"] == pytest.approx(expect_rev, rel=1e-9)
    # C 属 IND2（日收益 -0.002）
    mom_c = close["C"].iloc[t] / close["C"].iloc[t - 20] - 1.0
    assert panel.loc[(close.index[t], "C"), "ind_rel_mom_20"] == pytest.approx(
        mom_c - (0.998 ** 20 - 1.0), rel=1e-9)
    # ind_rel_vol_20：行业常数收益 → 行业 vol=0 → 相对波动 = 个股 vol
    vol_stock = close["A"].pct_change().rolling(20).std().iloc[t]
    assert panel.loc[(close.index[t], "A"), "ind_rel_vol_20"] == pytest.approx(vol_stock, rel=1e-9)

    # is_st / list_age
    assert panel["is_st"].dropna().isin([0.0, 1.0]).all()
    assert panel.loc[(close.index[0], "A"), "list_age"] == pytest.approx(
        np.log1p((close.index[0] - pd.Timestamp("2020-01-01")).days))


def test_causality_future_perturbation() -> None:
    """未来数据扰动（close/float_mv/is_st 改动）不影响历史特征值。"""
    market, status, ind_ret = _mk_inputs()
    base = _build(market, status, ind_ret)
    cut = 30
    pert_rows = market.rows.copy()
    future = pert_rows["trade_date"] >= market.rows["trade_date"].unique()[cut]
    pert_rows.loc[future, "close"] *= 2.0
    pert_rows.loc[future, "float_mv"] *= 10.0
    market2 = DailyMarket(rows=pert_rows)
    status2 = TradingStatus(rows=status.rows.assign(
        is_st=status.rows["is_st"] | (status.rows["trade_date"] >= market.rows["trade_date"].unique()[cut])))
    pert = _build(market2, status2, ind_ret)
    hist_idx = [i for i in base.index if i[0] < market.rows["trade_date"].unique()[cut]]
    new_cols = ["mv_log", "ind_rel_mom_20", "ind_rel_rev_5", "ind_rel_vol_20", "is_st", "list_age"]
    a = base.loc[hist_idx, new_cols]
    b = pert.loc[hist_idx, new_cols]
    pd.testing.assert_frame_equal(a, b)


def test_nan_warmup_pattern() -> None:
    """warmup：ind_rel_mom_20 前 19 日 NaN、第 20 日起有值；list_age/is_st 无 warmup。"""
    market, status, ind_ret = _mk_inputs()
    panel = _build(market, status, ind_ret)
    dates = sorted(panel.index.get_level_values("date").unique())
    assert panel["ind_rel_mom_20"].loc[dates[0]].isna().all()
    assert panel["ind_rel_mom_20"].loc[dates[18]].isna().all()
    assert panel["ind_rel_mom_20"].loc[dates[20]].notna().all()
    assert panel["list_age"].notna().all() and panel["is_st"].notna().all()


def test_extended_unchanged_and_plus_requires_inputs() -> None:
    """回归：extended 不传新参数行为不变；extended_plus 缺参显式报错。"""
    market, status, ind_ret = _mk_inputs()
    ext = build_finance_features(market, feature_set="extended")
    assert "mv_log" not in ext.columns and len(ext.columns) == 20
    with pytest.raises(ValueError, match="industry_index_returns"):
        build_finance_features(market, feature_set="extended_plus", status=status)
    with pytest.raises(ValueError, match="status"):
        build_finance_features(
            market, feature_set="extended_plus",
            industry_index_returns=ind_ret, industry_map=IND_MAP,
        )
    with pytest.raises(ValueError, match="未知 feature_set"):
        build_finance_features(market, feature_set="extended2")


def test_extended_plus_is_extended_plus_new_families() -> None:
    """extended_plus = extended 20 维原样 + 6 个新族（前置 bug 回归：曾漏 extended 部分）。"""
    market, status, ind_ret = _mk_inputs()
    plus = _build(market, status, ind_ret)
    ext = build_finance_features(market, feature_set="extended")
    assert len(plus.columns) == 26
    pd.testing.assert_frame_equal(plus[ext.columns], ext[ext.columns])
