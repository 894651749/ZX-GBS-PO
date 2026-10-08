"""扩展金融特征测试（B1）：列集与向后兼容、无未来函数、市场状态广播、手算核对。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qgbs.data.adapters.mock import MockAdapter, MockDataConfig
from qgbs.factors.finance import (
    build_finance_features,
    downside_volatility,
    market_state_features,
    rolling_max_drawdown,
    turnover_volatility,
    volume_abnormal,
)

BASE_COLUMNS = {
    "mom_5", "mom_20", "mom_60", "mom_120", "rev_5",
    "vol_20", "vol_60", "skew_20", "kurt_20", "amount_mean_20", "amihud_20",
}
EXTENDED_NEW_COLUMNS = {
    "downside_vol_20", "mdd_20", "mdd_60",
    "volume_abnormal_20", "turnover_vol_20",
    "mkt_vol_20", "mkt_corr_20", "mkt_disp_20", "mkt_breadth",
}
MARKET_COLUMNS = {"mkt_vol_20", "mkt_corr_20", "mkt_disp_20", "mkt_breadth"}


@pytest.fixture(scope="module")
def market():
    adapter = MockAdapter(
        MockDataConfig(
            n_assets=8, start="2021-01-01", end="2021-12-31",
            n_industries=3, n_sub_industries=6, seed=7,
        )
    )
    symbols = [f"MOCK{i + 1:04d}" for i in range(adapter.config.n_assets)]
    return adapter.load_daily_market(symbols, "2021-01-01", "2021-12-31")


def test_base_default_backward_compatible(market) -> None:
    """默认 feature_set="base"：列集与数值和旧版完全一致。"""
    feats_default = build_finance_features(market)
    feats_base = build_finance_features(market, feature_set="base")
    assert set(feats_default.columns) == BASE_COLUMNS
    pd.testing.assert_frame_equal(feats_default, feats_base)


def test_extended_columns_and_base_unchanged(market) -> None:
    feats_ext = build_finance_features(market, feature_set="extended")
    assert set(feats_ext.columns) == BASE_COLUMNS | EXTENDED_NEW_COLUMNS
    n_dates = market.rows["trade_date"].nunique()
    n_syms = market.rows["symbol"].nunique()
    assert len(feats_ext) == n_dates * n_syms
    assert feats_ext.index.names == ["date", "symbol"]
    # extended 不改动 base 特征数值
    feats_base = build_finance_features(market, feature_set="base")
    pd.testing.assert_frame_equal(feats_ext[list(feats_base.columns)], feats_base)


def test_unknown_feature_set_raises(market) -> None:
    with pytest.raises(ValueError, match="feature_set"):
        build_finance_features(market, feature_set="full")


def test_market_features_broadcast_by_date(market) -> None:
    """市场状态特征：同一 date 下所有股票取值相同，且在 [合理范围] 内。"""
    feats = build_finance_features(market, feature_set="extended")
    for col in MARKET_COLUMNS:
        per_date_nunique = feats[col].groupby(level="date").nunique()
        assert (per_date_nunique <= 1).all(), f"{col} 未按 date 广播"
    breadth = feats["mkt_breadth"].dropna()
    assert ((breadth >= 0.0) & (breadth <= 1.0)).all()
    assert feats["mdd_20"].dropna().ge(0.0).all()


def test_no_lookahead_extended(market) -> None:
    """截断未来数据后，t0 日 extended 特征值必须不变。"""
    feats_full = build_finance_features(market, feature_set="extended")
    dates = sorted(market.rows["trade_date"].unique())
    t0 = pd.Timestamp(dates[150])
    market_trunc = type(market)(rows=market.rows[market.rows["trade_date"] <= t0])
    feats_trunc = build_finance_features(market_trunc, feature_set="extended")
    sym = "MOCK0001"
    for col in feats_full.columns:
        a, b = feats_full.loc[(t0, sym), col], feats_trunc.loc[(t0, sym), col]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == pytest.approx(b, rel=1e-12, abs=1e-12), f"{col} 在 t0 使用了未来数据"


def test_warmup_missing_pattern(market) -> None:
    feats = build_finance_features(market, feature_set="extended")
    sym = "MOCK0001"
    mdd60 = feats["mdd_60"].xs(sym, level="symbol").sort_index()
    first_valid = int(np.argmax(mdd60.notna().to_numpy()))
    assert first_valid == 59  # 60 日窗口：第 60 个交易日起有值
    dvol = feats["downside_vol_20"].xs(sym, level="symbol").sort_index()
    first_valid_dvol = int(np.argmax(dvol.notna().to_numpy()))
    assert first_valid_dvol == 20  # pct_change 首日 NaN + 20 日窗口


# ---------------------------------------------------------------------------
# 手算核对（小样本确定值）
# ---------------------------------------------------------------------------


def test_downside_volatility_handcheck() -> None:
    prices = pd.DataFrame({"A": [100.0, 90.0, 90.0, 90.0, 90.0]})
    out = downside_volatility(prices, 4)
    # 收益 = [-0.1, 0, 0, 0] → 下行波动 = sqrt(0.01/4) = 0.05
    assert out["A"].iloc[-1] == pytest.approx(0.05, rel=1e-12)


def test_rolling_max_drawdown_handcheck() -> None:
    prices = pd.DataFrame({"A": [100.0, 120.0, 90.0, 110.0]})
    out = rolling_max_drawdown(prices, 4)
    # p/cummax = [1, 1, 0.75, 11/12] → mdd = 1 − 0.75 = 0.25
    assert out["A"].iloc[-1] == pytest.approx(0.25, rel=1e-12)
    assert out["A"].iloc[:3].isna().all()


def test_volume_abnormal_handcheck() -> None:
    volumes = pd.DataFrame({"A": [100.0] * 19 + [300.0]})
    out = volume_abnormal(volumes, 20)
    # 20 日均量 = (19×100 + 300)/20 = 110 → 300/110
    assert out["A"].iloc[-1] == pytest.approx(300.0 / 110.0, rel=1e-12)


def test_turnover_volatility_handcheck() -> None:
    turnover = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = turnover_volatility(turnover, 5)
    assert out["A"].iloc[-1] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1), rel=1e-12)


def test_market_state_features_handcheck() -> None:
    idx = pd.bdate_range("2024-01-01", periods=30)
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.01, 30)
    b = a + rng.normal(0.0, 1e-9, 30)  # 与 a 几乎完全相关
    ret = pd.DataFrame({"A": a, "B": b}, index=idx)
    out = market_state_features(ret, 5)
    # 完全相关 → 截面平均两两相关 ≈ 1
    assert out["mkt_corr_5"].iloc[-1] == pytest.approx(1.0, abs=1e-6)
    # 市场代理缺省为等权平均
    expected_vol = ret.mean(axis=1).rolling(5).std()
    pd.testing.assert_series_equal(out["mkt_vol_5"], expected_vol, check_names=False)
    # 传入外部指数收益时以其为准
    mkt_ext = pd.Series(0.01, index=idx)
    out_ext = market_state_features(ret, 5, market_ret=mkt_ext)
    assert out_ext["mkt_vol_5"].dropna().eq(0.0).all()  # 常数指数收益 → 波动为 0


def test_market_breadth_handcheck() -> None:
    idx = pd.bdate_range("2024-01-01", periods=25)
    ret = pd.DataFrame(
        {"A": 0.01, "B": -0.02, "C": 0.03, "D": -0.01}, index=idx
    )
    out = market_state_features(ret, 5)
    assert out["mkt_breadth"].iloc[-1] == pytest.approx(0.5, rel=1e-12)
    # 截面离散度 = 当日横截面 std 的 5 日均值（常数截面下逐日相同）
    expected_disp = ret.std(axis=1).iloc[-1]
    assert out["mkt_disp_5"].iloc[-1] == pytest.approx(expected_disp, rel=1e-12)
