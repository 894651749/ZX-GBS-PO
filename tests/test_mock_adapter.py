"""MockAdapter 七类数据的结构/主键/自洽性测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qgbs.data.adapters.base import (
    AssetPoolRecord,
    CorporateActionRecord,
    DailyMarketRow,
    TradingStatusRow,
)
from qgbs.data.adapters.mock import MockAdapter, MockDataConfig


@pytest.fixture(scope="module")
def adapter() -> MockAdapter:
    return MockAdapter(
        MockDataConfig(
            n_assets=40,
            start="2020-01-01",
            end="2021-12-31",
            n_industries=5,
            n_sub_industries=10,
            seed=7,
        )
    )


@pytest.fixture(scope="module")
def symbols(adapter: MockAdapter) -> list[str]:
    return [f"MOCK{i + 1:04d}" for i in range(adapter.config.n_assets)]


def test_daily_market_schema_and_self_consistency(adapter: MockAdapter, symbols: list[str]) -> None:
    mkt = adapter.load_daily_market(symbols, "2020-01-01", "2021-12-31")
    rows = mkt.rows
    # schema 字段齐全
    assert set(DailyMarketRow.model_fields) <= set(rows.columns)
    # 联合主键唯一
    assert not rows.duplicated(subset=["symbol", "trade_date"]).any()
    # OHLC 自洽
    assert (rows["high"] >= rows[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (rows["low"] <= rows[["open", "close"]].min(axis=1) + 1e-9).all()
    # prev_close 连续性 & 日收益口径
    for sym, g in rows.groupby("symbol"):
        g = g.sort_values("trade_date")
        assert (g["prev_close"].iloc[1:].to_numpy() == g["close"].iloc[:-1].to_numpy()).all()
        assert np.allclose(
            g["daily_return"], g["close"] / g["prev_close"] - 1.0, atol=1e-10
        )
    # 市值/换手非负
    assert (rows["total_mv"] >= rows["float_mv"]).all()
    assert (rows["turnover_rate"] >= 0).all()
    # 停牌日：价格冻结、量为 0、收益为 0
    susp = rows[rows["volume"] == 0]
    assert np.allclose(susp["daily_return"], 0.0)
    assert (susp["close"] == susp["prev_close"]).all()


def test_asset_pool_history(adapter: MockAdapter) -> None:
    pool29 = adapter.load_asset_pool_history("pool29")
    pool100 = adapter.load_asset_pool_history("pool100")
    assert set(AssetPoolRecord.model_fields) <= set(pool29.records.columns)
    stocks29 = pool29.records[~pool29.records["is_cash_asset"]]
    stocks100 = pool100.records[~pool100.records["is_cash_asset"]]
    assert len(stocks29) == adapter.config.pool29_size
    assert len(stocks100) == adapter.config.n_assets
    assert set(stocks29["symbol"]) <= set(stocks100["symbol"])
    # 区间自洽：生效 < 失效（失效非空时）
    exp = pd.to_datetime(pool100.records["expiry_date"])
    eff = pd.to_datetime(pool100.records["effective_date"])
    has_exp = exp.notna()
    assert (eff[has_exp] < exp[has_exp]).all()
    # 现金资产行
    cash = pool100.records[pool100.records["is_cash_asset"]]
    assert len(cash) == adapter.config.n_cash_assets
    assert cash["cash_category"].notna().all()
    # 未知 pool 报错
    with pytest.raises(KeyError):
        adapter.load_asset_pool_history("pool999")


def test_trading_status(adapter: MockAdapter, symbols: list[str]) -> None:
    sts = adapter.load_trading_status(symbols, "2020-01-01", "2021-12-31").rows
    assert set(TradingStatusRow.model_fields) <= set(sts.columns)
    assert not sts.duplicated(subset=["symbol", "trade_date"]).any()
    assert (sts["lot_size"] == 100).all()
    susp = sts[sts["is_suspended"]]
    assert (~susp["buyable"]).all() and (~susp["sellable"]).all()
    assert susp["limit_up_price"].isna().all()
    assert susp["resume_date"].notna().all() or len(susp) == 0
    # 停牌占比应为少量
    assert sts["is_suspended"].mean() < 0.05
    # 可交易日与行情量一致：停牌日 volume=0
    mkt = adapter.load_daily_market(symbols, "2020-01-01", "2021-12-31").rows
    merged = susp.merge(mkt, on=["symbol", "trade_date"])
    assert (merged["volume"] == 0).all()


def test_corporate_actions(adapter: MockAdapter, symbols: list[str]) -> None:
    ca = adapter.load_corporate_actions(symbols, "2020-01-01", "2021-12-31").records
    assert set(CorporateActionRecord.model_fields) <= set(ca.columns)
    assert len(ca) > 0
    assert ca["action_type"].isin(["dividend", "allotment", "split"]).all()
    assert (ca["adj_factor"] > 0).all()
    assert (
        (ca["ex_date"] >= pd.Timestamp("2020-01-01"))
        & (ca["ex_date"] <= pd.Timestamp("2021-12-31"))
    ).all()


def test_market_sector_factor(adapter: MockAdapter) -> None:
    msf = adapter.load_market_sector_factor("2020-01-01", "2021-12-31")
    assert msf.index_returns.shape[1] == 3
    assert msf.industry_index_returns.shape[1] == adapter.config.n_industries
    assert (msf.index_returns.index == msf.industry_index_returns.index).all()
    assert (msf.risk_free_rate > 0).all()
    assert np.allclose(msf.risk_free_rate, adapter.config.rf_annual / 252.0)


def test_constraints_costs(adapter: MockAdapter, symbols: list[str]) -> None:
    cc = adapter.load_constraints_costs("default")
    c = adapter.config
    assert cc.commission_bps == c.commission_bps
    assert cc.stamp_tax_bps == c.stamp_tax_bps
    assert set(symbols) <= set(cc.weight_upper)
    assert tuple(cc.cash_range) == tuple(c.cash_range)
    with pytest.raises(KeyError):
        adapter.load_constraints_costs("nonexistent")


def test_baseline_results_empty(adapter: MockAdapter) -> None:
    base = adapter.load_baseline_results()
    for name, f in base.__dataclass_fields__.items():
        assert getattr(base, name) is None, f"基线字段 {name} 应为 None"


def test_reproducibility(symbols: list[str]) -> None:
    cfg = MockDataConfig(n_assets=40, start="2020-01-01", end="2021-12-31",
                         n_industries=5, n_sub_industries=10, seed=7)
    a1 = MockAdapter(cfg).load_daily_market(symbols, "2020-01-01", "2021-12-31").rows
    a2 = MockAdapter(cfg).load_daily_market(symbols, "2020-01-01", "2021-12-31").rows
    pd.testing.assert_frame_equal(a1, a2)
