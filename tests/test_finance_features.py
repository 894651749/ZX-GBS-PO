"""金融特征测试：窗口对齐（无未来函数）、形状与缺失模式。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qgbs.data.adapters.mock import MockAdapter, MockDataConfig
from qgbs.factors.finance import build_finance_features

EXPECTED_COLUMNS = {
    "mom_5", "mom_20", "mom_60", "mom_120", "rev_5",
    "vol_20", "vol_60", "skew_20", "kurt_20", "amount_mean_20", "amihud_20",
}


@pytest.fixture(scope="module")
def adapter() -> MockAdapter:
    return MockAdapter(
        MockDataConfig(
            n_assets=20,
            start="2020-01-01",
            end="2021-12-31",
            n_industries=4,
            n_sub_industries=8,
            seed=11,
        )
    )


@pytest.fixture(scope="module")
def market(adapter: MockAdapter):
    symbols = [f"MOCK{i + 1:04d}" for i in range(adapter.config.n_assets)]
    return adapter.load_daily_market(symbols, "2020-01-01", "2021-12-31")


def test_shape_and_columns(market) -> None:
    feats = build_finance_features(market)
    assert set(feats.columns) == EXPECTED_COLUMNS
    assert feats.index.names == ["date", "symbol"]
    # 面板行数 = 交易日数 × 股票数
    n_dates = market.rows["trade_date"].nunique()
    n_syms = market.rows["symbol"].nunique()
    assert len(feats) == n_dates * n_syms


def test_missing_pattern_warmup(market) -> None:
    feats = build_finance_features(market)
    sym = "MOCK0001"
    s = feats["mom_120"].xs(sym, level="symbol").sort_index()
    # 自上市日起前 120 日为 NaN，第 121 个交易日起有值
    first_valid_pos = int(np.argmax(s.notna().to_numpy()))
    assert first_valid_pos == 120
    assert s.iloc[:120].isna().all()
    s5 = feats["mom_5"].xs(sym, level="symbol").sort_index()
    assert s5.iloc[:5].isna().all()
    assert s5.notna().sum() == len(s5) - 5


def test_no_lookahead(market) -> None:
    """截断未来数据后，t0 日的特征值必须不变。"""
    feats_full = build_finance_features(market)
    dates = sorted(market.rows["trade_date"].unique())
    t0 = pd.Timestamp(dates[300])
    market_trunc = type(market)(rows=market.rows[market.rows["trade_date"] <= t0])
    feats_trunc = build_finance_features(market_trunc)
    sym = "MOCK0001"
    full_row = feats_full.loc[(t0, sym)]
    trunc_row = feats_trunc.loc[(t0, sym)]
    for col in feats_full.columns:
        a, b = full_row[col], trunc_row[col]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == pytest.approx(b, rel=1e-12, abs=1e-12), f"{col} 在 t0 使用了未来数据"
