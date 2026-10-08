"""A3：CsvAdapter 测试——合成 CSV 套件、缺失容忍、与 MockAdapter 互换进引擎。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from qgbs.backtest.engine import BacktestEngine
from qgbs.backtest.rolling import WindowSpec
from qgbs.data.adapters.csv_adapter import CsvAdapter
from qgbs.data.adapters.mock import MockAdapter, MockDataConfig
from qgbs.portfolio.costs import CostModel
from qgbs.portfolio.optimizer import OptimizerParams, PortfolioConstraints

MOCK_CFG = MockDataConfig(n_assets=15, start="2020-01-01", end="2020-12-31",
                          n_industries=4, n_sub_industries=8, seed=5)


def _write_csv_suite(root, with_optional: bool = True, field_dict: bool = False) -> None:
    """从 MockAdapter 生成一套中信格式 CSV。"""
    adapter = MockAdapter(MOCK_CFG)
    symbols = [f"MOCK{i + 1:04d}" for i in range(MOCK_CFG.n_assets)]

    # 资产池：两个池合并到一个文件，pool_id 列区分
    pool_frames = []
    for pid in ("pool29", "pool100"):
        rec = adapter.load_asset_pool_history(pid).records.copy()
        rec.insert(0, "pool_id", pid)
        pool_frames.append(rec)
    pd.concat(pool_frames).to_csv(root / "asset_pool.csv", index=False)

    market = adapter.load_daily_market(symbols, MOCK_CFG.start, MOCK_CFG.end).rows
    market.to_csv(root / "daily_market.csv", index=False)
    status = adapter.load_trading_status(symbols, MOCK_CFG.start, MOCK_CFG.end).rows
    status.to_csv(root / "trading_status.csv", index=False)

    msf = adapter.load_market_sector_factor(MOCK_CFG.start, MOCK_CFG.end)
    long_parts = []
    for kind, frame in (("index", msf.index_returns), ("industry", msf.industry_index_returns)):
        melted = frame.reset_index().melt(id_vars="index", var_name="code", value_name="value")
        melted = melted.rename(columns={"index": "trade_date"})
        melted["kind"] = kind
        long_parts.append(melted)
    for kind, s in (("risk_free", msf.risk_free_rate), ("cash_mgmt", msf.cash_mgmt_yield)):
        long_parts.append(pd.DataFrame(
            {"trade_date": s.index, "code": s.name, "kind": kind, "value": s.to_numpy()}
        ))
    pd.concat(long_parts).to_csv(root / "market_sector_factor.csv", index=False)

    pd.DataFrame([{
        "config_id": "default", "rebalance_freq": "1m", "commission_bps": 2.5,
        "stamp_tax_bps": 10.0, "slippage_bps": 5.0, "impact_coef": 0.0,
        "cash_lower": 0.0, "cash_upper": 0.05, "max_turnover": None,
        "liquidity_limit": None, "weight_upper_json": json.dumps({s: 0.2 for s in symbols}),
        "blacklist": "",
    }]).to_csv(root / "constraints_costs.csv", index=False)

    if with_optional:
        ca = adapter.load_corporate_actions(symbols, MOCK_CFG.start, MOCK_CFG.end).records
        ca.to_csv(root / "corporate_actions.csv", index=False)
        pd.DataFrame([
            {"item": "pool_29", "payload": ",".join(symbols[:29])},
            {"item": "sharpe_convention", "payload": "annualized_252"},
        ]).to_csv(root / "baseline_results.csv", index=False)
    if field_dict:
        pd.DataFrame([
            {"column": "close", "description": "收盘价（元，未复权）"},
            {"column": "volume", "description": "成交量（股）"},
        ]).to_csv(root / "field_dict.csv", index=False)


@pytest.fixture()
def csv_dir(tmp_path):
    _write_csv_suite(tmp_path)
    return tmp_path


def test_all_seven_methods(csv_dir) -> None:
    adapter = CsvAdapter(csv_dir)
    symbols = [f"MOCK{i + 1:04d}" for i in range(MOCK_CFG.n_assets)]
    pool = adapter.load_asset_pool_history("pool29")
    assert pool.pool_id == "pool29"
    assert len(pool.records[pool.records["is_cash_asset"] == False]) > 0
    mkt = adapter.load_daily_market(symbols, "2020-01-01", "2020-12-31")
    assert {"symbol", "trade_date", "close", "adj_factor"} <= set(mkt.rows.columns)
    sts = adapter.load_trading_status(symbols, "2020-01-01", "2020-12-31")
    assert sts.rows["buyable"].dtype == bool
    ca = adapter.load_corporate_actions(symbols, "2020-01-01", "2020-12-31")
    assert len(ca.records) > 0
    msf = adapter.load_market_sector_factor("2020-01-01", "2020-12-31")
    assert msf.index_returns.shape[1] == 3
    assert msf.industry_index_returns.shape[1] == MOCK_CFG.n_industries
    cc = adapter.load_constraints_costs("default")
    assert cc.weight_upper[symbols[0]] == 0.2
    assert cc.cash_range == (0.0, 0.05)
    base = adapter.load_baseline_results()
    assert base.pool_29 is not None and base.sharpe_convention == "annualized_252"


def test_roundtrip_matches_mock(csv_dir) -> None:
    symbols = [f"MOCK{i + 1:04d}" for i in range(MOCK_CFG.n_assets)]
    mock = MockAdapter(MOCK_CFG).load_daily_market(symbols, MOCK_CFG.start, MOCK_CFG.end).rows
    csv = CsvAdapter(csv_dir).load_daily_market(symbols, MOCK_CFG.start, MOCK_CFG.end).rows
    key = ["symbol", "trade_date"]
    mock = mock.sort_values(key).reset_index(drop=True)
    csv = csv.sort_values(key).reset_index(drop=True)
    assert len(mock) == len(csv)
    assert np.allclose(mock["close"], csv["close"], rtol=1e-9)


def test_optional_missing_tolerated(csv_dir) -> None:
    (csv_dir / "corporate_actions.csv").unlink()
    (csv_dir / "baseline_results.csv").unlink()
    adapter = CsvAdapter(csv_dir)
    with pytest.warns(UserWarning, match="可选数据文件缺失"):
        ca = adapter.load_corporate_actions(["MOCK0001"], "2020-01-01", "2020-12-31")
    assert ca.records.empty
    base = adapter.load_baseline_results()
    assert base.pool_29 is None  # 空容器


def test_required_missing_raises_with_manifest(csv_dir) -> None:
    (csv_dir / "daily_market.csv").unlink()
    adapter = CsvAdapter(csv_dir)
    with pytest.raises(FileNotFoundError, match="期望文件清单"):
        adapter.load_daily_market(["MOCK0001"], "2020-01-01", "2020-12-31")


def test_missing_column_chinese_hint(csv_dir) -> None:
    _write_csv_suite(csv_dir, field_dict=True)
    df = pd.read_csv(csv_dir / "daily_market.csv")
    df = df.drop(columns=["close"])
    df.to_csv(csv_dir / "daily_market.csv", index=False)
    adapter = CsvAdapter(csv_dir)
    with pytest.raises(ValueError, match="收盘价"):
        adapter.load_daily_market(["MOCK0001"], "2020-01-01", "2020-12-31")


def test_adj_factor_default_and_symbol_normalization(csv_dir) -> None:
    df = pd.read_csv(csv_dir / "daily_market.csv")
    df = df.drop(columns=["adj_factor"])
    df["symbol"] = df["symbol"].str.lower()  # 小写代码应被规范化
    df.to_csv(csv_dir / "daily_market.csv", index=False)
    adapter = CsvAdapter(csv_dir)
    mkt = adapter.load_daily_market(["mock0001"], "2020-01-01", "2020-12-31")
    assert (mkt.rows["adj_factor"] == 1.0).all()
    assert set(mkt.rows["symbol"]) == {"MOCK0001"}


def test_csv_adapter_drives_engine_like_mock(csv_dir) -> None:
    """CsvAdapter 与 MockAdapter 输出可互换进回测引擎。"""
    symbols = [f"MOCK{i + 1:04d}" for i in range(MOCK_CFG.n_assets)]

    def _run(adapter) -> pd.Series:
        market = adapter.load_daily_market(symbols, MOCK_CFG.start, MOCK_CFG.end)
        status = adapter.load_trading_status(symbols, MOCK_CFG.start, MOCK_CFG.end)
        engine = BacktestEngine(
            window_spec=WindowSpec(train="6m", valid="2m", test="1m", purge="5d", embargo="2d"),
            constraints=PortfolioConstraints(
                weight_cap={s: 0.2 for s in symbols}, cash_range=(0.0, 0.05), exposure_bounds={}
            ),
            optimizer_params=OptimizerParams(5.0, 1e-3, 1e-4),
            cost_model=CostModel(),
        )
        return engine.run(market, status, risk_free_daily=0.02 / 252).nav

    nav_mock = _run(MockAdapter(MOCK_CFG))
    nav_csv = _run(CsvAdapter(csv_dir))
    assert len(nav_mock) == len(nav_csv)
    assert np.allclose(nav_mock.to_numpy(), nav_csv.to_numpy(), rtol=1e-6, atol=1e-12)
