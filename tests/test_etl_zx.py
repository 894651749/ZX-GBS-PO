"""qgbs.data.etl_zx 转换函数单元测试（小规模合成夹具，验证口径而非数据）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qgbs.data.etl_zx import (
    build_asset_pool,
    build_constraints_costs,
    build_daily_market,
    build_market_sector_factor,
    build_trading_status,
    industry_ci_code_map,
)


def _index_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SECU_ID": ["000300.SH", "CI005024.WI"],
            "SECU_NAME": ["沪深300指数", "中信交通运输"],
            "DT": ["2023-01-03 00:00:00", "2023-01-03 00:00:00"],
            "CLOSE": [4000.0, 2000.0],
            "PCTCHANGE": [0.001, -0.002],
        }
    )


def _static_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "证券代码": ["000001.SZ"],
            "交易所": ["SZ"],
            "上市日期": ["2000-01-01 00:00:00"],
            "退市日期": [None],
            "中信一级行业": ["交通运输"],
            "中信二级行业": ["物流"],
        }
    )


def _market_df() -> pd.DataFrame:
    # 第二行为除权日：复权前收盘(9.9) ≠ 上一日复权收盘(10.0)
    return pd.DataFrame(
        {
            "DT": ["2023-01-03 00:00:00", "2023-01-04 00:00:00"],
            "SECU_ID": ["000001.SZ", "000001.SZ"],
            "复权前收盘价": [9.5, 9.9],
            "复权开盘价": [9.6, 9.95],
            "复权收盘价": [10.0, 10.4],
            "成交量": [100.0, 200.0],
            "成交额": [9_600_000.0, 20_200_000.0],
            "换手率": [1.1, 2.2],
            "总市值": [1.0, 1.0],
            "流通市值": [0.8, 0.8],
            "ST/*ST": [None, "ST"],
            "可否买卖": ["是", "否"],
        }
    )


def test_industry_ci_code_map() -> None:
    mapping = industry_ci_code_map(_index_df())
    assert mapping == {"交通运输": "CI005024.WI"}


def test_daily_market_return_consistency_and_units() -> None:
    out = build_daily_market(_market_df())
    assert list(out["return_basis"].unique()) == ["adjusted"]
    # 日收益 = 复权收盘/复权前收盘 − 1（除权日不走 close 环比）
    assert out.loc[0, "daily_return"] == pytest.approx(10.0 / 9.5 - 1.0)
    assert out.loc[1, "daily_return"] == pytest.approx(10.4 / 9.9 - 1.0)
    # 一致复权价：close/prev_close − 1 ≡ daily_return（逐行成立，含除权日）
    assert np.allclose(out["close"] / out["prev_close"] - 1.0, out["daily_return"])
    # 量额恒等：volume × close ≡ amount
    assert np.allclose(out["volume"] * out["close"], out["amount"], rtol=1e-12)
    # 市值 万元 → 元
    assert out.loc[0, "total_mv"] == pytest.approx(1.0e4)
    assert out.loc[0, "float_mv"] == pytest.approx(0.8e4)


def test_trading_status_derivation() -> None:
    out = build_trading_status(_market_df(), _static_df())
    row_ok, row_stop = out.iloc[0], out.iloc[1]
    assert bool(row_ok["buyable"]) and not bool(row_ok["is_suspended"])
    assert bool(row_stop["is_suspended"]) and not bool(row_stop["buyable"])
    assert bool(row_stop["is_st"]) and not bool(row_ok["is_st"])
    assert row_ok["lot_size"] == 100
    assert row_ok["list_date"] == "2000-01-01"


def test_asset_pool_industry_mapping_and_pool_id() -> None:
    out = build_asset_pool(_static_df(), industry_ci_code_map(_index_df()))
    assert set(out["pool_id"]) == {"zz800"}
    assert out.loc[0, "citics_industry_l1"] == "CI005024.WI"
    assert out.loc[0, "citics_industry_l1_name"] == "交通运输"
    assert out.loc[0, "is_cash_asset"] is False or out.loc[0, "is_cash_asset"] == False  # noqa: E712
    assert out.loc[0, "effective_date"] == "2000-01-01"


def test_asset_pool_unknown_industry_raises() -> None:
    bad = _static_df()
    bad["中信一级行业"] = ["不存在的行业"]
    with pytest.raises(ValueError, match="无对应行业指数"):
        build_asset_pool(bad, industry_ci_code_map(_index_df()))


def test_market_sector_factor_kinds() -> None:
    out = build_market_sector_factor(_index_df())
    kind = dict(zip(out["code"], out["kind"]))
    assert kind == {"000300.SH": "index", "CI005024.WI": "industry"}
    assert out["value"].tolist() == [0.001, -0.002]


def test_constraints_costs_placeholder_marked() -> None:
    out = build_constraints_costs()
    assert out.loc[0, "config_id"] == "default"
    assert "占位待中信" in out.loc[0, "note"]
