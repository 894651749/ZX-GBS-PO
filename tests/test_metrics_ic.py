"""指标正确性测试：手算小样例核对 IC/ICIR/方向准确率/CVaR/HHI/分层收益。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qgbs.backtest.metrics import (
    cvar,
    direction_accuracy,
    exposure_report,
    ic_series,
    icir,
    layered_returns,
    pearson_ic,
    rank_ic,
    weight_concentration,
)


def test_rank_ic_hand_computed() -> None:
    # 完全单调 → 1；完全反向 → −1
    pred = pd.Series([1.0, 2.0, 3.0, 4.0])
    real = pd.Series([0.1, 0.2, 0.3, 0.4])
    assert rank_ic(pred, real) == pytest.approx(1.0)
    assert rank_ic(pred, -real) == pytest.approx(-1.0)
    # 手算：pred ranks [1,2,3,4]，real ranks [1,2,4,3] → 1 − 6·2/(4·15) = 0.8
    real2 = pd.Series([0.1, 0.2, 0.4, 0.3])
    assert rank_ic(pred, real2) == pytest.approx(0.8)


def test_pearson_ic_matches_numpy() -> None:
    pred = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    real = pd.Series([2.1, 1.9, 4.2, 3.8, 5.5])
    assert pearson_ic(pred, real) == pytest.approx(np.corrcoef(pred, real)[0, 1])


def test_ic_series_and_icir() -> None:
    dates = pd.bdate_range("2021-01-01", periods=3)
    idx = pd.MultiIndex.from_product([dates, ["a", "b", "c", "d"]], names=["date", "symbol"])
    pred = pd.Series(np.tile([1.0, 2.0, 3.0, 4.0], 3), index=idx)
    real = pd.Series(np.tile([0.1, 0.2, 0.3, 0.4], 3), index=idx)
    ics = ic_series(pred, real)
    assert len(ics) == 3 and np.allclose(ics, 1.0)
    assert np.isnan(icir(ics))  # 完全稳定时 std=0 → ICIR 定义为 NaN
    assert icir(pd.Series([1.0, 2.0, 3.0])) == pytest.approx(2.0 / 1.0)


def test_direction_accuracy() -> None:
    pred = pd.Series([1.0, -1.0, 1.0])
    real = pd.Series([0.5, -0.2, -0.1])
    assert direction_accuracy(pred, real) == pytest.approx(2 / 3)


def test_cvar_hand_computed() -> None:
    rets = pd.Series(np.arange(-5, 5) * 0.01)  # -0.05 … 0.04
    # level=0.8 → 左尾 20% 分位 = -0.0322，尾部 {-0.05,-0.04,-0.03...} 中 ≤ 阈值的均值
    thr = rets.quantile(0.2)
    expected = rets[rets <= thr].mean()
    assert cvar(rets, level=0.8) == pytest.approx(expected)


def test_weight_concentration() -> None:
    w = pd.DataFrame([[0.5, 0.5], [0.8, 0.2]], columns=["a", "b"],
                     index=pd.to_datetime(["2021-01-01", "2021-02-01"]))
    hhi = weight_concentration(w)
    assert hhi.iloc[0] == pytest.approx(0.5)
    assert hhi.iloc[1] == pytest.approx(0.68)


def test_layered_returns() -> None:
    dates = pd.bdate_range("2021-01-01", periods=2)
    idx = pd.MultiIndex.from_product([dates, ["a", "b", "c", "d"]], names=["date", "symbol"])
    pred = pd.Series(np.tile([1.0, 2.0, 3.0, 4.0], 2), index=idx)
    real = pd.Series(np.tile([0.01, 0.02, 0.03, 0.04], 2), index=idx)
    layered = layered_returns(pred, real, n_layers=2)
    assert layered.loc[0] == pytest.approx(0.015)  # 低预测层
    assert layered.loc[1] == pytest.approx(0.035)  # 高预测层


def test_exposure_report() -> None:
    w = pd.DataFrame([[0.6, 0.4]], columns=["a", "b"], index=pd.to_datetime(["2021-01-01"]))
    expo = pd.DataFrame({"IND1": [1.0, 0.0], "IND2": [0.0, 1.0]}, index=["a", "b"])
    rep = exposure_report(w, expo)
    assert rep.loc[pd.Timestamp("2021-01-01"), "IND1"] == pytest.approx(0.6)
    assert rep.loc[pd.Timestamp("2021-01-01"), "IND2"] == pytest.approx(0.4)
