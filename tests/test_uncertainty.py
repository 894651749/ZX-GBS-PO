"""A2：三种不确定性口径测试（异方差合成数据）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from qgbs.models.blending import (
    QuantileUncertaintyModel,
    estimate_uncertainty_by_method,
    uncertainty_block_bootstrap,
    uncertainty_residual,
)

T, N = 400, 20


@pytest.fixture(scope="module")
def heteroskedastic() -> dict:
    """逐资产条件波动不同的残差面板（sigma 已知）。"""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=T)
    symbols = [f"S{i:02d}" for i in range(N)]
    sigma = pd.Series(np.linspace(0.005, 0.05, N), index=symbols)  # 真实条件波动
    resid = pd.DataFrame(
        rng.normal(0, 1, (T, N)) * sigma.to_numpy(), index=dates, columns=symbols
    )
    return {"resid": resid, "sigma": sigma}


def test_residual_uncertainty_tracks_sigma(heteroskedastic: dict) -> None:
    unc = uncertainty_residual(heteroskedastic["resid"])
    rho = stats.spearmanr(unc, heteroskedastic["sigma"]).statistic
    assert rho > 0.9


def test_block_bootstrap_tracks_sigma(heteroskedastic: dict) -> None:
    unc = uncertainty_block_bootstrap(heteroskedastic["resid"], block_length=21, n_bootstrap=50, seed=0)
    rho = stats.spearmanr(unc, heteroskedastic["sigma"]).statistic
    assert rho > 0.5
    # 同 seed 可复现
    unc2 = uncertainty_block_bootstrap(heteroskedastic["resid"], block_length=21, n_bootstrap=50, seed=0)
    assert np.allclose(unc, unc2)


def test_block_bootstrap_short_series_raises() -> None:
    short = pd.DataFrame({"a": [0.01, -0.01]}, index=pd.bdate_range("2021-01-01", periods=2))
    with pytest.raises(ValueError, match="block_length"):
        uncertainty_block_bootstrap(short, block_length=21)


@pytest.fixture(scope="module")
def quantile_data() -> dict:
    """条件异方差面板：y = x1 + sigma(x0)·eps，sigma 依赖特征。"""
    rng = np.random.default_rng(1)
    n = 4000
    idx = pd.MultiIndex.from_arrays(
        [np.repeat(pd.bdate_range("2020-01-01", periods=100), 40), np.tile(np.arange(40), 100)],
        names=["date", "symbol"],
    )
    x0 = rng.normal(0, 1, n)
    x1 = rng.normal(0, 1, n)
    sigma = 0.01 + 0.05 * np.abs(x0)
    y = x1 + sigma * rng.normal(0, 1, n)
    X = pd.DataFrame({"x0": x0, "x1": x1}, index=idx)
    return {"X": X, "y": pd.Series(y, index=idx), "sigma": pd.Series(sigma, index=idx)}


def test_quantile_model_tracks_conditional_sigma(quantile_data: dict) -> None:
    X, y, sigma = quantile_data["X"], quantile_data["y"], quantile_data["sigma"]
    dates = X.index.get_level_values("date").unique()
    tr = X.index.get_level_values("date").isin(dates[:80])
    model = QuantileUncertaintyModel(seed=0).fit(X[tr], y[tr])
    width = model.predict_uncertainty(X[~tr])
    rho = stats.spearmanr(width, sigma[~tr]).statistic
    assert rho > 0.5
    assert (width >= 0).all()


def test_dispatcher_all_methods(heteroskedastic: dict, quantile_data: dict) -> None:
    resid = heteroskedastic["resid"]
    for method in ("residual", "block_bootstrap"):
        unc = estimate_uncertainty_by_method(method, residuals=resid, n_bootstrap=20, seed=0)
        assert isinstance(unc, pd.Series) and len(unc) == N
    X, y = quantile_data["X"], quantile_data["y"]
    unc = estimate_uncertainty_by_method("quantile", X_train=X, y_train=y, X_score=X)
    assert len(unc) == 40  # 逐 symbol
    with pytest.raises(ValueError, match="未知不确定性口径"):
        estimate_uncertainty_by_method("bogus", residuals=resid)
    with pytest.raises(ValueError, match="需要"):
        estimate_uncertainty_by_method("residual")
