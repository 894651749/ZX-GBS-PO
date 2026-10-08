"""读出模型测试：Ridge / GBM 可训练可预测，特征名记录与冻结语义。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from qgbs.models.readout import GBMReadout, RidgeReadout
from qgbs.models.selection import search_hyperparams, select_features

N_DATES, N_SYM, N_FEAT = 100, 30, 4


@pytest.fixture(scope="module")
def panel() -> dict:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2021-01-01", periods=N_DATES)
    symbols = [f"S{i:02d}" for i in range(N_SYM)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    X = pd.DataFrame(
        rng.normal(0, 1, (len(idx), N_FEAT)), index=idx, columns=[f"f{k}" for k in range(N_FEAT)]
    )
    y_linear = 2.0 * X["f0"] - X["f1"] + rng.normal(0, 0.1, len(idx))
    y_nonlinear = np.sign(X["f0"]) * np.abs(X["f1"]) + rng.normal(0, 0.1, len(idx))
    return {"X": X, "y_linear": y_linear.rename("y"), "y_nonlinear": y_nonlinear.rename("y")}


def _split(panel: dict):
    dates = panel["X"].index.get_level_values("date").unique()
    tr = panel["X"].index.get_level_values("date").isin(dates[:80])
    te = ~tr
    return tr, te


def test_ridge_fit_predict(panel: dict) -> None:
    tr, te = _split(panel)
    X, y = panel["X"], panel["y_linear"]
    model = RidgeReadout(alpha=1.0).fit(X[tr], y[tr])
    pred = model.predict(X[te])
    assert pred.index.equals(X[te].index)
    assert model.feature_names_ == list(X.columns)
    rho = stats.spearmanr(pred, y[te]).statistic
    assert rho > 0.9  # 线性真值下 ridge 应高度还原截面排序


def test_gbm_fit_predict(panel: dict) -> None:
    tr, te = _split(panel)
    X, y = panel["X"], panel["y_nonlinear"]
    model = GBMReadout(params={"max_iter": 100, "max_depth": 3, "random_state": 0}).fit(X[tr], y[tr])
    pred = model.predict(X[te])
    rho = stats.spearmanr(pred, y[te]).statistic
    assert rho > 0.7  # 非线性真值下 GBM 应显著优于随机


def test_predict_unfitted_raises(panel: dict) -> None:
    with pytest.raises(RuntimeError):
        RidgeReadout().predict(panel["X"])
    with pytest.raises(RuntimeError):
        GBMReadout().predict(panel["X"])


def test_predict_with_missing_feature(panel: dict) -> None:
    tr, _ = _split(panel)
    X, y = panel["X"], panel["y_linear"]
    model = RidgeReadout().fit(X[tr], y[tr])
    pred = model.predict(X[["f0", "f1"]])  # 缺列自动补 0
    assert len(pred) == len(X)


def test_select_features_ranking(panel: dict) -> None:
    tr, _ = _split(panel)
    X, y = panel["X"][tr], panel["y_linear"][tr]
    res = select_features(X, y, method="ic_mean", max_features=2)
    assert res.selected == ["f0", "f1"]  # 真值只含 f0/f1，且 f0 权重更大
    assert res.scores["f0"] > res.scores["f1"]


def test_search_hyperparams_picks_best(panel: dict) -> None:
    tr, te = _split(panel)
    half = int(np.where(np.cumsum(np.ones(N_DATES)) == 60)[0][0])
    dates = panel["X"].index.get_level_values("date").unique()
    m_tr = panel["X"].index.get_level_values("date").isin(dates[:60])
    m_va = panel["X"].index.get_level_values("date").isin(dates[60:80])
    X, y = panel["X"], panel["y_linear"]
    res = search_hyperparams(
        "ridge", X[m_tr], y[m_tr], X[m_va], y[m_va], {"alpha": [0.01, 1.0, 100.0]}
    )
    assert res.best_params["alpha"] in (0.01, 1.0)  # 低噪声线性数据不应选最强正则
    assert len(res.search_log) == 3
    assert np.isfinite(res.valid_score)
