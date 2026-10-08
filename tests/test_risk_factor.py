"""A1：FactorRiskModel 与引擎 cov_method='factor' 接线测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qgbs.backtest.engine import BacktestEngine
from qgbs.backtest.rolling import WindowSpec
from qgbs.data.adapters.mock import MockAdapter, MockDataConfig
from qgbs.portfolio.costs import CostModel
from qgbs.portfolio.optimizer import OptimizerParams, PortfolioConstraints
from qgbs.portfolio.risk import FactorRiskConfig, FactorRiskModel, check_psd

T, N, K = 300, 30, 4


@pytest.fixture(scope="module")
def synthetic() -> dict:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2021-01-01", periods=T)
    symbols = [f"S{i:02d}" for i in range(N)]
    industry = pd.Series([f"IND{(i % K) + 1:02d}" for i in range(N)], index=symbols)
    B = pd.get_dummies(industry).astype(float)
    # 已知因子协方差（强因子结构）与特质方差
    a = rng.normal(0, 1, (K, K))
    sf_true = a @ a.T * 1e-4 + np.eye(K) * 1e-4
    d_true = rng.uniform(0.5, 1.5, N) * 2e-4
    F = rng.multivariate_normal(np.zeros(K), sf_true, T)
    eps = rng.normal(0, 1, (T, N)) * np.sqrt(d_true)
    returns = pd.DataFrame(F @ B.to_numpy().T + eps, index=dates, columns=symbols)
    cov_true = B.to_numpy() @ sf_true @ B.to_numpy().T + np.diag(d_true)
    return {"returns": returns, "B": B, "cov_true": cov_true}


def test_factor_model_beats_sample_cov(synthetic: dict) -> None:
    rets, B, cov_true = synthetic["returns"], synthetic["B"], synthetic["cov_true"]
    frm = FactorRiskModel(FactorRiskConfig(factor_cov_method="sample")).fit(rets, B)
    cov_factor = frm.covariance()
    cov_sample = rets.cov().to_numpy()
    err_factor = np.linalg.norm(cov_factor - cov_true, "fro")
    err_sample = np.linalg.norm(cov_sample - cov_true, "fro")
    assert err_factor < err_sample  # 因子结构下因子模型应更接近真实 Σ
    assert check_psd(cov_factor)


def test_ewma_and_output_alignment(synthetic: dict) -> None:
    frm = FactorRiskModel(FactorRiskConfig(factor_cov_method="ewma", ewma_halflife=30)).fit(
        synthetic["returns"], synthetic["B"]
    )
    cov = frm.covariance()
    assert cov.shape == (N, N)
    assert check_psd(cov)
    assert frm.factor_covariance_.shape == (K, K)
    assert (frm.specific_risk_ > 0).all()
    # 输出顺序与暴露行顺序一致：调换暴露顺序 → 协方差块对应调换
    perm = list(range(N))[::-1]
    cov_perm = frm.covariance(synthetic["B"].iloc[perm])
    assert np.allclose(cov_perm, cov[np.ix_(perm, perm)])


def test_unfitted_and_min_obs(synthetic: dict) -> None:
    with pytest.raises(RuntimeError):
        FactorRiskModel().covariance()
    with pytest.raises(ValueError, match="min_obs"):
        FactorRiskModel(FactorRiskConfig(min_obs=1000)).fit(synthetic["returns"], synthetic["B"])


def test_scale_calibration_matches_sample_diag(synthetic: dict) -> None:
    """尺度校准：合成协方差对角线方差均值 == 估计窗样本协方差对角线方差均值。"""
    rets, B = synthetic["returns"], synthetic["B"]
    frm = FactorRiskModel(FactorRiskConfig(factor_cov_method="sample")).fit(rets, B)
    cov = frm.covariance()
    ref = float(rets.var(axis=0).mean())  # 样本协方差对角线方差均值（ddof=1）
    assert frm.scale_factor_ is not None
    assert frm.scale_note_ is None
    assert float(np.mean(np.diag(cov))) == pytest.approx(ref, rel=1e-8)
    assert check_psd(cov)  # 正数整体缩放不破坏 PSD
    # 换暴露合成时按同一参照校准
    cov_perm = frm.covariance(B.iloc[::-1])
    assert float(np.mean(np.diag(cov_perm))) == pytest.approx(ref, rel=1e-8)


def test_scale_calibration_disabled(synthetic: dict) -> None:
    """scale_calibrate=False 时输出未校准矩阵（无校准系数、无跳过说明）。"""
    frm = FactorRiskModel(
        FactorRiskConfig(factor_cov_method="sample", scale_calibrate=False)
    ).fit(synthetic["returns"], synthetic["B"])
    cov = frm.covariance()
    assert frm.scale_factor_ is None
    assert frm.scale_note_ is None
    assert check_psd(cov)


def test_scale_calibration_skipped_without_reference() -> None:
    """参照无效（估计窗收益零方差）时跳过校准并在 scale_note_ 注明。"""
    dates = pd.bdate_range("2021-01-01", periods=60)
    symbols = [f"S{i:02d}" for i in range(8)]
    rets = pd.DataFrame(0.0, index=dates, columns=symbols)
    B = pd.get_dummies(pd.Series([f"IND{(i % 2) + 1}" for i in range(8)], index=symbols)).astype(float)
    frm = FactorRiskModel(FactorRiskConfig(factor_cov_method="sample", min_obs=10)).fit(rets, B)
    cov = frm.covariance()
    assert frm.scale_factor_ is None
    assert frm.scale_note_ is not None and "跳过" in frm.scale_note_
    assert np.allclose(cov, 0.0)


def test_engine_cov_method_factor() -> None:
    """引擎接线：cov_method='factor' + 行业暴露矩阵，回测可跑通。"""
    adapter = MockAdapter(MockDataConfig(n_assets=20, start="2020-01-01", end="2021-06-30",
                                         n_industries=4, n_sub_industries=8, seed=9))
    symbols = [f"MOCK{i + 1:04d}" for i in range(20)]
    market = adapter.load_daily_market(symbols, "2020-01-01", "2021-06-30")
    status = adapter.load_trading_status(symbols, "2020-01-01", "2021-06-30")
    pool = adapter.load_asset_pool_history("pool100")
    stocks = pool.records[~pool.records["is_cash_asset"]]
    expo = pd.get_dummies(stocks.set_index("symbol")["citics_industry_l1"]).astype(float)
    engine = BacktestEngine(
        window_spec=WindowSpec(train="6m", valid="2m", test="1m", purge="5d", embargo="2d"),
        constraints=PortfolioConstraints(
            weight_cap={s: 0.2 for s in symbols}, cash_range=(0.0, 0.05), exposure_bounds={}
        ),
        optimizer_params=OptimizerParams(5.0, 1e-3, 1e-4),
        cost_model=CostModel(),
        cov_method="factor",
    )
    bt = engine.run(market, status, risk_free_daily=0.02 / 252, exposure_matrix=expo)
    assert len(bt.nav) > 100
    assert (bt.nav > 0).all()
    assert "cov_note" not in bt.ledger.columns or bt.ledger["cov_note"].isna().all()


def test_engine_factor_degrades_without_exposure() -> None:
    """无暴露矩阵时显式降级回 Ledoit-Wolf 并在 ledger 标注。"""
    adapter = MockAdapter(MockDataConfig(n_assets=20, start="2020-01-01", end="2021-06-30",
                                         n_industries=4, n_sub_industries=8, seed=9))
    symbols = [f"MOCK{i + 1:04d}" for i in range(20)]
    market = adapter.load_daily_market(symbols, "2020-01-01", "2021-06-30")
    status = adapter.load_trading_status(symbols, "2020-01-01", "2021-06-30")
    engine = BacktestEngine(
        window_spec=WindowSpec(train="6m", valid="2m", test="1m", purge="5d", embargo="2d"),
        constraints=PortfolioConstraints(
            weight_cap={s: 0.2 for s in symbols}, cash_range=(0.0, 0.05), exposure_bounds={}
        ),
        optimizer_params=OptimizerParams(5.0, 1e-3, 1e-4),
        cost_model=CostModel(),
        cov_method="factor",
    )
    bt = engine.run(market, status)  # 无 exposure_matrix
    assert len(bt.nav) > 100
    assert bt.ledger["cov_note"].str.contains("degraded").all()
