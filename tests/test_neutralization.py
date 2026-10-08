"""中性化标签测试：合成数据上构造已知因子结构，验证标签剥离市场/行业成分。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from qgbs.factors.neutralization import (
    MARKET_COL,
    JointFactorModel,
    build_alpha_label,
    cash_return_label,
)

T, N, K = 300, 12, 3
H = 10


@pytest.fixture(scope="module")
def synthetic() -> dict:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2021-01-01", periods=T)
    symbols = [f"S{i:02d}" for i in range(N)]
    ind_codes = [f"IND{k + 1:02d}" for k in range(K)]
    industry = pd.Series([ind_codes[i % K] for i in range(N)], index=symbols)
    mkt = rng.normal(0.0005, 0.010, T)
    ind_ret = rng.normal(0.0, 0.008, (T, K))
    alpha_true = np.linspace(-0.001, 0.001, N)  # 已知逐股票日 alpha（截距）
    beta = rng.uniform(0.6, 1.4, N)
    gamma = rng.uniform(0.5, 1.5, N)
    noise = rng.normal(0.0, 0.0005, (T, N))
    ind_idx = np.arange(N) % K
    R = beta * mkt[:, None] + ind_ret[:, ind_idx] * gamma + alpha_true + noise
    returns = pd.DataFrame(R, index=dates, columns=symbols)
    prices = 100.0 * np.exp(returns.cumsum())
    factor_df = pd.concat(
        [pd.Series(mkt, index=dates, name=MARKET_COL),
         pd.DataFrame(ind_ret, index=dates, columns=ind_codes)],
        axis=1,
    )
    return {
        "dates": dates, "symbols": symbols, "industry": industry,
        "returns": returns, "prices": prices, "factor_df": factor_df,
        "alpha_true": alpha_true, "beta": beta, "gamma": gamma,
    }


def test_factor_exposure_recovery(synthetic: dict) -> None:
    model = JointFactorModel().fit(
        synthetic["returns"], synthetic["industry"], synthetic["factor_df"]
    )
    assert model.is_fitted
    expo = model.exposures_
    # β̂ 应接近真值（带截距 OLS 无偏 + 低噪声）
    assert np.allclose(expo[MARKET_COL].to_numpy(), synthetic["beta"], atol=0.05)
    ind_col = [f"IND{(i % K) + 1:02d}" for i in range(N)]
    gamma_hat = [expo.loc[sym, ind_col[i]] for i, sym in enumerate(synthetic["symbols"])]
    assert np.allclose(gamma_hat, synthetic["gamma"], atol=0.05)


def test_fit_with_intercept_recovers_beta() -> None:
    """因子均值非零 + 个股截距：带截距回归应无偏恢复 β。

    无截距回归会把截距 a_i 部分摊入 β̂（偏差 ≈ a_i·E[f]/E[f²]），
    本用例的因子均值显著非零，旧实现无法通过以下容差。
    """
    rng = np.random.default_rng(7)
    n = 500
    dates = pd.bdate_range("2022-01-01", periods=n)
    symbols = ["S1", "S2"]
    industry = pd.Series({"S1": "IND01", "S2": "IND01"})  # IND01 不在因子表中 → 仅用 MKT
    mkt = rng.normal(0.004, 0.01, n)  # 因子均值显著非零
    a_true = np.array([0.003, -0.002])
    b_true = np.array([0.8, 1.2])
    returns = pd.DataFrame(
        a_true + b_true * mkt[:, None] + rng.normal(0.0, 5e-4, (n, 2)),
        index=dates, columns=symbols,
    )
    factor_df = pd.DataFrame({MARKET_COL: mkt}, index=dates)
    expo = JointFactorModel().fit(returns, industry, factor_df).exposures_
    assert np.allclose(expo[MARKET_COL].to_numpy(), b_true, atol=0.02)


def test_label_strips_factor_component(synthetic: dict) -> None:
    model = JointFactorModel().fit(
        synthetic["returns"], synthetic["industry"], synthetic["factor_df"]
    )
    labels = build_alpha_label(
        synthetic["prices"], synthetic["industry"], H, synthetic["factor_df"], model=model
    )
    assert labels.index.names == ["date", "symbol"]
    lab_mean = labels.groupby(level="symbol").mean()
    # 标签应恢复 h·alpha_true 的截面排序（截距不扣，留在标签中）
    rho = stats.spearmanr(lab_mean.to_numpy(), H * synthetic["alpha_true"]).statistic
    assert rho > 0.9
    # 标签与市场 beta 应基本无关（市场成分已被剥离）
    rho_mkt = abs(stats.spearmanr(lab_mean.to_numpy(), synthetic["beta"]).statistic)
    assert rho_mkt < 0.4
    # 末尾 horizon 日无标签
    last_dates = synthetic["dates"][-H:]
    assert labels.index.get_level_values("date").max() <= synthetic["dates"][-H - 1]
    assert not labels.index.get_level_values("date").isin(last_dates).any()


def test_label_deducts_factor_part_only() -> None:
    """标签口径：只减 β̂ᵀ·f_fwd，不扣截距 a_i（截距留给读出模型学习）。"""
    rng = np.random.default_rng(11)
    n, h = 120, 2
    dates = pd.bdate_range("2023-01-01", periods=n)
    industry = pd.Series({"S1": "IND01"})
    mkt = rng.normal(0.002, 0.01, n)
    returns = pd.DataFrame(
        {"S1": 0.004 + 0.9 * mkt + rng.normal(0.0, 5e-4, n)}, index=dates
    )
    prices = 100.0 * np.exp(returns.cumsum())
    factor_df = pd.DataFrame({MARKET_COL: mkt}, index=dates)
    model = JointFactorModel().fit(returns, industry, factor_df)
    labels = build_alpha_label(prices, industry, h, factor_df, model=model)
    t = dates[10]
    fwd_ret = prices["S1"].iloc[12] / prices["S1"].iloc[10] - 1.0
    fwd_mkt = (1.0 + mkt[11]) * (1.0 + mkt[12]) - 1.0  # 复利口径
    expected = fwd_ret - model.exposures_.loc["S1", MARKET_COL] * fwd_mkt
    assert labels.loc[(t, "S1")] == pytest.approx(expected, rel=1e-12)


def test_transform_residual_market_neutral(synthetic: dict) -> None:
    model = JointFactorModel().fit(
        synthetic["returns"], synthetic["industry"], synthetic["factor_df"]
    )
    out = model.transform(
        synthetic["returns"], synthetic["industry"], synthetic["factor_df"]
    )
    resid_mean = out.alpha.mean(axis=0)
    # 完整残差 r − a_i − β̂ᵀf：含截距 OLS 在训练窗内残差均值 ≈ 0
    # （个股均值 alpha_true 已被截距吸收，不再留在残差中）
    assert np.allclose(resid_mean.to_numpy(), 0.0, atol=1e-6)
    assert out.r_squared.mean() > 0.9  # 强因子结构下回归 R² 应高


def test_fit_requires_only_used_factor_columns() -> None:
    """fit 只对实际用到的因子列要求非缺失：其他行业列 NaN 不应丢整天。"""
    rng = np.random.default_rng(3)
    n = 120
    dates = pd.bdate_range("2023-01-01", periods=n)
    mkt = rng.normal(0.001, 0.01, n)
    ind1 = rng.normal(0.0, 0.005, n)
    ind2 = rng.normal(0.0, 0.005, n)
    ind1[5] = np.nan  # IND01 第 5 日缺失
    factor_df = pd.DataFrame({MARKET_COL: mkt, "IND01": ind1, "IND02": ind2}, index=dates)
    noise = rng.normal(0.0, 1e-4, (n, 2))
    r_in2 = 0.001 + 0.9 * mkt + 1.1 * ind2 + noise[:, 0]
    r_in1 = -0.001 + 1.2 * mkt + 0.8 * ind1 + noise[:, 1]
    returns = pd.DataFrame({"S1": r_in2, "S2": r_in1}, index=dates)
    industry = pd.Series({"S1": "IND02", "S2": "IND01"})
    expo = JointFactorModel().fit(returns, industry, factor_df).exposures_
    # S1 属于 IND02：第 5 日不应因 IND01 缺失而丢弃 → 等价于全样本带截距 OLS
    X_full = np.column_stack([np.ones(n), mkt, ind2])
    coef_full, *_ = np.linalg.lstsq(X_full, r_in2, rcond=None)
    assert expo.loc["S1", MARKET_COL] == pytest.approx(coef_full[1])
    assert expo.loc["S1", "IND02"] == pytest.approx(coef_full[2])
    # S2 属于 IND01：仅剔除第 5 日
    mask = np.ones(n, bool)
    mask[5] = False
    X_sub = np.column_stack([np.ones(n - 1), mkt[mask], ind1[mask]])
    coef_sub, *_ = np.linalg.lstsq(X_sub, r_in1[mask], rcond=None)
    assert expo.loc["S2", MARKET_COL] == pytest.approx(coef_sub[1])
    assert expo.loc["S2", "IND01"] == pytest.approx(coef_sub[2])


def test_transform_nan_on_missing_factor_days() -> None:
    """缺因子收益的日期不输出 alpha（置 NaN），不得以原始收益冒充。"""
    rng = np.random.default_rng(5)
    n = 60
    dates = pd.bdate_range("2023-01-01", periods=n)
    mkt = rng.normal(0.001, 0.01, n)
    ind1 = rng.normal(0.0, 0.005, n)
    ind2 = rng.normal(0.0, 0.005, n)
    factor_df = pd.DataFrame({MARKET_COL: mkt, "IND01": ind1, "IND02": ind2}, index=dates)
    noise = rng.normal(0.0, 1e-4, (n, 2))
    returns = pd.DataFrame(
        {
            "S1": 0.001 + 0.9 * mkt + 1.0 * ind1 + noise[:, 0],
            "S2": -0.001 + 1.1 * mkt + 0.8 * ind2 + noise[:, 1],
        },
        index=dates,
    )
    industry = pd.Series({"S1": "IND01", "S2": "IND02"})
    model = JointFactorModel().fit(returns, industry, factor_df)
    f2 = factor_df.copy()
    f2.loc[dates[7], "IND01"] = np.nan  # 仅 IND01 缺失
    f2.loc[dates[9]] = np.nan  # 整天缺失
    out = model.transform(returns, industry, f2)
    # 单列缺失只影响使用该列的股票
    assert np.isnan(out.alpha.loc[dates[7], "S1"])
    assert np.isfinite(out.alpha.loc[dates[7], "S2"])
    # 整天因子缺失 → 当日所有股票无 alpha
    assert out.alpha.loc[dates[9]].isna().all()
    # 正常日期为完整残差，不等于原始收益
    assert np.isfinite(out.alpha.loc[dates[8]]).all()
    assert not np.allclose(out.alpha.loc[dates[8]].to_numpy(), returns.loc[dates[8]].to_numpy())


def test_label_dropped_on_missing_factor_days() -> None:
    """前视窗 (t, t+h] 内因子收益缺失 → 对应标签不输出。"""
    rng = np.random.default_rng(13)
    n, h = 80, 3
    dates = pd.bdate_range("2023-01-01", periods=n)
    mkt = rng.normal(0.001, 0.01, n)
    ind1 = rng.normal(0.0, 0.005, n)
    factor_df = pd.DataFrame({MARKET_COL: mkt, "IND01": ind1}, index=dates)
    returns = pd.DataFrame(
        {"S1": 0.9 * mkt + ind1 + rng.normal(0.0, 1e-4, n)}, index=dates
    )
    prices = 100.0 * np.exp(returns.cumsum())
    industry = pd.Series({"S1": "IND01"})
    model = JointFactorModel().fit(returns, industry, factor_df)
    f2 = factor_df.copy()
    f2.loc[dates[20], MARKET_COL] = np.nan
    labels = build_alpha_label(prices, industry, h, f2, model=model)
    # 前视窗覆盖第 20 日的标签（t ∈ {17, 18, 19}）全部缺席
    for k in (17, 18, 19):
        assert (dates[k], "S1") not in labels.index
    # 视窗 (16, 19] 与 (20, 23] 不含缺失日 → 标签保留
    assert (dates[16], "S1") in labels.index
    assert (dates[20], "S1") in labels.index


def test_cash_return_label() -> None:
    # 常数利率：复利口径 (1+rf)^h − 1，而非简单求和 h·rf
    rf = pd.Series(0.02 / 252, index=pd.bdate_range("2021-01-01", periods=50))
    label = cash_return_label(rf, horizon=10)
    assert label.iloc[0] == pytest.approx((1.0 + 0.02 / 252) ** 10 - 1.0, rel=1e-12)
    assert label.iloc[-10:].isna().all()  # 末尾无未来数据
    # 非常数利率：复利与简单求和产生可分辨差异；窗口为 (t, t+h]（第 t+1..t+h 日）
    rf2 = pd.Series([0.01, 0.02, 0.03, 0.005], index=pd.bdate_range("2024-01-01", periods=4))
    label2 = cash_return_label(rf2, horizon=2)
    assert label2.iloc[0] == pytest.approx((1.02 * 1.03) - 1.0, rel=1e-12)
    assert label2.iloc[0] != pytest.approx(0.02 + 0.03, rel=1e-6)  # ≠ 简单求和
    assert label2.iloc[1] == pytest.approx((1.03 * 1.005) - 1.0, rel=1e-12)
    assert label2.iloc[2:].isna().all()
