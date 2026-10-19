"""GBS 风险先验（E8.1）测试：暴力对拍、PSD、δ=1 退化、随机先验可复现。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qgbs.gbs.risk_prior import (
    gbs_prior_corr,
    mix_covariance,
    prior_covariance,
    random_prior_corr,
    static_prior_corr,
)
from qgbs.portfolio.risk import check_psd


def _mk_rho(n_modes: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n_modes, n_modes))
    cov = a @ a.T
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def test_gbs_prior_corr_bruteforce() -> None:
    """K = U·ρ̂·Uᵀ/676 与逐对暴力 mean(ρ̂[A_i, A_j]) 逐元素一致。"""
    rho = _mk_rho(32)
    addr = pd.DataFrame(
        [[0, 5, 9], [1, 5, 30], [2, 7, 9]],  # 3 股 × 3 模式（含跨股共享模式 5、9）
        index=["s0", "s1", "s2"],
    )
    K = gbs_prior_corr(addr, rho)
    for i in range(3):
        for j in range(3):
            sub = rho[np.ix_(addr.iloc[i].to_numpy(), addr.iloc[j].to_numpy())]
            assert K.iloc[i, j] == pytest.approx(sub.mean(), rel=1e-12)
    # 对称性
    assert np.allclose(K.to_numpy(), K.to_numpy().T)
    # 对角：K_ii = mean(ρ̂[A_i,A_i]) ∈ (0,1]（含对角 1 + 组内互相关）
    assert (np.diag(K.to_numpy()) <= 1.0 + 1e-12).all()


def test_prior_covariance_psd_and_scale() -> None:
    """先验协方差：PSD、对角 = vol²·K_ii、尺度校准对齐参照。"""
    rho = _mk_rho(16, seed=1)
    addr = pd.DataFrame([np.arange(4), np.arange(4, 8)], index=["a", "b"])
    K = gbs_prior_corr(addr, rho)
    vols = pd.Series({"a": 0.02, "b": 0.03})
    cov = prior_covariance(K, vols)
    assert check_psd(cov)
    assert np.isclose(cov[0, 0], 0.02**2 * K.iloc[0, 0])
    # 尺度校准：对角均值对齐 ref
    ref = 0.001
    cov2 = prior_covariance(K, vols, ref_var_mean=ref)
    assert float(np.mean(np.diag(cov2))) == pytest.approx(ref, rel=1e-9)


def test_mix_covariance_delta_one_is_identity() -> None:
    """δ=1 严格退化为样本（LW）输入；δ=0 严格为先验。"""
    a = np.array([[0.04, 0.01], [0.01, 0.09]])
    p = np.array([[0.05, 0.0], [0.0, 0.05]])
    assert np.array_equal(mix_covariance(a, p, 1.0), a)
    assert np.array_equal(mix_covariance(a, p, 0.0), p)
    mix = mix_covariance(a, p, 0.5)
    assert np.allclose(mix, 0.5 * a + 0.5 * p)
    with pytest.raises(ValueError):
        mix_covariance(a, p, 1.5)


def test_random_prior_reproducible() -> None:
    """随机先验：同 seed 逐位一致，不同 seed 不同；PSD。"""
    syms = ["a", "b", "c"]
    r1 = random_prior_corr(syms, seed=42)
    r2 = random_prior_corr(syms, seed=42)
    r3 = random_prior_corr(syms, seed=7)
    assert r1.equals(r2)
    assert not r1.equals(r3)
    assert check_psd(r1.to_numpy())
    assert np.allclose(np.diag(r1.to_numpy()), 1.0)


def test_static_prior_time_average() -> None:
    """静态先验 = 各日 K 的均值。"""
    rho = _mk_rho(16, seed=2)
    a1 = pd.DataFrame([np.arange(4), np.arange(2, 6)], index=["a", "b"])
    a2 = pd.DataFrame([np.arange(1, 5), np.arange(4, 8)], index=["a", "b"])
    k1, k2 = gbs_prior_corr(a1, rho), gbs_prior_corr(a2, rho)
    s = static_prior_corr([k1, k2])
    assert np.allclose(s.to_numpy(), (k1.to_numpy() + k2.to_numpy()) / 2)
