"""精确小规模 GBS 采样后端（exact_gbs）测试。"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qgbs.gbs.backends.exact_gbs import ExactGBSConfig, ExactGBSSampler


def _fock_click_probs_pure(sampler: ExactGBSSampler, cutoff: int) -> np.ndarray:
    """独立路径：纯态 state_vector 聚合 click pattern 概率（截断归一化）。"""
    from thewalrus.quantum import state_vector

    n = sampler.n_modes
    psi = state_vector(np.zeros(2 * n), sampler.covariance, normalize=True, cutoff=cutoff)
    probs = np.zeros(2**n)
    for photons in np.ndindex(*([cutoff] * n)):
        s = sum((1 << j) for j in range(n) if photons[j] > 0)
        probs[s] += abs(psi[photons]) ** 2
    return probs


def _fock_click_probs_lossy(sampler: ExactGBSSampler, cutoff: int) -> np.ndarray:
    """独立路径：损耗态 density_matrix 聚合 click pattern 概率。

    注意 thewalrus density_matrix 返回 SF 交错轴序 (n_1, m_1, n_2, m_2, ...)。
    """
    from thewalrus.quantum import density_matrix

    n = sampler.n_modes
    rho = density_matrix(np.zeros(2 * n), sampler.covariance, normalize=True, cutoff=cutoff)
    probs = np.zeros(2**n)
    for photons in np.ndindex(*([cutoff] * n)):
        sf_idx = tuple(k for x in photons for k in (x, x))
        s = sum((1 << j) for j in range(n) if photons[j] > 0)
        probs[s] += rho[sf_idx].real
    return probs


@pytest.fixture(scope="module")
def exact6() -> ExactGBSSampler:
    return ExactGBSSampler(
        ExactGBSConfig(n_temporal=3, n_spatial=2, seed=7, system_efficiency=0.9)
    )


def test_satisfies_gbs_sampler_protocol(exact6: ExactGBSSampler) -> None:
    assert exact6.n_modes == 6
    batch = exact6.sample(500)
    assert batch.samples.shape == (500, 6)
    assert set(np.unique(batch.samples)) <= {0, 1}
    m = batch.meta
    assert m.source == "simulation"
    assert m.system_efficiency == 0.9
    assert m.simulation_params["method"] == "exact_gbs"
    assert m.simulation_params["seed"] == 7
    assert len(m.simulation_params["squeezing"]) == 6
    assert m.mode_config["n_temporal"] == 3 and m.mode_config["n_spatial"] == 2
    assert m.calibration_session_id == "exact-gbs-session-7"


def test_pattern_probabilities_normalized(exact6: ExactGBSSampler) -> None:
    probs = exact6.pattern_probabilities()
    assert probs.shape == (2**6,)
    assert probs.min() >= 0.0
    assert abs(probs.sum() - 1.0) < 1e-12


def test_vacuum_limit() -> None:
    """压缩趋零 → 态趋真空 → 全零 pattern 概率趋 1。"""
    sampler = ExactGBSSampler(
        ExactGBSConfig(
            n_temporal=3, n_spatial=2, seed=1, squeezing_low=0.005, squeezing_high=0.01
        )
    )
    probs = sampler.pattern_probabilities()
    assert probs[0] > 0.99
    assert probs[1:].sum() < 0.01


def test_pure_case_matches_fock_state_vector() -> None:
    """η=1 纯态：Torontonian 公式 vs 独立 Fock 空间计算（截断误差内一致）。"""
    sampler = ExactGBSSampler(
        ExactGBSConfig(
            n_temporal=3,
            n_spatial=2,
            seed=11,
            system_efficiency=1.0,
            squeezing_low=0.2,
            squeezing_high=0.6,
        )
    )
    p_tor = sampler.pattern_probabilities()
    p_fock = _fock_click_probs_pure(sampler, cutoff=7)
    assert p_fock.sum() > 0.999  # 截断质量损失可忽略
    assert np.abs(p_tor - p_fock).max() < 1e-3


def test_lossy_case_matches_fock_density_matrix() -> None:
    """η<1 损耗态：Torontonian 公式 vs 独立密度矩阵计算（截断误差内一致）。"""
    sampler = ExactGBSSampler(
        ExactGBSConfig(
            n_temporal=3,
            n_spatial=1,
            seed=5,
            system_efficiency=0.75,
            squeezing_low=0.2,
            squeezing_high=0.6,
        )
    )
    p_tor = sampler.pattern_probabilities()
    p_fock = _fock_click_probs_lossy(sampler, cutoff=7)
    assert p_fock.sum() > 0.999
    assert np.abs(p_tor - p_fock).max() < 1e-3


def test_reproducible_and_sample_seed_semantics() -> None:
    cfg = ExactGBSConfig(n_temporal=3, n_spatial=2, seed=7, system_efficiency=0.9)
    a = ExactGBSSampler(cfg)
    b1, b2 = a.sample(300), a.sample(300)
    assert np.array_equal(b1.samples, b2.samples)
    assert b1.meta.batch_id == b2.meta.batch_id
    # 同实例、不同采样种子：精确分布一致，样本独立（统计功效检验的基础）
    other = ExactGBSSampler(dataclasses.replace(cfg, sample_seed=99))
    assert np.array_equal(a.pattern_probabilities(), other.pattern_probabilities())
    assert not np.array_equal(a.sample(300).samples, other.sample(300).samples)


def test_samples_match_exact_marginals(exact6: ExactGBSSampler) -> None:
    """大样本边际 click 率与精确边际一致（采样涨落容差内）。"""
    batch = exact6.sample(20000)
    sample_marg = batch.samples.mean(axis=0)
    exact_marg = exact6.exact_marginals()
    # 每模式标准误 ≈ sqrt(p(1-p)/20000) ≈ 0.0035，容差取 ~6σ
    assert np.abs(sample_marg - exact_marg).max() < 0.02


def test_eta_lowers_exact_marginals() -> None:
    base = ExactGBSConfig(n_temporal=3, n_spatial=2, seed=7)
    hi = ExactGBSSampler(dataclasses.replace(base, system_efficiency=0.95))
    lo = ExactGBSSampler(dataclasses.replace(base, system_efficiency=0.3))
    assert hi.exact_marginals().mean() > lo.exact_marginals().mean()


def test_max_modes_enforced() -> None:
    """超出精确枚举上限 → 显式报错（大规模强制走 copula/真机路径）。"""
    with pytest.raises(ValueError, match="max_modes"):
        ExactGBSSampler(ExactGBSConfig(n_temporal=6, n_spatial=2, seed=1))  # 12 > 10
    # 8 模式仍可行
    ok = ExactGBSSampler(ExactGBSConfig(n_temporal=4, n_spatial=2, seed=1))
    assert ok.n_modes == 8
