"""分布级对照（distribution_compare）测试。

(a) exact vs exact（同实例、不同采样种子）：验证对照工具的统计功效——
    差异应落在采样噪声底的同量级；
(b) exact vs copula（平均边际已对齐）：记录结构性差异量级，作为 P3 真机
    接入前的基线认知（数值见 qgbs.gbs.distribution_compare 模块 docstring）。
"""

from __future__ import annotations

import numpy as np
import pytest

from qgbs.gbs.backends.exact_gbs import ExactGBSConfig, ExactGBSSampler
from qgbs.gbs.backends.simulator import GaussianSimulatorBackend, SimulatorConfig
from qgbs.gbs.distribution_compare import CompareReport, compare_backends, match_marginals

N_SHOTS = 20000


def _exact(seed: int = 7, sample_seed: int | None = None) -> ExactGBSSampler:
    return ExactGBSSampler(
        ExactGBSConfig(
            n_temporal=3,
            n_spatial=2,
            seed=seed,
            sample_seed=sample_seed,
            system_efficiency=0.9,
            squeezing_low=0.3,
            squeezing_high=0.9,
        )
    )


def _copula_matched(target_rates: np.ndarray) -> GaussianSimulatorBackend:
    # n_temporal=2 时时间包络平坦，残余边际失配仅来自空间权重（包络约束，
    # 逐模式无法完全对齐——该约束本身是基线认知的一部分）
    cfg = match_marginals(
        SimulatorConfig(n_temporal=2, n_spatial=3, seed=3, system_efficiency=0.9),
        target_rates,
    )
    return GaussianSimulatorBackend(cfg)


def test_identical_backend_zero_difference() -> None:
    """同一后端同一参数（确定性样本）对照 → 所有差异严格为 0。"""
    exact = _exact()
    report = compare_backends(exact, exact, N_SHOTS)
    assert float(report.table["value"].abs().max()) == 0.0
    assert isinstance(report.interpretation, str) and report.interpretation
    assert report.marginals_a.shape == (6,)


def test_exact_vs_exact_within_noise() -> None:
    """同实例、独立采样：差异应在噪声底同量级（比值 ≲ 3）。"""
    a, b = _exact(sample_seed=101), _exact(sample_seed=102)
    report = compare_backends(a, b, N_SHOTS, seed=0)
    table = report.table
    for metric, row in table.iterrows():
        assert row["value"] <= 3.0 * row["noise_floor"], (
            f"{metric}: value={row['value']:.4g} 超出 3×噪声底 {row['noise_floor']:.4g}"
        )


def test_match_marginals_achieves_target_mean() -> None:
    exact = _exact()
    target = exact.exact_marginals()
    matched = _copula_matched(target)
    achieved = matched.sample(N_SHOTS).samples.mean()
    assert abs(achieved - target.mean()) < 2e-3


def test_match_marginals_validates_shape() -> None:
    with pytest.raises(ValueError, match="target_rates"):
        match_marginals(SimulatorConfig(n_temporal=3, n_spatial=2), np.zeros(5))


def test_exact_vs_copula_structural_difference() -> None:
    """边际均值对齐后，copula 与精确 GBS 的联合分布存在显著结构差异。"""
    exact = _exact()
    copula = _copula_matched(exact.exact_marginals())
    report = compare_backends(exact, copula, N_SHOTS, seed=0)
    assert isinstance(report, CompareReport)
    table = report.table
    # 平均边际已对齐（match_marginals 收敛），但逐模式形状受包络约束：
    # 残余边际失配有界且为结构性（超出噪声）
    assert abs(report.marginals_a.mean() - report.marginals_b.mean()) < 2e-3
    assert table.loc["marginal_mean_abs", "value"] < 0.1
    assert table.loc["marginal_mean_abs", "ratio"] > 3.0
    # 二阶联合结构（含对边际失配免疫的相关系数指标）与
    # 子系统联合分布：显著超出噪声底（实测比值 ≈ 15）
    assert table.loc["second_order_frobenius_offdiag", "ratio"] > 5.0
    assert table.loc["second_order_corr_frobenius_offdiag", "ratio"] > 5.0
    assert table.loc["tv_k3_mean", "ratio"] > 5.0
    assert table.loc["tv_k4_mean", "ratio"] > 5.0
    # 标签与元数据
    assert report.labels == ("simulator-exact-gbs", "simulator-gaussian-copula")
    assert set(report.subsystem_tv) == {3, 4}


def test_n_modes_mismatch_rejected() -> None:
    exact = _exact()
    other = GaussianSimulatorBackend(SimulatorConfig(n_temporal=4, n_spatial=2, seed=3))
    with pytest.raises(ValueError, match="n_modes"):
        compare_backends(exact, other, 100)
