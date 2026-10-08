"""基选择与统计量测试：论文流程参数语义、手算小样例。"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from qgbs.gbs.backends.base import BatchMeta, SampleBatch
from qgbs.gbs.basis import BasisSelectionParams, select_higher_order_basis
from qgbs.gbs.statistics import (
    batch_dispersion,
    higher_order_stats,
    marginal_stats,
    second_order_stats,
    subbatch_resample,
)


def _meta(n_shots: int, n_modes: int, batch_id: str = "t") -> BatchMeta:
    return BatchMeta(
        batch_id=batch_id,
        device_version="test",
        n_modes=n_modes,
        n_shots=n_shots,
        started_at="2026-01-01T00:00:00",
        finished_at="2026-01-01T00:01:00",
        source="simulation",
        calibration_session_id="test-session",
    )


def test_marginal_stats_hand_computed() -> None:
    samples = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.uint8)
    batch = SampleBatch(meta=_meta(4, 3), samples=samples)
    assert np.allclose(marginal_stats(batch), [0.75, 0.25, 0.5])
    # 标定标准化响应
    cal = np.array([1.0, 0.5, 2.0])
    assert np.allclose(marginal_stats(batch, calibration=cal), [0.75, 0.5, 0.25])


def test_second_order_kinds() -> None:
    samples = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.uint8)
    batch = SampleBatch(meta=_meta(4, 3), samples=samples)
    joint = second_order_stats(batch, "joint")
    assert joint[0, 2] == pytest.approx(1 / 4)  # 模式 0,2 仅在第 0 行共同点击
    cov = second_order_stats(batch, "covariance")
    assert np.allclose(cov, np.cov(samples.astype(float).T))
    corr = second_order_stats(batch, "correlation")
    assert np.allclose(np.diag(corr), 1.0)
    assert corr[0, 1] == pytest.approx(np.corrcoef(samples[:, 0], samples[:, 1])[0, 1])


def test_higher_order_stats_counts() -> None:
    samples = np.array([[1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)
    batch = SampleBatch(meta=_meta(4, 3), samples=samples)
    basis = select_higher_order_basis(
        batch, (0, 1, 2), BasisSelectionParams(n1=4, n2=3, N=5, order=2, seed=0)
    )
    rates = higher_order_stats(batch, basis).toarray().ravel()
    # 基 (0,1) 联合频率 = 3/4
    idx = [i for i, c in enumerate(basis.basis) if tuple(c) == (0, 1)][0]
    assert rates[idx] == pytest.approx(0.75)


def test_basis_selection_recovers_planted_triple() -> None:
    rng = np.random.default_rng(0)
    n_modes, n_shots = 10, 2000
    samples = (rng.random((n_shots, n_modes)) < 0.05).astype(np.uint8)
    planted = (1, 3, 7)
    samples[:1000, list(planted)] = 1  # 植入高频三重组合
    batch = SampleBatch(meta=_meta(n_shots, n_modes), samples=samples)
    params = BasisSelectionParams(n1=500, n2=5, N=10, order=3, seed=0)
    basis = select_higher_order_basis(batch, tuple(range(n_modes)), params)
    assert basis.params == params
    assert basis.dedup_rule  # 去重语义显式记录
    # sample_window 记录采样批次的采集时间范围（取自批次 meta，非金融训练窗口）
    assert basis.sample_window == (batch.meta.started_at, batch.meta.finished_at)
    assert len(basis.basis) <= params.N
    assert any(tuple(row) == planted for row in basis.basis)
    # 基组合无重复
    assert len({tuple(r) for r in basis.basis}) == len(basis.basis)


def test_select_basis_cross_batch_stability_filter() -> None:
    """跨批次稳定性筛选（方案书 §4.2）：剔除相对标准差超阈值的基，保留稳定基。"""
    n_modes, n_shots = 8, 1000
    train = np.zeros((n_shots, n_modes), dtype=np.uint8)
    train[:500, [0, 1]] = 1
    train[:500, [2, 3]] = 1
    batch = SampleBatch(meta=_meta(n_shots, n_modes), samples=train)
    params = BasisSelectionParams(n1=1000, n2=4, N=10, order=2, seed=0, stability_max_rel_std=0.5)
    # 稳定性批次：(0,1) 频率 0.5 vs 0.0（相对标准差 1.0 > 0.5）→ 剔除；
    # (2,3) 两批均 0.5（相对标准差 0）→ 保留
    sa = np.zeros((n_shots, n_modes), dtype=np.uint8)
    sb = np.zeros((n_shots, n_modes), dtype=np.uint8)
    sa[:500, [0, 1]] = 1
    sa[:500, [2, 3]] = 1
    sb[:500, [2, 3]] = 1
    ba = SampleBatch(meta=_meta(n_shots, n_modes, "sa"), samples=sa)
    bb = SampleBatch(meta=_meta(n_shots, n_modes, "sb"), samples=sb)
    basis = select_higher_order_basis(batch, tuple(range(n_modes)), params, stability_batches=[ba, bb])
    kept = {tuple(c) for c in basis.basis}
    assert (2, 3) in kept
    assert (0, 1) not in kept


def test_select_basis_stability_filter_needs_two_batches() -> None:
    """稳定性筛选须 ≥2 个批次：单批次或不提供时不筛选，基原样保留。"""
    n_modes, n_shots = 8, 1000
    train = np.zeros((n_shots, n_modes), dtype=np.uint8)
    train[:500, [0, 1]] = 1
    train[:500, [2, 3]] = 1
    batch = SampleBatch(meta=_meta(n_shots, n_modes), samples=train)
    params = BasisSelectionParams(n1=1000, n2=4, N=10, order=2, seed=0, stability_max_rel_std=0.5)
    other = SampleBatch(meta=_meta(n_shots, n_modes, "o"), samples=np.zeros((n_shots, n_modes), np.uint8))
    for stab in (None, [other]):
        basis = select_higher_order_basis(batch, tuple(range(n_modes)), params, stability_batches=stab)
        kept = {tuple(c) for c in basis.basis}
        assert (0, 1) in kept and (2, 3) in kept


def test_subbatch_resample_and_dispersion() -> None:
    rng = np.random.default_rng(2)
    samples = (rng.random((400, 6)) < 0.3).astype(np.uint8)
    batch = SampleBatch(meta=_meta(400, 6), samples=samples)
    subs = subbatch_resample(batch, 4, seed=0)
    assert len(subs) == 4
    assert sum(s.meta.n_shots for s in subs) == 400  # 无重叠全覆盖
    assert all(s.samples.shape[1] == 6 for s in subs)
    # 同 seed 可复现
    subs2 = subbatch_resample(batch, 4, seed=0)
    assert all(np.array_equal(a.samples, b.samples) for a, b in zip(subs, subs2))
    disp = batch_dispersion(subs)
    assert set(disp.columns) == {"mean", "std", "min", "max"}
    assert (disp["std"] >= 0).all()
    assert bool((disp["max"] >= disp["min"]).all())
