"""诊断一（真机 vs 仿真）单测：统计功效、单批次适配、接口校验。"""

from __future__ import annotations

import numpy as np
import pytest

from qgbs.experiments.diagnostics import hardware_vs_simulation
from qgbs.gbs.backends.base import BatchMeta, SampleBatch


def _batch(samples: np.ndarray, batch_id: str, source: str, n_t: int = 4, n_s: int = 2) -> SampleBatch:
    return SampleBatch(
        meta=BatchMeta(
            batch_id=batch_id, device_version="t", n_modes=samples.shape[1],
            n_shots=samples.shape[0], started_at="t", finished_at="t",
            source=source, calibration_session_id="t",
            mode_config={"n_temporal": n_t, "n_spatial": n_s},
        ),
        samples=samples,
    )


def test_identical_distribution_within_noise() -> None:
    """同分布两源：边际/二阶差异应与噪声底同量级（ratio ~1）。"""
    rng = np.random.default_rng(0)
    p = np.linspace(0.1, 0.4, 8)
    a = (rng.random((20000, 8)) < p).astype(np.uint8)
    b = (rng.random((20000, 8)) < p).astype(np.uint8)
    r = hardware_vs_simulation([_batch(a, "hw", "hardware")], [_batch(b, "sim", "simulation")])
    row = r.table.loc["marginal_l1"]
    assert 0.3 < row["ratio"] < 3.0
    assert r.table.loc["second_order_corr_frobenius", "ratio"] < 3.0
    assert r.details["mean_click_hw"] == pytest.approx(float(a.mean()), rel=1e-3)
    assert "temporal_corr" in r.details  # 子系统剖面齐全


def test_shifted_distribution_detected() -> None:
    """边际整体抬升 + 植入强相关的仿真源：差异显著超噪声底。"""
    rng = np.random.default_rng(1)
    n = 20000
    hw = (rng.random((n, 8)) < 0.25).astype(np.uint8)
    # 仿真源：边际抬升 0.1，且 0/1 列完全共变（强联合结构差异）
    sim = (rng.random((n, 8)) < 0.35).astype(np.uint8)
    common = (rng.random(n) < 0.5).astype(np.uint8)
    sim[:, 0] = sim[:, 1] = common
    r = hardware_vs_simulation([_batch(hw, "hw", "hardware")], [_batch(sim, "sim", "simulation")])
    assert r.table.loc["marginal_l1", "ratio"] > 5
    assert r.table.loc["second_order_corr_frobenius", "ratio"] > 5
    assert "装置级差异" in r.verdict
    assert "η 未标定" in r.verdict  # 单批次现实声明


def test_multi_batch_merge_and_validation() -> None:
    """同来源多批次纵向合并；空列表与模式数不一致显式报错。"""
    rng = np.random.default_rng(2)
    a1 = (rng.random((3000, 8)) < 0.2).astype(np.uint8)
    a2 = (rng.random((2000, 8)) < 0.2).astype(np.uint8)
    b = (rng.random((5000, 8)) < 0.2).astype(np.uint8)
    r = hardware_vs_simulation(
        [_batch(a1, "h1", "hardware"), _batch(a2, "h2", "hardware")],
        [_batch(b, "s", "simulation")],
    )
    assert r.details["n_shots_hw"] == 5000 and r.details["n_shots_sim"] == 5000
    with pytest.raises(ValueError, match="不能为空"):
        hardware_vs_simulation([], [_batch(b, "s", "simulation")])
    with pytest.raises(ValueError, match="模式数不一致"):
        hardware_vs_simulation(
            [_batch(a1, "h1", "hardware")],
            [_batch((rng.random((100, 6)) < 0.2).astype(np.uint8), "s", "simulation", 3, 2)],
        )
