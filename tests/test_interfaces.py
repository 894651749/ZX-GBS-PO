"""接口骨架测试：mock 适配器 / 仿真后端 / 置换后端 / 编码器的 Protocol 结构检查。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qgbs.data.adapters.mock import MockAdapter
from qgbs.gbs.backends.base import BatchMeta, SampleBatch
from qgbs.gbs.backends.permuted import PermutedBackend
from qgbs.gbs.backends.simulator import GaussianSimulatorBackend, SimulatorConfig
from qgbs.gbs.encoder import EncoderParams, QuantileEncoder, ensure_default_mappings
from qgbs.gbs.statistics import second_order_stats

ADAPTER_METHODS = (
    "load_asset_pool_history",
    "load_daily_market",
    "load_trading_status",
    "load_corporate_actions",
    "load_market_sector_factor",
    "load_constraints_costs",
    "load_baseline_results",
)


def test_mock_adapter_satisfies_data_adapter() -> None:
    adapter = MockAdapter()
    for method in ADAPTER_METHODS:
        assert callable(getattr(adapter, method))


def test_simulator_backend_satisfies_gbs_sampler() -> None:
    backend = GaussianSimulatorBackend(SimulatorConfig(n_temporal=8, n_spatial=2, seed=0))
    assert backend.n_modes == 16
    batch = backend.sample(n_shots=100)
    assert batch.samples.shape == (100, 16)
    assert set(np.unique(batch.samples)) <= {0, 1}
    assert batch.meta.source == "simulation"
    assert batch.meta.system_efficiency == 0.7
    assert batch.meta.calibration_session_id


def test_permuted_backend_wraps_sampler() -> None:
    base = GaussianSimulatorBackend(SimulatorConfig(n_temporal=8, n_spatial=2, seed=0))
    backend = PermutedBackend(base, seed=0)
    assert backend.n_modes == base.n_modes
    raw = base.sample(n_shots=500)
    perm = backend.permute(raw)
    # 逐列置换精确保持边际统计
    assert np.allclose(raw.samples.mean(axis=0), perm.samples.mean(axis=0))
    assert perm.samples.shape == raw.samples.shape
    # 同 seed 确定性可复现
    assert np.array_equal(backend.permute(raw).samples, perm.samples)


def test_permuted_backend_breaks_joint_structure() -> None:
    """置换破坏联合结构：植入完全相关的两列，置换后二阶联合显著削弱。"""
    n = 5000
    base_col = (np.arange(n) % 2).astype(np.uint8)  # 边际 0.5 的确定列
    samples = np.tile(base_col[:, None], (1, 8))
    meta = BatchMeta(
        batch_id="t", device_version="t", n_modes=8, n_shots=n,
        started_at="t", finished_at="t", source="simulation", calibration_session_id="t",
    )
    batch = SampleBatch(meta=meta, samples=samples)
    assert second_order_stats(batch, "joint")[0, 1] == pytest.approx(0.5)  # 完全共同点击
    perm = PermutedBackend(GaussianSimulatorBackend(SimulatorConfig(n_temporal=4, n_spatial=2, seed=0)), seed=1).permute(batch)
    joint_p = second_order_stats(perm, "joint")
    assert joint_p[0, 1] == pytest.approx(0.25, abs=0.02)  # 独立化 → p0·p1
    assert np.allclose(perm.samples.mean(axis=0), 0.5)  # 边际不变


def test_quantile_encoder_freeze_semantics() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(0, 1, (200, 3)), columns=["a", "b", "c"])
    ensure_default_mappings(3, n_spatial=4, n_temporal=32)
    from qgbs.gbs.encoder import resolve_mapping_id

    encoder = QuantileEncoder(
        EncoderParams(n_bins_a=16, n_bins_b=16, partition_boundary=16),
        mapping_id=resolve_mapping_id("identity", 3, 4, 32),
    )
    with pytest.raises(RuntimeError):
        encoder.transform(X)  # 未冻结不可 transform
    encoder.fit(X)
    frozen = encoder.freeze()
    assert frozen.is_frozen and not encoder.is_frozen
    addr = frozen.transform(X)
    assert addr.mode_index.shape == (200, 3)
    with pytest.raises(RuntimeError):
        frozen.fit(X)  # 冻结后不可 fit
