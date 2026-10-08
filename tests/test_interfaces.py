"""接口骨架测试：mock 适配器 / 仿真后端 / 置换后端 / 编码器的 Protocol 结构检查。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qgbs.data.adapters.mock import MockAdapter
from qgbs.gbs.backends.permuted import PermutedBackend
from qgbs.gbs.backends.simulator import GaussianSimulatorBackend, SimulatorConfig
from qgbs.gbs.encoder import EncoderParams, QuantileEncoder, ensure_default_mappings

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
