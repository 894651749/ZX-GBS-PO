"""仿真器 / 样本库 / cached 后端测试。"""

from __future__ import annotations

import numpy as np
import pytest

from qgbs.gbs.backends.base import pack_samples, unpack_samples
from qgbs.gbs.backends.cached import CachedBackend
from qgbs.gbs.backends.simulator import GaussianSimulatorBackend, SimulatorConfig
from qgbs.gbs.sample_store import SampleStore


@pytest.fixture(scope="module")
def sim() -> GaussianSimulatorBackend:
    return GaussianSimulatorBackend(
        SimulatorConfig(n_temporal=8, n_spatial=4, seed=3, system_efficiency=0.7)
    )


def test_simulator_shape_values_and_meta(sim: GaussianSimulatorBackend) -> None:
    batch = sim.sample(500)
    assert batch.samples.shape == (500, 32)
    assert set(np.unique(batch.samples)) <= {0, 1}
    m = batch.meta
    assert m.source == "simulation"
    assert m.system_efficiency == 0.7
    assert m.simulation_params["method"] == "gaussian_copula_threshold"
    assert "derived_partition_boundary" in m.mode_config
    assert m.mode_config["n_temporal"] == 8 and m.mode_config["n_spatial"] == 4
    assert m.calibration_session_id == "sim-session-3"


def test_simulator_reproducible(sim: GaussianSimulatorBackend) -> None:
    b1, b2 = sim.sample(300), sim.sample(300)
    assert np.array_equal(b1.samples, b2.samples)
    assert b1.meta.batch_id == b2.meta.batch_id


def test_eta_lowers_click_rate() -> None:
    hi = GaussianSimulatorBackend(SimulatorConfig(n_temporal=8, n_spatial=4, seed=3, system_efficiency=0.9))
    lo = GaussianSimulatorBackend(SimulatorConfig(n_temporal=8, n_spatial=4, seed=3, system_efficiency=0.2))
    assert hi.sample(2000).samples.mean() > lo.sample(2000).samples.mean()


def test_packbits_roundtrip(sim: GaussianSimulatorBackend) -> None:
    batch = sim.sample(100)
    packed = pack_samples(batch.samples)
    assert packed.shape == (100, 4)  # 32 modes / 8
    assert np.array_equal(unpack_samples(packed, 32), batch.samples)


def test_sample_store_versioning() -> None:
    store = SampleStore()  # 纯内存
    sim_a = GaussianSimulatorBackend(SimulatorConfig(n_temporal=4, n_spatial=2, seed=1))
    b1 = sim_a.sample(50)
    v1 = store.register_batch(b1)
    assert (v1.namespace, v1.major, v1.minor) == ("simulation", 1, 1)
    # 同会话追加 → minor 递增
    import dataclasses

    b2 = dataclasses.replace(b1, meta=dataclasses.replace(b1.meta, batch_id="b2", n_shots=50))
    v2 = store.register_batch(b2)
    assert (v2.major, v2.minor) == (1, 2)
    # 新会话 → major 递增
    sim_b = GaussianSimulatorBackend(SimulatorConfig(n_temporal=4, n_spatial=2, seed=2))
    v3 = store.register_batch(sim_b.sample(50))
    assert (v3.major, v3.minor) == (2, 1)
    # 重复登记拒绝
    with pytest.raises(ValueError):
        store.register_batch(b1)


def test_sample_store_disk_roundtrip(tmp_path) -> None:
    store = SampleStore(tmp_path)
    sim = GaussianSimulatorBackend(SimulatorConfig(n_temporal=4, n_spatial=2, seed=5))
    batch = sim.sample(64)
    store.register_batch(batch)
    store.set_quality_flag(batch.meta.batch_id, "verified", True)
    # 重新扫描恢复
    store2 = SampleStore(tmp_path)
    loaded = store2.load_batch(batch.meta.batch_id)
    assert np.array_equal(loaded.samples, batch.samples)
    assert loaded.meta.quality_flags["verified"] is True
    listing = store2.list_batches(namespace="simulation")
    assert batch.meta.batch_id in set(listing["batch_id"])


def test_cached_backend_readonly(sim: GaussianSimulatorBackend) -> None:
    store = SampleStore()
    batch = sim.sample(200)
    store.register_batch(batch)
    cached = CachedBackend(store, batch.meta.batch_id)
    assert cached.n_modes == 32
    sub = cached.sample(50)
    assert sub.samples.shape == (50, 32)
    assert np.array_equal(sub.samples, batch.samples[:50])  # 确定性子集
    with pytest.raises(ValueError, match="样本不足"):
        cached.sample(10_000)


def _random_correlation(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    cov = a @ a.T
    return cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))


def test_set_state_before_first_sample_not_overwritten() -> None:
    """首次 sample 前调用 set_state：自定义相关矩阵不被实例构建静默覆盖。"""
    cfg = SimulatorConfig(n_temporal=4, n_spatial=2, seed=3)
    cov = _random_correlation(8)
    early = GaussianSimulatorBackend(cfg)
    early.set_state(cov)  # 首次 sample 之前
    late = GaussianSimulatorBackend(cfg)
    late.sample(10)  # 先触发实例构建
    late.set_state(cov)
    assert np.array_equal(early.sample(500).samples, late.sample(500).samples)
    # 自定义态确实生效（与默认固定实例的采样不同）
    default_samples = GaussianSimulatorBackend(cfg).sample(500).samples
    assert not np.array_equal(early.sample(500).samples, default_samples)


def test_set_state_validates_method_and_shape() -> None:
    sim = GaussianSimulatorBackend(SimulatorConfig(n_temporal=4, n_spatial=2, seed=3))
    with pytest.raises(ValueError, match="n_modes"):
        sim.set_state(np.eye(4))  # 形状与 n_modes=8 不符
    sim_ind = GaussianSimulatorBackend(
        SimulatorConfig(n_temporal=4, n_spatial=2, seed=3, method="independent")
    )
    with pytest.raises(ValueError, match="copula"):
        sim_ind.set_state(np.eye(8))  # independent 通道无相关结构可设置


def test_sample_store_mode_structure_isolation(tmp_path) -> None:
    """模式结构切换（16→9 空间模式）后：旧批次保留在库中但不被新配置命中。"""
    from qgbs.experiments.runner import _prepare_gbs_backend

    # 存量：16 空间模式批次入库
    old_sim = GaussianSimulatorBackend(SimulatorConfig(n_temporal=32, n_spatial=16, seed=42))
    old_batch = old_sim.sample(2000)
    from qgbs.gbs.sample_store import SampleStore

    store = SampleStore(tmp_path)
    store.register_batch(old_batch)

    # 9 空间模式配置：必须新采样入库，而非复用 16 模式批次
    cfg = {
        "gbs": {
            "sample_store": str(tmp_path),
            "n_temporal": 32,
            "n_spatial": 9,
            "n_shots": 2000,
            "system_efficiency": 0.7,
            "seed": 42,
        }
    }
    backend, batch = _prepare_gbs_backend(cfg)
    assert batch.meta.mode_config["n_spatial"] == 9
    assert batch.meta.n_modes == 32 * 9
    assert backend.n_modes == 32 * 9
    # 旧批次仍在库中、未被命中（重新扫描磁盘索引）
    listing = SampleStore(tmp_path).list_batches(namespace="simulation")
    ids = set(listing["batch_id"])
    assert old_batch.meta.batch_id in ids
    assert batch.meta.batch_id in ids
    assert len(ids) == 2
