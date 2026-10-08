"""GBSFeatureService 测试：状态机、双通道一致性、特征形状与稀疏性。"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from qgbs.gbs.backends.base import SampleBatch
from qgbs.gbs.backends.simulator import GaussianSimulatorBackend, SimulatorConfig
from qgbs.gbs.basis import (
    DEDUP_RULE,
    BasisSelectionParams,
    CandidateBasis,
    select_higher_order_basis,
)
from qgbs.gbs.encoder import EncoderParams, QuantileEncoder, ensure_default_mappings
from qgbs.gbs.feature_service import GBSFeatureConfig, GBSFeatureService

N_TEMPORAL, N_SPATIAL = 32, 4
N_MODES = N_TEMPORAL * N_SPATIAL


@pytest.fixture(scope="module")
def env() -> dict:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2021-01-01", periods=30)
    symbols = [f"S{i}" for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    X = pd.DataFrame(rng.normal(0, 1, (len(idx), 3)), index=idx, columns=["a", "b", "c"])
    ensure_default_mappings(3, N_SPATIAL, N_TEMPORAL)
    from qgbs.gbs.encoder import resolve_mapping_id

    enc = QuantileEncoder(
        EncoderParams(n_bins_a=16, n_bins_b=16, partition_boundary=16),
        mapping_id=resolve_mapping_id("identity", 3, N_SPATIAL, N_TEMPORAL),
    ).fit(X)
    enc = enc.freeze()
    addresses = enc.transform(X)
    backend = GaussianSimulatorBackend(SimulatorConfig(n_temporal=N_TEMPORAL, n_spatial=N_SPATIAL, seed=1))
    batch = backend.sample(3000)
    basis = select_higher_order_basis(
        batch, tuple(range(N_MODES)), BasisSelectionParams(n1=200, n2=6, N=20, order=3, seed=0)
    )
    return {"X": X, "idx": idx, "addresses": addresses, "batch": batch, "basis": basis, "backend": backend}


def _make_service(env: dict, kinds: tuple[str, ...]) -> GBSFeatureService:
    svc = GBSFeatureService(GBSFeatureConfig(stat_kinds=kinds))
    svc.fit(env["addresses"], [env["batch"]], basis=env["basis"] if "higher_order" in kinds else None)
    return svc


def test_freeze_state_machine(env: dict) -> None:
    svc = _make_service(env, ("marginal",))
    with pytest.raises(RuntimeError):
        svc.build_features(env["addresses"], env["idx"])  # 未冻结
    frozen = svc.freeze()
    assert frozen.is_frozen
    with pytest.raises(RuntimeError):
        frozen.fit(env["addresses"], [env["batch"]])  # 冻结后不可 fit


def test_feature_shapes_and_names(env: dict) -> None:
    svc = _make_service(env, ("marginal", "second_order", "higher_order")).freeze()
    out = svc.build_features(env["addresses"], env["idx"])
    n_expected = 3 + 3 + len(env["basis"].basis)  # 3 边际 + C(3,2)=3 二阶 + N 高阶
    assert out.shape == (len(env["idx"]), n_expected)
    assert out.columns[0] == "gbs_marg_f0"
    assert any(c.startswith("gbs_pair_") for c in out.columns)
    assert any(c.startswith("gbs_ho_") for c in out.columns)
    assert np.isfinite(out.to_numpy()).all()


def test_marginal_feature_matches_address(env: dict) -> None:
    """边际特征值应等于寻址模式的边际 click 率。"""
    svc = _make_service(env, ("marginal",)).freeze()
    out = svc.build_features(env["addresses"], env["idx"])
    marg = env["batch"].samples.mean(axis=0)
    addr = env["addresses"].mode_index
    assert np.allclose(out["gbs_marg_f0"].to_numpy(), marg[addr[:, 0]])
    assert np.allclose(out["gbs_marg_f2"].to_numpy(), marg[addr[:, 2]])


def test_dual_channel_cache_consistency(env: dict) -> None:
    svc = _make_service(env, ("marginal", "second_order"))
    svc.precompute_cache(env["addresses"])
    frozen = svc.freeze()
    out1 = frozen.build_features(env["addresses"], env["idx"])
    out2 = frozen.build_features(env["addresses"], env["idx"])  # 缓存命中路径
    pd.testing.assert_frame_equal(out1, out2)
    assert len(frozen._cache) > 0


def test_higher_order_sparsity(env: dict) -> None:
    svc = _make_service(env, ("higher_order",)).freeze()
    out = svc.build_features(env["addresses"], env["idx"])
    ho = out[[c for c in out.columns if c.startswith("gbs_ho_")]]
    assert (ho.to_numpy() >= 0).all()
    # 软激活：值 = 基频率 × 重合数/阶数，不应超过基频率本身
    assert (ho.to_numpy() <= svc._basis_rates.max() + 1e-12).all()
    assert (ho.to_numpy() == 0).mean() > 0.3  # 仍应相当稀疏


def test_fit_multi_batch_enables_stability_filter(env: dict) -> None:
    """多训练批次 fit 启用跨批次稳定性筛选（方案书 §4.2）：不稳定基被剔除。"""
    n_shots = 1000
    s1 = np.zeros((n_shots, N_MODES), dtype=np.uint8)
    s2 = np.zeros((n_shots, N_MODES), dtype=np.uint8)
    # 稳定基 (2,3,4)：两批均以 50% 频率出现（相对标准差 0）
    s1[:500, [2, 3, 4]] = 1
    s2[:500, [2, 3, 4]] = 1
    # 不稳定基 (0,1,5)：批1 频率 50%、批2 为 0（相对标准差 1.0 > 阈值 0.5）
    s1[:500, [0, 1, 5]] = 1
    meta = env["batch"].meta
    b1 = SampleBatch(meta=dataclasses.replace(meta, n_shots=n_shots), samples=s1)
    b2 = SampleBatch(meta=dataclasses.replace(meta, n_shots=n_shots), samples=s2)
    basis = CandidateBasis(
        basis=np.array([[2, 3, 4], [0, 1, 5]]),
        sample_window=(meta.started_at, meta.finished_at),
        params=BasisSelectionParams(order=3, stability_max_rel_std=0.5),
        mode_group=tuple(range(N_MODES)),
        dedup_rule=DEDUP_RULE,
    )
    svc = GBSFeatureService(GBSFeatureConfig(stat_kinds=("higher_order",)))
    svc.fit(env["addresses"], [b1, b2], basis=basis)
    assert {tuple(c) for c in svc._basis.basis} == {(2, 3, 4)}
    # 单批次 fit 不启用筛选：基原样保留
    svc_single = GBSFeatureService(GBSFeatureConfig(stat_kinds=("higher_order",)))
    svc_single.fit(env["addresses"], [b1], basis=basis)
    assert len(svc_single._basis.basis) == 2
