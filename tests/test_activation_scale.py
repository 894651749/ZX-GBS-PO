"""修正 4：真机规模（511×9 模式、256/255 bins）下软硬激活对比。

用 simulator 的 independent 快速通道（无协方差分解，适合大规模诊断），
样本量 20k（≤ 50 万预算内）。

可直接运行查看数字：``PYTHONPATH=src .venv/bin/python -m tests.test_activation_scale``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qgbs.gbs.backends.simulator import GaussianSimulatorBackend, SimulatorConfig
from qgbs.gbs.basis import BasisSelectionParams, select_higher_order_basis
from qgbs.gbs.encoder import (
    EncoderParams,
    QuantileEncoder,
    ensure_default_mappings,
    resolve_mapping_id,
)
from qgbs.gbs.feature_service import GBSFeatureConfig, GBSFeatureService

# 我方真机规模：511 时间模式 × 9 空间模式（3×3），bin 数 256（A 区）/255（B 区），分区边界 256
N_TEMPORAL, N_SPATIAL = 511, 9
N_SHOTS = 20_000  # 样本量（≤ 50 万预算，控制耗时/内存 ~92MB）
N_FEATURES = 11  # 与 finance_base 特征数一致


def _run_compare() -> dict[str, float]:
    backend = GaussianSimulatorBackend(
        SimulatorConfig(
            n_temporal=N_TEMPORAL, n_spatial=N_SPATIAL, seed=7,
            system_efficiency=0.7, method="independent",
        )
    )
    batch = backend.sample(N_SHOTS)
    basis = select_higher_order_basis(
        batch, tuple(range(backend.n_modes)),
        BasisSelectionParams(n1=1000, n2=100, N=300, order=3, seed=7, max_combos_per_shot=2000),
    )
    # 合成金融特征面板（20 期 × 40 资产 × 11 特征）
    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2021-01-01", periods=20), [f"S{i}" for i in range(40)]],
        names=["date", "symbol"],
    )
    X = pd.DataFrame(rng.normal(0, 1, (len(idx), N_FEATURES)), index=idx,
                     columns=[f"f{k}" for k in range(N_FEATURES)])
    ensure_default_mappings(N_FEATURES, N_SPATIAL, N_TEMPORAL)
    # 注册表全局共享：经 resolve_mapping_id 解析，避免与其他测试文件的维度撞车
    mapping_id = resolve_mapping_id("identity", N_FEATURES, N_SPATIAL, N_TEMPORAL)
    enc = QuantileEncoder(
        EncoderParams(n_bins_a=256, n_bins_b=255, partition_boundary=256), mapping_id=mapping_id
    )
    enc.fit(X)
    addresses = enc.freeze().transform(X)

    rates = {}
    for activation in ("soft", "hard"):
        svc = GBSFeatureService(
            GBSFeatureConfig(stat_kinds=("higher_order",), activation=activation),
        ).fit(addresses, [batch], basis=basis)
        svc = svc.freeze()
        out = svc.build_features(addresses, idx)
        rates[activation] = float((out.to_numpy() > 0).mean())
    rates["n_basis"] = float(len(basis.basis))
    return rates


@pytest.fixture(scope="module")
def rates() -> dict[str, float]:
    return _run_compare()


def test_hard_is_strictly_sparser(rates: dict[str, float]) -> None:
    assert 0.0 <= rates["hard"] < rates["soft"] <= 1.0
    assert rates["n_basis"] > 0


def test_soft_activation_nonzero_at_paper_scale(rates: dict[str, float]) -> None:
    # 真机规模下软激活仍应有一定非零率；硬激活应近乎全零
    assert rates["soft"] > 0.001
    assert rates["hard"] < 0.05


if __name__ == "__main__":
    r = _run_compare()
    print("=== 真机规模（511×9 模式、256/255 bins、20k shots）软硬激活对比 ===")
    print(f"候选基数量        : {int(r['n_basis'])}")
    print(f"软激活非零率      : {r['soft']:.4%}")
    print(f"硬激活非零率      : {r['hard']:.6%}")
    if r["hard"] > 0:
        print(f"非零率比值 soft/hard: {r['soft'] / r['hard']:.1f}x")
    else:
        print("非零率比值 soft/hard: ∞（硬激活全零——demo 特征数下严格子集命中概率为零）")
