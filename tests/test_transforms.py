"""C4/C5 特征变换测试：维度、冻结语义、确定性。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qgbs.gbs.compression import (
    NystromConfig,
    NystromFeatures,
    QuantileExplicitConfig,
    QuantileExplicitFeatures,
    RFFConfig,
    RFFeatures,
)

N_TR, N_TE, N_FEAT = 150, 50, 5


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(5)
    idx_tr = pd.RangeIndex(N_TR, name="row")
    idx_te = pd.RangeIndex(N_TR, N_TR + N_TE, name="row")
    cols = [f"f{k}" for k in range(N_FEAT)]
    return (
        pd.DataFrame(rng.normal(0, 1, (N_TR, N_FEAT)), index=idx_tr, columns=cols),
        pd.DataFrame(rng.normal(0, 1, (N_TE, N_FEAT)), index=idx_te, columns=cols),
    )


def test_quantile_explicit_dim_and_freeze(data) -> None:
    X_tr, X_te = data
    cfg = QuantileExplicitConfig(n_bins=4, max_interactions=3)
    qe = QuantileExplicitFeatures(cfg)
    with pytest.raises(RuntimeError):
        qe.transform(X_te)  # 未 fit 不可 transform
    qe.fit(X_tr)
    out = qe.transform(X_te)
    assert out.shape[0] == N_TE
    assert out.shape[1] == N_FEAT * 4 + 3  # one-hot + 交互
    assert qe.is_frozen
    with pytest.raises(RuntimeError):
        qe.fit(X_tr)  # 冻结后不可重复 fit


def test_quantile_explicit_clips_out_of_range(data) -> None:
    X_tr, _ = data
    qe = QuantileExplicitFeatures(QuantileExplicitConfig(n_bins=4, max_interactions=0)).fit(X_tr)
    extreme = pd.DataFrame(
        {c: [1e6, -1e6] for c in X_tr.columns}, index=pd.Index([0, 1], name="row")
    )
    out = qe.transform(extreme)
    # 越界值截断到端点 bin：max → 最后一个 bin，min → 第一个 bin
    for c in X_tr.columns:
        assert out.loc[0, f"{c}__bin3"] == 1.0
        assert out.loc[1, f"{c}__bin0"] == 1.0


def test_rff_shape_and_determinism(data) -> None:
    X_tr, X_te = data
    cfg = RFFConfig(target_dim=16, gamma=1.0, seed=7)
    rf1 = RFFeatures(cfg).fit(X_tr)
    rf2 = RFFeatures(cfg).fit(X_tr)
    z1, z2 = rf1.transform(X_te), rf2.transform(X_te)
    assert z1.shape == (N_TE, 16)
    assert np.allclose(z1, z2)  # 同 seed 可复现
    assert np.abs(z1).max() <= np.sqrt(2.0 / 16) + 1e-12  # cos 有界
    with pytest.raises(RuntimeError):
        RFFeatures(cfg).transform(X_te)  # 未 fit


def test_nystrom_shape_and_determinism(data) -> None:
    X_tr, X_te = data
    cfg = NystromConfig(target_dim=8, n_landmarks=20, seed=7)
    ny1 = NystromFeatures(cfg).fit(X_tr)
    ny2 = NystromFeatures(cfg).fit(X_tr)
    z1, z2 = ny1.transform(X_te), ny2.transform(X_te)
    assert z1.shape == (N_TE, 8)
    assert np.allclose(z1, z2)
    assert np.isfinite(z1).all()
    with pytest.raises(RuntimeError):
        NystromFeatures(cfg).transform(X_te)


def test_nystrom_insufficient_train_raises(data) -> None:
    X_tr, _ = data
    with pytest.raises(ValueError):
        NystromFeatures(NystromConfig(target_dim=8, n_landmarks=1000)).fit(X_tr)
