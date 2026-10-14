"""E6 阶段 0：分组分区编码器变体 + pattern 直方图引擎测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qgbs.gbs.backends.base import pack_samples
from qgbs.gbs.encoder import GroupedEncoderParams, GroupedQuantileEncoder
from qgbs.gbs.pattern_service import (
    ColumnPackedStore,
    histogram_from_row_packed,
    top_n_patterns,
)


def _mk_train(n: int = 400, n_feat: int = 26, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, n_feat)), columns=[f"f{i}" for i in range(n_feat)])


def test_grouped_encoder_address_ranges_no_collision() -> None:
    """A 组模式 ∈ [0,4096)、B 组 ∈ [4096,8176)；组内同 bin 不撞模式。"""
    X = _mk_train()
    enc = GroupedQuantileEncoder(GroupedEncoderParams()).fit(X).freeze()
    addr = enc.transform(X.head(50))
    a, b = addr.mode_index_a, addr.mode_index_b
    assert a.shape == (50, 16) and b.shape == (50, 10)
    assert a.min() >= 0 and a.max() < 4096
    assert b.min() >= 4096 and b.max() < 8176
    # 组内互异（同 bin 不同特征 → spatial 不同 → mode 不同）
    for row in a:
        assert len(np.unique(row)) == 16
    for row in b:
        assert len(np.unique(row)) == 10
    # B 组首特征（f16）落入 bin=0 时 → mode = 256*16+0 = 4096（注入极小值触发端点 bin）
    X2 = X.head(50).copy()
    X2.iloc[0, 16] = X["f16"].min() - 10.0
    b2 = enc.transform(X2).mode_index_b
    assert b2[0, 0] == 4096


def test_grouped_encoder_hand_computed_and_frozen() -> None:
    """手算对照：构造已知分位边界的小例；状态机约束。"""
    X = _mk_train(n=100, n_feat=26, seed=3)
    enc = GroupedQuantileEncoder().fit(X)
    with pytest.raises(RuntimeError):
        enc.transform(X)  # 未冻结
    enc = enc.freeze()
    with pytest.raises(RuntimeError):
        enc.fit(X)  # 已冻结
    addr = enc.transform(X)
    # 手算 f0（A 组第 0 特征）：bin = digitize(x, 255 个分位边界)；mode = bin*16+0
    edges = enc._edges["f0"]
    x0 = X["f0"].to_numpy()
    bins0 = np.clip(np.digitize(x0, edges), 0, 255)
    assert np.array_equal(addr.mode_index_a[:, 0], bins0 * 16)
    # f16（B 组第 0 特征）：mode = (256+bin)*16+0
    edges16 = enc._edges["f16"]
    bins16 = np.clip(np.digitize(X["f16"].to_numpy(), edges16), 0, 254)
    assert np.array_equal(addr.mode_index_b[:, 0], (256 + bins16) * 16)


def test_grouped_encoder_rejects_small_feature_count() -> None:
    with pytest.raises(ValueError, match="group_split"):
            GroupedQuantileEncoder().fit(_mk_train(n_feat=16))


def _brute_histogram(samples: np.ndarray, modes: list[int]) -> np.ndarray:
    """暴力对照：逐 shot 取位组装 pattern 索引后计数。"""
    n, k = samples.shape[0], len(modes)
    idx = np.zeros(n, dtype=int)
    for s in range(n):
        for i, m in enumerate(modes):
            idx[s] |= int(samples[s, m]) << i
    return np.bincount(idx, minlength=1 << k) / n


def test_histogram_matches_bruteforce() -> None:
    """直方图引擎（行/列两路径）与暴力逐 shot 循环逐元素一致。"""
    rng = np.random.default_rng(0)
    samples = (rng.random((1000, 32)) < 0.27).astype(np.uint8)
    packed = pack_samples(samples)
    modes = np.array([3, 7, 11, 15, 19, 23])  # 6 模 → 64 维
    want = _brute_histogram(samples, modes.tolist())
    got_row = histogram_from_row_packed(packed, modes)
    assert np.allclose(got_row, want)
    store = ColumnPackedStore.from_row_packed(packed, 32)
    got_col = store.histogram(modes)
    assert np.allclose(got_col, want)
    # 16 模全组（65536 维）抽验
    modes16 = np.arange(16)
    want16 = _brute_histogram(samples, modes16.tolist())
    assert np.allclose(histogram_from_row_packed(packed, modes16), want16)
    assert np.allclose(store.histogram(modes16), want16)


def test_top_n_patterns() -> None:
    hist = np.array([0.5, 0.1, 0.3, 0.05, 0.05])
    top2 = top_n_patterns(hist, 2)
    assert set(top2) == {0, 2}  # 最高频两项
    assert list(top2) == sorted(top2)  # 规范升序
EOF_MARKER = None
