"""编码器测试：二维寻址正确性、多映射、防泄漏状态机、越界截断。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qgbs.gbs.encoder import (
    EncoderParams,
    QuantileEncoder,
    ensure_default_mappings,
    get_mapping,
    resolve_mapping_id,
)

N_SPATIAL, N_TEMPORAL = 4, 32
PARAMS = EncoderParams(n_bins_a=16, n_bins_b=16, partition_boundary=16)


def _identity_id(n_feat: int = 3) -> str:
    """解析与当前维度兼容的 identity 映射（注册表全局共享，须防维度撞车）。"""
    ensure_default_mappings(n_feat, N_SPATIAL, N_TEMPORAL)
    return resolve_mapping_id("identity", n_feat, N_SPATIAL, N_TEMPORAL)


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(rng.normal(0, 1, (300, 3)), columns=["a", "b", "c"])


@pytest.fixture(scope="module")
def frozen_encoder(data: pd.DataFrame) -> QuantileEncoder:
    enc = QuantileEncoder(PARAMS, mapping_id=_identity_id())
    enc.fit(data)
    return enc.freeze()


def test_two_dimensional_addressing(frozen_encoder: QuantileEncoder, data: pd.DataFrame) -> None:
    addr = frozen_encoder.transform(data)
    assert addr.mode_index.shape == (300, 3)
    # mode_index = temporal × n_spatial + spatial
    assert (addr.mode_index == addr.temporal_index * N_SPATIAL + addr.spatial_index).all()
    # identity 映射：空间模式 = 特征索引（n_features ≤ n_spatial）
    assert (addr.spatial_index == addr.feature_index).all()
    # 分区：bin < n_bins_a → A，否则 B
    assert ((addr.bin_index < PARAMS.n_bins_a) == (addr.partition == "A")).all()
    # A 区时间模式 = bin；B 区 = boundary + (bin − n_bins_a)
    a_mask = addr.partition == "A"
    assert (addr.temporal_index[a_mask] == addr.bin_index[a_mask]).all()
    assert (
        addr.temporal_index[~a_mask]
        == PARAMS.partition_boundary + (addr.bin_index[~a_mask] - PARAMS.n_bins_a)
    ).all()


def test_quantile_bins_uniform_on_train(frozen_encoder: QuantileEncoder, data: pd.DataFrame) -> None:
    addr = frozen_encoder.transform(data)
    # 分位数编码在训练分布上近似均匀
    counts = np.bincount(addr.bin_index[:, 0], minlength=32)
    assert counts.min() > 0
    assert counts.max() / counts.min() < 2.0


def test_out_of_range_clipped_to_edge_bins(frozen_encoder: QuantileEncoder) -> None:
    extreme = pd.DataFrame({"a": [1e9, -1e9], "b": [0.0, 0.0], "c": [0.0, 0.0]})
    addr = frozen_encoder.transform(extreme)
    assert addr.bin_index[0, 0] == 31  # 最大值 → 末 bin
    assert addr.bin_index[1, 0] == 0  # 最小值 → 首 bin


def test_mapping_locked_and_selectable(data: pd.DataFrame) -> None:
    ids = ensure_default_mappings(3, N_SPATIAL, N_TEMPORAL)
    assert len(ids) >= 2
    enc_a = QuantileEncoder(PARAMS, mapping_id=_identity_id()).fit(data).freeze()
    enc_b = QuantileEncoder(PARAMS, mapping_id=ids[1]).fit(data, mapping_id=ids[1]).freeze()
    addr_a, addr_b = enc_a.transform(data), enc_b.transform(data)
    # 不同映射（置换）给出不同寻址
    assert not (addr_a.mode_index == addr_b.mode_index).all()
    # fit 时换映射 → 新映射生效并锁定
    assert enc_b.mapping_id == ids[1]
    with pytest.raises(KeyError):
        get_mapping("nonexistent-mapping")


def test_nan_filled_with_train_median(frozen_encoder: QuantileEncoder) -> None:
    row = pd.DataFrame({"a": [np.nan], "b": [0.0], "c": [0.0]})
    addr = frozen_encoder.transform(row)
    assert np.isfinite(addr.mode_index).all()


def test_default_params_match_paper() -> None:
    """论文口径默认值：A 区 256 箱（[0,255]）、B 区 255 箱（[0,254]），共 511 时间模式。"""
    p = EncoderParams()
    assert (p.n_bins_a, p.n_bins_b, p.partition_boundary) == (256, 255, 256)
    assert p.n_bins_a + p.n_bins_b == 511


def test_address_budget_validation(data: pd.DataFrame) -> None:
    """bin 可能落到映射时间模式预算之外时，fit 抛带参数信息的 ValueError。"""
    mid = _identity_id()
    # B 区最高基准时间模式 = 16 + 20 - 1 = 35，超出 32 个时间模式预算 [0, 31]
    bad_b = EncoderParams(n_bins_a=16, n_bins_b=20, partition_boundary=16)
    with pytest.raises(ValueError, match="时间模式预算"):
        QuantileEncoder(bad_b, mapping_id=mid).fit(data)
    # A 区 bin 数越过分区边界 → A 区地址侵入 B 区
    overlap = EncoderParams(n_bins_a=20, n_bins_b=4, partition_boundary=16)
    with pytest.raises(ValueError, match="侵入 B 区"):
        QuantileEncoder(overlap, mapping_id=mid).fit(data)


def test_paper_scale_budget() -> None:
    """论文规模（511 时间模式）：正确配置 (256,255,256) 通过；写反的 (255,256,256) 抛错。

    写反配置下 B 区最高基准时间模式 = 256 + 256 - 1 = 511，超出 511 个时间
    模式的预算 [0, 510]——修复前会被 np.clip 静默压到模式 510。
    """
    ensure_default_mappings(3, N_SPATIAL, 511)
    mid = resolve_mapping_id("identity", 3, N_SPATIAL, 511)
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(0, 1, (200, 3)), columns=["a", "b", "c"])
    enc = QuantileEncoder(EncoderParams(), mapping_id=mid).fit(X).freeze()
    addr = enc.transform(X)
    assert addr.temporal_index.max() <= 510  # 全部落在预算内
    swapped = QuantileEncoder(
        EncoderParams(n_bins_a=255, n_bins_b=256, partition_boundary=256), mapping_id=mid
    )
    with pytest.raises(ValueError, match="时间模式预算"):
        swapped.fit(X)


def test_transform_internal_state_guard(data: pd.DataFrame) -> None:
    """内部状态被破坏时 transform 抛显式异常（不再依赖 assert）。"""
    enc = QuantileEncoder(PARAMS, mapping_id=_identity_id()).fit(data).freeze()
    enc._mapping = None  # 模拟内部状态不完整
    with pytest.raises(RuntimeError, match="内部状态"):
        enc.transform(data)
