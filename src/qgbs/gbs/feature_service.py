"""GBSFeatureService：模式地址 → GBS 特征向量（双通道 + 防泄漏状态机）。

特征构造机制（论文 arXiv:2506.19707 的 click pattern 频率特征）：

- ``marginal``：每个特征寻址模式的边际 click 率（值 → 时间 bin → 模式频率）；
- ``second_order``：特征对寻址模式之间的归一化关联量；
- ``higher_order``：候选基（CandidateBasis）中各模式组合的联合 click 频率，
  按该资产地址集合与基组合的重合比例软激活（值 = 基频率 × 重合数/阶数）。
  注：硬子集激活（全包含才取值）在 demo 小规模寻址下会退化为全零特征，
  软激活是显式的设计决策。

双通道：

- 预计算缓存通道：fit 时从训练批次一次性算出统计量词表（边际/二阶/基频率），
  并对高频出现的地址行预计算特征向量；
- 按需统计通道：缓存未命中的地址行现场基于冻结词表计算（``build_features``
  内回退到 ``_compute_rows``，统计来源仍是 fit 时冻结的训练批次词表），
  不设独立接口。

防泄漏状态机：``fit``（训练/验证期确定映射、候选基、压缩）→ ``freeze``
后锁定，测试期仅允许 ``build_features``。
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from qgbs.gbs.backends.base import BatchMeta, SampleBatch
from qgbs.gbs.basis import CandidateBasis, stability_filter_basis
from qgbs.gbs.compression import CompressionTransform
from qgbs.gbs.encoder import ModeAddress
from qgbs.gbs.statistics import higher_order_stats, marginal_stats, second_order_stats


@dataclass(frozen=True)
class GBSFeatureConfig:
    """GBS 特征生成配置。"""

    stat_kinds: tuple[str, ...] = ("marginal", "second_order")  # 使用的统计量族
    cache_top_k: int = 10_000  # 离线预计算缓存的高频地址行数
    activation: str = "soft"  # 高阶基激活方式："soft"（重合度加权）/ "hard"（严格子集）


def _merged_meta(batches: list[SampleBatch]) -> BatchMeta:
    """拼接批次的合并 meta：n_shots 求和、采集区间取并集，质量标记注明拼接来源。"""
    metas = [b.meta for b in batches]
    if len(metas) == 1:
        return metas[0]
    return dataclasses.replace(
        metas[0],
        batch_id="+".join(m.batch_id for m in metas),
        n_shots=sum(m.n_shots for m in metas),
        started_at=min(m.started_at for m in metas),
        finished_at=max(m.finished_at for m in metas),
        quality_flags={
            **metas[0].quality_flags,
            "merged_from_batches": [m.batch_id for m in metas],
        },
    )


class GBSFeatureService:
    """将编码后的模式地址转换为可入模的 GBS 特征向量（见模块 docstring）。"""

    def __init__(
        self,
        config: GBSFeatureConfig | None = None,
        compression: CompressionTransform | None = None,
    ) -> None:
        self.config = config or GBSFeatureConfig()
        self.compression = compression
        self._fitted: bool = False
        self._frozen: bool = False
        self._marginal: np.ndarray | None = None
        self._corr: np.ndarray | None = None
        self._basis: CandidateBasis | None = None
        self._basis_rates: np.ndarray | None = None
        self._n_features: int = 0
        self._pairs: list[tuple[int, int]] = []
        self._names: list[str] = []
        self._cache: dict[bytes, np.ndarray] = {}

    # ------------------------------------------------------------------
    # 状态机
    # ------------------------------------------------------------------

    def fit(
        self,
        train_addresses: ModeAddress,
        train_batches: list[SampleBatch],
        basis: CandidateBasis | None = None,
    ) -> "GBSFeatureService":
        """训练/验证期：从训练批次建立统计量词表，确定高阶基与压缩。

        对已冻结实例调用抛 RuntimeError。多个训练批次拼接时合并 meta
        （n_shots 求和、质量标记注明拼接）；收到 ≥2 个批次且启用
        higher_order 时，对候选基做跨批次稳定性筛选（方案书 §4.2，
        阈值取 ``basis.params.stability_max_rel_std``）。
        """
        if self._frozen:
            raise RuntimeError("GBSFeatureService 已冻结：不允许再 fit")
        if not train_batches:
            raise ValueError("train_batches 为空")
        kinds = self.config.stat_kinds
        # 单批次直接引用（零拷贝）；多批次才拼接
        X = (
            np.asarray(train_batches[0].samples)
            if len(train_batches) == 1
            else np.concatenate([np.asarray(b.samples) for b in train_batches], axis=0)
        )
        joint_batch = SampleBatch(meta=_merged_meta(train_batches), samples=X)
        if "marginal" in kinds:
            self._marginal = marginal_stats(joint_batch)
        if "second_order" in kinds:
            self._corr = second_order_stats(joint_batch, kind="correlation")
        if "higher_order" in kinds:
            if basis is None:
                raise ValueError("stat_kinds 含 higher_order 时必须提供 CandidateBasis")
            if len(train_batches) >= 2:
                basis = stability_filter_basis(basis, train_batches)
            self._basis = basis
            self._basis_rates = higher_order_stats(joint_batch, basis).toarray().ravel()
        self._n_features = train_addresses.mode_index.shape[1]
        self._pairs = list(combinations(range(self._n_features), 2))
        self._names = self._build_names()
        if self.compression is not None:
            train_feats = self._compute_rows(train_addresses.mode_index)
            self.compression.fit(pd.DataFrame(train_feats))
        self._fitted = True
        return self

    def freeze(self) -> "GBSFeatureService":
        """冻结服务（基/压缩/映射锁定），返回冻结副本。"""
        if not self._fitted:
            raise RuntimeError("GBSFeatureService 未 fit，无法 freeze")
        frozen = copy.copy(self)
        frozen._cache = dict(self._cache)
        frozen._frozen = True
        if self.compression is not None:
            frozen.compression = self.compression.freeze()
        return frozen

    def vocab_snapshot(self) -> "GBSFeatureService":
        """词表快照：浅拷贝共享统计量词表（_marginal/_corr/_basis_rates 只读）、
        清空地址行缓存。供跨折复用词表（同一采样批次词表与折无关）。"""
        if not self._fitted:
            raise RuntimeError("GBSFeatureService 未 fit，无法快照词表")
        snap = copy.copy(self)
        snap._cache = {}
        return snap

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    # ------------------------------------------------------------------
    # 特征名
    # ------------------------------------------------------------------

    def _build_names(self) -> list[str]:
        names: list[str] = []
        if "marginal" in self.config.stat_kinds:
            names += [f"gbs_marg_f{j}" for j in range(self._n_features)]
        if "second_order" in self.config.stat_kinds:
            names += [f"gbs_pair_f{i}_f{j}" for i, j in self._pairs]
        if "higher_order" in self.config.stat_kinds and self._basis is not None:
            names += [f"gbs_ho_{k}" for k in range(len(self._basis.basis))]
        return names

    def feature_names(self) -> list[str]:
        """返回当前配置下全部 GBS 特征名。"""
        return list(self._names)

    # ------------------------------------------------------------------
    # 统计通道
    # ------------------------------------------------------------------

    def _compute_rows(self, addr_mode_index: np.ndarray) -> np.ndarray:
        """按需统计通道：基于冻结词表对地址行矩阵逐行计算特征向量。"""
        R, F = addr_mode_index.shape
        parts: list[np.ndarray] = []
        if "marginal" in self.config.stat_kinds:
            assert self._marginal is not None
            parts.append(self._marginal[addr_mode_index])  # (R, F)
        if "second_order" in self.config.stat_kinds:
            assert self._corr is not None
            cols = [self._corr[addr_mode_index[:, i], addr_mode_index[:, j]] for i, j in self._pairs]
            parts.append(np.stack(cols, axis=1) if cols else np.empty((R, 0)))
        if "higher_order" in self.config.stat_kinds:
            assert self._basis is not None and self._basis_rates is not None
            order = self._basis.basis.shape[1]
            ho = np.zeros((R, len(self._basis.basis)))
            for k, combo in enumerate(self._basis.basis):
                if self.config.activation == "soft":
                    # 软激活：地址集合与基组合的重合比例 × 基联合频率
                    hits = np.zeros(R, dtype=float)
                    for m in combo:
                        hits += (addr_mode_index == m).any(axis=1)
                    ho[:, k] = self._basis_rates[k] * (hits / order)
                elif self.config.activation == "hard":
                    # 硬激活：基组合被地址集合严格包含时取联合 click 频率
                    mask = np.ones(R, dtype=bool)
                    for m in combo:
                        mask &= (addr_mode_index == m).any(axis=1)
                    ho[mask, k] = self._basis_rates[k]
                else:
                    raise ValueError(f"未知激活方式: {self.config.activation!r}")
            parts.append(ho)
        return np.concatenate(parts, axis=1) if parts else np.empty((R, 0))

    def precompute_cache(self, addresses: ModeAddress) -> None:
        """预计算缓存通道：对高频出现的地址行预算特征向量（fit 阶段调用）。"""
        if not self._fitted:
            raise RuntimeError("GBSFeatureService 未 fit")
        rows = addresses.mode_index
        uniq, counts = np.unique(rows, axis=0, return_counts=True)
        order = np.argsort(-counts)[: self.config.cache_top_k]
        for r in uniq[order]:
            key = r.tobytes()
            if key not in self._cache:
                self._cache[key] = self._compute_rows(r[None, :])[0]

    def build_features(self, addresses: ModeAddress, index: pd.MultiIndex) -> pd.DataFrame:
        """按模式地址生成特征表（双通道：缓存命中直接查表，否则按需统计）。

        仅允许在冻结副本上调用。

        Returns:
            索引为 (date, symbol)、列为 gbs_* 特征名的 DataFrame。
        """
        if not self._frozen:
            raise RuntimeError("GBSFeatureService 未冻结：仅允许在 freeze() 后 build_features")
        rows = addresses.mode_index
        uniq, inverse = np.unique(rows, axis=0, return_inverse=True)
        feats = np.empty((len(uniq), len(self._names)))
        for k, r in enumerate(uniq):
            key = r.tobytes()
            vec = self._cache.get(key)
            if vec is None:
                vec = self._compute_rows(r[None, :])[0]
                self._cache[key] = vec
            feats[k] = vec
        out = pd.DataFrame(feats[inverse], index=index, columns=self._names)
        if self.compression is not None:
            out = self.compression.transform(out)
        return out
