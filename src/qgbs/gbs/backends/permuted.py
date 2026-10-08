"""置换对照后端：保持边际统计、破坏联合结构（方案书 §7.1 诊断）。"""

from __future__ import annotations

import dataclasses

import numpy as np

from qgbs.gbs.backends.base import GBSSampler, SampleBatch


class PermutedBackend:
    """置换对照后端。

    包装任意 GBSSampler：对其样本在模式维度做独立置换（逐模式列独立
    shuffle），边际分布精确保持不变、模式间联合结构被破坏。

    用途：置换对照实验——仅用于判断联合结构是否有贡献；归因需结合
    匹配低阶统计的经典随机特征对照（见 experiments/diagnostics.py）。
    """

    def __init__(self, base: GBSSampler, seed: int = 42) -> None:
        self.base = base
        self.seed = seed

    def sample(self, n_shots: int) -> SampleBatch:
        """采样并对样本做模式维度置换后返回。"""
        return self.permute(self.base.sample(n_shots))

    @property
    def n_modes(self) -> int:
        return self.base.n_modes

    def permute(self, batch: SampleBatch) -> SampleBatch:
        """对一个批次做逐模式列独立置换（保持边际、破坏联合）。"""
        rng = np.random.default_rng((self.seed, batch.meta.n_shots))
        permuted = np.apply_along_axis(rng.permutation, 0, batch.samples)
        meta = dataclasses.replace(
            batch.meta,
            batch_id=f"{batch.meta.batch_id}-permuted-s{self.seed}",
            quality_flags={**batch.meta.quality_flags, "permuted": True},
        )
        return SampleBatch(meta=meta, samples=permuted.astype(np.uint8))
