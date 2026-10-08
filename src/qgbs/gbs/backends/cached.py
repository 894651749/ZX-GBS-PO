"""只读样本库后端：复用已采集并质检的样本批次。"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from qgbs.gbs.backends.base import SampleBatch
from qgbs.gbs.sample_store import SampleStore


class CachedBackend:
    """从样本库按批次读取样本的 GBSSampler 实现。

    只读：不触发任何新的真机/仿真采样，保证实验可复现；
    样本不足时显式抛错而不是静默补齐。
    """

    def __init__(self, store: SampleStore | Path, batch_id: str) -> None:
        self.store = store if isinstance(store, SampleStore) else SampleStore(Path(store))
        self.batch_id = batch_id
        self._batch: SampleBatch | None = None

    def _load(self) -> SampleBatch:
        if self._batch is None:
            self._batch = self.store.load_batch(self.batch_id)
        return self._batch

    def sample(self, n_shots: int) -> SampleBatch:
        """取该批次前 ``n_shots`` 个样本（确定性子集）。"""
        batch = self._load()
        if n_shots > batch.meta.n_shots:
            raise ValueError(
                f"样本不足：批次 {self.batch_id!r} 仅有 {batch.meta.n_shots} shots，"
                f"请求 {n_shots}；cached 后端不生成新样本"
            )
        meta = dataclasses.replace(batch.meta, n_shots=n_shots)
        return SampleBatch(meta=meta, samples=batch.samples[:n_shots])

    @property
    def n_modes(self) -> int:
        return self._load().meta.n_modes
