"""样本库：批次 / 装置版本 / 质量标记的统一管理。

版本管理以标定会话为边界：

- 新标定会话（装置重启/复标定）→ major 版本递增；
- 同会话追加批次 → minor 版本递增；
- 真机与仿真样本使用隔离的命名空间（namespace 由 BatchMeta.source 推导）。

持久化：内存为主；``root`` 非空时每批次落盘
``{root}/{namespace}/{batch_id}/``（samples.npz 位压缩样本 + meta.json），
重启后可扫描恢复。选择 npz+json 而非 parquet：避免 pyarrow 依赖。
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from qgbs.gbs.backends.base import (
    BatchMeta,
    SampleBatch,
    pack_samples,
    unpack_samples,
)


@dataclass(frozen=True)
class SampleVersion:
    """样本库版本号：标定会话边界为 major，同会话追加为 minor。"""

    namespace: Literal["hardware", "simulation"]  # 真机/仿真命名空间隔离
    major: int  # 标定会话序号
    minor: int  # 会话内批次序号

    @property
    def version_str(self) -> str:
        """形如 "hardware-v3.2" 的版本字符串。"""
        return f"{self.namespace}-v{self.major}.{self.minor}"


class SampleStore:
    """GBS 样本库（内存 + 可选落盘）。"""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else None
        self._batches: dict[str, SampleBatch] = {}
        self._versions: dict[str, SampleVersion] = {}
        self._session_major: dict[tuple[str, str], int] = {}
        if self.root is not None and self.root.exists():
            self._scan_disk()

    # ------------------------------------------------------------------
    # 版本语义
    # ------------------------------------------------------------------

    def resolve_version(self, calibration_session_id: str, namespace: str = "simulation") -> SampleVersion:
        """由标定会话 ID 解析当前 minor 版本（不存在则建立新 major）。"""
        key = (namespace, calibration_session_id)
        if key not in self._session_major:
            major = 1 + max(
                (m for (ns, _), m in self._session_major.items() if ns == namespace),
                default=0,
            )
            self._session_major[key] = major
        minor = sum(
            1 for bid, v in self._versions.items()
            if v.namespace == namespace and v.major == self._session_major[key]
        )
        return SampleVersion(namespace=namespace, major=self._session_major[key], minor=minor)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # 读写
    # ------------------------------------------------------------------

    def register_batch(self, batch: SampleBatch) -> SampleVersion:
        """登记一个新批次（内存 + 可选落盘），按标定会话分配版本号。"""
        meta = batch.meta
        if meta.batch_id in self._batches:
            raise ValueError(f"批次 {meta.batch_id!r} 已存在，拒绝重复登记")
        namespace = "hardware" if meta.source == "hardware" else "simulation"
        current = self.resolve_version(meta.calibration_session_id, namespace)
        version = SampleVersion(
            namespace=namespace, major=current.major, minor=current.minor + 1
        )
        self._batches[meta.batch_id] = batch
        self._versions[meta.batch_id] = version
        if self.root is not None:
            self._persist(batch, version)
        return version

    def load_batch(self, batch_id: str) -> SampleBatch:
        """按批次号读取样本（内存优先，其次磁盘）。"""
        if batch_id in self._batches:
            return self._batches[batch_id]
        if self.root is None:
            raise KeyError(f"批次 {batch_id!r} 不存在")
        for ns in ("simulation", "hardware"):
            d = self.root / ns / batch_id
            if (d / "meta.json").exists():
                meta = BatchMeta(**json.loads((d / "meta.json").read_text(encoding="utf-8")))
                packed = np.load(d / "samples.npz")["packed"]
                batch = SampleBatch(meta=meta, samples=unpack_samples(packed, meta.n_modes))
                self._batches[batch_id] = batch
                return batch
        raise KeyError(f"批次 {batch_id!r} 不存在")

    def list_batches(
        self,
        namespace: Literal["hardware", "simulation"] | None = None,
        device_version: str | None = None,
    ) -> pd.DataFrame:
        """列出批次清单（可按命名空间/装置版本过滤），含质量标记与版本。"""
        rows = []
        for bid, batch in self._batches.items():
            v = self._versions[bid]
            m = batch.meta
            rows.append(
                {
                    "batch_id": bid,
                    "namespace": v.namespace,
                    "version": v.version_str,
                    "device_version": m.device_version,
                    "n_modes": m.n_modes,
                    "n_shots": m.n_shots,
                    "calibration_session_id": m.calibration_session_id,
                    "system_efficiency": m.system_efficiency,
                    "quality_flags": dict(m.quality_flags),
                }
            )
        df = pd.DataFrame(rows)
        if len(df) == 0:
            return df
        if namespace is not None:
            df = df[df["namespace"] == namespace]
        if device_version is not None:
            df = df[df["device_version"] == device_version]
        return df.reset_index(drop=True)

    def set_quality_flag(self, batch_id: str, key: str, value: object) -> None:
        """更新批次质量标记（如标记某批次不可用）。"""
        batch = self._batches.get(batch_id) or self.load_batch(batch_id)
        flags = {**batch.meta.quality_flags, key: value}
        batch.meta = dataclasses.replace(batch.meta, quality_flags=flags)
        if self.root is not None:
            v = self._versions[batch_id]
            (self.root / v.namespace / batch_id / "meta.json").write_text(
                json.dumps(dataclasses.asdict(batch.meta), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    # ------------------------------------------------------------------
    # 落盘
    # ------------------------------------------------------------------

    def _persist(self, batch: SampleBatch, version: SampleVersion) -> None:
        assert self.root is not None
        d = self.root / version.namespace / batch.meta.batch_id
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "samples.npz", packed=pack_samples(batch.samples))
        (d / "meta.json").write_text(
            json.dumps(dataclasses.asdict(batch.meta), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _scan_disk(self) -> None:
        assert self.root is not None
        for ns in ("simulation", "hardware"):
            ns_dir = self.root / ns
            if not ns_dir.exists():
                continue
            for d in sorted(ns_dir.iterdir()):
                meta_path = d / "meta.json"
                if not meta_path.exists() or d.name in self._batches:
                    continue
                meta = BatchMeta(**json.loads(meta_path.read_text(encoding="utf-8")))
                packed = np.load(d / "samples.npz")["packed"]
                batch = SampleBatch(meta=meta, samples=unpack_samples(packed, meta.n_modes))
                version = self.resolve_version(meta.calibration_session_id, ns)
                self._batches[d.name] = batch
                self._versions[d.name] = SampleVersion(
                    namespace=ns, major=version.major, minor=version.minor + 1  # type: ignore[arg-type]
                )
