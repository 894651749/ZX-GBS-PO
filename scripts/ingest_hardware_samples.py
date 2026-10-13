"""真机 GBS 样本入库：扁平位压缩 .npy → 样本库 hardware 命名空间。

源文件约定（与 data/data_5M_0.bin.npy 一致）：合法 .npy（带 header）、
扁平 uint8、总字节数 = n_shots × (n_modes // 8)，每行为一个 click pattern
的 np.packbits 位压缩形式（与 qgbs.gbs.backends.base.pack_samples 约定一致）。

入库方式：优先硬链接（os.link，同文件系统零拷贝）把源文件链为
``{store_root}/hardware/{batch_id}/packed.npy``；跨设备失败时退化为拷贝。
meta.json 按 BatchMeta 序列化；时间戳取源文件 mtime（真实采集时间待
装置元数据补充，quality_flags.metadata_pending 标记）。

用法：``PYTHONPATH=src .venv/bin/python scripts/ingest_hardware_samples.py``
"""

from __future__ import annotations

import argparse
import dataclasses
import errno
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from qgbs.gbs.backends.base import BatchMeta
from qgbs.gbs.sample_store import SampleStore

ROOT = Path(__file__).resolve().parents[1]


def ingest(
    src: Path,
    batch_id: str,
    store_root: Path,
    n_modes: int,
    n_temporal: int,
    n_spatial: int,
) -> Path:
    """把扁平位压缩真机样本入库为 packed.npy 批次，返回批次目录。"""
    src = Path(src)
    if n_temporal * n_spatial != n_modes:
        raise ValueError(
            f"n_temporal×n_spatial={n_temporal}×{n_spatial} 与 n_modes={n_modes} 不一致"
        )
    row_bytes = n_modes // 8
    raw = np.load(src, mmap_mode="r")
    if raw.dtype != np.uint8 or raw.ndim != 1:
        raise ValueError(f"源文件须为扁平 uint8 .npy，实际 dtype={raw.dtype} shape={raw.shape}")
    if raw.nbytes % row_bytes != 0:
        raise ValueError(
            f"源文件总字节数 {raw.nbytes} 不能被行字节数 {row_bytes}（n_modes//8）整除"
        )
    n_shots = raw.nbytes // row_bytes

    batch_dir = Path(store_root) / "hardware" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    dest = batch_dir / "packed.npy"
    if dest.exists():
        if not os.path.samefile(src, dest):
            raise FileExistsError(f"{dest} 已存在且非源文件硬链接，拒绝覆盖")
        action = "已存在（硬链接），跳过"
    else:
        try:
            os.link(src, dest)
            action = "硬链接"
        except OSError as e:
            if e.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES):
                raise
            shutil.copy2(src, dest)
            action = "拷贝（硬链接失败，跨设备或权限受限）"

    mtime_iso = datetime.fromtimestamp(src.stat().st_mtime, tz=timezone.utc).isoformat()
    meta = BatchMeta(
        batch_id=batch_id,
        device_version="unknown-pending-device-metadata",
        n_modes=n_modes,
        n_shots=n_shots,
        started_at=mtime_iso,  # 文件接收时间占位，真实采集时间待装置元数据
        finished_at=mtime_iso,
        source="hardware",
        calibration_session_id="unknown-pending-calibration",
        system_efficiency=None,
        mode_config={"n_temporal": n_temporal, "n_spatial": n_spatial},
        quality_flags={
            "timestamps_are_file_receipt": True,
            "metadata_pending": True,
            "bit_packing": "np.packbits axis=1 big-endian",
        },
    )
    (batch_dir / "meta.json").write_text(
        json.dumps(dataclasses.asdict(meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[入库] {src} → {dest}（{action}）；n_shots={n_shots}, n_modes={n_modes}")
    return batch_dir


def verify(store_root: Path, batch_id: str, n_check: int = 100_000) -> float:
    """只读验证：load_batch 切片加载前 n_check shots，返回平均点击数。"""
    store = SampleStore(Path(store_root))
    meta = store.get_meta(batch_id)
    batch = store.load_batch(batch_id, n_shots=min(n_check, meta.n_shots))
    mean_clicks = float(batch.samples.sum(axis=1).mean())
    print(
        f"[验证] load_batch({batch_id!r}, n_shots={batch.meta.n_shots})："
        f"samples{batch.samples.shape}，平均点击数 {mean_clicks:.1f}"
    )
    return mean_clicks


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", default="data/data_5M_0.bin.npy", help="扁平位压缩 .npy 源文件")
    p.add_argument("--batch-id", default="hw-5M-b0", help="入库批次号")
    p.add_argument("--store-root", default="sample_store/", help="样本库根目录")
    p.add_argument("--n-modes", type=int, default=8176, help="模式数（= n_temporal × n_spatial）")
    p.add_argument("--n-temporal", type=int, default=511, help="时间模式数")
    p.add_argument("--n-spatial", type=int, default=16, help="空间模式数")
    args = p.parse_args(argv)

    src = Path(args.src)
    if not src.is_absolute():
        src = ROOT / src
    store_root = Path(args.store_root)
    if not store_root.is_absolute():
        store_root = ROOT / store_root

    ingest(src, args.batch_id, store_root, args.n_modes, args.n_temporal, args.n_spatial)
    verify(store_root, args.batch_id)


if __name__ == "__main__":
    main()
