"""真机样本入库 / 样本库惰性化 / runner hardware 分支测试。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from qgbs.gbs.backends.base import pack_samples
from qgbs.gbs.backends.cached import CachedBackend
from qgbs.gbs.sample_store import SampleStore

ROOT = Path(__file__).resolve().parents[1]
INGEST_SCRIPT = ROOT / "scripts" / "ingest_hardware_samples.py"


@pytest.fixture()
def hw_src(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """小尺寸真机格式源文件：扁平 uint8 .npy（加载端 reshape 视图）。"""
    rng = np.random.default_rng(7)
    samples = (rng.random((100, 16)) < 0.3).astype(np.uint8)
    src = tmp_path / "src.npy"
    np.save(src, pack_samples(samples).ravel())
    return src, samples


def _run_ingest(src: Path, store_root: Path, batch_id: str = "hw-test") -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(
        [
            sys.executable, str(INGEST_SCRIPT),
            "--src", str(src), "--batch-id", batch_id,
            "--store-root", str(store_root),
            "--n-modes", "16", "--n-temporal", "2", "--n-spatial", "8",
        ],
        check=True, env=env, capture_output=True, text=True,
    )


def test_ingest_cli_roundtrip(hw_src, tmp_path: Path) -> None:
    """入库：硬链接零拷贝 + meta.json 字段 + 切片解包与源样本一致。"""
    src, samples = hw_src
    store_root = tmp_path / "store"
    _run_ingest(src, store_root)
    batch_dir = store_root / "hardware" / "hw-test"
    assert os.path.samefile(src, batch_dir / "packed.npy")  # 硬链接（同 inode）
    meta = json.loads((batch_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["source"] == "hardware" and meta["n_shots"] == 100
    assert meta["mode_config"] == {"n_temporal": 2, "n_spatial": 8}
    assert meta["quality_flags"]["metadata_pending"] is True

    store = SampleStore(store_root)
    full = store.load_batch("hw-test")
    assert np.array_equal(full.samples, samples)
    sub = store.load_batch("hw-test", n_shots=40)
    assert sub.samples.shape == (40, 16)
    assert sub.meta.n_shots == 40
    assert np.array_equal(sub.samples, samples[:40])
    with pytest.raises(ValueError, match="样本不足"):
        store.load_batch("hw-test", n_shots=101)


def test_scan_disk_lazy(hw_src, tmp_path: Path) -> None:
    """_scan_disk 只建 meta 索引不物化样本；get_meta / list_batches 可用。"""
    src, _ = hw_src
    store_root = tmp_path / "store"
    _run_ingest(src, store_root)
    store = SampleStore(store_root)
    assert len(store._batches) == 0  # 未物化
    listing = store.list_batches(namespace="hardware")
    assert set(listing["batch_id"]) == {"hw-test"}
    assert store.get_meta("hw-test").n_modes == 16


def test_cached_backend_sliced_loading(hw_src, tmp_path: Path) -> None:
    """CachedBackend.sample(n) 走切片加载；扩容请求重新加载更大切片。"""
    src, samples = hw_src
    store_root = tmp_path / "store"
    _run_ingest(src, store_root)
    backend = CachedBackend(store_root, "hw-test")
    assert backend.n_modes == 16
    assert len(backend.store._batches) == 0  # n_modes 不触发物化
    b40 = backend.sample(40)
    assert b40.samples.shape == (40, 16)
    assert len(backend.store._batches) == 0  # packed.npy 批次不进全量缓存
    assert np.array_equal(b40.samples, samples[:40])
    b90 = backend.sample(90)  # 超出已缓存切片 → 重新加载更大切片
    assert np.array_equal(b90.samples, samples[:90])
    assert np.array_equal(backend.sample(30).samples, samples[:30])
    with pytest.raises(ValueError, match="cached 后端不生成新样本"):
        backend.sample(10_000)


def test_runner_hardware_branch(hw_src, tmp_path: Path) -> None:
    """runner hardware 分支：按 batch_id 直挂真机批次，不触发仿真生成。"""
    from qgbs.experiments.runner import _prepare_gbs_backend

    src, samples = hw_src
    store_root = tmp_path / "store"
    _run_ingest(src, store_root)
    cfg = {
        "gbs": {
            "backend": "cached",
            "namespace": "hardware",
            "batch_id": "hw-test",
            "sample_store": str(store_root),
            "n_temporal": 2,
            "n_spatial": 8,
            "n_shots": 60,
        }
    }
    backend, batch = _prepare_gbs_backend(cfg)
    assert batch.meta.source == "hardware"
    assert batch.meta.n_shots == 60
    assert np.array_equal(batch.samples, samples[:60])
    assert backend.n_modes == 16
    # 不触发仿真：simulation 命名空间为空
    assert len(SampleStore(store_root).list_batches(namespace="simulation")) == 0
    # mode_config 不一致 → 显式报错
    cfg_bad = {"gbs": {**cfg["gbs"], "n_temporal": 4}}
    with pytest.raises(ValueError, match="模式结构不一致"):
        _prepare_gbs_backend(cfg_bad)
    # 缺 batch_id → 显式报错
    cfg_no_id = {"gbs": {k: v for k, v in cfg["gbs"].items() if k != "batch_id"}}
    with pytest.raises(ValueError, match="batch_id"):
        _prepare_gbs_backend(cfg_no_id)


def test_q1_hardware_config_wiring() -> None:
    """Q1 真机配置：地址预算校验口径与 n_shots 不超批次库存。"""
    raw = yaml.safe_load(
        (ROOT / "configs" / "experiments" / "q1_marginal_hardware.yaml").read_text(encoding="utf-8")
    )
    gbs, enc = raw["gbs"], raw["encoder"]
    assert (gbs["backend"], gbs["namespace"]) == ("cached", "hardware")
    assert gbs["n_temporal"] * gbs["n_spatial"] == 8176
    assert enc["n_bins_a"] + enc["n_bins_b"] == gbs["n_temporal"]
    assert enc["n_bins_a"] <= enc["partition_boundary"]
    assert enc["partition_boundary"] + enc["n_bins_b"] - 1 <= gbs["n_temporal"] - 1
    store = SampleStore(ROOT / gbs["sample_store"])
    assert store.get_meta(gbs["batch_id"]).n_shots >= gbs["n_shots"]
