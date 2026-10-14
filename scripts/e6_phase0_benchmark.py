"""E6 阶段 0 性能基准：单行直方图耗时/内存实测 + 470 万行推算。

用法：``PYTHONPATH=src .venv/bin/python scripts/e6_phase0_benchmark.py``
产物：``artifacts/e6_paper_replica/phase0/benchmark.json`` + ``benchmark_report.md``。
"""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from qgbs.gbs.pattern_service import ColumnPackedStore, histogram_from_row_packed

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "e6_paper_replica" / "phase0"
SRC = ROOT / "sample_store" / "hardware" / "hw-5M-b0" / "packed.npy"
N_MODES = 8176
REPEAT = 5  # 单行计时重复次数取均值
ROWS_TOTAL = 4_700_000  # 阶段 2 推算口径：23.4 万行 × 20 折
N_PROC = 8


def _bench(fn, *args) -> tuple[float, float]:
    """返回（单次耗时秒均值, tracemalloc 峰值字节）。"""
    fn(*args)  # 预热
    times = []
    for _ in range(REPEAT):
        tracemalloc.start()
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return float(np.mean(times)), float(peak)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    modes_a = np.sort(rng.choice(4096, size=16, replace=False))  # A 组 16 模
    modes_b = 4096 + np.sort(rng.choice(4080, size=10, replace=False))  # B 组 10 模
    results: dict = {}

    for n_shots in (1_000_000, 5_000_000):
        packed = np.load(SRC, mmap_mode="r").reshape(5_000_000, 1022)[:n_shots]
        rec: dict = {"n_shots": n_shots}
        for grp, modes in (("A16", modes_a), ("B10", modes_b)):
            t, mem = _bench(histogram_from_row_packed, packed, modes)
            rec[f"row_packed_{grp}"] = {"sec_per_row": t, "tracemalloc_peak_mb": mem / 1e6}
            print(f"[{n_shots}] 行打包 {grp}: {t*1000:.1f}ms/行, 峰值 {mem/1e6:.0f}MB", flush=True)

        # 列打包路径：转置一次性成本 + 单行耗时
        t0 = time.perf_counter()
        store = ColumnPackedStore.from_row_packed(packed, N_MODES)
        rec["colpack_build_s"] = time.perf_counter() - t0
        rec["colpack_store_mb"] = store.cols.nbytes / 1e6
        print(f"[{n_shots}] 列打包转置: {rec['colpack_build_s']:.1f}s, 存储 {rec['colpack_store_mb']:.0f}MB", flush=True)
        for grp, modes in (("A16", modes_a), ("B10", modes_b)):
            t, mem = _bench(store.histogram, modes)
            rec[f"col_packed_{grp}"] = {"sec_per_row": t, "tracemalloc_peak_mb": mem / 1e6}
            print(f"[{n_shots}] 列打包 {grp}: {t*1000:.1f}ms/行, 峰值 {mem/1e6:.0f}MB", flush=True)
        results[str(n_shots)] = rec
        del packed, store

    # 推算 470 万行总耗时（行打包路径 + 8 进程并行两档；列打包另加一次性转置）
    est = {}
    for ns in ("1000000", "5000000"):
        r = results[ns]
        per_row = r["row_packed_A16"]["sec_per_row"] + r["row_packed_B10"]["sec_per_row"]
        est[ns] = {
            "per_row_s_rowpacked": per_row,
            "total_h_single_core": per_row * ROWS_TOTAL / 3600,
            "total_h_8proc": per_row * ROWS_TOTAL / 3600 / N_PROC,
            "colpack_total_h_8proc": (
                (r["col_packed_A16"]["sec_per_row"] + r["col_packed_B10"]["sec_per_row"]) * ROWS_TOTAL / 3600 / N_PROC
                + r["colpack_build_s"] / 3600
            ),
        }
        print(f"[推算 {ns}] 单行(AB合计) {per_row*1000:.0f}ms → 470万行 单核 {est[ns]['total_h_single_core']:.1f}h / "
              f"8进程 {est[ns]['total_h_8proc']:.1f}h（列打包 {est[ns]['colpack_total_h_8proc']:.1f}h）", flush=True)
    results["estimates"] = est
    (OUT / "benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved {OUT / 'benchmark.json'}")


if __name__ == "__main__":
    main()
