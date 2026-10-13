"""E4.1 诊断一：真机 vs 仿真样本对照（方案书 §7.1）。

真机 hw-5M-b0 前 200k shots（冻结口径）vs 511×16 高斯 copula 仿真
（mean_click_rate 按真机平均 click 率 0.2703 对齐，η=0.7 占位）。

仿真器分块采样：copula 通道的 z ~ N(0,R) 一次性物化为 n_shots×8176
float64（200k → 13GB），超出 16GB 内存；此处复用仿真器已构建实例的
_chol/_thresholds 按块生成（块内 matmul out= 复用缓冲），峰值 ~2GB。
注：分块 rng 消费序列与 simulator.sample 一次性生成不同，本诊断批次的
样本不与样本库 sim 批次逐位一致（同分布、同实例结构），诊断结论不受影响。

用法：``PYTHONPATH=src .venv/bin/python scripts/e41_hw_vs_sim.py``
产物：``artifacts/e41_hw_vs_sim/``（table.csv / details.json / run.log）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from qgbs.experiments.diagnostics import hardware_vs_simulation
from qgbs.gbs.backends.base import BatchMeta, SampleBatch
from qgbs.gbs.backends.simulator import GaussianSimulatorBackend, SimulatorConfig
from qgbs.gbs.sample_store import SampleStore

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "e41_hw_vs_sim"
N_SHOTS = 200_000
SEED = 42
ETA = 0.7  # 占位（真机 η 未标定）；mean_click_rate 反解对齐真机平均 click 率


def sim_batch_chunked(n_shots: int, chunk: int = 25_000) -> SampleBatch:
    """511×16 copula 仿真分块采样（复用已构建实例，控制内存峰值）。"""
    cfg = SimulatorConfig(
        n_temporal=511, n_spatial=16, seed=SEED, system_efficiency=ETA,
        mean_click_rate=0.2703 / ETA,  # η=1 口径；实际平均 click 率对齐真机 0.2703
    )
    sim = GaussianSimulatorBackend(cfg)
    sim.sample(1)  # 触发实例构建（ chol / 阈值 / 边际）
    rng = np.random.default_rng((SEED, n_shots))
    parts = []
    for s in range(0, n_shots, chunk):
        e = min(s + chunk, n_shots)
        z = rng.standard_normal((e - s, sim.n_modes))
        np.matmul(z, sim._chol.T, out=z)  # 同形状 out= 复用，零额外分配
        parts.append((z > sim._thresholds).astype(np.uint8))
        del z
        print(f"  仿真采样 {e}/{n_shots}", flush=True)
    samples = np.concatenate(parts)
    now = "2026-10-12T00:00:00+00:00"
    meta = BatchMeta(
        batch_id=f"sim-copula-511x16-s{SEED}-n{n_shots}-eta{ETA}",
        device_version="simulator-gaussian-copula",
        n_modes=sim.n_modes, n_shots=n_shots,
        started_at=now, finished_at=now,
        source="simulation", calibration_session_id=f"sim-session-{SEED}",
        simulation_params={"method": "gaussian_copula_threshold", "seed": SEED,
                           "mean_click_rate": cfg.mean_click_rate,
                           "note": "分块采样（内存控制），与样本库 sim 批次不逐位一致"},
        system_efficiency=ETA,
        mode_config={"n_temporal": 511, "n_spatial": 16,
                     **(sim._profile_summary or {})},
    )
    return SampleBatch(meta=meta, samples=samples)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    store = SampleStore(ROOT / "sample_store")
    print(f"[1/3] 真机批次 hw-5M-b0 前 {N_SHOTS} shots 加载", flush=True)
    t0 = time.time()
    hw = store.load_batch("hw-5M-b0", n_shots=N_SHOTS)
    print(f"  done {time.time()-t0:.1f}s, mean_click={hw.samples.mean():.4f}", flush=True)

    print("[2/3] 仿真批次分块采样", flush=True)
    t0 = time.time()
    sim = sim_batch_chunked(N_SHOTS)
    print(f"  done {time.time()-t0:.1f}s, mean_click={sim.samples.mean():.4f}", flush=True)

    print("[3/3] 诊断对比", flush=True)
    t0 = time.time()
    r = hardware_vs_simulation([hw], [sim])
    print(f"  done {time.time()-t0:.1f}s", flush=True)
    print("\n" + r.verdict)
    r.table.to_csv(OUT / "diag1_hw_vs_sim_table.csv")
    (OUT / "diag1_hw_vs_sim_details.json").write_text(
        json.dumps({"verdict": r.verdict, **r.details}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n产物已写入 {OUT}")


if __name__ == "__main__":
    main()
