"""分布级对照：copula 近似仿真 vs 精确 GBS（真机接入前的基线验证）。

背景（审计结论）：高斯 copula 阈值采样可表达任意 pairwise 相关，而真机
threshold GBS 的联合分布受 Torontonian/hafnian 物理约束，两者的可达分布
集合不同。因此真机接入（P3）前，需要在**匹配边际 click 率**的前提下做
分布级对照，量化近似仿真与精确 GBS 分布的差异量级，作为后续真机对照的
基线认知。

对照内容（:func:`compare_backends`）：

1. 逐模式边际 click 率差异（L1 / L∞）；
2. 二阶联合 click 概率矩阵（非对角元）Frobenius 差异——与指标 1 正交
   （对角元即边际）；另有对边际失配**免疫**的 click 相关系数矩阵
   Frobenius 差异（隔离联合结构差异，其噪声底用 split-half 经验估计）；
3. 若干 3–4 模式子系统的 click pattern 联合分布 TV 距离；
4. 每个指标附**采样噪声底**估计（两独立有限样本对照在一阶高斯近似下的
   期望差异），用于区分"统计涨落"与"结构性差异"。

边际对齐：copula 仿真器的逐模式边际由固定包络决定，对外只有
``mean_click_rate`` 一个标量自由度；:func:`match_marginals` 对其做二分
搜索，使平均边际 click 率与目标（通常取精确后端的 ``exact_marginals()``）
对齐。逐模式形状不可逐一对齐（包络约束），残余差异本身即对照结果的
一部分——判读联合结构差异时以相关系数指标为准。

基线观测（P3 真机接入前，n_shots=20000；exact：6 模式、r∈U(0.3,0.9)、
η=0.9、Haar 实例 seed=7；copula：平坦时间包络 2×3、平均边际已对齐至
0.306，残余逐模式失配 |Δp| 均值 0.046 / 最大 0.081——包络形状约束所致）：

- exact vs exact（同实例、独立采样）：各指标与噪声底比值 0.7–1.9，
  验证对照工具统计功效（噪声底估计准确）；
- exact vs copula：二阶联合 Frobenius 0.262（15× 噪声底）、
  click 相关系数 Frobenius 1.03（15× 噪声底，对边际失配免疫）、
  3 模式子系统 TV 均值 0.147 / 最大 0.241（15–25× 噪声底）、
  4 模式子系统 TV 均值 0.219 / 最大 0.282（16–20× 噪声底）。

结论：即使平均边际对齐，copula 近似的二阶与高阶联合结构与精确 GBS
分布存在**显著结构性差异**（远超采样噪声），证实审计判断——copula
可达分布集合与真机 threshold GBS 不同；涉及联合结构的结论须以
本精确基准（小规模）或真机样本为准。
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from qgbs.gbs.backends.base import GBSSampler, SampleBatch
from qgbs.gbs.backends.simulator import GaussianSimulatorBackend, SimulatorConfig
from qgbs.gbs.statistics import second_order_stats

__all__ = ["CompareReport", "compare_backends", "match_marginals"]

# 高斯近似下两独立比例估计之差的期望绝对值系数 E|N(0,σ²)| = σ·√(2/π)
_MEAN_ABS_NORMAL = float(np.sqrt(2.0 / np.pi))


# ----------------------------------------------------------------------
# 边际对齐
# ----------------------------------------------------------------------


def match_marginals(
    base_config: SimulatorConfig,
    target_rates: np.ndarray,
    *,
    n_calib: int = 20000,
    tol: float = 1e-3,
    max_iter: int = 50,
) -> SimulatorConfig:
    """二分搜索 ``mean_click_rate``，使 copula 仿真器的平均边际 click 率对齐目标。

    仿真器采样在给定 (seed, n_shots) 下完全确定，因此校准是确定性的。
    逐模式边际由仿真器固定包络决定，无法逐一对齐（见模块 docstring）。

    Args:
        base_config: 待调整的仿真器配置（仅改 ``mean_click_rate``，其余保留）。
        target_rates: 目标逐模式边际 click 率，长度须等于 ``n_modes``
            （典型取 ``ExactGBSSampler.exact_marginals()``）。
        n_calib: 每次评估使用的校准样本数。
        tol: 平均边际的收敛容差。

    Returns:
        平均边际与 ``target_rates`` 均值对齐后的新 :class:`SimulatorConfig`。
    """
    target_rates = np.asarray(target_rates, dtype=float)
    n_modes = base_config.n_temporal * base_config.n_spatial
    if target_rates.shape != (n_modes,):
        raise ValueError(
            f"target_rates 长度 {target_rates.shape[0]} 与 n_modes={n_modes} 不一致"
        )
    target = float(target_rates.mean())

    def mean_rate(mean_click_rate: float) -> float:
        cfg = dataclasses.replace(base_config, mean_click_rate=mean_click_rate)
        return float(GaussianSimulatorBackend(cfg).sample(n_calib).samples.mean())

    lo, hi = 1e-4, 1.0
    hi_rate = mean_rate(hi)
    while hi_rate < target and hi < 8.0:
        hi *= 2.0  # 目标过高时扩展上界（受边际截断 0.95 限制，可能不可达）
        hi_rate = mean_rate(hi)
    if hi_rate < target:
        raise ValueError(
            f"目标平均边际 {target:.4f} 超出该配置可达范围（mean_click_rate={hi:.2f} "
            f"时仍仅 {hi_rate:.4f}）；请降低目标或调整 system_efficiency"
        )
    mid = 0.5 * (lo + hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if mean_rate(mid) < target:
            lo = mid
        else:
            hi = mid
        if abs(mean_rate(0.5 * (lo + hi)) - target) < tol:
            mid = 0.5 * (lo + hi)
            break
    return dataclasses.replace(base_config, mean_click_rate=mid)


# ----------------------------------------------------------------------
# 分布对照
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CompareReport:
    """分布级对照结果。

    Attributes:
        table: 指标对照表（metric / value / noise_floor / ratio / note）。
        interpretation: 中文判读（各指标是否超出采样噪声底的结论）。
        marginals_a / marginals_b: 两后端的逐模式边际 click 率估计。
        joint_a / joint_b: 两后端的二阶联合 click 概率矩阵。
        subsystem_tv: 各子系统规模的逐子系统 TV 距离列表。
        n_shots: 对照使用的样本数（每后端）。
        labels: 两后端的标识（device_version/batch_id）。
    """

    table: pd.DataFrame
    interpretation: str
    marginals_a: np.ndarray
    marginals_b: np.ndarray
    joint_a: np.ndarray
    joint_b: np.ndarray
    subsystem_tv: dict[int, list[float]]
    n_shots: int
    labels: tuple[str, str] = field(default=("", ""))
    extra: dict[str, Any] = field(default_factory=dict)


def _subsystem_codes(samples: np.ndarray, idx: tuple[int, ...]) -> np.ndarray:
    """样本在子系统 idx 上的 pattern 编码（列序即 idx 序，低位在前）。"""
    x = np.asarray(samples[:, list(idx)], dtype=np.int64)
    return x @ (1 << np.arange(len(idx)))


def _empirical_probs(samples: np.ndarray, idx: tuple[int, ...]) -> np.ndarray:
    codes = _subsystem_codes(samples, idx)
    return np.bincount(codes, minlength=2 ** len(idx)) / len(codes)


def _tv_floor(p: np.ndarray, n: int) -> float:
    """两独立 n 样本经验分布在 p 附近的 TV 距离期望（一阶高斯近似）。"""
    return 0.5 * _MEAN_ABS_NORMAL * float(np.sqrt(2.0 * p * (1.0 - p) / n).sum())


def _corr_offdiag(batch: SampleBatch, offdiag: np.ndarray) -> np.ndarray:
    """click 相关系数矩阵的非对角元（零方差模式处置为 0）。"""
    return second_order_stats(batch, "correlation")[offdiag]


def _corr_split_half_floor(batch: SampleBatch, offdiag: np.ndarray) -> float:
    """相关系数差异的经验噪声底：同批样本对半切分后的 Frobenius 差异。

    半样本差异的方差是全样本独立对照方差的 4 倍（v(m) ∝ 1/m），
    故全样本对照噪声底 = sqrt(fro_a² + fro_b²) / 2（在调用处合并）。
    """
    half = batch.meta.n_shots // 2
    first = SampleBatch(meta=batch.meta, samples=batch.samples[:half])
    meta2 = dataclasses.replace(batch.meta, n_shots=batch.meta.n_shots - half)
    second = SampleBatch(meta=meta2, samples=batch.samples[half:])
    diff = _corr_offdiag(first, offdiag) - _corr_offdiag(second, offdiag)
    return float(np.sqrt((diff**2).sum()))


def compare_backends(
    backend_a: GBSSampler,
    backend_b: GBSSampler,
    n_shots: int = 20000,
    *,
    subsystem_sizes: tuple[int, ...] = (3, 4),
    n_subsystems: int = 12,
    seed: int = 0,
) -> CompareReport:
    """对两个 GBS 采样后端做分布级对照（假定边际已按需对齐）。

    Args:
        backend_a / backend_b: 满足 GBSSampler 的两个后端，n_modes 须一致。
        n_shots: 每后端采样数（两后端独立采样）。
        subsystem_sizes: 子系统模式数（每个规模随机抽 ``n_subsystems`` 个）。
        n_subsystems: 每个规模抽取的子系统个数（超出组合数时取全部）。
        seed: 子系统抽样种子（两后端使用相同子系统，保证可比）。

    Returns:
        :class:`CompareReport`，含指标对照表与中文判读。
    """
    if backend_a.n_modes != backend_b.n_modes:
        raise ValueError(f"两后端 n_modes 不一致：{backend_a.n_modes} vs {backend_b.n_modes}")
    n = backend_a.n_modes
    batch_a = backend_a.sample(n_shots)
    batch_b = backend_b.sample(n_shots)
    xa = np.asarray(batch_a.samples, dtype=float)
    xb = np.asarray(batch_b.samples, dtype=float)

    # 1) 边际 click 率
    ma = xa.mean(axis=0)
    mb = xb.mean(axis=0)
    d_marg = np.abs(ma - mb)
    pooled = 0.5 * (ma + mb)
    marg_se = _MEAN_ABS_NORMAL * np.sqrt(2.0 * pooled * (1.0 - pooled) / n_shots)
    marg_l1_floor = float(marg_se.sum())

    # 2) 二阶联合 click 概率（非对角元与边际指标正交）
    ja = (xa.T @ xa) / n_shots
    jb = (xb.T @ xb) / n_shots
    offdiag = ~np.eye(n, dtype=bool)
    dj = (ja - jb)[offdiag]
    fro = float(np.sqrt((dj**2).sum()))
    pooled_j = 0.5 * (ja + jb)[offdiag]
    fro_floor = float(np.sqrt((2.0 * pooled_j * (1.0 - pooled_j) / n_shots).sum()))

    # 2b) click 相关系数矩阵差异（对边际失配免疫，隔离联合结构差异）
    ca = _corr_offdiag(batch_a, offdiag)
    cb = _corr_offdiag(batch_b, offdiag)
    corr_fro = float(np.sqrt(((ca - cb) ** 2).sum()))
    corr_floor = 0.5 * float(
        np.hypot(
            _corr_split_half_floor(batch_a, offdiag),
            _corr_split_half_floor(batch_b, offdiag),
        )
    )

    # 3) 子系统 click pattern 联合分布 TV
    rng = np.random.default_rng(seed)
    tv_by_k: dict[int, list[float]] = {}
    tv_floor_by_k: dict[int, list[float]] = {}
    for k in subsystem_sizes:
        if k > n:
            continue
        combos = list(itertools.combinations(range(n), k))
        take = min(n_subsystems, len(combos))
        chosen = rng.choice(len(combos), size=take, replace=False)
        tvs, floors = [], []
        for ci in chosen:
            idx = combos[int(ci)]
            pa = _empirical_probs(batch_a.samples, idx)
            pb = _empirical_probs(batch_b.samples, idx)
            tvs.append(float(0.5 * np.abs(pa - pb).sum()))
            floors.append(_tv_floor(0.5 * (pa + pb), n_shots))
        tv_by_k[k] = tvs
        tv_floor_by_k[k] = floors

    # 4) 汇总表与判读
    rows: list[dict[str, Any]] = []

    def add(metric: str, value: float, floor: float, note: str) -> None:
        rows.append(
            {
                "metric": metric,
                "value": value,
                "noise_floor": floor,
                "ratio": value / floor if floor > 0 else np.inf,
                "note": note,
            }
        )

    add("marginal_l1", float(d_marg.sum()), marg_l1_floor, "逐模式边际 click 率差异 L1")
    add("marginal_linf", float(d_marg.max()), float(marg_se.max()), "逐模式边际 click 率差异 L∞")
    add("marginal_mean_abs", float(d_marg.mean()), float(marg_se.mean()), "逐模式边际平均绝对差")
    add(
        "second_order_frobenius_offdiag",
        fro,
        fro_floor,
        "二阶联合 click 概率矩阵非对角元 Frobenius 差异",
    )
    add(
        "second_order_corr_frobenius_offdiag",
        corr_fro,
        corr_floor,
        "click 相关系数矩阵非对角元 Frobenius 差异（对边际失配免疫；split-half 噪声底）",
    )
    for k, tvs in tv_by_k.items():
        floors = tv_floor_by_k[k]
        add(
            f"tv_k{k}_mean",
            float(np.mean(tvs)),
            float(np.mean(floors)),
            f"{k} 模式子系统 click pattern 联合分布 TV（{len(tvs)} 个子系统均值）",
        )
        add(
            f"tv_k{k}_max",
            float(np.max(tvs)),
            float(np.mean(floors)),
            f"{k} 模式子系统 click pattern 联合分布 TV（最大值）",
        )
    table = pd.DataFrame(rows).set_index("metric")

    def verdict(ratio: float) -> str:
        if ratio <= 2.0:
            return "与采样噪声一致"
        if ratio <= 5.0:
            return "边缘超出噪声"
        return "显著结构性差异"

    lines = [
        f"分布级对照（n_shots={n_shots}，噪声底为两独立有限样本对照的一阶期望）：",
    ]
    for row in rows:
        lines.append(
            f"- {row['metric']}: {row['value']:.4g}（噪声底 {row['noise_floor']:.3g}，"
            f"比值 {row['ratio']:.1f}）→ {verdict(row['ratio'])}；{row['note']}"
        )
    interpretation = "\n".join(lines)

    labels = (batch_a.meta.device_version, batch_b.meta.device_version)
    return CompareReport(
        table=table,
        interpretation=interpretation,
        marginals_a=ma,
        marginals_b=mb,
        joint_a=ja,
        joint_b=jb,
        subsystem_tv=tv_by_k,
        n_shots=n_shots,
        labels=labels,
        extra={"tv_floor": tv_floor_by_k},
    )
