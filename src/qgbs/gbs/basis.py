"""高阶候选基选择（论文 arXiv:2506.19707 流程的参数化）。

论文流程：随机抽 n1 个训练样本 → 各取 top-n2 计算基 → 合并取 top-N。
默认参数 (n1=1000, n2=2000, N=3000)；demo 用小参数（如 n1=200, n2=8, N=50）。

我方设计决策（论文未明确，显式记录）：

- 每个被抽中样本（click pattern）贡献其点击模式集合中按批次边际频率
  排序的 top-n2 个模式，候选基为这些模式的所有 order 阶组合；
  单样本组合数超过 ``max_combos_per_shot`` 时随机抽取组合（防爆炸）；
- 合并去重语义（``dedup_rule``）：组合先排序成规范元组再去重，
  按跨样本共现次数降序取 top-N；
- 可选跨批次稳定性筛选（方案书 §4.2）：提供 ≥2 个稳定性批次时，剔除
  各批次联合 click 频率相对标准差超 ``stability_max_rel_std`` 的基。
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np

from qgbs.gbs.backends.base import SampleBatch

DEDUP_RULE = "组合排序为规范元组后去重，按跨样本共现次数降序取 top-N"


@dataclass(frozen=True)
class BasisSelectionParams:
    """高阶基选择参数（论文默认值；demo 用小参数）。"""

    n1: int = 1000  # 随机抽取的训练样本数
    n2: int = 2000  # 每个样本保留的 top 点击模式数
    N: int = 3000  # 合并后保留 top-N
    order: int = 3  # 高阶统计阶数
    seed: int = 42  # 抽样随机种子
    max_combos_per_shot: int = 5000  # 单样本组合数安全上限
    stability_max_rel_std: float = 0.5  # 跨批次频率相对标准差（std/mean）上限；仅在提供 ≥2 个稳定性批次时生效


@dataclass(frozen=True)
class CandidateBasis:
    """高阶候选基集合及其 provenance 记录。"""

    basis: np.ndarray  # (n_basis, order) 全局模式编号组合
    sample_window: tuple[str, str]  # 采样批次的采集时间范围（取自批次 meta.started_at/finished_at，非金融训练窗口）
    params: BasisSelectionParams  # 选择参数记录
    mode_group: tuple[int, ...]  # 作用的局部模式组
    dedup_rule: str  # 合并去重语义（论文模糊，我方设计决策，须显式记录）


def _combo_frequencies(samples: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """各基组合在一组样本中的联合 click 频率，形状 (n_basis,)。"""
    Xb = np.asarray(samples, dtype=bool)
    n = max(Xb.shape[0], 1)
    freqs = np.zeros(len(basis), dtype=float)
    for k, combo in enumerate(basis):
        freqs[k] = float(Xb[:, list(combo)].all(axis=1).sum()) / n
    return freqs


def stability_filter_basis(
    basis: CandidateBasis,
    batches: list[SampleBatch],
    max_rel_std: float | None = None,
) -> CandidateBasis:
    """跨批次稳定性筛选（方案书 §4.2）：剔除相对标准差超阈值的基。

    对每个候选基计算其在各批次的联合 click 频率，相对标准差（std/mean）
    超过阈值的基被剔除；批次内平均频率为 0 的基视为无信号一并剔除。

    Args:
        basis: 待筛选的候选基。
        batches: 稳定性评估批次；不足 2 个时不筛选，原样返回。
        max_rel_std: 相对标准差上限；缺省取 ``basis.params.stability_max_rel_std``。
    """
    if len(batches) < 2 or len(basis.basis) == 0:
        return basis
    threshold = basis.params.stability_max_rel_std if max_rel_std is None else max_rel_std
    freqs = np.stack([_combo_frequencies(b.samples, basis.basis) for b in batches])
    mean = freqs.mean(axis=0)
    std = freqs.std(axis=0)
    rel_std = np.where(mean > 0, std / np.where(mean > 0, mean, 1.0), np.inf)
    keep = rel_std <= threshold
    return dataclasses.replace(basis, basis=basis.basis[keep])


def select_higher_order_basis(
    train_samples: SampleBatch,
    mode_group: tuple[int, ...],
    params: BasisSelectionParams | None = None,
    stability_batches: list[SampleBatch] | None = None,
) -> CandidateBasis:
    """在训练样本上选择局部高阶候选基（仅允许训练期调用）。

    Args:
        stability_batches: 可选的跨批次稳定性筛选样本（方案书 §4.2）。
            提供 ≥2 个批次时，按各批次联合 click 频率的相对标准差
            （阈值 ``params.stability_max_rel_std``）剔除不稳定基，
            最终基数可能少于 ``params.N``。
    """
    params = params or BasisSelectionParams()
    group = np.asarray(mode_group, dtype=int)
    samples = np.asarray(train_samples.samples)[:, group]
    marginals = samples.mean(axis=0)
    n_shots = samples.shape[0]

    rng = np.random.default_rng(params.seed)
    shot_idx = rng.choice(n_shots, size=min(params.n1, n_shots), replace=False)

    counts: Counter[tuple[int, ...]] = Counter()
    for row in samples[shot_idx]:
        clicked = np.flatnonzero(row)
        if len(clicked) < params.order:
            continue
        top = clicked[np.argsort(-marginals[clicked])[: params.n2]]
        top_sorted = np.sort(top)
        n_combos = comb(len(top_sorted), params.order)
        if n_combos <= params.max_combos_per_shot:
            for combo in combinations(top_sorted.tolist(), params.order):
                counts[combo] += 1
        else:  # 组合爆炸防护：随机抽样组合
            for _ in range(params.max_combos_per_shot):
                combo = tuple(sorted(rng.choice(top_sorted, size=params.order, replace=False).tolist()))
                counts[combo] += 1

    top_n = counts.most_common(params.N)
    basis_local = np.array([list(c) for c, _ in top_n], dtype=int) if top_n else np.empty((0, params.order), dtype=int)
    basis_global = group[basis_local] if len(basis_local) else basis_local

    meta = train_samples.meta
    selected = CandidateBasis(
        basis=basis_global,
        sample_window=(meta.started_at, meta.finished_at),
        params=params,
        mode_group=tuple(int(g) for g in mode_group),
        dedup_rule=DEDUP_RULE,
    )
    if stability_batches is not None and len(stability_batches) >= 2:
        selected = stability_filter_basis(selected, stability_batches)
    return selected
