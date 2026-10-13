"""GBS 样本统计量：边际 / 二阶 / 局部高阶 click pattern 统计与批次诊断。"""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse

from qgbs.gbs.backends.base import SampleBatch
from qgbs.gbs.basis import CandidateBasis


def marginal_stats(batch: SampleBatch, calibration: np.ndarray | None = None) -> np.ndarray:
    """一阶边际统计：每个模式的 click 概率，形状 (n_modes,)。

    Args:
        calibration: 经装置标定的各模式响应系数（标定会话产物）。
            提供时返回标准化响应（click 概率除以标定响应）；为空返回原始边际。
    """
    # 整型累加再转 float：与 mean 数值一致，避免 0/1 大矩阵物化为 float64
    X = np.asarray(batch.samples)
    marg = X.sum(axis=0, dtype=np.uint64).astype(float) / X.shape[0]
    if calibration is not None:
        cal = np.asarray(calibration, dtype=float)
        marg = np.where(cal > 0, marg / np.where(cal > 0, cal, 1.0), 0.0)
    return marg


_SECOND_ORDER_CHUNK_ROWS = 20_000  # 分块行数（float32 块 ≈ 0.65GB @8176 模式）


def _gram_and_colsum(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """分块累加 Gram 矩阵 G = XᵀX（float64）与列和（uint64），避免全量 float 物化。

    0/1 样本下块内部分和为不超过块行数的整数（< 2^24），块转 float32 的
    matmul 累加无舍入，G 各元素为精确整数；分块与全量结果 bit 级一致。
    """
    n, n_modes = X.shape
    G = np.zeros((n_modes, n_modes), dtype=np.float64)
    colsum = np.zeros(n_modes, dtype=np.uint64)
    for s in range(0, n, _SECOND_ORDER_CHUNK_ROWS):
        blk = X[s : s + _SECOND_ORDER_CHUNK_ROWS]
        colsum += blk.sum(axis=0, dtype=np.uint64)
        G += blk.astype(np.float32).T @ blk
    return G, colsum


def second_order_stats(
    batch: SampleBatch,
    kind: Literal["joint", "covariance", "correlation"] = "joint",
) -> np.ndarray:
    """二阶统计，形状 (n_modes, n_modes)。

    分块累加 G = XᵀX 后按口径推导：joint = G/n；协方差 = (G − ccᵀ/n)/(n−1)
    （与 np.cov 的 ddof=1 口径一致）；correlation 零方差模式处置为 0。

    Args:
        kind: "joint" 联合 click 概率；"covariance" 模式对协方差；
            "correlation" 相关系数/归一化关联量（零方差模式处置为 0）。
    """
    X = np.asarray(batch.samples)
    n = X.shape[0]
    G, colsum = _gram_and_colsum(X)
    if kind == "joint":
        return G / n
    c = colsum.astype(np.float64)
    cov = (G - np.outer(c, c) / n) / (n - 1)
    if kind == "covariance":
        return cov
    if kind == "correlation":
        std = np.sqrt(np.diag(cov))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = cov / np.outer(std, std)
        return np.where(np.isfinite(corr), corr, 0.0)
    raise ValueError(f"未知二阶统计类型: {kind!r}")


def higher_order_stats(
    batch: SampleBatch,
    basis: CandidateBasis,
) -> sparse.spmatrix:
    """局部高阶统计：在 :class:`CandidateBasis` 选出的基组合上计算联合 click 频率。

    Returns:
        1 × n_basis 的 CSR 稀疏矩阵（基索引 → 联合 click 频率）。
    """
    X = np.asarray(batch.samples, dtype=bool)
    n = X.shape[0]
    rates = np.zeros(len(basis.basis), dtype=float)
    for k, combo in enumerate(basis.basis):
        rates[k] = float(X[:, list(combo)].all(axis=1).sum()) / n
    return sparse.csr_matrix(rates[None, :])


def batch_dispersion(batches: list[SampleBatch]) -> pd.DataFrame:
    """跨批次统计量离散度：同一编码态在多个批次上的边际/二阶统计离散度。

    Returns:
        索引为统计量坐标（"m_{j}" 边际 / "c_{i}_{j}" 二阶上三角），
        列为 mean/std/min/max。
    """
    if not batches:
        raise ValueError("batches 为空")
    margs = np.stack([marginal_stats(b) for b in batches])
    joints = np.stack([second_order_stats(b, "joint") for b in batches])
    n_modes = margs.shape[1]
    iu = np.triu_indices(n_modes, k=1)

    stats: dict[str, pd.DataFrame] = {}

    def _summarize(arr: np.ndarray, names: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "mean": arr.mean(axis=0),
                "std": arr.std(axis=0),
                "min": arr.min(axis=0),
                "max": arr.max(axis=0),
            },
            index=pd.Index(names, name="stat"),
        )

    marg_df = _summarize(margs, [f"m_{j}" for j in range(n_modes)])
    joint_flat = joints[:, iu[0], iu[1]]
    joint_df = _summarize(joint_flat, [f"c_{i}_{j}" for i, j in zip(*iu)])
    return pd.concat([marg_df, joint_df])


def subbatch_resample(batch: SampleBatch, n_sub: int, seed: int = 42) -> list[SampleBatch]:
    """子批次重采样：将一批样本随机划分为 ``n_sub`` 个无重叠子批次。

    用于 GBS 采样敏感性分析（统计量随 shot 数的波动）。
    """
    if n_sub < 1:
        raise ValueError("n_sub 须 ≥ 1")
    rng = np.random.default_rng(seed)
    order = rng.permutation(batch.meta.n_shots)
    subs = []
    for k, idx in enumerate(np.array_split(order, n_sub)):
        meta = dataclasses.replace(
            batch.meta,
            batch_id=f"{batch.meta.batch_id}-sub{k}",
            n_shots=len(idx),
        )
        subs.append(SampleBatch(meta=meta, samples=batch.samples[np.sort(idx)]))
    return subs
