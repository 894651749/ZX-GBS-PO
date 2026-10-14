"""pattern 直方图引擎（E6 论文级忠实复现的核心计算）。

论文原版特征（arXiv:2506.19707）：每行寻址 16 模子集上的**联合点击 pattern
频率直方图**——把全部 shots 投影到该行寻址的 16 个模式上，每个 shot 得到
一个 16-bit pattern 索引，bincount 得 2^16 维直方图（B 组 10 模 → 2^10）。

存储约定与实现路径：真机采样流是**行打包**（packed.npy，每 shot 一行
1022 字节，np.packbits axis=1 big-endian）。本引擎提供两条路径：

- 行打包直提（默认）：模式 m 的位 = ``packed[:, m//8]`` 字节的第 ``7-m%8``
  位（big-endian 约定，与 unpack_samples 一致），跨步字节列读 + 移位组装，
  零预处理、无 5.1GB 转置拷贝；
- 列打包预转置（可选）：``ColumnPackedStore.from_row_packed`` 一次性转置为
  （8176 模式 × n_shots 位）列打包，取列为连续内存读。基准实测两路径择优。

内存纪律：单行直方图的中间量为 n_shots×16 bit（5M → 80MB uint8 视图）；
直方图输出 65536 维 float64 = 512KB。全程不物化全量解包样本。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _assemble_index(bits: np.ndarray) -> np.ndarray:
    """位矩阵（n_shots, k 的 0/1）→ pattern 索引（低位在前）。

    uint16 位平面组装（k ≤ 16 时流量比 uint32 减半；实测最快口径）。
    """
    k = bits.shape[1]
    dtype = np.uint16 if k <= 16 else np.uint32
    idx = np.zeros(bits.shape[0], dtype=dtype)
    for i in range(k):
        idx |= bits[:, i].astype(dtype) << np.uint16(i)
    return idx


def histogram_from_row_packed(packed: np.ndarray, modes: np.ndarray) -> np.ndarray:
    """单行直方图（行打包直提路径）。

    Args:
        packed: (n_shots, ceil(n_modes/8)) uint8 行打包采样流（可为 mmap）。
        modes: 该行寻址的模式编号（长度 ≤ 16，如 A 组 16 / B 组 10）。

    Returns:
        2^len(modes) 维 float64 直方图：索引为 pattern（第 i 位 = modes[i]
        是否点击，低位在前），值为经验频率。
    """
    modes = np.asarray(modes, dtype=np.int64)
    n_shots = packed.shape[0]
    bits = np.empty((n_shots, len(modes)), dtype=np.uint8)
    for i, m in enumerate(modes):
        bits[:, i] = (packed[:, m // 8] >> (7 - m % 8)) & 1
    idx = _assemble_index(bits)
    return np.bincount(idx, minlength=1 << len(modes)).astype(np.float64) / n_shots


@dataclass(frozen=True)
class ColumnPackedStore:
    """列打包采样存储：8176 个模式各存为 n_shots 位的位打包列。

    预转置一次（``from_row_packed`` 分块进行，峰值内存受控）；取 16 列为
    连续内存读，适合同一批 shots 上反复投影（E6 阶段 1/2 的批量行构建）。
    """

    cols: np.ndarray  # (n_modes, ceil(n_shots/8)) uint8
    n_shots: int
    n_modes: int

    @classmethod
    def from_row_packed(
        cls, packed: np.ndarray, n_modes: int, chunk_shots: int = 250_000
    ) -> "ColumnPackedStore":
        """行打包 → 列打包转置（分块控制峰值内存：块 ≈ chunk×n_modes 字节）。"""
        if chunk_shots % 8:
            raise ValueError("chunk_shots 须为 8 的倍数（位打包边界）")
        n_shots = packed.shape[0]
        n_bytes = (n_shots + 7) // 8
        cols = np.zeros((n_modes, n_bytes), dtype=np.uint8)
        for s in range(0, n_shots, chunk_shots):
            blk = np.unpackbits(packed[s : s + chunk_shots], axis=1, count=n_modes)
            packed_blk = np.packbits(blk.T, axis=1)
            cols[:, s // 8 : s // 8 + packed_blk.shape[1]] = packed_blk
        return cls(cols=cols, n_shots=n_shots, n_modes=n_modes)

    def histogram(self, modes: np.ndarray) -> np.ndarray:
        """单行直方图（列打包路径）：取列 → unpack → 移位组装 → bincount。

        位组装在行视图（连续内存）上进行，uint16 位平面（k≤16 时流量减半）——
        基准实测比"转置 + uint32"快约 2×。
        """
        modes = np.asarray(modes, dtype=np.int64)
        k = len(modes)
        bits = np.unpackbits(self.cols[modes], axis=1, count=self.n_shots)  # (k, n_shots)
        dtype = np.uint16 if k <= 16 else np.uint32
        idx = np.zeros(self.n_shots, dtype=dtype)
        for i in range(k):
            idx |= bits[i].astype(dtype) << np.uint16(i)
        return np.bincount(idx, minlength=1 << k).astype(np.float64) / self.n_shots


def top_n_patterns(hist: np.ndarray, n: int) -> np.ndarray:
    """直方图取 top-N 高频 pattern 索引（降序取前 N，返回规范升序索引）。"""
    order = np.lexsort((np.arange(len(hist)), -hist))[:n]
    return np.sort(order)
