"""GBS 采样后端抽象与样本数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np


@dataclass(frozen=True)
class BatchMeta:
    """采样批次元数据。

    装置版本、采集区间、来源（真机/仿真）、系统效率、时空模式配置、
    探测器设置、标定会话与质量标记。
    """

    batch_id: str
    device_version: str  # "simulator-gaussian" / 真机装置编号
    n_modes: int
    n_shots: int
    started_at: str  # 采集起始 ISO 时间戳
    finished_at: str  # 采集结束 ISO 时间戳
    source: Literal["hardware", "simulation"]  # 样本来源
    calibration_session_id: str  # 标定会话 ID（版本边界：装置重启/复标定 → 新 major 版本）
    simulation_params: dict[str, Any] = field(default_factory=dict)  # 仿真方法与参数（如 {"method": "MPS", "chi": 1000}）
    system_efficiency: float | None = None  # η，连续质量指标（一阶影响因素）；未知为 None
    mode_config: dict[str, Any] = field(default_factory=dict)  # 时空模式结构（如 511×9、分区边界、click 剖面摘要）
    detector_config: dict[str, Any] = field(default_factory=dict)
    quality_flags: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleBatch:
    """一个批次的 click pattern 样本。

    samples 为 (n_shots, n_modes) 的 0/1 数组（或位压缩存储）。
    """

    meta: BatchMeta
    samples: np.ndarray


class GBSSampler(Protocol):
    """GBS 采样后端抽象：simulator / hardware / cached / permuted 可切换。"""

    def sample(self, n_shots: int) -> SampleBatch:
        """采集 ``n_shots`` 个 click pattern 样本。"""
        ...

    @property
    def n_modes(self) -> int:
        """模式数（编码后特征空间维度）。"""
        ...


def pack_samples(samples: np.ndarray) -> np.ndarray:
    """0/1 样本位压缩：(n_shots, n_modes) → (n_shots, ceil(n_modes/8)) uint8。"""
    return np.packbits(np.asarray(samples, dtype=np.uint8), axis=1)


def unpack_samples(packed: np.ndarray, n_modes: int) -> np.ndarray:
    """位压缩样本还原为 (n_shots, n_modes) 0/1 uint8 数组。"""
    return np.unpackbits(packed, axis=1, count=n_modes)
