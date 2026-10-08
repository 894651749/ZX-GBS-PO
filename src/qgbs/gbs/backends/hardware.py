"""真机 SDK 适配器骨架。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qgbs.gbs.backends.base import SampleBatch


@dataclass(frozen=True)
class HardwareConfig:
    """真机连接与采样配置。"""

    device_id: str  # 真机装置编号
    n_modes: int
    endpoint: str = ""  # SDK 接入地址
    credentials_ref: str = ""  # 凭证引用（不直接存放密钥）
    detector_config: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 600.0  # 单次采样超时


class HardwareBackend:
    """真机 GBS 装置的 GBSSampler 适配器骨架。

    负责 SDK 会话管理、任务提交、结果回收与质量标记采集；
    所有批次落库到样本库，供 cached 后端复用。
    """

    def __init__(self, config: HardwareConfig) -> None:
        self.config = config

    def sample(self, n_shots: int) -> SampleBatch:
        raise NotImplementedError

    @property
    def n_modes(self) -> int:
        return self.config.n_modes

    def health_check(self) -> dict[str, Any]:
        """装置连通性与状态检查，返回质量标记候选。"""
        raise NotImplementedError
