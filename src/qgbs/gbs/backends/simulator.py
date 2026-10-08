"""高斯态仿真后端（固定实例近似采样器）。

**近似口径（重要）**：本后端不模拟完整 GBS 量子光学过程（那需要
hafnian/Torontonian 采样）。采用高斯 copula 阈值模型近似：

1. 固定实例：seed → 随机正交矩阵 U（模拟干涉结构）+ 各模式压缩参数
   s_k → 协方差 C = U·diag(tanh s)·Uᵀ → 归一化为相关矩阵 R；
2. 边际 click 率按时空模式包络设定（时间模式呈包络剖面，空间模式加权，
   整体按系统效率 η 缩放——损耗降低 click 率）；
3. 采样：Z ~ N(0, R)，click_j = 1[Z_j > Φ⁻¹(1−p_j)]，得到相关 0/1 模式。

该近似保留了"固定设备 + 时空二维模式 + click 剖面 + 联合结构"的机制
形态，用于开发调试与管线验证，**不替代真机**（真机结论需 hardware 后端 +
§7.1 诊断）。时空模式默认 32×9 以控制 demo 耗时，我方真机为 511×9
（3×3=9 空间模式；九章论文为 4×4=16，注意区分）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from scipy.stats import norm

from qgbs.gbs.backends.base import BatchMeta, SampleBatch


@dataclass(frozen=True)
class SimulatorConfig:
    """仿真器参数：固定高斯实例与采样配置。"""

    n_temporal: int = 32  # 时间模式数（论文 511）
    n_spatial: int = 9  # 空间模式数（我方真机 3×3=9；九章论文为 4×4=16）
    seed: int = 42  # 固定实例种子（干涉结构 + 压缩参数）
    system_efficiency: float = 0.7  # η：系统效率（损耗降低 click 率）
    mean_click_rate: float = 0.3  # 目标平均 click 率（η=1 时）
    method: str = "copula"  # "copula" 相关采样 / "independent" 独立 Bernoulli（仅用于大规模诊断）


class GaussianSimulatorBackend:
    """固定实例高斯 copula 阈值采样器（近似仿真，见模块 docstring）。

    同一 (seed, n_shots) 的 sample 调用结果完全可复现。
    """

    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config
        self._R: np.ndarray | None = None  # 相关矩阵
        self._chol: np.ndarray | None = None
        self._thresholds: np.ndarray | None = None
        self._marginals: np.ndarray | None = None
        self._profile_summary: dict[str, float] | None = None

    @property
    def n_modes(self) -> int:
        return self.config.n_temporal * self.config.n_spatial

    def _build_instance(self) -> None:
        """构造固定实例：干涉结构、压缩谱、边际包络、判定阈值。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        n = self.n_modes
        if cfg.method == "copula":
            # 随机正交干涉结构 + 压缩谱 → 相关矩阵
            u, _ = np.linalg.qr(rng.normal(0.0, 1.0, (n, n)))
            squeezing = rng.uniform(0.2, 1.0, n)
            cov = (u * np.tanh(squeezing)) @ u.T
            d = np.sqrt(np.diag(cov))
            self._R = cov / np.outer(d, d)
            self._chol = np.linalg.cholesky(self._R + 1e-10 * np.eye(n))
        elif cfg.method != "independent":
            raise ValueError(f"未知仿真方法: {cfg.method!r}")
        # 时间模式包络剖面（sin^k 包络 + 基底），空间模式随机权重
        t = np.arange(cfg.n_temporal)
        envelope = np.sin(np.pi * t / max(cfg.n_temporal - 1, 1)) ** 1.5 + 0.05
        spatial_w = rng.uniform(0.5, 1.5, cfg.n_spatial)
        profile = np.outer(envelope, spatial_w).ravel()  # mode = temporal * n_spatial + spatial
        p = cfg.mean_click_rate * cfg.system_efficiency * profile / max(float(profile.mean()), 1e-12)
        self._marginals = np.clip(p, 1e-4, 0.95)
        self._thresholds = norm.ppf(1.0 - self._marginals)
        # 时间模式 click 剖面摘要与分区边界推导（累计包络质量中位数）
        temporal_profile = profile.reshape(cfg.n_temporal, cfg.n_spatial).mean(axis=1)
        cum = np.cumsum(temporal_profile) / temporal_profile.sum()
        self._profile_summary = {
            "temporal_peak_mode": float(np.argmax(temporal_profile)),
            "temporal_mean_click": float(temporal_profile.mean()),
            "derived_partition_boundary": float(np.searchsorted(cum, 0.5) + 1),
        }

    def sample(self, n_shots: int) -> SampleBatch:
        """采样 ``n_shots`` 个 click pattern（同参数调用可复现）。"""
        if self._marginals is None:
            self._build_instance()
        assert self._thresholds is not None and self._marginals is not None
        rng = np.random.default_rng((self.config.seed, n_shots))
        if self.config.method == "independent":
            # 独立 Bernoulli 快速通道：无模式间相关，仅用于大规模诊断
            clicks = (rng.random((n_shots, self.n_modes)) < self._marginals).astype(np.uint8)
        else:
            assert self._chol is not None
            z = rng.standard_normal((n_shots, self.n_modes)) @ self._chol.T
            clicks = (z > self._thresholds).astype(np.uint8)
        now = datetime.now(timezone.utc).isoformat()
        cfg = self.config
        meta = BatchMeta(
            batch_id=(
                f"sim-{cfg.method}-{cfg.n_temporal}x{cfg.n_spatial}"
                f"-s{cfg.seed}-n{n_shots}-eta{cfg.system_efficiency}"
            ),
            device_version="simulator-gaussian-copula",
            n_modes=self.n_modes,
            n_shots=n_shots,
            started_at=now,
            finished_at=now,
            source="simulation",
            calibration_session_id=f"sim-session-{cfg.seed}",
            simulation_params={
                "method": f"gaussian_copula_threshold" if cfg.method == "copula" else "independent_bernoulli",
                "seed": cfg.seed,
                "mean_click_rate": cfg.mean_click_rate,
                "note": "近似仿真，不替代真机 GBS（hafnian/Torontonian 采样）",
            },
            system_efficiency=cfg.system_efficiency,
            mode_config={
                "n_temporal": cfg.n_temporal,
                "n_spatial": cfg.n_spatial,
                **(self._profile_summary or {}),
            },
        )
        return SampleBatch(meta=meta, samples=clicks)

    def set_state(self, covariance: np.ndarray) -> None:
        """设置自定义相关矩阵（覆盖固定实例）。

        首次 ``sample()`` 之前调用时先构建实例（边际包络/阈值沿用固定实例），
        再覆盖相关矩阵，保证自定义态不被随后的 ``_build_instance()`` 静默覆盖。

        用途边界（方案书 §3.3）：仅限开发调试与噪声诊断；正式实验
        一律使用固定实例采样器（编码态在实验设计阶段固定，实验中
        不逐期重设），不得用本方法在测试期动态改态。
        """
        if self.config.method != "copula":
            raise ValueError(
                f"set_state 仅适用于 copula 方法（当前 {self.config.method!r} 无相关结构可设置）"
            )
        if self._marginals is None:
            self._build_instance()  # 先建实例，避免自定义相关矩阵被构建覆盖
        sym = (np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T) / 2.0
        if sym.shape != (self.n_modes, self.n_modes):
            raise ValueError(f"相关矩阵形状 {sym.shape} 与 n_modes={self.n_modes} 不符")
        self._R = sym
        self._chol = np.linalg.cholesky(sym + 1e-10 * np.eye(sym.shape[0]))
