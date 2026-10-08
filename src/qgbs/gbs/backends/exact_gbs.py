"""精确小规模 threshold GBS 采样后端（真机分布的数值基准）。

物理模型（零位移高斯态 → 阈值探测器）：

1. M 个单模压缩真空态（SMSV），压缩参数 r_j 可逐模式不同
   （x 正交分量压缩，V_in = diag(e^{-2r}, e^{2r})，ℏ=2 约定下真空 V=I）；
2. Haar 随机酉干涉 U（scipy.stats.unitary_group，seed 固定 → 固定实例），
   辛变换 S = [[Re U, −Im U], [Im U, Re U]]；
3. 各模式均匀损耗 η：V = η·(S·V_in·Sᵀ) + (1−η)·I（真空环境）；
4. 阈值探测器：click_j = 1[模式 j 至少一个光子]。

采样算法（精确，非近似）：M ≤ ``max_modes``（默认 10）时枚举全部 2^M 个
click pattern，用 Torontonian 公式（Quesada et al., PRA 98, 062322）计算
精确概率：

    Q = Qmat(V)（Husimi Q 矩阵），O = I − Q⁻¹，
    P(∅) = 1/√det(Q)，P(S) = Tor(O_S)/√det(Q)，

再从该分类分布中 i.i.d. 采样。实现已通过数值恒等式自验
（概率和 = 1 至 1e-15；与 Fock 空间 state_vector/density_matrix
独立计算一致至截断误差 ~1e-5；纯态特例与 sech(r) 解析值一致）。

**与 copula 仿真器的关系**：:mod:`~qgbs.gbs.backends.simulator` 的高斯
copula 阈值采样可表达任意 pairwise 相关，而真机 threshold GBS 的联合分布
受 Torontonian/hafnian 物理约束，两者可达集合不同。本后端是真机接入前
（P3）做**分布级对照验证**的基准：见 :mod:`qgbs.gbs.distribution_compare`。

**规模约束**：枚举复杂度 ~3^M，仅限小模式数（默认 6，上限 10）；更大规模
请走 copula 仿真或真机后端（实例化时显式报错）。时空模式标签
（n_temporal × n_spatial）仅作分区语义，小实例不受论文 511×16 规模约束（我方真机为 511×9）。

**运行依赖**：``thewalrus``（pyproject ``gbs-sim`` extras 已声明；
未安装时实例化会显式报错，模块本身可导入）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from qgbs.gbs.backends.base import BatchMeta, SampleBatch

try:  # thewalrus 为可选依赖（gbs-sim extras）
    import thewalrus
    from thewalrus import tor as _torontonian
    from thewalrus.quantum import Qmat as _qmat

    _THEWALRUS_ERROR: str | None = None
except ImportError:  # pragma: no cover - 取决于环境
    thewalrus = None  # type: ignore[assignment]
    _torontonian = None  # type: ignore[assignment]
    _qmat = None  # type: ignore[assignment]
    _THEWALRUS_ERROR = (
        "ExactGBSSampler 依赖 thewalrus（Torontonian/Qmat），"
        "请运行 `pip install thewalrus` 或安装 gbs-sim extras"
    )


@dataclass(frozen=True)
class ExactGBSConfig:
    """精确 GBS 采样器配置：固定实例 + 采样配置。

    时空模式标签仅提供分区语义（mode = temporal × n_spatial + spatial，
    与 simulator 一致）；小实例不受论文 511×16 规模约束（我方真机为 511×9）。
    """

    n_temporal: int = 3  # 时间模式数（标签语义）
    n_spatial: int = 2  # 空间模式数（标签语义）
    seed: int = 42  # 固定实例种子（Haar 干涉结构 + 压缩参数谱）
    sample_seed: int | None = None  # 采样种子；None 时与实例种子同源。
    # 同实例不同采样噪声（统计功效检验）需显式设置不同的 sample_seed
    system_efficiency: float = 0.9  # η：各模式均匀损耗后的系统效率
    squeezing_low: float = 0.2  # 压缩参数 r_j ~ U(low, high)
    squeezing_high: float = 1.0
    max_modes: int = 10  # 精确枚举的模式数上限（2^M pattern 空间）


class ExactGBSSampler:
    """精确小规模 threshold GBS 采样器（满足 GBSSampler 结构式 Protocol）。

    同一 (seed, sample_seed, n_shots) 的 sample 调用结果完全可复现；
    ``pattern_probabilities()`` 给出枚举空间上的精确联合分布，
    可作为分布级对照的无采样噪声基准。
    """

    def __init__(self, config: ExactGBSConfig) -> None:
        if _THEWALRUS_ERROR is not None:
            raise ImportError(_THEWALRUS_ERROR)
        n = config.n_temporal * config.n_spatial
        if n < 1:
            raise ValueError(f"n_modes 须 ≥ 1，当前 {n}")
        if n > config.max_modes:
            raise ValueError(
                f"精确枚举要求 n_modes ≤ max_modes={config.max_modes}，当前 {n}；"
                "大规模请走 copula 仿真（GaussianSimulatorBackend）或真机后端"
            )
        if not 0.0 < config.system_efficiency <= 1.0:
            raise ValueError(f"system_efficiency 须落在 (0, 1]，当前 {config.system_efficiency}")
        if not 0.0 <= config.squeezing_low <= config.squeezing_high:
            raise ValueError("squeezing 区间非法")
        self.config = config
        self._built = False
        self._squeezing: np.ndarray
        self._interferometer: np.ndarray
        self._covariance: np.ndarray
        self._probs: np.ndarray | None = None
        self._marginals: np.ndarray | None = None

    # ------------------------------------------------------------------
    # 固定实例构造（确定性）
    # ------------------------------------------------------------------

    def _build_instance(self) -> None:
        """构造固定实例：压缩谱、Haar 干涉、损耗后的协方差矩阵。"""
        from scipy.stats import unitary_group

        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        n = self.n_modes
        self._squeezing = rng.uniform(cfg.squeezing_low, cfg.squeezing_high, n)
        self._interferometer = unitary_group.rvs(n, random_state=rng)
        u = self._interferometer
        # xp 排序 (x_1..x_M, p_1..p_M)，ℏ=2（真空 V = I）
        v_in = np.diag(np.concatenate([np.exp(-2.0 * self._squeezing), np.exp(2.0 * self._squeezing)]))
        s = np.block([[u.real, -u.imag], [u.imag, u.real]])
        v_pure = s @ v_in @ s.T
        eta = cfg.system_efficiency
        self._covariance = eta * v_pure + (1.0 - eta) * np.eye(2 * n)
        self._built = True

    def _ensure_built(self) -> None:
        if not self._built:
            self._build_instance()

    # ------------------------------------------------------------------
    # 精确分布
    # ------------------------------------------------------------------

    def pattern_probabilities(self) -> np.ndarray:
        """全部 2^M 个 click pattern 的精确概率（Torontonian 公式）。

        Returns:
            形状 (2^M,) 的概率向量；pattern 索引 s 的第 j 位对应模式 j
            的 click（s = Σ_j click_j·2^j）。概率和 = 1（数值残差 < 1e-12，
            返回前已归一化）。
        """
        if self._probs is not None:
            return self._probs
        self._ensure_built()
        n = self.n_modes
        q = _qmat(self._covariance)  # Husimi Q 矩阵（复 (α*, α) 表象）
        o = np.eye(2 * n) - np.linalg.inv(q)
        p_vac = 1.0 / np.sqrt(np.linalg.det(q))  # P(∅) = 1/√det(Q)
        probs = np.empty(2**n)
        for s in range(2**n):
            clicked = [j for j in range(n) if (s >> j) & 1]
            if not clicked:
                probs[s] = p_vac.real
                continue
            # 模式 j click → 取 Q 表象中第 j 与 j+M 两行/列
            sub = clicked + [j + n for j in clicked]
            val = _torontonian(o[np.ix_(sub, sub)]) * p_vac
            assert abs(val.imag) < 1e-8, f"Torontonian 概率虚部异常: {val}"
            probs[s] = val.real
        # 数值守卫：截断微小负数并归一化（概率和恒等式自验）
        if probs.min() < -1e-9:
            raise ArithmeticError(f"出现显著负概率（min={probs.min():.3e}），数值异常")
        probs = np.clip(probs, 0.0, None)
        total = float(probs.sum())
        if abs(total - 1.0) > 1e-9:
            raise ArithmeticError(f"pattern 概率和偏离 1：{total!r}")
        self._probs = probs / total
        return self._probs

    def exact_marginals(self) -> np.ndarray:
        """逐模式精确边际 click 率（由精确联合分布聚合，无采样噪声）。"""
        if self._marginals is None:
            probs = self.pattern_probabilities()
            n = self.n_modes
            patterns = np.arange(2**n, dtype=np.int64)
            bits = ((patterns[:, None] >> np.arange(n)) & 1).astype(float)
            self._marginals = bits.T @ probs  # = Σ_s p_s · bits[s]
        return self._marginals

    @property
    def covariance(self) -> np.ndarray:
        """损耗后高斯态的 2M×2M xp 协方差矩阵（ℏ=2 约定）。"""
        self._ensure_built()
        return self._covariance

    @property
    def squeezing(self) -> np.ndarray:
        """各模式压缩参数 r_j（实例固定）。"""
        self._ensure_built()
        return self._squeezing

    @property
    def interferometer(self) -> np.ndarray:
        """Haar 随机干涉酉矩阵 U（实例固定）。"""
        self._ensure_built()
        return self._interferometer

    # ------------------------------------------------------------------
    # GBSSampler 协议
    # ------------------------------------------------------------------

    @property
    def n_modes(self) -> int:
        return self.config.n_temporal * self.config.n_spatial

    def sample(self, n_shots: int) -> SampleBatch:
        """从精确分类分布采样 ``n_shots`` 个 click pattern（同参数可复现）。"""
        probs = self.pattern_probabilities()
        n = self.n_modes
        cfg = self.config
        eff_sample_seed = cfg.sample_seed if cfg.sample_seed is not None else cfg.seed
        rng = np.random.default_rng((cfg.seed, eff_sample_seed, n_shots))
        idx = rng.choice(2**n, size=n_shots, p=probs)
        samples = ((idx[:, None] >> np.arange(n)) & 1).astype(np.uint8)
        now = datetime.now(timezone.utc).isoformat()
        meta = BatchMeta(
            batch_id=(
                f"exact-gbs-{cfg.n_temporal}x{cfg.n_spatial}"
                f"-s{cfg.seed}-ss{eff_sample_seed}-n{n_shots}-eta{cfg.system_efficiency}"
            ),
            device_version="simulator-exact-gbs",
            n_modes=n,
            n_shots=n_shots,
            started_at=now,
            finished_at=now,
            source="simulation",
            calibration_session_id=f"exact-gbs-session-{cfg.seed}",
            simulation_params={
                "method": "exact_gbs",
                "engine": f"thewalrus-{thewalrus.__version__}",
                "formula": "P(S) = Tor(O_S)/sqrt(det(Q))（Quesada PRA 98, 062322）",
                "seed": cfg.seed,
                "sample_seed": eff_sample_seed,
                "squeezing": [round(float(r), 6) for r in self._squeezing],
                "squeezing_range": [cfg.squeezing_low, cfg.squeezing_high],
                "max_modes": cfg.max_modes,
                "note": "枚举 2^M 精确概率采样，真机接入前的分布级基准；仅适用小模式数",
            },
            system_efficiency=cfg.system_efficiency,
            mode_config={"n_temporal": cfg.n_temporal, "n_spatial": cfg.n_spatial},
        )
        return SampleBatch(meta=meta, samples=samples)
