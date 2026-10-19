"""GBS 风险先验（E8，机制 A）：把 LW 收缩目标从常相关矩阵换成 GBS 时变先验。

构造（设计文档 §2.2）：
- 每只股票 i 在调仓日 t 的 26 维特征经（各折 train 窗拟合冻结的）寻址器
  得到 26 个模式地址集合 A_i(t)；
- 量子相似性 K(i,j,t) = A_i(t) 与 A_j(t) 在真机相关矩阵 ρ̂ 上的跨组平均
  相关；矩阵化 K = U·ρ̂·Uᵀ/26²（U 为 n×8176 地址指示，每行 26 个 1）；
- 先验协方差 Σ_prior = D·K·D（D = train 窗已实现波动率对角阵），经
  对称化 → 最近 PSD 投影 → 尺度校准（对齐样本协方差对角均值）三步处理；
- 混合 Σ = δ·Σ_LW + (1−δ)·Σ_prior；δ=1 严格退化为现有 LW 路径（R0）。

防泄漏：ρ̂ 来自冻结真机批次（与金融数据无关）；寻址器各折 train 窗
拟合冻结；D 用 train 窗回看收益；δ 在 valid 窗择优。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qgbs.portfolio.risk import check_psd, nearest_psd


@dataclass(frozen=True)
class PriorMatrix:
    """调仓日的先验相关矩阵与诊断信息。"""

    corr: pd.DataFrame  # symbol × symbol 先验相关矩阵 K
    note: str = ""


def gbs_prior_corr(
    addr_by_symbol: pd.DataFrame,
    rho_hat: np.ndarray,
) -> pd.DataFrame:
    """K = U·ρ̂·Uᵀ/26²：地址集合在真机相关矩阵上的跨组平均相关。

    Args:
        addr_by_symbol: symbol × 特征 的寻址模式编号表（每行 26 个模式）。
        rho_hat: (n_modes, n_modes) 真机 click 相关矩阵（对角为 1）。
    """
    symbols = list(addr_by_symbol.index)
    modes = addr_by_symbol.to_numpy(dtype=np.int64)  # (n, 26)
    n, k = modes.shape
    # U·ρ̂·Uᵀ/676 的等价稀疏计算：K[i,j] = mean(ρ̂[np.ix_(A_i, A_j)])
    U = np.zeros((n, rho_hat.shape[0]), dtype=np.float64)
    U[np.arange(n)[:, None], modes] = 1.0
    K = U @ rho_hat @ U.T / float(k * k)
    K = (K + K.T) / 2.0
    return pd.DataFrame(K, index=symbols, columns=symbols)


def random_prior_corr(symbols: list[str], seed: int = 42) -> pd.DataFrame:
    """随机先验（R2 归因对照）：随机 PSD 相关矩阵，seed 固定可复现。"""
    rng = np.random.default_rng(seed)
    n = len(symbols)
    a = rng.normal(size=(n, n))
    cov = a @ a.T
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    return pd.DataFrame((corr + corr.T) / 2.0, index=symbols, columns=symbols)


def static_prior_corr(daily: list[pd.DataFrame]) -> pd.DataFrame:
    """静态 GBS 先验（R3' 消融）：各调仓日 K 的时间平均。"""
    if not daily:
        raise ValueError("daily 为空")
    mean = sum(daily) / len(daily)
    return pd.DataFrame((mean + mean.T) / 2.0, index=daily[0].index, columns=daily[0].columns)


def prior_covariance(
    corr: pd.DataFrame,
    vols: pd.Series,
    ref_var_mean: float | None = None,
) -> np.ndarray:
    """先验协方差 Σ_prior = D·K·D + 三步处理（对称化/PSD 投影/尺度校准）。

    Args:
        corr: 先验相关矩阵 K（symbol × symbol）。
        vols: 各股 train 窗已实现波动率（对角阵 D 的内容）。
        ref_var_mean: 尺度校准参照（样本协方差对角方差均值）；提供时把
            Σ_prior 整体缩放使对角方差均值对齐参照（沿用 FactorRiskModel 口径）。
    """
    syms = corr.index
    d = vols.reindex(syms).fillna(float(vols.median())).to_numpy(dtype=float)
    cov = (corr.to_numpy(dtype=float) * np.outer(d, d))
    cov = (cov + cov.T) / 2.0
    if not check_psd(cov):
        cov = nearest_psd(cov)
    if ref_var_mean is not None and np.isfinite(ref_var_mean) and ref_var_mean > 0:
        synth = float(np.mean(np.diag(cov)))
        if synth > 0:
            cov = cov * (ref_var_mean / synth)
    return cov


def mix_covariance(
    sample_cov: np.ndarray,
    prior_cov: np.ndarray,
    delta: float,
) -> np.ndarray:
    """Σ = δ·Σ_LW + (1−δ)·Σ_prior；δ=1 严格退化为 LW 输入本身。"""
    if not 0.0 <= delta <= 1.0:
        raise ValueError(f"delta 须在 [0,1]，收到 {delta}")
    out = delta * np.asarray(sample_cov, dtype=float) + (1.0 - delta) * np.asarray(prior_cov, dtype=float)
    return (out + out.T) / 2.0


class PriorProvider:
    """调仓日先验协方差供给（引擎回调）：(date, sleeve, hist) → Σ_prior。

    kind ∈ {"gbs_prior", "random_prior", "static_gbs_prior"}：
    - gbs_prior：当日地址集合在 ρ̂ 上的跨组平均相关（时变）；
    - random_prior：随机 PSD 相关矩阵（seed 固定；归因对照）；
    - static_gbs_prior：K_static 时间平均版（消融时变性；由调用方在
      train+valid 调仓日上预平均，防泄漏——不含 test 日地址）。
    """

    def __init__(
        self,
        kind: str,
        addr_by_date: dict[pd.Timestamp, pd.DataFrame] | None = None,
        rho_hat: np.ndarray | None = None,
        static_corr: pd.DataFrame | None = None,
        seed: int = 42,
    ) -> None:
        if kind not in ("gbs_prior", "random_prior", "static_gbs_prior"):
            raise ValueError(f"未知先验类型: {kind!r}")
        self.kind = kind
        self.addr_by_date = addr_by_date or {}
        self.rho_hat = rho_hat
        self.static_corr = static_corr
        self.seed = seed

    def __call__(self, d: pd.Timestamp, sleeve: list[str], hist: pd.DataFrame) -> np.ndarray:
        if self.kind == "random_prior":
            K = random_prior_corr(sleeve, seed=self.seed)
        elif self.kind == "static_gbs_prior":
            assert self.static_corr is not None
            arr = self.static_corr.reindex(index=sleeve, columns=sleeve).fillna(0.0).to_numpy(copy=True)
            # 缺失 symbol（整行 0）的对角按相关矩阵约定补 1
            diag = np.diag(arr).copy()
            np.fill_diagonal(arr, np.where(diag == 0.0, 1.0, diag))
            K = pd.DataFrame(arr, index=sleeve, columns=sleeve)
        else:
            assert self.rho_hat is not None
            addr = self.addr_by_date[d].reindex(sleeve)
            K = gbs_prior_corr(addr, self.rho_hat)
        vols = hist.std(axis=0)
        ref = float(hist.var(axis=0).mean())
        return prior_covariance(K, vols, ref_var_mean=ref)


def build_static_corr(
    addr_by_date: dict[pd.Timestamp, pd.DataFrame],
    rho_hat: np.ndarray,
    dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """静态先验：给定日期集合（train+valid 调仓日）上 K 的时间平均。"""
    daily = [gbs_prior_corr(addr_by_date[d], rho_hat) for d in dates if d in addr_by_date]
    return static_prior_corr(daily)
