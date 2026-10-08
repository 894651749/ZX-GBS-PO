"""风险模型：Ledoit-Wolf 收缩协方差 / 因子风险模型，PSD 检查与尺度校准。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf 收缩协方差估计（经典基线 C1 的风险口径）。

    Args:
        returns: 日收益宽表（date × symbol），含 NaN 的列须先剔除。

    Returns:
        (n, n) 收缩协方差矩阵（PD 保证优于样本协方差）。
    """
    lw = LedoitWolf()
    lw.fit(returns.to_numpy())
    return lw.covariance_


@dataclass(frozen=True)
class FactorRiskConfig:
    """因子风险模型配置。"""

    factor_cov_method: str = "ewma"  # 因子协方差估计："sample" / "ewma"
    ewma_halflife: float = 60.0  # EWMA 半衰期（交易日）
    min_obs: int = 30  # 截面回归所需的最少观测日数
    scale_calibrate: bool = True  # 尺度校准：合成协方差对角线方差均值对齐同一估计窗样本口径


class FactorRiskModel:
    """因子风险模型（Barra 式截面回归）：Σ = B·Σf·Bᵀ + D。

    - 暴露矩阵 B：行业哑变量 + 可选风格因子暴露（symbol × factor，
      由调用方以 ``exposures`` 传入）；
    - 因子收益 f_t：逐日截面回归 r_t = B·f_t + ε_t 估计；
    - 因子协方差 Σf：因子收益的样本协方差或 EWMA（half-life 参数化）；
    - 特质风险 D：回归残差方差（对角阵）。

    fit 仅允许在训练窗口调用。输出协方差前依次完成方案书 §6.1 三步处理：
    对称化 → 最近半正定投影 → 尺度校准（``scale_calibrate`` 开启时，整体
    缩放使对角线方差均值等于 fit 窗口内样本协方差的对角线方差均值；
    参照缺失/无效时跳过并记录于 ``scale_note_``）。
    """

    def __init__(self, config: FactorRiskConfig | None = None) -> None:
        self.config = config or FactorRiskConfig()
        self._fitted: bool = False
        self._B: pd.DataFrame | None = None  # symbol × factor
        self._factor_cov: np.ndarray | None = None
        self._specific_var: pd.Series | None = None  # 逐股票特质方差
        self._factor_returns: pd.DataFrame | None = None  # 估计的因子收益（诊断用）
        self._ref_var_mean: float | None = None  # fit 窗内样本协方差对角线方差均值（校准参照）
        self._scale_factor: float | None = None  # 最近一次 covariance() 的校准系数（None=未校准）
        self._scale_note: str | None = None  # 校准跳过说明

    def fit(self, returns: pd.DataFrame, exposures: pd.DataFrame) -> "FactorRiskModel":
        """在训练窗内估计 Σf 与 D。

        Args:
            returns: 日收益宽表（date × symbol）。
            exposures: 暴露矩阵 B（symbol × factor），行业哑变量 + 可选风格暴露。
        """
        B = exposures.reindex(index=returns.columns).fillna(0.0).astype(float)
        Bv = B.to_numpy()
        k = Bv.shape[1]
        f_rows: dict[pd.Timestamp, np.ndarray] = {}
        resid_sq_sum = pd.Series(0.0, index=returns.columns)
        resid_cnt = pd.Series(0, index=returns.columns, dtype=int)
        for t, r in returns.iterrows():
            mask = r.notna().to_numpy()
            if int(mask.sum()) < k + 2:
                continue
            beta, *_ = np.linalg.lstsq(Bv[mask], r.to_numpy()[mask], rcond=None)
            f_rows[t] = beta
            eps = r.to_numpy()[mask] - Bv[mask] @ beta
            resid_sq_sum[mask] += eps**2
            resid_cnt[mask] += 1
        if len(f_rows) < self.config.min_obs:
            raise ValueError(
                f"有效观测日 {len(f_rows)} 少于 min_obs={self.config.min_obs}，无法拟合因子风险模型"
            )
        F = pd.DataFrame(f_rows).T  # date × factor
        F.columns = list(B.columns)

        if self.config.factor_cov_method == "ewma":
            decay = 0.5 ** (1.0 / self.config.ewma_halflife)
            w = decay ** np.arange(len(F) - 1, -1, -1)
            w = w / w.sum()
            mu = (F.to_numpy() * w[:, None]).sum(axis=0)
            Fc = F.to_numpy() - mu
            factor_cov = (Fc * w[:, None]).T @ Fc
        elif self.config.factor_cov_method == "sample":
            factor_cov = F.cov().to_numpy()
        else:
            raise ValueError(f"未知因子协方差方法: {self.config.factor_cov_method!r}")

        specific_var = resid_sq_sum / resid_cnt.clip(lower=1)
        specific_var = specific_var.fillna(float(specific_var.median()))

        # 尺度校准参照：同一估计窗内样本协方差的对角线方差均值（ddof=1，逐列跳过 NaN）
        col_var = returns.var(axis=0, skipna=True).dropna()
        self._ref_var_mean = float(col_var.mean()) if len(col_var) else None

        self._B = B
        self._factor_cov = factor_cov
        self._specific_var = specific_var
        self._factor_returns = F
        self._fitted = True
        return self

    def covariance(self, exposures: pd.DataFrame | None = None) -> np.ndarray:
        """合成资产协方差矩阵 Σ = B·Σf·Bᵀ + D（对称化 → PSD 投影 → 尺度校准后输出）。

        Args:
            exposures: 为空用 fit 时的 B；提供时按新暴露合成（行顺序即输出顺序）。

        尺度校准（``config.scale_calibrate`` 开启时）：将合成协方差整体缩放，
        使其对角线方差均值等于 fit 窗口内样本协方差的对角线方差均值；参照
        缺失/无效或合成对角线均值非正时跳过，并在 ``scale_note_`` 注明。
        """
        if not self._fitted or self._B is None or self._factor_cov is None or self._specific_var is None:
            raise RuntimeError("FactorRiskModel 未 fit")
        B = self._B if exposures is None else exposures.reindex(columns=self._B.columns).fillna(0.0)
        Bv = B.to_numpy(dtype=float)
        d = self._specific_var.reindex(B.index).fillna(float(self._specific_var.median())).to_numpy()
        cov = Bv @ self._factor_cov @ Bv.T + np.diag(d)
        cov = (cov + cov.T) / 2.0
        if not check_psd(cov):
            cov = nearest_psd(cov)
        cov = self._apply_scale_calibration(cov)
        return cov

    def _apply_scale_calibration(self, cov: np.ndarray) -> np.ndarray:
        """尺度校准：正数整体缩放（不破坏 PSD）；无法校准时记录原因并原样返回。"""
        self._scale_factor = None
        self._scale_note = None
        if not self.config.scale_calibrate:
            return cov
        ref = self._ref_var_mean
        synth = float(np.mean(np.diag(cov))) if cov.size else 0.0
        if ref is None or not np.isfinite(ref) or ref <= 0.0:
            self._scale_note = "尺度校准跳过：估计窗样本方差参照缺失或无效"
            return cov
        if synth <= 0.0:
            self._scale_note = "尺度校准跳过：合成协方差对角线方差均值非正"
            return cov
        self._scale_factor = ref / synth
        return cov * self._scale_factor

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def factor_covariance_(self) -> np.ndarray:
        """因子协方差 Σf。"""
        if self._factor_cov is None:
            raise RuntimeError("FactorRiskModel 未 fit")
        return self._factor_cov

    @property
    def specific_risk_(self) -> pd.Series:
        """逐股票特质风险（残差标准差）。"""
        if self._specific_var is None:
            raise RuntimeError("FactorRiskModel 未 fit")
        return np.sqrt(self._specific_var)

    @property
    def scale_factor_(self) -> float | None:
        """最近一次 covariance() 的尺度校准系数；未校准（关闭或跳过）为 None。"""
        return self._scale_factor

    @property
    def scale_note_(self) -> str | None:
        """尺度校准跳过原因；正常校准或未校准时为 None。"""
        return self._scale_note


def check_psd(cov: np.ndarray, tol: float = 1e-10) -> bool:
    """检查协方差矩阵是否半正定（PSD）：最小特征值 ≥ −tol。"""
    cov = np.asarray(cov, dtype=float)
    if cov.shape[0] != cov.shape[1]:
        return False
    min_eig = np.linalg.eigvalsh((cov + cov.T) / 2.0).min()
    return bool(min_eig >= -tol)


def nearest_psd(cov: np.ndarray) -> np.ndarray:
    """将非 PSD 矩阵投影到最近的 PSD 矩阵（对称化 + 负特征值截断为 0）。"""
    sym = (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, 0.0, None)
    out = (eigvecs * eigvals) @ eigvecs.T
    return (out + out.T) / 2.0
