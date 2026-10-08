"""ρ 收缩混合与逐资产预测不确定性估计。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qgbs.models.selection import _per_date_rank_ic


@dataclass(frozen=True)
class BlendResult:
    """混合结果：收缩后的预测与其不确定性估计。"""

    prediction: pd.Series  # 混合后截面预测
    uncertainty: pd.Series  # 各资产预测不确定性
    rho: float  # 实际使用的混合系数


def shrinkage_blend(
    pred_gbs: pd.Series,
    pred_classic: pd.Series,
    rho: float,
    uncertainty: pd.Series | None = None,
) -> BlendResult:
    """ρ 收缩混合：rho·pred_gbs + (1−rho)·pred_classic。

    ``rho`` 在验证窗口上估计（见 :func:`estimate_rho`）。提供
    ``uncertainty`` 时，逐资产有效系数 rho_i = rho·(1−u_i^norm)，
    即不确定性高的预测向经典端额外收缩（u^norm 为 min-max 归一化）。
    """
    idx = pred_gbs.index.union(pred_classic.index)
    pg = pred_gbs.reindex(idx).fillna(0.0)
    pc = pred_classic.reindex(idx).fillna(0.0)
    if uncertainty is None:
        unc = pd.Series(0.0, index=idx)
        rho_eff = rho
    else:
        unc = uncertainty.reindex(idx).fillna(float(uncertainty.median()))
        span = float(unc.max() - unc.min())
        u_norm = (unc - unc.min()) / span if span > 0 else unc * 0.0
        rho_eff = rho * (1.0 - u_norm.clip(0.0, 1.0))
    pred = rho_eff * pg + (1.0 - rho_eff) * pc
    return BlendResult(prediction=pred.rename("alpha_blend"), uncertainty=unc, rho=rho)


def estimate_rho(
    pred_gbs: pd.Series,
    pred_classic: pd.Series,
    y_valid: pd.Series,
    grid: np.ndarray | None = None,
) -> float:
    """在验证窗口上网格搜索最优 ρ（逐日 rank IC 均值最大化）。"""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 11)
    best_rho, best_score = 0.0, float("-inf")
    for rho in grid:
        blended = shrinkage_blend(pred_gbs, pred_classic, float(rho)).prediction
        ics = _per_date_rank_ic(blended, y_valid)
        score = float(ics.mean()) if len(ics) else float("nan")
        if pd.notna(score) and score > best_score:
            best_rho, best_score = float(rho), score
    return best_rho


def estimate_uncertainty(predictions: pd.DataFrame) -> pd.Series:
    """由多折/多种子预测集合估计各资产不确定性（列间离散度的逐资产均值）。

    Args:
        predictions: (date, symbol) × fold 的预测面板。
    """
    per_row = predictions.std(axis=1)
    return per_row.groupby(level="symbol").mean().rename("uncertainty")


def uncertainty_block_bootstrap(
    pred_by_block: pd.DataFrame,
    block_length: int = 21,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> pd.Series:
    """逐资产不确定性（口径一）：时间分块 bootstrap。

    对时间轴按 ``block_length`` 分块有放回重采样，估计逐资产预测/残差
    序列的波动；多次重采样的均值作为不确定性。

    Args:
        pred_by_block: date × symbol 的预测或残差面板。
    """
    df = pred_by_block.sort_index()
    values = df.to_numpy(dtype=float)
    t_n = len(df)
    if t_n < block_length:
        raise ValueError(f"样本长度 {t_n} 小于 block_length={block_length}")
    n_blocks = int(np.ceil(t_n / block_length))
    rng = np.random.default_rng(seed)
    stds = []
    for _ in range(n_bootstrap):
        starts = rng.integers(0, t_n - block_length + 1, n_blocks)
        sample = np.concatenate([values[s : s + block_length] for s in starts], axis=0)
        stds.append(np.nanstd(sample, axis=0))
    unc = pd.Series(np.nanmean(stds, axis=0), index=df.columns)
    return unc.fillna(float(unc.mean())).rename("uncertainty")


def uncertainty_residual(residuals: pd.DataFrame) -> pd.Series:
    """逐资产不确定性（口径二）：历史残差分布的逐资产标准差。

    Args:
        residuals: 训练/验证期残差（date × symbol）。
    """
    unc = residuals.std(axis=0, skipna=True)
    return unc.fillna(float(unc.mean())).rename("uncertainty")


class QuantileUncertaintyModel:
    """逐资产不确定性（口径三）：条件分位数模型区间宽度。

    用 HGB 的 quantile loss 拟合条件分位数（默认 0.16/0.84），
    预测区间宽度作为不确定性。fit 仅训练期。
    """

    def __init__(
        self,
        quantiles: tuple[float, float] = (0.16, 0.84),
        params: dict | None = None,
        seed: int = 42,
    ) -> None:
        self.quantiles = quantiles
        self.params = {"max_iter": 100, "max_depth": 3, **(params or {})}
        self.seed = seed
        self._models: dict[float, object] = {}
        self._features: list[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "QuantileUncertaintyModel":
        """在训练窗拟合上下分位数模型。"""
        from sklearn.ensemble import HistGradientBoostingRegressor

        y = y_train.reindex(X_train.index)
        mask = y.notna() & X_train.notna().all(axis=1)
        self._features = list(X_train.columns)
        for q in self.quantiles:
            model = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, random_state=self.seed, **self.params
            )
            model.fit(X_train[mask].to_numpy(dtype=float), y[mask].to_numpy(dtype=float))
            self._models[q] = model
        return self

    def predict_uncertainty(self, X: pd.DataFrame) -> pd.Series:
        """输出条件分位数区间宽度（索引与 X 一致）。"""
        if not self._models:
            raise RuntimeError("QuantileUncertaintyModel 未 fit")
        X2 = X.reindex(columns=self._features).fillna(0.0).to_numpy(dtype=float)
        q_lo, q_hi = self.quantiles
        width = self._models[q_hi].predict(X2) - self._models[q_lo].predict(X2)
        return pd.Series(np.clip(width, 0.0, None), index=X.index, name="uncertainty")


def uncertainty_quantile_model(q_low: pd.Series, q_high: pd.Series) -> pd.Series:
    """逐资产不确定性（口径三）：分位数模型区间宽度（q_high − q_low）。

    条件分位数的拟合见 :class:`QuantileUncertaintyModel`；本函数只做
    已预测分位数的宽度合成。
    """
    width = (q_high - q_low).clip(lower=0.0)
    return width.rename("uncertainty")


def estimate_uncertainty_by_method(
    method: str,
    *,
    residuals: pd.DataFrame | None = None,
    X_train: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    X_score: pd.DataFrame | None = None,
    block_length: int = 21,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> pd.Series:
    """不确定性估计统一入口（runner 的 uncertainty.method 配置位）。

    - ``residual``：残差分布 std（需 residuals）；
    - ``block_bootstrap``：时间分块 bootstrap（需 residuals）；
    - ``quantile``：条件分位数模型（需 X_train/y_train 训练、X_score 评分）。

    返回值均为逐 symbol 的 Series（面板化由调用方按日期广播）。
    """
    if method == "residual":
        if residuals is None:
            raise ValueError("method='residual' 需要 residuals")
        return uncertainty_residual(residuals)
    if method == "block_bootstrap":
        if residuals is None:
            raise ValueError("method='block_bootstrap' 需要 residuals")
        return uncertainty_block_bootstrap(residuals, block_length, n_bootstrap, seed)
    if method == "quantile":
        if X_train is None or y_train is None or X_score is None:
            raise ValueError("method='quantile' 需要 X_train / y_train / X_score")
        model = QuantileUncertaintyModel(seed=seed).fit(X_train, y_train)
        # 截面宽度在评分截面上取逐 symbol 均值，与其他口径的逐 symbol 输出对齐
        width = model.predict_uncertainty(X_score)
        return width.groupby(level="symbol").mean().rename("uncertainty")
    raise ValueError(f"未知不确定性口径: {method!r}（residual/block_bootstrap/quantile）")
