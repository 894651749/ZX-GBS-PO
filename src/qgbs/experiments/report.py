"""实验报告：分块置信区间、消融对比表与多模型选择偏差控制（方案书 §7.2）。

统计口径约定：

- 置信区间用环形移动分块 bootstrap（保留时序相关），默认块长 21 个交易日；
- White Reality Check 用平稳 bootstrap（Politis–Romano，几何分布块长）；
- Deflated Sharpe Ratio 按 Bailey & López de Prado (2014)：输入 Sharpe 与
  ``n_obs`` 必须是同一频率口径（如日频 Sharpe = 年化 Sharpe / √252）。
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from qgbs.backtest.metrics import annual_return, max_drawdown, sharpe, sortino

_EULER_GAMMA = 0.5772156649015329  # Euler–Mascheroni 常数


def _stat_func(statistic: str) -> Callable[[pd.Series], float]:
    """收益序列统计量表：statistic 名 → 函数。"""
    funcs: dict[str, Callable[[pd.Series], float]] = {
        "sharpe": sharpe,
        "sortino": sortino,
        "annual_return": annual_return,
        "max_drawdown": lambda r: max_drawdown((1.0 + r).cumprod()),
        "mean": lambda r: float(r.mean()),
    }
    if statistic not in funcs:
        raise ValueError(f"未知统计量: {statistic!r}（支持 {sorted(funcs)}）")
    return funcs[statistic]


def _moving_block_indices(
    n: int, block_length: int, n_bootstrap: int, rng: np.random.Generator
) -> np.ndarray:
    """环形移动分块 bootstrap 索引：(n_bootstrap, n)，块起点均匀随机、越界回绕。"""
    n_blocks = math.ceil(n / block_length)
    starts = rng.integers(0, n, size=(n_bootstrap, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_length)) % n
    return idx.reshape(n_bootstrap, -1)[:, :n]


def _stationary_bootstrap_indices(
    n: int, avg_block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """平稳 bootstrap 索引（Politis–Romano）：块长 ~ Geometric(1/avg_block_length)。"""
    p = 1.0 / max(1, avg_block_length)
    idx = np.empty(n, dtype=int)
    pos = 0
    while pos < n:
        start = int(rng.integers(0, n))
        length = max(1, int(rng.geometric(p)))
        seg = (start + np.arange(length)) % n
        take = min(length, n - pos)
        idx[pos : pos + take] = seg[:take]
        pos += take
    return idx


def paired_metric_bootstrap_ci(
    returns_a: pd.Series,
    returns_b: pd.Series,
    metric: "Callable[[pd.Series], float]",
    n_bootstrap: int = 2000,
    block_length: int = 5,
    level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """配对块 bootstrap 指标差 CI（E7.2 组合层推断）。

    两序列按日期 inner join 对齐后，同一组环形块索引同时作用于两序列
    （保持配对），每次重采样重算 ``metric``（日收益 → 标量，如 Sharpe/
    IR/Calmar）并取差。返回（点估计差, 下界, 上界）。
    """
    df = pd.concat([returns_a.rename("a"), returns_b.rename("b")], axis=1,
                   join="inner").dropna()
    n = len(df)
    if n < 4:
        raise ValueError(f"样本长度 {n} 太短，无法分块 bootstrap（至少 4）")
    block_length = max(1, min(int(block_length), n))
    a = df["a"].to_numpy(dtype=float)
    b = df["b"].to_numpy(dtype=float)
    point = float(metric(df["a"]) - metric(df["b"]))
    rng = np.random.default_rng(seed)
    idx = _moving_block_indices(n, block_length, n_bootstrap, rng)
    diffs = np.array(
        [metric(pd.Series(a[row])) - metric(pd.Series(b[row])) for row in idx]
    )
    alpha = 1.0 - level
    lo, hi = np.quantile(diffs, [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(lo), float(hi)


def block_bootstrap_ci(
    returns: pd.Series,
    statistic: str = "sharpe",
    n_bootstrap: int = 1000,
    block_length: int = 21,
    level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """分块 bootstrap 置信区间（环形移动分块，保留时序相关性）。

    Args:
        returns: 日频收益序列（NaN 自动剔除）。
        statistic: 统计量名——"sharpe"/"sortino"/"annual_return"/"max_drawdown"/"mean"。
        block_length: 分块长度（默认 21，超过样本长度时自动截断）。

    Returns:
        (下界, 上界)，分位数口径。
    """
    r = returns.dropna().to_numpy(dtype=float)
    n = len(r)
    if n < 4:
        raise ValueError(f"样本长度 {n} 太短，无法分块 bootstrap（至少 4）")
    block_length = max(1, min(int(block_length), n))
    func = _stat_func(statistic)
    rng = np.random.default_rng(seed)
    idx = _moving_block_indices(n, block_length, n_bootstrap, rng)
    stats = np.array([func(pd.Series(r[row])) for row in idx])
    alpha = 1.0 - level
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _coerce_returns(obj: object) -> pd.Series:
    """接受收益序列或带 ``extra["returns"]`` 的 ExperimentResult（鸭子类型）。"""
    if isinstance(obj, pd.Series):
        return obj
    extra = getattr(obj, "extra", None)
    if isinstance(extra, dict) and isinstance(extra.get("returns"), pd.Series):
        return extra["returns"]
    raise TypeError(
        "ablation_table 的取值须为日收益 pd.Series，或 extra 含 'returns' 的 ExperimentResult；"
        f"收到 {type(obj).__name__}"
    )


def ablation_table(
    results: dict[str, object],
    baseline_id: str = "C2",
    metrics: tuple[str, ...] = ("sharpe", "annual_return", "max_drawdown"),
    n_bootstrap: int = 1000,
    block_length: int = 21,
    level: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """消融对比表：各实验相对基线的指标增量 + 配对分块 bootstrap 置信区间。

    Args:
        results: 实验编号 → 日收益 pd.Series（或 extra 含 "returns" 的
            ExperimentResult）。
        baseline_id: 基线实验编号（经典基线对照锚点）。
        metrics: 参与对比的统计量（见 :func:`_stat_func`）。
        增量 CI 用同一套分块索引对两序列配对重采样（保留横截面/时序对应）。

    Returns:
        行索引为实验编号的 DataFrame；每个指标四列：
        ``{m}``（本实验值）、``{m}_delta``（相对基线增量）、
        ``{m}_ci_lo`` / ``{m}_ci_hi``（增量的置信区间）。
        基线行 delta 与 CI 恒为 0。
    """
    series = {k: _coerce_returns(v) for k, v in results.items()}
    if baseline_id not in series:
        raise KeyError(f"基线 {baseline_id!r} 不在 results 中: {sorted(series)}")
    funcs = [(m, _stat_func(m)) for m in metrics]
    base = series[baseline_id]
    rng = np.random.default_rng(seed)
    alpha = 1.0 - level
    rows: dict[str, dict[str, float]] = {}
    for exp_id, r in series.items():
        pair = pd.DataFrame({"a": r, "b": base}).dropna()
        n = len(pair)
        a = pair["a"].to_numpy(dtype=float)
        b = pair["b"].to_numpy(dtype=float)
        row: dict[str, float] = {}
        idx = (
            _moving_block_indices(n, max(1, min(int(block_length), n)), n_bootstrap, rng)
            if exp_id != baseline_id and n >= 4
            else None
        )
        for m, func in funcs:
            va = func(pair["a"])
            vb = func(pair["b"])
            row[m] = va
            if idx is None:
                row[f"{m}_delta"] = 0.0 if exp_id == baseline_id else float("nan")
                row[f"{m}_ci_lo"] = 0.0 if exp_id == baseline_id else float("nan")
                row[f"{m}_ci_hi"] = 0.0 if exp_id == baseline_id else float("nan")
                continue
            deltas = np.array(
                [func(pd.Series(a[i])) - func(pd.Series(b[i])) for i in idx]
            )
            lo, hi = np.quantile(deltas, [alpha / 2.0, 1.0 - alpha / 2.0])
            row[f"{m}_delta"] = va - vb
            row[f"{m}_ci_lo"] = float(lo)
            row[f"{m}_ci_hi"] = float(hi)
        rows[exp_id] = row
    table = pd.DataFrame(rows).T
    table.index.name = "experiment"
    return table


def deflated_sharpe(
    sharpe_observed: float,
    sharpe_trials: pd.Series,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio（Bailey & López de Prado, 2014，p 值口径）。

    基准 Sharpe 取 N 次试算在零技能假设下的期望最大值

    SR0 = √V · [(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))]，

    其中 V 为试算 Sharpe 的样本方差、γ 为 Euler–Mascheroni 常数；
    再以 SR0 为基准计算概率 Sharpe：

    DSR = Φ( (SR̂−SR0)·√(T−1) / √(1−γ₃·SR̂+(γ₄−1)/4·SR̂²) )。

    Args:
        sharpe_observed: 被评估策略的观测 Sharpe（与 ``n_obs`` 同频率口径）。
        sharpe_trials: 全部已尝试模型/配置的 Sharpe 集合（同口径）。
        n_obs: 收益样本长度 T。
        skew: 收益偏度 γ₃（缺省 0）；kurt: 收益峰度 γ₄（非超额，正态取 3）。

    Returns:
        DSR ∈ [0, 1]（PSR 口径）；分母非正时返回 NaN。
    """
    trials = pd.Series(sharpe_trials, dtype=float).dropna()
    n_trials = len(trials)
    if n_trials == 0:
        raise ValueError("sharpe_trials 为空，无法估计期望最大 Sharpe")
    if n_obs < 2:
        raise ValueError(f"n_obs={n_obs} 太短（至少 2）")
    if n_trials == 1:
        sr0 = float(trials.iloc[0])
    else:
        v = float(trials.var(ddof=1))
        sr0 = math.sqrt(v) * (
            (1.0 - _EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n_trials)
            + _EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        )
    sr = float(sharpe_observed)
    var_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if var_term <= 0.0:
        return float("nan")
    z = (sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(var_term)
    return float(norm.cdf(z))


def white_reality_check(
    benchmark_returns: pd.Series,
    strategy_returns: dict[str, pd.Series],
    n_bootstrap: int = 1000,
    block_length: int = 21,
    seed: int = 42,
) -> pd.DataFrame:
    """White Reality Check：最优策略相对基准的超额是否整体显著（平稳 bootstrap）。

    统计量 V = max_k √T·d̄_k，d̄_k 为策略 k 相对基准的日均超额；bootstrap
    复样本用中心化超额（d_k,t − d̄_k）重采样，模拟零假设（期望超额为 0）下
    的分布，整体 p 值 = P*(max_k √T·d̄_k* ≥ V)。平稳 bootstrap（块长
    Geometric(1/block_length)）保留收益时序相关。

    Returns:
        行索引为策略名的 DataFrame：
        - ``mean_excess``：日均超额收益；
        - ``t_stat``：超额均值的朴素 t 统计量；
        - ``p_individual``：单策略 bootstrap p 值（未控制多重比较）；
        - ``p_reality_check``：整体检验 p 值（控制多重比较，逐行相同）；
        - ``is_best``：是否为观测期最优策略。
    """
    names = list(strategy_returns)
    if not names:
        raise ValueError("strategy_returns 为空")
    df = pd.DataFrame({k: v for k, v in strategy_returns.items()})
    df["_benchmark_"] = benchmark_returns
    df = df.dropna()
    n = len(df)
    if n < 8:
        raise ValueError(f"对齐后样本长度 {n} 太短（至少 8）")
    excess = df[names].to_numpy(dtype=float) - df["_benchmark_"].to_numpy(dtype=float)[:, None]
    obs_mean = excess.mean(axis=0)
    obs_stat = float(obs_mean.max())
    rng = np.random.default_rng(seed)
    boot_max = np.empty(n_bootstrap)
    boot_indiv = np.empty((n_bootstrap, len(names)))
    for b in range(n_bootstrap):
        idx = _stationary_bootstrap_indices(n, block_length, rng)
        boot_mean = (excess[idx] - obs_mean).mean(axis=0)
        boot_max[b] = float(boot_mean.max())
        boot_indiv[b] = boot_mean
    p_joint = float((boot_max >= obs_stat).mean())
    p_indiv = (boot_indiv >= obs_mean).mean(axis=0)
    std = excess.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t_stat = np.where(std > 0, obs_mean / std * math.sqrt(n), np.nan)
    table = pd.DataFrame(
        {
            "mean_excess": obs_mean,
            "t_stat": t_stat,
            "p_individual": p_indiv,
            "p_reality_check": p_joint,
            "is_best": np.arange(len(names)) == int(np.argmax(obs_mean)),
        },
        index=pd.Index(names, name="strategy"),
    )
    return table


def multiple_testing_correction(p_values: pd.Series, method: str = "benjamini_hochberg") -> pd.Series:
    """多重比较校正，返回校正后 p 值（索引与输入对齐）。

    method:
        - ``"benjamini_hochberg"``：BH 法控制 FDR，qᵢ = p₍ᵢ₎·m/i 并自最大
          p 值起强制单调；
        - ``"bonferroni"``：p·m（截断到 1）。
    """
    p = p_values.astype(float)
    m = len(p)
    if m == 0:
        return p
    if method == "bonferroni":
        return (p * m).clip(upper=1.0)
    if method != "benjamini_hochberg":
        raise ValueError(f"未知校正方法: {method!r}（支持 'benjamini_hochberg'/'bonferroni'）")
    ordered = p.sort_values()
    adj = ordered.to_numpy() * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    return pd.Series(adj, index=ordered.index).reindex(p_values.index)
