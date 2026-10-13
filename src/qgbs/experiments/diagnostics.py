"""辅助诊断（方案书 §7.1）：四类对照诊断。

判读逻辑通则：任何单一诊断都不足以归因 GBS 增益的来源，
须四类联合判读（详见各函数 docstring）。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from qgbs.backtest.metrics import ic_series
from qgbs.experiments.report import block_bootstrap_ci
from qgbs.gbs.backends.base import SampleBatch
from qgbs.gbs.statistics import (
    batch_dispersion,
    marginal_stats,
    second_order_stats,
    subbatch_resample,
)
from qgbs.models.readout import RidgeReadout


@dataclass(frozen=True)
class DiagnosticResult:
    """一项诊断的输出：对比表、判读结论与原始统计量。"""

    diagnostic: str  # 诊断名称
    table: pd.DataFrame  # 对比指标表
    verdict: str  # 判读结论（结构化文本）
    details: dict[str, object] = field(default_factory=dict)


def _merge_samples(batches: list[SampleBatch]) -> np.ndarray:
    """同来源多批次纵向拼接（单批次零拷贝）。"""
    if len(batches) == 1:
        return np.asarray(batches[0].samples)
    return np.concatenate([np.asarray(b.samples) for b in batches], axis=0)


def hardware_vs_simulation(
    hardware_batches: list[SampleBatch],
    simulation_batches: list[SampleBatch],
) -> DiagnosticResult:
    """诊断一：真机 vs 仿真样本对照（§7.1）。

    对比真机与仿真样本的三层统计差异，均附采样噪声底（比值 >1 即超出
    有限样本涨落）：

    1. 逐模式边际 click 率：L1 / L∞ / 分位数概览；
    2. 二阶 click 相关系数矩阵（非对角元，对边际失配免疫）：Frobenius
       差异与逐元素最大差；噪声底用真机批次 split-half 经验估计；
    3. 子系统分布：按 mode_config 的 (n_temporal, n_spatial) 把逐模式边际
       聚合成时间/空间两个一维剖面，对比剖面 L1 与 Pearson 相关。

    单批次现实适配：原设计"差异随 η 单调变化"的判读需要多批次不同 η
    （或标定数据）；当前真机仅单批次且 meta.system_efficiency 为占位
    None，η 单调性判读不做——多批次接口签名保留，同来源多批次只做
    纵向合并。二阶统计经分块 Gram 累加计算（statistics.second_order_stats），
    支持 200k+ shots 大模式批次。
    """
    if not hardware_batches or not simulation_batches:
        raise ValueError("hardware_batches / simulation_batches 均不能为空")
    hw = _merge_samples(hardware_batches)
    sim = _merge_samples(simulation_batches)
    if hw.shape[1] != sim.shape[1]:
        raise ValueError(f"两源模式数不一致: {hw.shape[1]} vs {sim.shape[1]}")
    n_hw, n_sim = hw.shape[0], sim.shape[0]
    b_hw = hardware_batches[0]
    b_sim = simulation_batches[0]

    # --- 1) 边际 ---
    m_hw = hw.sum(axis=0, dtype=np.uint64).astype(float) / n_hw
    m_sim = sim.sum(axis=0, dtype=np.uint64).astype(float) / n_sim
    d_marg = np.abs(m_hw - m_sim)
    pooled = 0.5 * (m_hw + m_sim)
    # 两独立有限样本边际差异的期望噪声底（E|N(0,σ²)|=σ√(2/π)，σ²=p(1-p)(1/n1+1/n2)）
    marg_se = np.sqrt(2.0 / np.pi) * np.sqrt(pooled * (1.0 - pooled) * (1.0 / n_hw + 1.0 / n_sim))
    marg_l1_floor = float(marg_se.sum())
    q = np.quantile(d_marg, [0.5, 0.9, 0.99])

    # --- 2) 二阶（click 相关矩阵，非对角元）---
    corr_hw = second_order_stats(b_hw, "correlation")
    corr_sim = second_order_stats(b_sim, "correlation")
    offdiag = ~np.eye(hw.shape[1], dtype=bool)
    d_corr = (corr_hw - corr_sim)[offdiag]
    fro = float(np.sqrt((d_corr**2).sum()))
    # 噪声底：真机批次 split-half 相关矩阵 Frobenius 差异 / 2
    # （半样本差异方差是全样本对照方差的 4 倍；v ∝ 1/m）
    half = n_hw // 2
    meta2 = dataclasses.replace(b_hw.meta, n_shots=n_hw - half)
    half_batch = [SampleBatch(meta=b_hw.meta, samples=b_hw.samples[:half]),
                  SampleBatch(meta=meta2, samples=b_hw.samples[half:])]
    c1 = second_order_stats(half_batch[0], "correlation")
    c2 = second_order_stats(half_batch[1], "correlation")
    corr_floor = 0.5 * float(np.sqrt((((c1 - c2)[offdiag]) ** 2).sum()))

    # --- 3) 子系统剖面（时间/空间聚合）---
    mc = b_hw.meta.mode_config
    n_t, n_s = int(mc.get("n_temporal", 0)), int(mc.get("n_spatial", 0))
    temporal: dict[str, float] = {}
    if n_t * n_s == hw.shape[1] and n_t > 0:
        # mode = temporal × n_spatial + spatial（与编码/仿真器约定一致）
        t_hw = m_hw.reshape(n_t, n_s).mean(axis=1)
        t_sim = m_sim.reshape(n_t, n_s).mean(axis=1)
        s_hw = m_hw.reshape(n_t, n_s).mean(axis=0)
        s_sim = m_sim.reshape(n_t, n_s).mean(axis=0)
        temporal = {
            "temporal_l1": float(np.abs(t_hw - t_sim).sum()),
            "temporal_corr": float(np.corrcoef(t_hw, t_sim)[0, 1]),
            "spatial_l1": float(np.abs(s_hw - s_sim).sum()),
            "spatial_corr": float(np.corrcoef(s_hw, s_sim)[0, 1]),
        }

    rows = [
        {"metric": "marginal_l1", "value": float(d_marg.sum()),
         "noise_floor": marg_l1_floor, "note": "逐模式边际 click 率差异 L1"},
        {"metric": "marginal_linf", "value": float(d_marg.max()),
         "noise_floor": float(marg_se.max()), "note": "逐模式边际差异 L∞"},
        {"metric": "marginal_mean_abs", "value": float(d_marg.mean()),
         "noise_floor": float(marg_se.mean()), "note": "边际平均绝对差"},
        {"metric": "second_order_corr_frobenius", "value": fro,
         "noise_floor": corr_floor, "note": "click 相关矩阵非对角元 Frobenius 差异（split-half 噪声底）"},
        {"metric": "second_order_corr_max_abs", "value": float(np.abs(d_corr).max()),
         "noise_floor": float("nan"), "note": "相关矩阵逐元素最大差"},
        *[
            {"metric": k, "value": v, "noise_floor": float("nan"),
             "note": "时间/空间聚合剖面差异" + ("（Pearson 相关）" if k.endswith("corr") else "（L1）")}
            for k, v in temporal.items()
        ],
    ]
    table = pd.DataFrame(rows).set_index("metric")
    table["ratio"] = table["value"] / table["noise_floor"].where(table["noise_floor"] > 0)

    marg_ratio = float(d_marg.sum()) / marg_l1_floor if marg_l1_floor > 0 else float("inf")
    corr_ratio = fro / corr_floor if corr_floor > 0 else float("inf")
    verdict = (
        f"真机 vs 仿真（hw {n_hw} shots / sim {n_sim} shots）："
        f"边际 L1 {d_marg.sum():.2f}（{marg_ratio:.1f}× 噪声底）、"
        f"L∞ {d_marg.max():.4f}；"
        f"二阶相关 Frobenius {fro:.2f}（{corr_ratio:.1f}× split-half 噪声底）、"
        f"逐元素最大 {np.abs(d_corr).max():.3f}。"
        + (f"时间剖面相关 {temporal['temporal_corr']:.3f}、空间剖面相关 {temporal['spatial_corr']:.3f}。"
           if temporal else "")
        + "单批次现实：真机 η 未标定（占位 None），η 单调性判读不做。"
        + ("判读：二阶差异远超噪声底 → 联合结构存在装置级差异（噪声/干涉结构），"
           "GBS 联合类特征的仿真结论不可外推真机。"
           if corr_ratio > 5 else
           "判读：二阶差异与噪声底同量级 → 仿真联合结构与真机无显著差异。")
    )
    return DiagnosticResult(
        diagnostic="hardware_vs_simulation",
        table=table,
        verdict=verdict,
        details={
            "n_shots_hw": n_hw, "n_shots_sim": n_sim,
            "batch_hw": b_hw.meta.batch_id, "batch_sim": b_sim.meta.batch_id,
            "marginal_abs_quantiles": {"p50": float(q[0]), "p90": float(q[1]), "p99": float(q[2])},
            "mean_click_hw": float(m_hw.mean()), "mean_click_sim": float(m_sim.mean()),
            **temporal,
        },
    )


def _panel_mean_ic(feats: pd.DataFrame, y: pd.Series) -> pd.Series:
    """面板各特征列对 y 的逐日 rank IC 均值。"""
    out = {}
    for col in feats.columns:
        ics = ic_series(feats[col], y)
        out[col] = float(ics.mean()) if len(ics) else float("nan")
    return pd.Series(out)


def permutation_control(
    features_gbs: pd.DataFrame,
    features_permuted: pd.DataFrame,
    y_valid: pd.Series,
) -> DiagnosticResult:
    """诊断二：置换对照（§7.1）——保持边际、破坏联合结构的 GBS 特征。

    对比原 GBS 特征面板与置换后端（PermutedBackend）产出面板的逐特征
    rank IC。判读：仅判断联合结构是否有贡献（置换后联合类特征增益消失
    ⇒ 联合结构有信息量）；**归因需结合诊断三**（匹配低阶统计的经典随机
    特征对照），否则无法区分"量子关联"与"任意非线性升维"。
    """
    ic_gbs = _panel_mean_ic(features_gbs, y_valid)
    ic_perm = _panel_mean_ic(features_permuted, y_valid)
    table = pd.DataFrame({"ic_gbs": ic_gbs, "ic_permuted": ic_perm})
    table["ic_diff"] = table["ic_gbs"] - table["ic_permuted"]
    pair_cols = [c for c in table.index if "pair" in c or "ho" in c]
    joint_diff = float(table.loc[pair_cols, "ic_diff"].mean()) if pair_cols else 0.0
    verdict = (
        f"联合类特征 IC 差均值 {joint_diff:+.4f}；"
        + ("置换后联合特征增益消失，联合结构有信息量。" if joint_diff > 0 else "置换未削弱联合特征，联合结构贡献未获支持。")
        + "注意：归因须结合匹配低阶统计的经典随机特征对照（诊断三）。"
    )
    return DiagnosticResult(
        diagnostic="permutation_control",
        table=table,
        verdict=verdict,
        details={"joint_ic_diff_mean": joint_diff},
    )


def _quantile_linear(sorted_ref: np.ndarray, q: np.ndarray) -> np.ndarray:
    """有序参考序列上的线性插值分位数（与 np.quantile 默认 linear 口径数值一致）。

    np.quantile 对多查询点会逐点 partial-sort（O(nq·m)），大面板下不可用；
    此处排序一次后向量化插值（O(m log m + nq log m)）。
    """
    m = len(sorted_ref)
    pos = np.clip(q, 0.0, 1.0) * (m - 1)
    lo = np.floor(pos).astype(np.int64)
    frac = pos - lo
    hi = np.minimum(lo + 1, m - 1)
    return sorted_ref[lo] * (1.0 - frac) + sorted_ref[hi] * frac


class GaussianCopulaMatchedFeatures:
    """与参照面板低阶统计匹配的等维经典随机特征生成器（诊断三用）。

    ``fit`` 学习参照面板的逐列经验边际与协方差；``sample`` 生成同维随机
    面板：先抽多元高斯 Z ~ N(μ̂, Σ̂)（Σ̂ 加微小岭项保证半正定），再逐列做
    经验分位数映射（Gaussian copula），使各列边际分布与参照一致、列间
    相关结构近似保留。生成的行是纯随机抽样，不含参照面板任何逐行
    (date, symbol) 的真实信息，用于检验"GBS 增益是否可由低阶统计 +
    随机升维本身解释"。
    """

    def __init__(self, seed: int = 42, ridge: float = 1e-8) -> None:
        self.seed = seed
        self.ridge = ridge
        self._columns: list[str] = []
        self._mu: np.ndarray | None = None
        self._chol: np.ndarray | None = None
        self._marginals: list[np.ndarray] = []

    def fit(self, X_ref: pd.DataFrame) -> "GaussianCopulaMatchedFeatures":
        """在参照特征面板上拟合逐列均值/协方差与经验边际。"""
        X = X_ref.dropna()
        if len(X) < 2:
            raise ValueError("参照面板有效行不足 2 行，无法拟合低阶统计")
        self._columns = list(X.columns)
        arr = X.to_numpy(dtype=float)
        self._mu = arr.mean(axis=0)
        cov = np.atleast_2d(np.cov(arr, rowvar=False))
        scale = float(np.trace(cov)) / cov.shape[0] if cov.shape[0] else 1.0
        cov = cov + self.ridge * max(scale, 1e-12) * np.eye(cov.shape[0])
        self._chol = np.linalg.cholesky(cov)
        self._marginals = [np.sort(arr[:, j]) for j in range(arr.shape[1])]
        return self

    def sample(self, index: pd.Index) -> pd.DataFrame:
        """生成 ``len(index)`` 行的等维随机特征面板（列名沿用参照面板）。"""
        if self._chol is None or self._mu is None:
            raise RuntimeError("GaussianCopulaMatchedFeatures 未 fit")
        rng = np.random.default_rng(self.seed)
        n = len(index)
        z = rng.standard_normal((n, len(self._columns))) @ self._chol.T + self._mu
        out = np.empty_like(z)
        u = (np.arange(n) + 0.5) / n
        for j in range(len(self._columns)):
            order = np.argsort(z[:, j], kind="stable")
            out[order, j] = _quantile_linear(self._marginals[j], u)
        return pd.DataFrame(out, index=index, columns=self._columns)

    @property
    def output_dim(self) -> int:
        return len(self._columns)


def _panel_readout_ic(
    X: pd.DataFrame, y: pd.Series, train_frac: float, alpha: float
) -> pd.Series:
    """诊断样本内按时间先后因果切分 train/eval 的 Ridge 读出逐日 rank IC。

    复用与实验链路相同的读出模型（RidgeReadout）：前 ``train_frac`` 的日期
    训练、后段日期逐日评估，保持诊断内部的时间因果性。
    """
    dates = pd.Index(sorted(X.index.get_level_values("date").unique()))
    n_train = max(2, int(len(dates) * train_frac))
    train_dates = dates[:n_train]
    eval_dates = dates[n_train:]
    if len(eval_dates) < 3:
        return pd.Series(dtype=float)
    Xd = X.dropna()
    yd = y.reindex(Xd.index)
    date_level = Xd.index.get_level_values("date")
    model = RidgeReadout(alpha=alpha).fit(Xd[date_level.isin(train_dates)], yd)
    eval_mask = date_level.isin(eval_dates)
    pred = model.predict(Xd[eval_mask])
    return ic_series(pred, yd[eval_mask])


def matched_random_features_control(
    features_gbs: pd.DataFrame,
    features_classical_rf: pd.DataFrame | None,
    y_valid: pd.Series,
    *,
    seed: int = 42,
    n_bootstrap: int = 500,
    block_length: int = 21,
    level: float = 0.95,
    train_frac: float = 0.5,
    readout_alpha: float = 1.0,
) -> DiagnosticResult:
    """诊断三：与 GBS 低阶统计匹配的经典随机特征对照（§7.1）。

    ``features_classical_rf`` 为 None 时用 :class:`GaussianCopulaMatchedFeatures`
    自动生成与 GBS 特征等维且逐列边际/协方差匹配的随机特征面板。

    输出两层对比：
    - 逐特征 rank IC 均值表（与诊断二同风格）；
    - 面板级读出对照：同一 RidgeReadout 链路下 GBS vs 匹配随机特征的
      逐日 rank IC 差及分块 bootstrap 置信区间。

    判读：若等维低阶匹配的经典随机特征获得同等增益（IC 差 CI 含 0），
    则增益可由随机升维 + 低阶结构解释，不支持 GBS 联合结构的独特贡献；
    反之（CI 下界 > 0）支持 GBS 特征含有超出低阶统计匹配的增量。
    置换实验（诊断二）仅判断联合结构是否有贡献，归因须结合本诊断。
    """
    matched_source = "provided"
    if features_classical_rf is None:
        gen = GaussianCopulaMatchedFeatures(seed=seed).fit(features_gbs)
        features_classical_rf = gen.sample(features_gbs.index)
        matched_source = "auto_gaussian_copula"

    ic_gbs = _panel_mean_ic(features_gbs, y_valid)
    ic_matched = _panel_mean_ic(features_classical_rf, y_valid)
    table = pd.DataFrame({"ic_gbs": ic_gbs, "ic_matched": ic_matched})
    table["ic_diff"] = table["ic_gbs"] - table["ic_matched"]

    ic_gbs_ts = _panel_readout_ic(features_gbs, y_valid, train_frac, readout_alpha)
    ic_rf_ts = _panel_readout_ic(features_classical_rf, y_valid, train_frac, readout_alpha)
    diff = (ic_gbs_ts - ic_rf_ts).dropna()
    gbs_mean = float(ic_gbs_ts.mean()) if len(ic_gbs_ts) else float("nan")
    rf_mean = float(ic_rf_ts.mean()) if len(ic_rf_ts) else float("nan")
    diff_mean = float(diff.mean()) if len(diff) else float("nan")
    if len(diff) >= 8:
        # 短样本保护：块长截断到 √n，避免块长接近 n 时 CI 退化偏窄
        eff_block = max(2, min(block_length, int(np.sqrt(len(diff)))))
        lo, hi = block_bootstrap_ci(
            diff, statistic="mean", n_bootstrap=n_bootstrap,
            block_length=eff_block, level=level, seed=seed,
        )
    else:
        lo, hi = float("nan"), float("nan")

    if np.isnan(diff_mean):
        conclusion = "评估日期不足，无法给出面板级判读。"
    elif lo > 0:
        conclusion = "IC 差置信区间下界为正：支持 GBS 特征含有超出低阶统计匹配的增量信息。"
    elif hi < 0:
        conclusion = (
            "IC 差置信区间上界为负：GBS 特征读出表现显著低于等维低阶匹配经典随机特征，"
            "本口径下不支持 GBS 独特结构贡献（且方向为负）。"
        )
    else:
        conclusion = (
            "IC 差置信区间含 0：等维低阶匹配经典随机特征达到同等读出表现，"
            "GBS 增量可由低阶统计 + 随机升维解释，未支持独特结构贡献。"
        )
    verdict = (
        f"面板读出 rank IC：GBS {gbs_mean:+.4f} vs 匹配随机 {rf_mean:+.4f}，"
        f"差 {diff_mean:+.4f}，{level:.0%} 分块 bootstrap CI [{lo:+.4f}, {hi:+.4f}]。"
        + conclusion
        + "结合诊断二（置换对照）联合判读：置换判断联合结构是否有贡献，本诊断判断该贡献能否由经典低阶匹配复制。"
    )
    return DiagnosticResult(
        diagnostic="matched_random_features_control",
        table=table,
        verdict=verdict,
        details={
            "panel_ic_gbs": gbs_mean,
            "panel_ic_matched": rf_mean,
            "ic_diff_mean": diff_mean,
            "ic_diff_ci": (lo, hi),
            "n_eval_dates": len(diff),
            "matched_source": matched_source,
            "seed": seed,
        },
    )


def mapping_batch_stability(
    features_by_mapping: dict[str, pd.DataFrame],
    batches: list[SampleBatch],
) -> DiagnosticResult:
    """诊断四：模式映射与批次稳定性（§7.1）。

    - 批次稳定性：跨批次边际/二阶统计量离散度（batch_dispersion），
      以及子批次重采样的边际统计离散度随样本量的曲线；
    - 映射稳定性：不同预先固定映射下同名特征面板的相关性。

    判读：统计量离散度随 shots 增大而下降、跨映射特征高度相关，
    方可认为增益对映射选择/批次抽样稳健；不稳健则降级为仿真结论并记录原因。
    """
    if not batches:
        raise ValueError("mapping_batch_stability 需要至少一个采样批次：batches 为空")
    disp = batch_dispersion(batches)
    marg_rows = [i for i in disp.index if str(i).startswith("m_")]
    marg_std_mean = float(disp.loc[marg_rows, "std"].mean()) if marg_rows else float("nan")

    big = max(batches, key=lambda b: b.meta.n_shots)
    curve: dict[str, float] = {}
    for n_sub in (2, 4, 8):
        subs = subbatch_resample(big, n_sub)
        margs = np.stack([marginal_stats(s) for s in subs])
        curve[f"1/{n_sub}_shots"] = float(margs.std(axis=0).mean())

    keys = list(features_by_mapping)
    mapping_corr = float("nan")
    if len(keys) >= 2:
        a, b = features_by_mapping[keys[0]], features_by_mapping[keys[1]]
        common = a.columns.intersection(b.columns)
        if len(common):
            mapping_corr = float(a[common].corrwith(b[common], axis=0).mean())

    table = pd.DataFrame(
        {
            "batch_marginal_std_mean": [marg_std_mean],
            "mapping_cross_corr": [mapping_corr],
            **{f"subbatch_dispersion_{k}": [v] for k, v in curve.items()},
        }
    )
    decreasing = list(curve.values()) == sorted(curve.values(), reverse=True)
    verdict = (
        f"批次边际离散度 {marg_std_mean:.5f}；子批次离散度随样本量"
        + ("单调下降" if decreasing else "未单调下降")
        + f"；跨映射相关 {mapping_corr:.3f}。"
    )
    return DiagnosticResult(
        diagnostic="mapping_batch_stability",
        table=table,
        verdict=verdict,
        details={"dispersion_curve": curve, "batch_dispersion_head": disp.head(20)},
    )
