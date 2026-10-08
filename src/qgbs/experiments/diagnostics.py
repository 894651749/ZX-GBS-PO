"""辅助诊断（方案书 §7.1）：四类对照诊断。

判读逻辑通则：任何单一诊断都不足以归因 GBS 增益的来源，
须四类联合判读（详见各函数 docstring）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from qgbs.backtest.metrics import ic_series
from qgbs.experiments.report import block_bootstrap_ci
from qgbs.gbs.backends.base import SampleBatch
from qgbs.gbs.statistics import batch_dispersion, marginal_stats, subbatch_resample
from qgbs.models.readout import RidgeReadout


@dataclass(frozen=True)
class DiagnosticResult:
    """一项诊断的输出：对比表、判读结论与原始统计量。"""

    diagnostic: str  # 诊断名称
    table: pd.DataFrame  # 对比指标表
    verdict: str  # 判读结论（结构化文本）
    details: dict[str, object] = field(default_factory=dict)


def hardware_vs_simulation(
    hardware_batches: list[SampleBatch],
    simulation_batches: list[SampleBatch],
) -> DiagnosticResult:
    """诊断一：真机 vs 仿真样本对照（§7.1）。

    对比同源编码态下真机与仿真样本的边际/二阶统计差异与系统效率 η
    的影响。判读：差异是否随 η 单调变化——若是，装置噪声为一阶因素，
    特征构造须基于标定后的标准化响应。
    """
    raise NotImplementedError  # P3：等真机样本


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
            out[order, j] = np.quantile(self._marginals[j], u)
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
