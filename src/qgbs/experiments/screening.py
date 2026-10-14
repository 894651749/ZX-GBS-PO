"""三闸门 GBS 特征筛选器（E5：二阶/高阶候选池 → 稳定且有效的特征子集）。

筛选口径（参数冻结：ρ₀=0.8、δ=0.012、k=50、跨折一致性 60%、种子 42）：

1. **闸门 1 采样稳定性**（无监督）：样本批次切分为 4×50k 子批次
   （:func:`~qgbs.gbs.statistics.subbatch_resample`），各子批次独立构建词表
   （二阶相关 + 高阶基频率），同一批地址行的候选特征值跨子批次两两
   Pearson 相关取平均 ≥ ρ₀ 通过；
2. **闸门 2 预测有效性**（仅用本折 train+valid）：候选特征对标签的逐日
   rank IC（按日向量化秩相关），通过条件 = |IC 均值| ≥ δ 且 |IC 均值|
   排名在 top-k 内且 train 段与 valid 段 IC 均值同号；
3. **闸门 3 因果跨折一致性**：最小历史规则——可用折数（含本折）≤ 5
   时筛选集为空（退回 Q1 纯边际口径）；历史充足后，第 k 折最终筛选集 =
   本折过闸 1∩2 且在第 1..k 折中通过比例 ≥ 60%（分母固定全历史折数）
   且各通过折 IC 均值同号的特征。绝不允许使用 j>k 折的信息（泄漏）。

候选池：extended_plus 26 维编码的二阶联合特征（C(26,2)=325）+ 高阶基
特征（N=3000）。边际族不进入筛选（已整族证有效）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from qgbs.gbs.backends.base import SampleBatch
from qgbs.gbs.basis import CandidateBasis
from qgbs.gbs.statistics import higher_order_stats, second_order_stats, subbatch_resample


@dataclass(frozen=True)
class ScreeningParams:
    """三闸门筛选参数（E5 冻结值）。"""

    rho0: float = 0.8  # 闸门 1：跨子批次平均相关下限
    delta: float = 0.012  # 闸门 2：|IC 均值| 下限
    top_k: int = 50  # 闸门 2：|IC 均值| 排名前 k
    consistency_ratio: float = 0.6  # 闸门 3：跨折通过比例下限
    min_history: int = 5  # 闸门 3：可用折数（含本折）≤ 此值时筛选集为空（退回 Q1 纯边际口径）
    n_sub: int = 4  # 闸门 1 子批次数
    seed: int = 42
    gate1_n_rows: int = 5000  # 闸门 1 代表性地址行抽样数


@dataclass(frozen=True)
class Vocab:
    """词表：二阶相关矩阵 + 高阶基频率（筛选只需这两族）。"""

    corr: np.ndarray  # (n_modes, n_modes) float64
    basis: np.ndarray  # (n_basis, order) 全局模式编号
    basis_rates: np.ndarray  # (n_basis,)


def build_vocab(batch: SampleBatch, basis: CandidateBasis) -> Vocab:
    """从一个样本批次构建词表（分块二阶统计，内存安全）。"""
    corr = second_order_stats(batch, "correlation")
    rates = higher_order_stats(batch, basis).toarray().ravel()
    return Vocab(corr=corr, basis=np.asarray(basis.basis), basis_rates=rates)


def candidate_names(n_features: int, n_basis: int) -> list[str]:
    """候选特征名：gbs_pair_f{i}_f{j}（组合序，与 GBSFeatureService 一致）+ gbs_ho_{k}。"""
    pairs = [f"gbs_pair_f{i}_f{j}" for i, j in combinations(range(n_features), 2)]
    return pairs + [f"gbs_ho_{k}" for k in range(n_basis)]


def candidate_values(
    vocab: Vocab,
    addr_mode_index: np.ndarray,
    n_features: int,
    *,
    chunk_rows: int = 40_000,
) -> np.ndarray:
    """候选特征值矩阵（R, n_pairs + n_basis），float32 分块求值。

    二阶：行地址对查表 corr[a_i, a_j]；高阶：软激活（地址集合与基组合的
    重合比例 × 基频率，与 GBSFeatureService 的 activation="soft" 同口径）。
    """
    R = addr_mode_index.shape[0]
    pairs = list(combinations(range(n_features), 2))
    n_out = len(pairs) + len(vocab.basis_rates)
    out = np.empty((R, n_out), dtype=np.float32)
    for s in range(0, R, chunk_rows):
        a = addr_mode_index[s : s + chunk_rows]
        block = np.empty((a.shape[0], n_out), dtype=np.float32)
        for k, (i, j) in enumerate(pairs):
            block[:, k] = vocab.corr[a[:, i], a[:, j]]
        order = vocab.basis.shape[1] if len(vocab.basis) else 0
        for k, combo in enumerate(vocab.basis):
            hits = np.zeros(a.shape[0], dtype=np.float32)
            for m in combo:
                hits += (a == m).any(axis=1)
            block[:, len(pairs) + k] = hits * np.float32(vocab.basis_rates[k] / max(order, 1))
        out[s : s + chunk_rows] = block
    return out


def _daily_rank_ic_matrix(values: np.ndarray, y: np.ndarray, day_slices: list[np.ndarray]) -> np.ndarray:
    """按日分组的秩相关矩阵：返回 (n_days, n_features)。

    逐日对特征列与标签做秩变换后一次性矩阵乘（等价逐日 Spearman），
    避免逐特征 groupby 的开销（3325 列 × 数百日）。
    """
    n_days, n_feat = len(day_slices), values.shape[1]
    out = np.full((n_days, n_feat), np.nan)
    for d, idx in enumerate(day_slices):
        X = values[idx]
        yy = y[idx]
        ok = ~np.isnan(yy)
        n = int(ok.sum())
        if n < 3:
            continue
        X = X[ok]
        yy = yy[ok]
        rx = np.argsort(np.argsort(X, axis=0), axis=0).astype(float)
        ry = np.argsort(np.argsort(yy)).astype(float)
        rx -= rx.mean(axis=0)
        ry -= ry.mean()
        denom = np.sqrt((rx**2).sum(axis=0)) * float(np.sqrt((ry**2).sum()))
        denom = np.where(denom > 0, denom, np.nan)
        out[d] = (rx * ry[:, None]).sum(axis=0) / denom
    return out


def gate1_stability(
    batch: SampleBatch,
    basis: CandidateBasis,
    addr_rows: np.ndarray,
    n_features: int,
    params: ScreeningParams,
) -> tuple[np.ndarray, np.ndarray]:
    """闸门 1：候选特征在 4 个子批次词表下的取值稳定性。

    Returns:
        (passed mask (n_candidates,), 平均两两相关 (n_candidates,))。
    """
    rng = np.random.default_rng(params.seed)
    n_addr = addr_rows.shape[0]
    take = rng.choice(n_addr, size=min(params.gate1_n_rows, n_addr), replace=False)
    addr_sub = addr_rows[np.sort(take)]
    subs = subbatch_resample(batch, params.n_sub, seed=params.seed)
    vals = []
    for sb in subs:
        v = build_vocab(sb, basis)
        vals.append(candidate_values(v, addr_sub, n_features))
    arr = np.stack(vals)  # (n_sub, n_rows, n_candidates)
    n_cand = arr.shape[2]
    avg_corr = np.full(n_cand, np.nan)
    for c in range(n_cand):
        sub_vals = arr[:, :, c]  # (n_sub, n_rows)
        cors = []
        for i in range(len(subs)):
            for j in range(i + 1, len(subs)):
                a_, b_ = sub_vals[i], sub_vals[j]
                # 常数列（float32 残留 std ~1e-8 量级，不能用 ==0 判断）：
                # 相关未定义，按取值是否一致处置——一致视为稳定（1），否则 0
                if float(a_.std()) < 1e-6 or float(b_.std()) < 1e-6:
                    same = (
                        float(a_.std()) < 1e-6 and float(b_.std()) < 1e-6
                        and np.allclose(a_, b_, atol=1e-6)
                    )
                    cors.append(1.0 if same else 0.0)
                    continue
                cors.append(float(np.corrcoef(a_, b_)[0, 1]))
        avg_corr[c] = float(np.mean(cors)) if cors else float("nan")
    passed = np.nan_to_num(avg_corr, nan=0.0) >= params.rho0
    return passed, avg_corr


def gate2_predictive(
    values: np.ndarray,
    labels: pd.Series,
    index: pd.MultiIndex,
    train_dates: pd.DatetimeIndex,
    valid_dates: pd.DatetimeIndex,
    params: ScreeningParams,
) -> tuple[np.ndarray, pd.DataFrame]:
    """闸门 2：逐候选特征对标签的逐日 rank IC（仅用 train+valid）。

    Returns:
        (passed mask, 明细表 DataFrame: ic_mean / ic_train / ic_valid / rank)。
    """
    dates = index.get_level_values("date")
    uniq = pd.DatetimeIndex(dates.unique()).sort_values()
    day_slices = [np.flatnonzero(dates == d) for d in uniq]
    y = labels.reindex(index).to_numpy(dtype=float)
    ic_daily = _daily_rank_ic_matrix(values, y, day_slices)  # (n_days, n_cand)
    tr_mask = uniq.isin(train_dates)
    va_mask = uniq.isin(valid_dates)
    tv_mask = tr_mask | va_mask  # 闸门 2 口径：仅用本折 train+valid，绝不触及 test
    if (~tv_mask).any():
        raise ValueError("gate2_predictive 输入含 train+valid 之外的日期（防泄漏红线）")
    ic_mean = np.nanmean(ic_daily[tv_mask], axis=0) if tv_mask.any() else np.full(ic_daily.shape[1], np.nan)
    ic_tr = np.nanmean(ic_daily[tr_mask], axis=0) if tr_mask.any() else np.full(ic_daily.shape[1], np.nan)
    ic_va = np.nanmean(ic_daily[va_mask], axis=0) if va_mask.any() else np.full(ic_daily.shape[1], np.nan)
    abs_ic = np.abs(ic_mean)
    # top-k：|IC 均值| 降序排名（从 1 起），NaN 排最后
    order = np.argsort(-np.nan_to_num(abs_ic, nan=-np.inf))
    rank = np.empty(len(abs_ic), dtype=int)
    rank[order] = np.arange(1, len(abs_ic) + 1)
    same_sign = np.sign(ic_tr) == np.sign(ic_va)
    passed = (abs_ic >= params.delta) & (rank <= params.top_k) & same_sign
    detail = pd.DataFrame({"ic_mean": ic_mean, "ic_train": ic_tr, "ic_valid": ic_va, "rank": rank})
    return passed, detail


@dataclass(frozen=True)
class FoldGateRecord:
    """单折闸门 1∩2 结果（闸门 3 的输入）。"""

    split_id: int
    passed: frozenset[str]
    ic_mean: dict[str, float]  # 本折过闸特征的 IC 均值（方向一致性用）


def gate3_consistency(
    records: list[FoldGateRecord],
    names: list[str],
    params: ScreeningParams,
) -> frozenset[str]:
    """闸门 3（因果）：当前折 = records[-1]，仅用 records 内（折 ≤ 当前）信息。

    最小历史规则：可用折数（含本折，len(records)）≤ ``params.min_history``
    时筛选集为空（该折退回 Q1 纯边际口径，不使用任何筛选特征）——防止
    前若干折因"比例口径"在噪声上全员入选。历史充足后按正常规则：
    当前折过闸 1∩2 且跨折通过比例 ≥ consistency_ratio（分母 = 固定全
    历史折数 len(records)）且各通过折 IC 均值同号。
    """
    if not records:
        raise ValueError("records 为空")
    if len(records) <= params.min_history:
        return frozenset()
    current = records[-1]
    out = set()
    for feat in current.passed:
        n_pass = sum(feat in r.passed for r in records)
        if n_pass / len(records) < params.consistency_ratio:
            continue
        signs = {
            np.sign(r.ic_mean[feat]) for r in records
            if feat in r.passed and np.isfinite(r.ic_mean.get(feat, np.nan))
        }
        if len(signs) == 1:
            out.add(feat)
    return frozenset(out)


def near_miss_diagnostic(
    records: list[FoldGateRecord],
    names: list[str],
    n_folds_total: int,
) -> dict:
    """near-miss 诊断：全部候选的跨折通过（闸 1∩2）次数分布 + top 明细。

    只用各折 train/valid 的闸门结果聚合，不涉及任何 test 段。
    """
    counts = {n: sum(n in r.passed for r in records) for n in names}
    hist = pd.Series(counts).value_counts().sort_index()
    histogram = {int(k): int(v) for k, v in hist.items()}
    # top 候选：按通过折数降序（同数按名字典序保证确定性）
    top = sorted(names, key=lambda n: (-counts[n], n))[:20]
    top_detail = []
    for n in top:
        signs = {
            int(np.sign(r.ic_mean[n])) for r in records
            if n in r.passed and np.isfinite(r.ic_mean.get(n, np.nan))
        }
        top_detail.append({
            "feature": n,
            "n_folds_passed": counts[n],
            "ic_sign_consistent": len(signs) == 1,
            "ic_signs": sorted(signs),
        })
    return {
        "n_folds_evaluated": len(records),
        "n_folds_total": n_folds_total,
        "histogram": histogram,  # 通过折数 → 候选数
        "top_candidates": top_detail,
    }
