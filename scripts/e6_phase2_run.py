"""E6 阶段 2：双臂论文复现实验（P1 冻结协议 20 折，1M shots，多进程列打包）。

每折流程（设计文档 §3 阶段 2）：
1. GroupedQuantileEncoder 在该折 train 窗 fit 分位边界后冻结；
2. 基选择（无泄漏）：train 行 seed=42 抽 n1=1000 行 → A 组 top-3000 pattern，
   B 组 1024 全保留；
3. 1M shots 列打包存储（转置一次落盘缓存、worker 进程 mmap 只读共享）构建
   全部行的 A3000+B1024 计数特征 → train 行统计归一化 → sigmoid；
4. 臂 A（论文忠实）：26 经典 + 4024 pattern 全量拼接 + ridge（λ valid 择优）；
5. 臂 B（择优拼接，规则冻结）：候选 4024 sigmoid 特征，按该折 valid rank IC
   绝对值降序贪心、跳过与已选集合（26 经典+已选）|corr|≥0.8 者（corr 用
   train 行），K ∈ {10,25,50,100,200} 按 ridge valid IC 择优后拼接 ridge。

防泄漏：分位边界/基选择/归一化/筛选/λ 全部只用各折 train（必要时 valid）
数据；test 仅用于最终 IC 评估（一次性使用）。

用法（本地/远端通用）：
``PYTHONPATH=src python scripts/e6_phase2_run.py --shots 1000000 --processes 8``
断点续跑：同一 --outdir 重跑同一命令即自动跳过已完成折（fold_results.jsonl）。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qgbs.backtest.metrics import rank_ic
from qgbs.backtest.rolling import WindowSpec, rolling_splits
from qgbs.backtest.rolling import protocol_splits as protocol_splits_
from qgbs.data.adapters.csv_adapter import CsvAdapter
from qgbs.experiments.runner import load_config
from qgbs.factors.finance import build_finance_features
from qgbs.factors.neutralization import MARKET_COL, JointFactorModel, build_alpha_label
from qgbs.factors.pipeline import FactorPipeline
from qgbs.gbs.encoder import GroupedQuantileEncoder
from qgbs.gbs.pattern_service import ColumnPackedStore, top_n_patterns
from qgbs.models.readout import RidgeReadout
from qgbs.models.selection import _per_date_rank_ic, search_hyperparams

ROOT = Path(__file__).resolve().parents[1]
N1, N2, N_TOP = 1000, 2000, 3000
K_GRID = (10, 25, 50, 100, 200)
ALPHA_GRID = [0.1, 1.0, 10.0]  # 与 Q1H 口径一致
CORR_SKIP = 0.8
# 参照系（历史冻结结果，G0/E4.4 产物；不重跑）
REFERENCES = {"C2ext-plus": -0.0163, "Q1H-plus": 0.0169, "Q1H-plus−C2ext-plus": 0.0332}


# ---------------------------------------------------------------------------
# 多进程 worker（模块级，可 pickle；spawn/fork 均兼容）
# ---------------------------------------------------------------------------

def _build_block(task: dict) -> np.ndarray:
    """构建一块行的计数特征（worker 内 mmap 只读打开列存储）。"""
    cols = np.load(task["colpack_path"], mmap_mode="r")
    store = ColumnPackedStore(cols=cols, n_shots=task["n_shots"], n_modes=cols.shape[0])
    addr_a, addr_b = task["addr_a"], task["addr_b"]
    basis_a = task["basis_a"]
    n_a = len(basis_a)
    out = np.empty((addr_a.shape[0], n_a + task["n_b"]), dtype=np.float32)
    for r in range(addr_a.shape[0]):
        out[r, :n_a] = store.histogram(addr_a[r])[basis_a] * task["n_shots"]
        out[r, n_a:] = store.histogram(addr_b[r]) * task["n_shots"]
    return out


# ---------------------------------------------------------------------------
# 臂 B：贪心筛选（规则冻结，见模块 docstring）
# ---------------------------------------------------------------------------

def greedy_select(
    cand: pd.DataFrame,
    classic: pd.DataFrame,
    y_valid: pd.Series,
    corr_ref: pd.DataFrame,
    k_max: int,
    corr_skip: float = CORR_SKIP,
) -> list[str]:
    """按 valid rank IC 绝对值降序贪心：跳过与已选集合 |corr|≥corr_skip 者。

    Args:
        cand: 候选 pattern 特征（train+valid 行）；
        classic: 26 经典特征（同索引）——已选集合的初始成员；
        y_valid: valid 标签（IC 排序只用 valid 段行）；
        corr_ref: 计算相关的参考行集（train 段）；
        k_max: 最多入选数。
    """
    va_mask = cand.index.get_level_values("date").isin(
        y_valid.index.get_level_values("date").unique())
    ic_abs = {
        c: abs(float(_per_date_rank_ic(cand.loc[va_mask, c], y_valid).mean()))
        for c in cand.columns
    }
    selected: list[str] = []
    base_corr = corr_ref[classic.columns].to_numpy(dtype=float)
    for c in sorted(ic_abs, key=lambda c: -ic_abs[c]):
        if len(selected) >= k_max:
            break
        col = corr_ref[c].to_numpy(dtype=float)
        pool = np.column_stack([base_corr] + [corr_ref[s].to_numpy(dtype=float) for s in selected]) \
            if selected else base_corr
        # 与已选集合（26 经典 + 已选 pattern）的 |corr|
        pool_std = pool.std(axis=0)
        c_std = col.std()
        if c_std == 0:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            covs = ((pool - pool.mean(axis=0)) * (col - col.mean())[:, None]).mean(axis=0) / (
                pool_std * c_std)
        if np.nanmax(np.abs(covs)) >= corr_skip:
            continue
        selected.append(c)
    return selected


def _safe_ic_mean(pred: pd.Series, y: pd.Series) -> float:
    """逐日 rank IC 均值；空/退化输入返回 nan（不崩）。"""
    if len(pred) == 0:
        print("警告：IC 输入为空，记 nan", flush=True)
        return float("nan")
    if isinstance(pred, pd.DataFrame):
        pred = pred.squeeze(axis=1)
    ics = _per_date_rank_ic(pred, y.reindex(pred.index))
    arr = np.asarray(ics, dtype=float).ravel()
    return float(np.nanmean(arr)) if arr.size else float("nan")


def _fit_ridge_ic(X_tr, y_tr, X_va, y_va) -> float:
    """λ 在 valid 择优后的 valid IC（用于 K 择优；train/valid 口径，不碰 test）。"""
    res = search_hyperparams("ridge", X_tr, y_tr, X_va, y_va, {"alpha": ALPHA_GRID})
    m = RidgeReadout(alpha=float(res.best_params.get("alpha", 1.0))).fit(X_tr, y_tr)
    ics = _per_date_rank_ic(m.predict(X_va), y_va)
    return float(ics.mean()) if len(ics) else float("nan")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E6 阶段 2 双臂论文复现实验")
    p.add_argument("--shots", type=int, default=1_000_000)
    p.add_argument("--processes", type=int, default=8)
    p.add_argument("--folds", type=str, default="all",
                   help="'all' 或逗号分隔折号（如 0,1,2 或 0-3）")
    p.add_argument("--outdir", type=str, default="artifacts/e6_paper_replica/phase2")
    p.add_argument("--max-rows", type=int, default=None, help="调试截断：每折行数上限")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-features", choices=["none", "test", "all"], default="none")
    return p.parse_args(argv)


def _parse_folds(spec: str) -> set[int] | None:
    if spec == "all":
        return None
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _done_folds(jsonl: Path) -> set[int]:
    if not jsonl.exists():
        return set()
    return {json.loads(line)["split_id"] for line in jsonl.read_text().splitlines() if line.strip()}


def main(argv=None) -> None:
    args = _parse_args(argv)
    out = Path(args.outdir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "fold_results.jsonl"
    folds_wanted = _parse_folds(args.folds)
    done = _done_folds(jsonl)

    # --- 数据（与 q4_full_enhanced_hardware_p1ext 冻结口径同源） ---
    cfg = load_config(ROOT / "configs" / "experiments" / "q4_full_enhanced_hardware_p1ext.yaml")
    data_cfg = cfg["data"]
    adapter = CsvAdapter(ROOT / data_cfg["store_dir"])
    pool = adapter.load_asset_pool_history(data_cfg["pool"])
    stock_records = pool.records[~pool.records["is_cash_asset"]]
    symbols = sorted(stock_records["symbol"].unique())
    industry = stock_records.drop_duplicates("symbol").set_index("symbol")["citics_industry_l1"]
    market = adapter.load_daily_market(symbols, data_cfg["start"], data_cfg["end"])
    msf = adapter.load_market_sector_factor(data_cfg["start"], data_cfg["end"])
    close = market.rows.pivot(index="trade_date", columns="symbol", values="close").sort_index()
    ret_wide = market.rows.pivot(index="trade_date", columns="symbol", values="daily_return").sort_index()
    factor_df = pd.concat([msf.index_returns[data_cfg["market_index"]].rename(MARKET_COL),
                           msf.industry_index_returns], axis=1)
    feats = build_finance_features(market, feature_set="extended_plus",
                                   market_returns=msf.index_returns[data_cfg["market_index"]],
                                   industry_index_returns=msf.industry_index_returns,
                                   industry_map=industry,
                                   status=adapter.load_trading_status(symbols, data_cfg["start"], data_cfg["end"]))
    horizon = int(cfg.get("label_horizon", 21))
    spec = WindowSpec(**{k: cfg["rolling"][k] for k in ("train", "valid", "test", "purge", "embargo")})

    # --- 列打包存储（转置一次落盘缓存；worker 只读 mmap 共享） ---
    colpack_path = out / f"colpack_{args.shots}.npy"
    if colpack_path.exists():
        print(f"列存储缓存命中 {colpack_path}", flush=True)
    else:
        t0 = time.time()
        packed = np.load(ROOT / "sample_store/hardware/hw-5M-b0/packed.npy",
                         mmap_mode="r").reshape(5_000_000, 1022)[: args.shots]
        store = ColumnPackedStore.from_row_packed(packed, 8176)
        np.save(colpack_path, store.cols)
        print(f"列打包转置 {time.time()-t0:.0f}s → {colpack_path}（{store.cols.nbytes/1e6:.0f}MB）", flush=True)

    import multiprocessing as mp
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")

    # 协议折过滤（剔除 test 窗无再平衡日的尾折）
    protocol_splits = protocol_splits_(close.index, spec)
    rng = np.random.default_rng(args.seed)
    for split in protocol_splits:
        if folds_wanted is not None and split.split_id not in folds_wanted:
            continue
        if split.split_id in done:
            print(f"fold {split.split_id} 已完成，跳过（断点续跑）", flush=True)
            continue
        t_fold = time.time()
        print(f"\n=== fold {split.split_id}（train {split.train[0].date()}~{split.train[-1].date()} "
              f"valid {split.valid[0].date()}~{split.valid[-1].date()} "
              f"test {split.test[0].date()}~{split.test[-1].date()}）===", flush=True)

        jfm = JointFactorModel().fit(ret_wide.loc[split.train], industry, factor_df.loc[split.train])
        labels = build_alpha_label(close, industry, horizon, factor_df, model=jfm)
        pipe = FactorPipeline()
        train_mask = feats.index.get_level_values("date").isin(split.train)
        pipe.fit(feats[train_mask].dropna())
        feats_p = pipe.transform(feats)

        enc = GroupedQuantileEncoder().fit(feats_p[train_mask].dropna()).freeze()
        all_idx = feats_p.index
        addr = enc.transform(feats_p)  # 全期地址（encoder 已冻结，无泄漏）
        addr_a, addr_b = addr.mode_index_a, addr.mode_index_b

        if args.max_rows:
            take = np.sort(rng.choice(len(all_idx), size=min(args.max_rows, len(all_idx)), replace=False))
            row_idx = take
        else:
            row_idx = np.arange(len(all_idx))
        cur_idx = all_idx[row_idx]
        aa, ab = addr_a[row_idx], addr_b[row_idx]
        dates_cur = cur_idx.get_level_values("date")
        tr_rows = np.flatnonzero(dates_cur.isin(split.train))
        va_rows = np.flatnonzero(dates_cur.isin(split.valid))
        te_rows = np.flatnonzero(dates_cur.isin(split.test))

        # --- 基选择（仅 train 行） ---
        t0 = time.time()
        store = ColumnPackedStore(cols=np.load(colpack_path, mmap_mode="r"),
                                  n_shots=args.shots, n_modes=8176)
        pick = rng.choice(len(tr_rows), size=min(N1, len(tr_rows)), replace=False)
        basis_rows = tr_rows[np.sort(pick)]
        freq_sum = np.zeros(65536, dtype=np.float64)
        for r in basis_rows:
            freq_sum += store.histogram(aa[r])
        basis_a = top_n_patterns(freq_sum, N_TOP)
        basis_b_n = 1024
        np.save(out / f"basis_a_fold{split.split_id}.npy", basis_a)
        print(f"基选择 {time.time()-t0:.0f}s（{len(basis_rows)} train 行 → top-{len(basis_a)}）", flush=True)

        # --- 特征构建（多进程分块） ---
        t0 = time.time()
        block = max(2000, int(np.ceil(len(row_idx) / (args.processes * 4))))
        tasks = []
        for s in range(0, len(row_idx), block):
            e = min(s + block, len(row_idx))
            tasks.append({"colpack_path": str(colpack_path), "n_shots": args.shots,
                          "addr_a": aa[s:e], "addr_b": ab[s:e],
                          "basis_a": basis_a, "n_b": basis_b_n})
        if args.processes > 1:
            with ctx.Pool(args.processes) as pool_:
                parts = pool_.map(_build_block, tasks)
        else:
            parts = [_build_block(t) for t in tasks]
        X_cnt = np.concatenate(parts)
        build_s = time.time() - t0
        print(f"特征构建 {build_s:.0f}s（{len(row_idx)} 行 × {X_cnt.shape[1]} 列，{args.processes} 进程）", flush=True)

        # --- 归一化 + sigmoid（参数仅 train 行） ---
        tr_cnt = X_cnt[tr_rows]
        mu, sd = tr_cnt.mean(axis=0), tr_cnt.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        X_sig = (1.0 / (1.0 + np.exp(-(X_cnt - mu) / sd))).astype(np.float32)
        cols = [f"pat_a_{i}" for i in range(len(basis_a))] + [f"pat_b_{i}" for i in range(basis_b_n)]
        df_sig = pd.DataFrame(X_sig, index=cur_idx, columns=cols)
        df_classic = feats_p.loc[cur_idx]

        # --- 两臂评估 ---
        X_tr_c = df_classic.iloc[tr_rows]
        y_tr = labels.reindex(df_classic.iloc[tr_rows].index)
        y_va = labels.reindex(df_classic.iloc[va_rows].index)
        y_te = labels.reindex(df_classic.iloc[te_rows].index)
        X_va_c, X_te_c = df_classic.iloc[va_rows], df_classic.iloc[te_rows]

        def _eval_ridge(X_tr_full, X_va_full, X_te_full) -> dict:
            res = search_hyperparams("ridge", X_tr_full, y_tr, X_va_full, y_va, {"alpha": ALPHA_GRID})
            best_alpha = float(res.best_params.get("alpha", 1.0))
            m = RidgeReadout(alpha=best_alpha).fit(X_tr_full, y_tr)
            ic_va = _safe_ic_mean(m.predict(X_va_full), y_va)
            pred_te = m.predict(X_te_full)
            ic_te = _safe_ic_mean(pred_te, y_te)
            return {"alpha": best_alpha, "valid_ic": ic_va, "test_ic": ic_te}

        # 臂 A：26 经典 + 4024 全量拼接
        X_tr_a = pd.concat([X_tr_c, df_sig.iloc[tr_rows]], axis=1)
        X_va_a = pd.concat([X_va_c, df_sig.iloc[va_rows]], axis=1)
        X_te_a = pd.concat([X_te_c, df_sig.iloc[te_rows]], axis=1)
        arm_a = _eval_ridge(X_tr_a, X_va_a, X_te_a)

        # 臂 B：贪心筛选 + K 择优
        cand = df_sig.iloc[np.concatenate([tr_rows, va_rows])]
        selected = greedy_select(cand, pd.concat([X_tr_c, X_va_c]), y_va,
                                 corr_ref=pd.concat([X_tr_c, df_sig.iloc[tr_rows]], axis=1),
                                 k_max=max(K_GRID))
        k_scores = {}
        for k in K_GRID:
            sel_k = selected[:k]
            X_tr_k = pd.concat([X_tr_c, df_sig[sel_k].iloc[tr_rows]], axis=1)
            X_va_k = pd.concat([X_va_c, df_sig[sel_k].iloc[va_rows]], axis=1)
            k_scores[k] = _fit_ridge_ic(X_tr_k, y_tr, X_va_k, y_va)
        best_k = max(k_scores, key=lambda k: (k_scores[k], -k))  # 同分取小 K
        sel = selected[:best_k]
        X_tr_b = pd.concat([X_tr_c, df_sig[sel].iloc[tr_rows]], axis=1)
        X_va_b = pd.concat([X_va_c, df_sig[sel].iloc[va_rows]], axis=1)
        X_te_b = pd.concat([X_te_c, df_sig[sel].iloc[te_rows]], axis=1)
        arm_b = _eval_ridge(X_tr_b, X_va_b, X_te_b)
        arm_b["k"] = best_k
        arm_b["selected"] = sel
        arm_b["k_scores"] = {str(k): v for k, v in k_scores.items()}

        if args.save_features in ("test", "all"):
            keep = te_rows if args.save_features == "test" else np.arange(len(cur_idx))
            np.save(out / f"features_fold{split.split_id}_{args.save_features}.npy",
                    X_sig[keep])

        rec = {
            "split_id": split.split_id,
            "n_rows": int(len(row_idx)),
            "arm_a": arm_a, "arm_b": {k: v for k, v in arm_b.items() if k != "selected"},
            "arm_b_selected": sel,
            "timing": {"build_s": build_s, "fold_s": time.time() - t_fold},
        }
        with jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"fold {split.split_id} done {rec['timing']['fold_s']:.0f}s："
              f"臂A test_ic={arm_a['test_ic']:+.4f} 臂B test_ic={arm_b['test_ic']:+.4f}（K={best_k}）", flush=True)
        del X_cnt, X_sig, df_sig

    _write_summary(out, jsonl)


def _write_summary(out: Path, jsonl: Path) -> None:
    """汇总两臂 test IC + 参照系（C2/Q1 冻结历史结果）。"""
    if not jsonl.exists():
        return
    from qgbs.experiments.report import block_bootstrap_ci
    recs = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    ic_a = pd.Series({r["split_id"]: r["arm_a"]["test_ic"] for r in recs}).sort_index()
    ic_b = pd.Series({r["split_id"]: r["arm_b"]["test_ic"] for r in recs}).sort_index()
    d = ic_a - ic_b
    summary = {
        "n_folds": len(recs),
        "arm_a": {"test_ic_mean": float(ic_a.mean()), "test_ic_std": float(ic_a.std())},
        "arm_b": {"test_ic_mean": float(ic_b.mean()), "test_ic_std": float(ic_b.std()),
                  "k_dist": pd.Series([r["arm_b"]["k"] for r in recs]).value_counts().sort_index().to_dict()},
        "arm_a_minus_b": {"mean": float(d.mean())},
        "references": REFERENCES,
    }
    if len(d) >= 8:
        lo, hi = block_bootstrap_ci(d, statistic="mean", n_bootstrap=2000, block_length=5, level=0.95, seed=42)
        summary["arm_a_minus_b"]["ci95"] = [float(lo), float(hi)]
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n汇总：臂A test IC {ic_a.mean():+.4f} / 臂B {ic_b.mean():+.4f}（n={len(recs)}）", flush=True)


if __name__ == "__main__":
    main()
