"""E6 阶段 1：单折小规模端到端验证（fold 0，~2 万行子集，1M shots）。

链路：26 维 extended_plus → GroupedQuantileEncoder（train 窗 fit/冻结）→
A/B 组寻址 → pattern 直方图（列打包引擎）→ 基选择（train 行 1000 行抽样、
无泄漏）→ 特征矩阵（A top-3000 + B 1024，计数→归一化→sigmoid，归一化参数
只用 train 行）。

红线：全程只用 fold 0 的 train/valid；test 段不进入任何计算。
用法：``PYTHONPATH=src .venv/bin/python scripts/e6_phase1_single_fold.py``
产物：``artifacts/e6_paper_replica/phase1/``。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qgbs.backtest.rolling import WindowSpec, rolling_splits
from qgbs.data.adapters.csv_adapter import CsvAdapter
from qgbs.factors.finance import build_finance_features
from qgbs.factors.neutralization import MARKET_COL, JointFactorModel, build_alpha_label
from qgbs.factors.pipeline import FactorPipeline
from qgbs.gbs.encoder import GroupedQuantileEncoder
from qgbs.gbs.pattern_service import ColumnPackedStore, top_n_patterns
from qgbs.models.readout import RidgeReadout
from qgbs.models.selection import _per_date_rank_ic

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "e6_paper_replica" / "phase1"
N_SHOTS = 1_000_000
N_ROWS = 20_000  # 行子集规模（train+valid）
N1, N2, N_TOP = 1000, 2000, 3000  # 基选择：抽 1000 train 行 × 各取 top-2000 → top-3000
SEED = 42


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log: dict = {"leak_audit": []}

    # --- 数据与 fold 0 ---
    adapter = CsvAdapter(ROOT / "data_store")
    pool = adapter.load_asset_pool_history("zz800")
    stock_records = pool.records[~pool.records["is_cash_asset"]]
    symbols = sorted(stock_records["symbol"].unique())
    industry = stock_records.drop_duplicates("symbol").set_index("symbol")["citics_industry_l1"]
    market = adapter.load_daily_market(symbols, "2023-01-03", "2025-12-31")
    msf = adapter.load_market_sector_factor("2023-01-03", "2025-12-31")
    close = market.rows.pivot(index="trade_date", columns="symbol", values="close").sort_index()
    ret_wide = market.rows.pivot(index="trade_date", columns="symbol", values="daily_return").sort_index()
    factor_df = pd.concat([msf.index_returns["000906.SH"].rename(MARKET_COL), msf.industry_index_returns], axis=1)
    feats = build_finance_features(market, feature_set="extended_plus",
                                   market_returns=msf.index_returns["000906.SH"],
                                   industry_index_returns=msf.industry_index_returns,
                                   industry_map=industry,
                                   status=adapter.load_trading_status(symbols, "2023-01-03", "2025-12-31"))
    spec = WindowSpec(train="1y", valid="3m", test="1m", purge="5d", embargo="2d")
    split = next(iter(rolling_splits(close.index, spec)))  # fold 0：数据起点边界，warmup NaN 最多
    print(f"fold 0: train {split.train[0].date()}~{split.train[-1].date()}, "
          f"valid {split.valid[0].date()}~{split.valid[-1].date()}（不碰 test {split.test[0].date()}~）", flush=True)
    log["leak_audit"].append("仅用 fold 0 的 train/valid 窗；test 段（{} 起）未读取".format(split.test[0].date()))

    jfm = JointFactorModel().fit(ret_wide.loc[split.train], industry, factor_df.loc[split.train])
    labels = build_alpha_label(close, industry, 21, factor_df, model=jfm)
    pipe = FactorPipeline()
    train_mask = feats.index.get_level_values("date").isin(split.train)
    pipe.fit(feats[train_mask].dropna())
    feats_p = pipe.transform(feats)

    # --- 分组编码（train 窗拟合分位边界 → 冻结） ---
    enc = GroupedQuantileEncoder().fit(feats_p[train_mask].dropna()).freeze()
    log["leak_audit"].append("分位边界仅由 fold 0 train 窗拟合（GroupedQuantileEncoder.fit），冻结后 transform")
    tv_mask = feats_p.index.get_level_values("date").isin(split.train.union(split.valid))
    idx_tv = feats_p.index[tv_mask]
    addr = enc.freeze().transform(feats_p.loc[idx_tv])  # train+valid 行地址
    print(f"train+valid 行数 {len(idx_tv)}，A 组地址 {addr.mode_index_a.shape}、B 组 {addr.mode_index_b.shape}", flush=True)

    # 行子集抽样（seed 42）
    rng = np.random.default_rng(SEED)
    take = np.sort(rng.choice(len(idx_tv), size=min(N_ROWS, len(idx_tv)), replace=False))
    idx_sub = idx_tv[take]
    addr_a_sub, addr_b_sub = addr.mode_index_a[take], addr.mode_index_b[take]
    train_rows = np.flatnonzero(idx_sub.get_level_values("date").isin(split.train))
    print(f"行子集 {len(idx_sub)}（其中 train {len(train_rows)}）", flush=True)

    # --- 列打包（1M shots，一次性转置） ---
    t0 = time.time()
    packed = np.load(ROOT / "sample_store/hardware/hw-5M-b0/packed.npy", mmap_mode="r").reshape(5_000_000, 1022)[:N_SHOTS]
    store = ColumnPackedStore.from_row_packed(packed, 8176)
    print(f"列打包转置 {time.time()-t0:.1f}s，存储 {store.cols.nbytes/1e6:.0f}MB", flush=True)

    # --- 基选择（论文口径：train 行随机 1000 行 → 各行 A 组 top-2000 → 汇总 top-3000）---
    t0 = time.time()
    pick = rng.choice(len(train_rows), size=min(N1, len(train_rows)), replace=False)
    basis_rows = train_rows[np.sort(pick)]
    log["leak_audit"].append(f"基选择仅用在 fold 0 train 窗内的 {len(basis_rows)} 行（seed=42）")
    freq_sum = np.zeros(65536, dtype=np.float64)
    for r in basis_rows:
        h = store.histogram(addr_a_sub[r])
        freq_sum += h  # 汇总口径：频率和（等价于行内计数和的均值排序）
    basis_a = top_n_patterns(freq_sum, N_TOP)
    basis_b = np.arange(1024)  # B 组 1024 种全保留（设计文档 §6 决策 2）
    print(f"基选择 {time.time()-t0:.1f}s：A 组 top-{len(basis_a)}（{N1} 行×top-{N2} 汇总），B 组全 {len(basis_b)}", flush=True)
    np.save(OUT / "basis_a.npy", basis_a)
    np.save(OUT / "basis_b.npy", basis_b)

    # --- 特征矩阵（行子集 × (3000+1024)） ---
    t0 = time.time()
    n_sub = len(idx_sub)
    X_pat = np.empty((n_sub, len(basis_a) + len(basis_b)), dtype=np.float32)
    for r in range(n_sub):
        ha = store.histogram(addr_a_sub[r])
        hb = store.histogram(addr_b_sub[r])
        X_pat[r, : len(basis_a)] = ha[basis_a] * N_SHOTS  # 计数口径
        X_pat[r, len(basis_a):] = hb * N_SHOTS
        if (r + 1) % 5000 == 0:
            print(f"  特征构建 {r+1}/{n_sub}（{time.time()-t0:.0f}s）", flush=True)
    build_s = time.time() - t0
    print(f"特征矩阵构建 {build_s:.1f}s（{n_sub} 行 × {X_pat.shape[1]} 列）", flush=True)

    # --- 归一化 + sigmoid（归一化参数只用 train 行） ---
    tr_X = X_pat[train_rows]
    mu, sd = tr_X.mean(axis=0), tr_X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    log["leak_audit"].append("归一化 mean/std 仅由行子集 ∩ train 窗估计")
    X_sig = (1.0 / (1.0 + np.exp(-(X_pat - mu) / sd))).astype(np.float32)
    feat_index = idx_sub
    cols = [f"pat_a_{i}" for i in range(len(basis_a))] + [f"pat_b_{i}" for i in range(len(basis_b))]
    # 落盘用 npy（项目惯例：sample_store 已因避免 pyarrow 依赖选 npz/json）
    np.save(OUT / "features_sigmoid.npy", X_sig)
    np.save(OUT / "features_count.npy", X_pat)
    (OUT / "features_index.json").write_text(json.dumps(
        {"index": [f"{d.date()}|{s}" for d, s in feat_index], "columns": cols},
        ensure_ascii=False), encoding="utf-8")
    print(f"特征矩阵落盘（sigmoid/count 两版）", flush=True)

    # --- 检查项 1：稀疏度与值域 ---
    sparsity = float((X_pat == 0).mean())
    sig_range = (float(X_sig.min()), float(X_sig.max()))
    print(f"稀疏度（计数为 0 占比）{sparsity:.3f}；sigmoid 值域 [{sig_range[0]:.4f}, {sig_range[1]:.4f}]", flush=True)

    # --- 检查项 3：与边际特征的相关结构 ---
    # 对照：Q1 口径边际特征 = 该行 A/B 寻址模式的边际频率（同一 1M 列存储上 popcount）
    from numpy import bitwise_count
    marg_freq = bitwise_count(store.cols).sum(axis=1) / store.n_shots  # (8176,)
    marg_feats = np.empty((n_sub, 26), dtype=np.float64)
    marg_feats[:, :16] = marg_freq[addr_a_sub]
    marg_feats[:, 16:] = marg_freq[addr_b_sub]
    # 逐列 Pearson：pattern 列（sigmoid 版）× 26 维边际列
    P = X_sig - X_sig.mean(axis=0)
    M = marg_feats - marg_feats.mean(axis=0)
    Pn = P / np.where(P.std(axis=0) > 0, P.std(axis=0), 1.0)
    Mn = M / np.where(M.std(axis=0) > 0, M.std(axis=0), 1.0)
    corr_pm = (Pn.T @ Mn) / n_sub  # (4024, 26)
    abs_corr = np.abs(corr_pm)
    mean_abs = float(abs_corr.mean())
    flat = np.dstack(np.unravel_index(np.argsort(-abs_corr, axis=None)[:10], abs_corr.shape))[0]
    top_pairs = [(f"{cols[i]}↔marg_f{j}", float(abs_corr[i, j])) for i, j in flat]
    print(f"pattern vs 边际：平均 |corr|={mean_abs:.4f}；top 对 {top_pairs[:5]}", flush=True)

    # --- 检查项 4：valid IC 探针（sanity check，不作结论） ---
    idx_sub_list = idx_sub
    df_sig = pd.DataFrame(X_sig, index=idx_sub_list, columns=cols)
    X_tr = df_sig.iloc[train_rows]
    y_tr = labels.reindex(X_tr.index)
    ridge = RidgeReadout(alpha=1.0).fit(X_tr, y_tr)
    va_rows = np.flatnonzero(idx_sub.get_level_values("date").isin(split.valid))
    X_va = df_sig.iloc[va_rows]
    y_va = labels.reindex(X_va.index)
    ic_pat = _per_date_rank_ic(ridge.predict(X_va), y_va).mean()
    # 同折 26 经典 + ridge
    Xc_tr = feats_p.loc[idx_sub_list].iloc[train_rows]
    ridge_c = RidgeReadout(alpha=1.0).fit(Xc_tr, y_tr)
    ic_classic = _per_date_rank_ic(ridge_c.predict(feats_p.loc[idx_sub_list].iloc[va_rows]), y_va).mean()
    # 拼接 26 经典 + pattern（论文口径）
    Xb_tr = pd.concat([Xc_tr, X_tr], axis=1)
    ridge_b = RidgeReadout(alpha=1.0).fit(Xb_tr, y_tr)
    Xb_va = pd.concat([feats_p.loc[idx_sub_list].iloc[va_rows], X_va], axis=1)
    ic_blend = _per_date_rank_ic(ridge_b.predict(Xb_va), y_va).mean()
    print(f"valid IC 探针（fold 0 行子集）：pattern-only {ic_pat:+.4f} / 26经典 {ic_classic:+.4f} / 拼接 {ic_blend:+.4f}", flush=True)

    # --- 报告 ---
    total_s = time.time() - t_start
    log.update({
        "fold": "fold 0（train %s~%s, valid %s~%s）" % (split.train[0].date(), split.train[-1].date(), split.valid[0].date(), split.valid[-1].date()),
        "n_shots": N_SHOTS, "n_rows": int(n_sub), "n_train_rows": int(len(train_rows)),
        "basis": {"n1": N1, "n2": N2, "N": len(basis_a), "b_group": "全 1024 保留"},
        "feature_matrix": {"shape": list(X_sig.shape), "sparsity": sparsity,
                           "sigmoid_range": sig_range},
        "corr_vs_marginal": {"mean_abs": mean_abs, "top_pairs": top_pairs},
        "valid_ic_probe": {"pattern_only": float(ic_pat), "classic26": float(ic_classic), "blended": float(ic_blend)},
        "timing": {"colpack_s": None, "basis_s": None, "build_s": build_s, "total_s": total_s},
        "leak_audit_note": "全程仅用 fold 0 train/valid；test 段未读取",
    })
    (OUT / "phase1_report.json").write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n总耗时 {total_s:.0f}s；产物已写入 {OUT}", flush=True)


if __name__ == "__main__":
    main()
