"""E8.2 全 20 折正式实验：四配置（R0/R1/R2/R3'）× δ 网格，fold 级并行。

结构：主进程预构建共享只读数据（样本批次、ρ̂ 词表、特征面板）→ fork 共享；
worker 以折为单位（20 单元），每折内 Q1 alpha 链构建一次、四配置串行评估
（δ 网格在 valid 窗择优，test 窗冻结评估）。断点续跑：fold_results.jsonl
按 (fold, config) 单元增量写，已完成跳过。

用法：``PYTHONPATH=src python scripts/e8_phase2_run.py --processes 16``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qgbs.backtest.engine import BacktestEngine
from qgbs.backtest.metrics import annual_volatility, calmar, information_ratio, sharpe
from qgbs.backtest.rolling import WindowSpec, rolling_splits
from qgbs.backtest.rolling import protocol_splits as protocol_splits_
from qgbs.data.adapters.csv_adapter import CsvAdapter
from qgbs.data.calendar import rebalance_dates
from qgbs.experiments.runner import _build_gbs_panel, _prepare_gbs_backend, load_config
from qgbs.factors.finance import build_finance_features
from qgbs.factors.neutralization import MARKET_COL, JointFactorModel, build_alpha_label
from qgbs.factors.pipeline import FactorPipeline
from qgbs.gbs.encoder import EncoderParams, QuantileEncoder, ensure_default_mappings, resolve_mapping_id
from qgbs.gbs.risk_prior import PriorProvider, build_static_corr
from qgbs.gbs.statistics import second_order_stats
from qgbs.models.readout import RidgeReadout
from qgbs.models.selection import _per_date_rank_ic
from qgbs.portfolio.costs import CostModel
from qgbs.portfolio.optimizer import OptimizerParams, PortfolioConstraints

ROOT = Path(__file__).resolve().parents[1]
DELTA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
CONFIGS = ["R0_lw", "R1_gbs", "R2_random", "R3_static"]

# 主进程预构建的共享只读状态（fork 共享；spawn 下为 None，worker 重建）
_SHARED: dict = {}


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E8.2 全 20 折四配置实验")
    p.add_argument("--processes", type=int, default=16)
    p.add_argument("--folds", type=str, default="all", help="'all' 或 0,2,5 / 1-3")
    p.add_argument("--outdir", type=str, default="artifacts/e8_risk_prior/phase2")
    p.add_argument("--seed", type=int, default=42)
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


def _done_units(jsonl: Path) -> set[tuple[int, str]]:
    if not jsonl.exists():
        return set()
    return {(r["split_id"], r["config"]) for r in
            (json.loads(line) for line in jsonl.read_text().splitlines() if line.strip())}


def _run_engine(shared: dict, split, dates, delta, provider, alpha_panel, labels) -> dict:
    """单配置单窗（valid 或 test）评估。

    口径与 runner.py 标准路径（L534-536/L795-825）一致：
    - alpha 只给窗内月度调仓日（rebalance_dates(freq="1m") ∩ 窗口），
      直接传整窗日频 alpha 会导致引擎在窗内每日调仓；
    - 收益切片到窗口——引擎在全行情区间记账，最后一个调仓日之后的
      漂移段不属于本窗，必须切除（否则跨折指标被同一漂移段主导）。
    """
    eng = BacktestEngine(window_spec=shared["spec"], constraints=shared["constraints"],
                         optimizer_params=shared["opt"], cost_model=shared["cost_model"],
                         risk_prior_delta=delta)
    date_set = set(dates)
    rebal = [d for d in shared["rebal_all"] if d in date_set]
    assert rebal, f"窗口无月度调仓日（{dates[0]}~{dates[-1]}）"
    panel = alpha_panel.loc[alpha_panel.index.isin(rebal)]
    bt = eng.run(shared["market"], shared["status"], risk_free_daily=shared["rf_daily"],
                 alpha_panel=panel, exposure_matrix=shared["exposure"],
                 risk_prior_provider=provider)
    r = bt.returns[bt.returns.index.isin(date_set)]
    nav = (1.0 + r).cumprod()
    return {"sharpe": sharpe(r), "ir": information_ratio(r, shared["bench"]),
            "calmar": calmar(nav), "realized_vol": annual_volatility(r),
            "turnover": float(bt.turnover.mean()), "n_rebalances": len(bt.weights),
            "returns": {str(d.date()): float(v) for d, v in r.items()},
            "nav": {str(d.date()): float(v) for d, v in nav.items()}}


def _safe_fold_ic(pred_panel: pd.DataFrame, labels: pd.Series) -> float:
    """test 窗逐日 rank IC 均值；空/退化窗口返回 nan 并记日志（不崩）。

    防御点：空面板的 groupby.apply 返回 DataFrame 而非 Series（float() 会崩），
    故显式判空 + 结果类型校验。
    """
    if pred_panel.empty or pred_panel.shape[1] == 0:
        print("警告：test 窗 alpha 面板为空，rank IC 记 nan", flush=True)
        return float("nan")
    pred = pred_panel.stack()
    if isinstance(pred, pd.DataFrame):  # 异常形态防御
        pred = pred.squeeze(axis=1)
    pred.index.names = ["date", "symbol"]  # stack 后 names 丢失，groupby 需要
    ics = _per_date_rank_ic(pred, labels.reindex(pred.index))
    arr = np.asarray(ics, dtype=float).ravel()
    return float(np.nanmean(arr)) if arr.size else float("nan")


def _run_fold(split_id: int) -> list[dict]:
    """worker：一折内 Q1 alpha 链 + 四配置评估。"""
    shared = _SHARED
    split = shared["splits"][split_id]
    cfg = shared["cfg"]
    t0 = time.time()

    jfm = JointFactorModel().fit(shared["ret_wide"].loc[split.train], shared["industry"],
                                 shared["factor_df"].loc[split.train])
    labels = build_alpha_label(shared["close"], shared["industry"], shared["horizon"],
                               shared["factor_df"], model=jfm)
    pipe = FactorPipeline()
    train_mask = shared["feats"].index.get_level_values("date").isin(split.train)
    pipe.fit(shared["feats"][train_mask].dropna())
    feats_p = pipe.transform(shared["feats"])
    gbs_panel = _build_gbs_panel(cfg, ["gbs_marginal"], feats_p, split,
                                 shared["backend"], shared["batch"])
    X_all = pd.concat([feats_p, gbs_panel], axis=1)
    X_tr = X_all[train_mask].dropna()
    ridge = RidgeReadout(alpha=1.0).fit(X_tr, labels.reindex(X_tr.index))
    alpha_panel = ridge.predict(X_all).unstack("symbol")

    enc_cfg = dict(cfg.get("encoder") or {})
    ensure_default_mappings(26, 16, 511)
    mid = resolve_mapping_id(str(enc_cfg.get("mapping_id", "identity")), 26, 16, 511)
    enc = QuantileEncoder(
        EncoderParams(n_bins_a=int(enc_cfg.get("n_bins_a", 255)),
                      n_bins_b=int(enc_cfg.get("n_bins_b", 256)),
                      partition_boundary=int(enc_cfg.get("partition_boundary", 255))),
        mapping_id=mid)
    enc.fit(feats_p[train_mask].dropna())
    addr = enc.freeze().transform(feats_p)
    mode_df = pd.DataFrame(addr.mode_index, index=feats_p.index).rename_axis(["date", "symbol"])
    addr_by_date = {d: mode_df.loc[d] for d in mode_df.index.get_level_values("date").unique()}
    static_K = build_static_corr(addr_by_date, shared["rho_hat"], list(split.train.union(split.valid)))

    va_dates, te_dates = pd.DatetimeIndex(split.valid), pd.DatetimeIndex(split.test)
    providers = {
        "R0_lw": None,
        "R1_gbs": PriorProvider("gbs_prior", addr_by_date=addr_by_date, rho_hat=shared["rho_hat"]),
        "R2_random": PriorProvider("random_prior", seed=shared["seed"]),
        "R3_static": PriorProvider("static_gbs_prior", static_corr=static_K),
    }
    # rank IC（四配置共享同一 alpha，风险引擎不改变预测）；退化/空窗记 nan 不崩
    pred_te = alpha_panel.loc[alpha_panel.index.isin(te_dates)]
    ic_te = _safe_fold_ic(pred_te, labels)

    recs = []
    for name, prov in providers.items():
        t1 = time.time()
        if prov is None:
            res = _run_engine(shared, split, te_dates, 1.0, None, alpha_panel, labels)
            recs.append({"split_id": split_id, "config": name, "delta": 1.0,
                         "valid_scores": {}, "rank_ic_test": ic_te,
                         **{k: v for k, v in res.items() if k not in ("returns", "nav")},
                         "returns": res["returns"], "nav": res["nav"],
                         "elapsed_s": time.time() - t1})
            continue
        scores = {}
        for delta in DELTA_GRID:
            r = _run_engine(shared, split, va_dates, delta, prov, alpha_panel, labels)
            scores[str(delta)] = r["sharpe"]
        # 平票时取更大 δ（保守方向：valid 无法区分时偏向退回经典 LW）
        best = max(scores, key=lambda k: (scores[k], float(k)))
        res = _run_engine(shared, split, te_dates, float(best), prov, alpha_panel, labels)
        recs.append({"split_id": split_id, "config": name, "delta": float(best),
                     "valid_scores": scores, "rank_ic_test": ic_te,
                     **{k: v for k, v in res.items() if k not in ("returns", "nav")},
                     "returns": res["returns"], "nav": res["nav"],
                     "elapsed_s": time.time() - t1})
    print(f"fold {split_id} done（{time.time()-t0:.0f}s）", flush=True)
    return recs


def main(argv=None) -> None:
    args = _parse_args(argv)
    out = Path(args.outdir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "fold_results.jsonl"
    folds_wanted = _parse_folds(args.folds)
    done = _done_units(jsonl)

    cfg = load_config(ROOT / "configs/experiments/q1_marginal_hardware_p1extplus.yaml")
    data_cfg = cfg["data"]
    adapter = CsvAdapter(ROOT / data_cfg["store_dir"])
    pool = adapter.load_asset_pool_history(data_cfg["pool"])
    stock_records = pool.records[~pool.records["is_cash_asset"]]
    symbols = sorted(stock_records["symbol"].unique())
    industry = stock_records.drop_duplicates("symbol").set_index("symbol")["citics_industry_l1"]
    market = adapter.load_daily_market(symbols, data_cfg["start"], data_cfg["end"])
    status = adapter.load_trading_status(symbols, data_cfg["start"], data_cfg["end"])
    msf = adapter.load_market_sector_factor(data_cfg["start"], data_cfg["end"])
    close = market.rows.pivot(index="trade_date", columns="symbol", values="close").sort_index()
    ret_wide = market.rows.pivot(index="trade_date", columns="symbol", values="daily_return").sort_index()
    mkt_code = data_cfg["market_index"]
    factor_df = pd.concat([msf.index_returns[mkt_code].rename(MARKET_COL), msf.industry_index_returns], axis=1)
    feats = build_finance_features(market, feature_set="extended_plus",
                                   market_returns=msf.index_returns[mkt_code],
                                   industry_index_returns=msf.industry_index_returns,
                                   industry_map=industry, status=status)
    backend, batch = _prepare_gbs_backend(cfg)
    t0 = time.time()
    rho_hat = second_order_stats(batch, "correlation")
    print(f"ρ̂ 词表 {time.time()-t0:.0f}s", flush=True)

    # 协议折过滤（剔除 test 窗无再平衡日的尾折）——P1 正式协议 20 折（0–19）
    spec = WindowSpec(train="1y", valid="3m", test="1m", purge="5d", embargo="2d")
    protocol_splits = protocol_splits_(close.index, spec)
    assert len(protocol_splits) == 20, f"协议折数 {len(protocol_splits)} ≠ 20"
    print(f"协议折：0~{protocol_splits[-1].split_id}", flush=True)

    cc = adapter.load_constraints_costs("default")
    _SHARED.update({
        "cfg": cfg, "spec": WindowSpec(train="1y", valid="3m", test="1m", purge="5d", embargo="2d"),
        "market": market, "status": status, "close": close, "ret_wide": ret_wide,
        "factor_df": factor_df, "industry": industry, "feats": feats,
        "backend": backend, "batch": batch, "rho_hat": rho_hat,
        "horizon": int(cfg.get("label_horizon", 21)),
        "constraints": PortfolioConstraints(
            weight_cap={s: 0.10 for s in symbols}, cash_range=(0.0, 0.05),
            exposure_bounds={}, max_turnover=None, liquidity_limit=None, blacklist=()),
        "cost_model": CostModel(commission_bps=cc.commission_bps, stamp_tax_bps=cc.stamp_tax_bps,
                                slippage_bps=cc.slippage_bps, impact_coef=cc.impact_coef),
        "opt": OptimizerParams(risk_aversion=5.0, turnover_penalty=0.001, l2_regularization=0.0001),
        "rf_daily": 0.02 / 252.0, "bench": msf.index_returns[mkt_code],
        "exposure": pd.get_dummies(industry).astype(float),
        "rebal_all": rebalance_dates(close.index, spec.test),
        "splits": {s.split_id: s for s in protocol_splits},
        "seed": args.seed,
    })
    # 词表缓存预热（fold 无关）：worker fork 后命中
    s0 = _SHARED["splits"][0]
    pipe0 = FactorPipeline()
    tm0 = feats.index.get_level_values("date").isin(s0.train)
    pipe0.fit(feats[tm0].dropna())
    _build_gbs_panel(cfg, ["gbs_marginal"], pipe0.transform(feats), s0, backend, batch)

    todo = [sid for sid in sorted(_SHARED["splits"])
            if (folds_wanted is None or sid in folds_wanted)
            and (sid, CONFIGS[-1]) not in done]  # 以最后配置为折完成标记
    print(f"待跑折：{todo}", flush=True)

    import gc
    import multiprocessing as mp
    gc.collect()
    gc.freeze()  # 预建对象移入永久代：fork 后保持写时复制共享，抑制 worker 内存膨胀
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    if args.processes > 1 and len(todo) > 1:
        with ctx.Pool(args.processes) as pool_:
            for recs in pool_.imap_unordered(_run_fold, todo):
                with jsonl.open("a", encoding="utf-8") as f:
                    for r in recs:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        for sid in todo:
            for r in _run_fold(sid):
                with jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    _write_summary(out, jsonl)


def _rec_returns(rec: dict) -> pd.Series:
    """jsonl 记录的 returns 字段还原为 DatetimeIndex Series（折间 test 窗互斥，拼接无重复）。"""
    s = pd.Series(rec["returns"], dtype=float)
    s.index = pd.to_datetime(s.index)
    return s


def _write_summary(out: Path, jsonl: Path) -> None:
    """汇总：四配置 × 指标的 20 折均值 + 配对 CI（R1−R0/R2/R3'）。"""
    if not jsonl.exists():
        return
    from qgbs.experiments.report import paired_metric_bootstrap_ci
    recs = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    by_cfg: dict[str, list[dict]] = {}
    for r in recs:
        by_cfg.setdefault(r["config"], []).append(r)
    metrics = ("sharpe", "ir", "calmar", "realized_vol", "turnover", "rank_ic_test")
    summary: dict = {"n_units": len(recs), "configs": {}}
    for name, rs in by_cfg.items():
        summary["configs"][name] = {
            m: float(np.nanmean([r[m] for r in rs])) for m in metrics}
        summary["configs"][name]["delta_dist"] = pd.Series(
            [r["delta"] for r in rs]).value_counts().sort_index().to_dict()
    # 配对 CI：R1 vs 各对照，对 test 日收益序列
    if "R1_gbs" in by_cfg and "R0_lw" in by_cfg:
        bench = _SHARED.get("bench")
        for ref in ("R0_lw", "R2_random", "R3_static"):
            if ref not in by_cfg:
                continue
            ra = {r["split_id"]: r for r in by_cfg["R1_gbs"]}
            rb = {r["split_id"]: r for r in by_cfg[ref]}
            common = sorted(set(ra) & set(rb))
            if len(common) < 8:
                continue
            rets_a = pd.concat([_rec_returns(ra[k]) for k in common]).sort_index()
            rets_b = pd.concat([_rec_returns(rb[k]) for k in common]).sort_index()
            # IR 基准位置化（重采样 Series 为位置索引；基准按交集日期对齐后取 values）
            common_dates = rets_a.index.intersection(rets_b.index)
            if bench is not None:
                common_dates = common_dates.intersection(bench.index)
            bench_pos = pd.Series(bench.loc[common_dates].to_numpy()) if bench is not None else None

            def fn_ir(x: pd.Series) -> float:
                assert bench_pos is not None
                return information_ratio(pd.Series(x.to_numpy()), bench_pos)

            point_lo_hi = {}
            for mname, fn in (("sharpe", sharpe),
                              ("calmar", lambda x: calmar((1 + x).cumprod())),
                              ("ir", fn_ir)):
                if mname == "ir" and bench is None:
                    continue
                point, lo, hi = paired_metric_bootstrap_ci(rets_a, rets_b, fn,
                                                           n_bootstrap=2000, block_length=5, seed=42)
                point_lo_hi[mname] = {"delta": point, "ci95": [lo, hi]}
            summary.setdefault("paired", {})[f"R1−{ref}"] = point_lo_hi
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"汇总已写入 {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
