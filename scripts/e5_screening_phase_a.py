"""E5 阶段 A 干跑：三闸门筛选器在 fold 0–2 上验证（只用 train/valid，红线：不碰 test）。

基座：extended_plus 26 维编码（P1 协议、hw-5M-b0、200k shots、b=255、identity）。
候选池：二阶联合特征 C(26,2)=325 + 高阶基特征 N=3000（冻结参数 n1=200/n2=8/order=3/seed=42，
N 按筛选规格升至 3000）。边际族 26 个不参与筛选。

用法：``PYTHONPATH=src .venv/bin/python scripts/e5_screening_phase_a.py``
产物：``artifacts/e5_screening/phase_a/``（逐折 JSONL + phase_a_report.json + run.log）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qgbs.backtest.rolling import WindowSpec, rolling_splits
from qgbs.data.adapters.csv_adapter import CsvAdapter
from qgbs.experiments.runner import _prepare_gbs_backend, load_config
from qgbs.experiments.screening import (
    FoldGateRecord,
    ScreeningParams,
    build_vocab,
    candidate_names,
    candidate_values,
    gate1_stability,
    gate2_predictive,
    gate3_consistency,
    near_miss_diagnostic,
)
from qgbs.factors.finance import build_finance_features
from qgbs.factors.neutralization import MARKET_COL, JointFactorModel, build_alpha_label
from qgbs.factors.pipeline import FactorPipeline
from qgbs.gbs.basis import BasisSelectionParams, select_higher_order_basis
from qgbs.gbs.encoder import EncoderParams, QuantileEncoder, ensure_default_mappings, resolve_mapping_id

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "e5_screening" / "phase_a"
N_FOLDS_DRYRUN = 8  # 阶段 A 修正版：跑 fold 0-7 验证最小历史规则（fold 0-4 空集）
PARAMS = ScreeningParams()  # 冻结：ρ₀=0.8、δ=0.012、top-50、一致性 60%、种子 42
BASIS_PARAMS = BasisSelectionParams(n1=200, n2=8, N=3000, order=3, seed=42)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "experiments" / "q4_full_enhanced_hardware_p1ext.yaml")
    data_cfg = cfg["data"]
    enc_cfg = dict(cfg.get("encoder") or {})

    adapter = CsvAdapter(ROOT / data_cfg["store_dir"])
    pool = adapter.load_asset_pool_history(data_cfg["pool"])
    stock_records = pool.records[~pool.records["is_cash_asset"]]
    symbols = sorted(stock_records["symbol"].unique())
    industry = stock_records.drop_duplicates("symbol").set_index("symbol")["citics_industry_l1"]
    market = adapter.load_daily_market(symbols, data_cfg["start"], data_cfg["end"])
    msf = adapter.load_market_sector_factor(data_cfg["start"], data_cfg["end"])
    close = market.rows.pivot(index="trade_date", columns="symbol", values="close").sort_index()
    ret_wide = market.rows.pivot(index="trade_date", columns="symbol", values="daily_return").sort_index()
    mkt_code = data_cfg["market_index"]
    factor_df = pd.concat(
        [msf.index_returns[mkt_code].rename(MARKET_COL), msf.industry_index_returns], axis=1
    )
    feats = build_finance_features(market, feature_set="extended_plus",
                                   market_returns=msf.index_returns[mkt_code],
                                   industry_index_returns=msf.industry_index_returns,
                                   industry_map=industry,
                                   status=adapter.load_trading_status(symbols, data_cfg["start"], data_cfg["end"]))
    horizon = int(cfg.get("label_horizon", 21))
    spec = WindowSpec(**{k: cfg["rolling"][k] for k in ("train", "valid", "test", "purge", "embargo")})

    backend, batch = _prepare_gbs_backend(cfg)
    n_feat = feats.shape[1]
    print(f"特征数 {n_feat}，候选池 = C({n_feat},2)={n_feat*(n_feat-1)//2} 二阶 + {BASIS_PARAMS.N} 高阶基", flush=True)

    # 基选择：同一冻结批次上做一次（与金融折无关，跨折共用）
    t0 = time.time()
    basis = select_higher_order_basis(batch, tuple(range(backend.n_modes)), BASIS_PARAMS)
    print(f"基选择 done {time.time()-t0:.1f}s，实际选中 {len(basis.basis)} 个基", flush=True)
    t0 = time.time()
    vocab = build_vocab(batch, basis)
    print(f"全量词表构建 done {time.time()-t0:.1f}s", flush=True)

    names = candidate_names(n_feat, len(basis.basis))
    records: list[FoldGateRecord] = []
    jsonl = OUT / "folds.jsonl"
    jsonl.write_text("", encoding="utf-8")

    for split in rolling_splits(close.index, spec):
        if split.split_id >= N_FOLDS_DRYRUN:
            break
        print(f"\n=== fold {split.split_id}（train {split.train[0].date()}~{split.train[-1].date()}，"
              f"valid {split.valid[0].date()}~{split.valid[-1].date()}）===", flush=True)
        t_fold = time.time()
        jfm = JointFactorModel().fit(ret_wide.loc[split.train], industry, factor_df.loc[split.train])
        labels = build_alpha_label(close, industry, horizon, factor_df, model=jfm)
        pipe = FactorPipeline()
        train_mask = feats.index.get_level_values("date").isin(split.train)
        pipe.fit(feats[train_mask].dropna())
        feats_p = pipe.transform(feats)

        ensure_default_mappings(n_feat, 16, 511)
        mid = resolve_mapping_id(str(enc_cfg.get("mapping_id", "identity")), n_feat, 16, 511)
        enc = QuantileEncoder(
            EncoderParams(n_bins_a=int(enc_cfg.get("n_bins_a", 255)),
                          n_bins_b=int(enc_cfg.get("n_bins_b", 256)),
                          partition_boundary=int(enc_cfg.get("partition_boundary", 255))),
            mapping_id=mid,
        )
        enc.fit(feats_p[train_mask].dropna())
        addr = enc.freeze().transform(feats_p).mode_index  # (R, 26) 全期地址

        # 闸门 1：采样稳定性（代表性地址行 = 本折 train 窗行，函数内抽样 5000）
        t0 = time.time()
        addr_train = addr[train_mask]
        g1_pass, g1_corr = gate1_stability(batch, basis, addr_train, n_feat, PARAMS)
        print(f"闸门1 done {time.time()-t0:.1f}s：存活 {int(g1_pass.sum())}/{len(names)}", flush=True)

        # 闸门 2：预测有效性（train+valid 行切片——红线：test 段不进入）
        t0 = time.time()
        tv_mask = feats_p.index.get_level_values("date").isin(split.train.union(split.valid))
        idx_tv = feats_p.index[tv_mask]
        vals_tv = candidate_values(vocab, addr[tv_mask], n_feat)
        g2_pass, g2_detail = gate2_predictive(
            vals_tv, labels, idx_tv, split.train, split.valid, PARAMS)
        print(f"闸门2 done {time.time()-t0:.1f}s：存活 {int(g2_pass.sum())}/{len(names)}", flush=True)

        passed = frozenset(np.array(names)[g1_pass & g2_pass])
        ic_map = {n: float(g2_detail["ic_mean"].iloc[i]) for i, n in enumerate(names) if g2_pass[i]}
        records.append(FoldGateRecord(split_id=split.split_id, passed=passed, ic_mean=ic_map))

        # 闸门 3：因果跨折一致性（当前折 = split_id，仅用 fold ≤ split_id 的记录）
        g3 = gate3_consistency(records, names, PARAMS)
        rec = {
            "split_id": split.split_id,
            "gate1_passed": int(g1_pass.sum()),
            "gate2_passed": int(g2_pass.sum()),
            "gate12_intersection": len(passed),
            "gate3_selected": sorted(g3),
            "gate3_n": len(g3),
            "elapsed_s": time.time() - t_fold,
        }
        with jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"闸门3（因果折0~{split.split_id}）：入选 {len(g3)} 个；本折耗时 {rec['elapsed_s']:.0f}s", flush=True)
        print(f"  入选样例：{sorted(g3)[:10]}", flush=True)

    # near-miss 诊断（全部候选跨折通过次数分布 + top20 明细；仅闸1∩2聚合，不涉及 test）
    diag = near_miss_diagnostic(records, names, n_folds_total=20)
    (OUT / "near_miss_diagnostic.json").write_text(
        json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[near-miss] 通过折数分布:", diag["histogram"], flush=True)
    print("[near-miss] top5:", [(c["feature"], c["n_folds_passed"]) for c in diag["top_candidates"][:5]], flush=True)

    report = {
        "params": {k: getattr(PARAMS, k) for k in ("rho0", "delta", "top_k", "consistency_ratio", "min_history", "n_sub", "seed", "gate1_n_rows")},
        "candidate_pool": {"n_features": n_feat, "n_pair": n_feat * (n_feat - 1) // 2,
                           "n_basis_selected": len(basis.basis), "n_basis_requested": BASIS_PARAMS.N},
        "folds": [r.split_id for r in records],
        "note": "阶段 A 修正版干跑：fold 0-7，仅用 train/valid；闸门 3 最小历史规则（可用折≤5 时空集）",
    }
    (OUT / "phase_a_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物已写入 {OUT}")


if __name__ == "__main__":
    main()
