"""E4.2 诊断三（真机版）：与 GBS 低阶统计匹配的经典随机特征对照（§7.1 / §7.5 归因）。

口径：
- 特征面板：Q3H 冻结口径三族全量（gbs_marginal + gbs_second_order + gbs_higher_order，
  hw-5M-b0、200k shots、b=255、identity、P1 协议、extended 语境）——与 P2 仿真版
  （q3 配置三族、第一折 valid 窗）结构可比；配置差异（P0→P1 协议、valid 6m→3m）
  在报告中注明。
- bootstrap 参数对齐 G0 判定口径（n_bootstrap=2000, block_length=5, seed=42）；
  P2 版用默认（500, 21），差异注明。
- 主结果：第一折 valid 窗（P2 可比单口径）；增强：20 折逐折面板读出 IC 差均值
  序列 → 配对块 bootstrap CI（折级口径，规避相邻折 valid 窗重叠）。

注意与 G0 的 C5 对照区分：C5 是等维 RFF **无关**随机特征；本诊断是逐列边际+
协方差**匹配**的相关随机特征（更强的对照，§7.5 归因条款两者并用）。

用法：``PYTHONPATH=src .venv/bin/python scripts/e42_matched_rf_hardware.py``
产物：``artifacts/e42_matched_rf_hardware/``（diag3 表/详情 + 逐折 JSONL + run.log）。
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qgbs.backtest.rolling import WindowSpec, rolling_splits
from qgbs.data.adapters.csv_adapter import CsvAdapter
from qgbs.experiments.diagnostics import matched_random_features_control
from qgbs.experiments.report import block_bootstrap_ci
from qgbs.experiments.runner import _build_gbs_panel, _prepare_gbs_backend, load_config
from qgbs.factors.finance import build_finance_features
from qgbs.factors.neutralization import MARKET_COL, JointFactorModel, build_alpha_label
from qgbs.factors.pipeline import FactorPipeline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "e42_matched_rf_hardware"
GBS_FAMILIES = ["gbs_marginal", "gbs_second_order", "gbs_higher_order"]  # Q3H 口径（P2 可比）


def _restrict(df, dates):
    return df[df.index.get_level_values("date").isin(dates)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "experiments" / "q3_higher_order_hardware_p1ext.yaml")
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
    mkt_code = data_cfg["market_index"]
    factor_df = pd.concat(
        [msf.index_returns[mkt_code].rename(MARKET_COL), msf.industry_index_returns], axis=1
    )
    feats = build_finance_features(market, feature_set="extended", market_returns=msf.index_returns[mkt_code])
    horizon = int(cfg.get("label_horizon", 21))
    spec = WindowSpec(**{k: cfg["rolling"][k] for k in ("train", "valid", "test", "purge", "embargo")})

    backend, batch = _prepare_gbs_backend(cfg)
    print(f"真机批次 {batch.meta.batch_id} n_shots={batch.meta.n_shots}", flush=True)

    fold_diffs: list[dict] = []
    jsonl = OUT / "diag3_folds.jsonl"
    for split in rolling_splits(close.index, spec):
        t0 = time.time()
        jfm = JointFactorModel().fit(ret_wide.loc[split.train], industry, factor_df.loc[split.train])
        labels = build_alpha_label(close, industry, horizon, factor_df, model=jfm)
        y_valid = _restrict(labels, split.valid)

        pipe = FactorPipeline()
        train_mask = feats.index.get_level_values("date").isin(split.train)
        pipe.fit(feats[train_mask].dropna())
        feats_p = pipe.transform(feats)

        panel = _build_gbs_panel(cfg, GBS_FAMILIES, feats_p, split, backend, batch)
        panel_v = _restrict(panel, split.valid)

        r = matched_random_features_control(
            panel_v, None, y_valid,
            seed=42, n_bootstrap=2000, block_length=5,  # G0 对齐口径
            train_frac=0.5, readout_alpha=1.0,
        )
        d = r.details
        rec = {
            "split_id": split.split_id,
            "panel_ic_gbs": d["panel_ic_gbs"],
            "panel_ic_matched": d["panel_ic_matched"],
            "ic_diff_mean": d["ic_diff_mean"],
            "ic_diff_ci": list(d["ic_diff_ci"]),
            "n_eval_dates": d["n_eval_dates"],
            "elapsed_s": time.time() - t0,
        }
        fold_diffs.append(rec)
        with jsonl.open("a", encoding="utf-8") as f:  # 逐折立即落盘
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[fold {split.split_id}] ΔIC={d['ic_diff_mean']:+.4f} "
              f"(GBS {d['panel_ic_gbs']:+.4f} vs matched {d['panel_ic_matched']:+.4f}) "
              f"{rec['elapsed_s']:.0f}s", flush=True)
        if split.split_id == 0:
            # 主口径（P2 可比）：第一折完整结果落盘
            r.table.to_csv(OUT / "diag3_matched_table_hardware.csv")
            (OUT / "diag3_matched_hardware_details.json").write_text(
                json.dumps({"verdict": r.verdict,
                            **{k: v for k, v in d.items()}}, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

    # 跨折汇总（折级配对块 bootstrap，n=20）
    diffs = pd.Series([r["ic_diff_mean"] for r in fold_diffs])
    lo, hi = block_bootstrap_ci(diffs, statistic="mean", n_bootstrap=2000, block_length=5, level=0.95, seed=42)
    summary = {
        "n_folds": len(diffs),
        "ic_diff_fold_mean": float(diffs.mean()),
        "ic_diff_fold_ci95": [float(lo), float(hi)],
        "positive_fold_share": float((diffs > 0).mean()),
        "folds": fold_diffs,
    }
    (OUT / "diag3_cross_fold_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n跨折汇总：ΔIC 折均值 {diffs.mean():+.4f}，CI95 [{lo:+.4f}, {hi:+.4f}]，"
          f"正折占比 {(diffs > 0).mean():.0%}")
    print(f"产物已写入 {OUT}")


if __name__ == "__main__":
    main()
