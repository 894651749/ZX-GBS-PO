"""Q2H 置换诊断（真机版诊断二，方案书 §7.1）：保持边际、破坏联合的对照。

口径与 scripts/p2_diagnostics_csv.py 的诊断二一致（第一滚动折、valid 窗、
逐特征 rank IC 对比），仅 GBS 后端换为真机批次 hw-5M-b0。

用法：``PYTHONPATH=src .venv/bin/python scripts/q2h_permutation_diag.py``
产物：``artifacts/q2_hardware/diag2_permutation_table_hardware.csv`` 与摘要 json。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from qgbs.backtest.rolling import WindowSpec, rolling_splits
from qgbs.data.adapters.csv_adapter import CsvAdapter
from qgbs.experiments.diagnostics import permutation_control
from qgbs.experiments.runner import _build_gbs_panel, _prepare_gbs_backend, load_config
from qgbs.factors.finance import build_finance_features
from qgbs.factors.neutralization import MARKET_COL, JointFactorModel, build_alpha_label
from qgbs.factors.pipeline import FactorPipeline
from qgbs.gbs.backends.permuted import PermutedBackend

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "q2_hardware"
GBS_FAMILIES = ["gbs_marginal", "gbs_second_order"]  # 与 q2_second_order_hardware.yaml 一致


def _restrict(df: pd.DataFrame, dates) -> pd.DataFrame:
    return df[df.index.get_level_values("date").isin(dates)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "experiments" / "q2_second_order_hardware.yaml")
    data_cfg = cfg["data"]

    adapter = CsvAdapter(ROOT / data_cfg["store_dir"])
    pool = adapter.load_asset_pool_history(data_cfg["pool"])
    stock_records = pool.records[~pool.records["is_cash_asset"]]
    symbols = sorted(stock_records["symbol"].unique())
    industry = stock_records.set_index("symbol")["citics_industry_l1"]
    market = adapter.load_daily_market(symbols, data_cfg["start"], data_cfg["end"])
    msf = adapter.load_market_sector_factor(data_cfg["start"], data_cfg["end"])
    close = market.rows.pivot(index="trade_date", columns="symbol", values="close").sort_index()
    ret_wide = market.rows.pivot(index="trade_date", columns="symbol", values="daily_return").sort_index()
    mkt_code = data_cfg["market_index"]
    factor_df = pd.concat(
        [msf.index_returns[mkt_code].rename(MARKET_COL), msf.industry_index_returns], axis=1
    )
    feats = build_finance_features(market, feature_set="base", market_returns=msf.index_returns[mkt_code])

    spec = WindowSpec(**{k: cfg["rolling"][k] for k in ("train", "valid", "test", "purge", "embargo")})
    split0 = next(iter(rolling_splits(close.index, spec)))
    print(f"诊断折：train {split0.train[0].date()}~{split0.train[-1].date()}  "
          f"valid {split0.valid[0].date()}~{split0.valid[-1].date()}", flush=True)

    jfm = JointFactorModel().fit(ret_wide.loc[split0.train], industry, factor_df.loc[split0.train])
    labels = build_alpha_label(close, industry, int(cfg.get("label_horizon", 21)), factor_df, model=jfm)
    y_valid = _restrict(labels, split0.valid)

    pipe = FactorPipeline()
    train_mask = feats.index.get_level_values("date").isin(split0.train)
    pipe.fit(feats[train_mask].dropna())
    feats_p = pipe.transform(feats)

    backend, batch = _prepare_gbs_backend(cfg)
    t0 = time.time()
    panel_gbs = _build_gbs_panel(cfg, GBS_FAMILIES, feats_p, split0, backend, batch)
    print(f"[panel_gbs] {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    batch_perm = PermutedBackend(backend, seed=123).permute(batch)
    print(f"[permute] {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    panel_perm = _build_gbs_panel(cfg, GBS_FAMILIES, feats_p, split0, backend, batch_perm)
    print(f"[panel_perm] {time.time()-t0:.1f}s", flush=True)

    r = permutation_control(_restrict(panel_gbs, split0.valid), _restrict(panel_perm, split0.valid), y_valid)
    print("\n[诊断二 置换对照·真机]", r.verdict, flush=True)
    r.table.to_csv(OUT / "diag2_permutation_table_hardware.csv")
    (OUT / "diag2_permutation_summary_hardware.json").write_text(
        json.dumps({"verdict": r.verdict, **r.details}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n产物已写入 {OUT}")


if __name__ == "__main__":
    main()
