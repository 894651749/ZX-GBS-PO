"""P2 辅助诊断驱动（真实金融数据 × 仿真 GBS 样本，方案书 §7.1）。

在第一个滚动折上执行三类诊断：
- 诊断二：置换对照（保持边际、破坏联合）——联合结构是否有信息量；
- 诊断三：低阶统计匹配的经典随机特征对照——增量能否由经典低阶匹配复制；
- 诊断四：模式映射与批次稳定性——多映射一致性与多批次离散度
  （批次库扩充：额外生成 seed 43/44/45 仿真批次并入库，方案书 §4.4 B≥4）。

诊断一为真机 vs 仿真对照，留待 P3 真机样本。

用法：``PYTHONPATH=src .venv/bin/python scripts/p2_diagnostics_csv.py``
产物：``artifacts/p2_csv/`` 下各诊断对比表与摘要。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from qgbs.backtest.rolling import WindowSpec, rolling_splits
from qgbs.data.adapters.csv_adapter import CsvAdapter
from qgbs.experiments.diagnostics import (
    mapping_batch_stability,
    matched_random_features_control,
    permutation_control,
)
from qgbs.experiments.runner import _build_gbs_panel, _prepare_gbs_backend, load_config
from qgbs.factors.finance import build_finance_features
from qgbs.factors.neutralization import MARKET_COL, JointFactorModel, build_alpha_label
from qgbs.factors.pipeline import FactorPipeline
from qgbs.gbs.backends.permuted import PermutedBackend
from qgbs.gbs.backends.simulator import GaussianSimulatorBackend, SimulatorConfig
from qgbs.gbs.sample_store import SampleStore

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "p2_csv"
GBS_FAMILIES = ["gbs_marginal", "gbs_second_order", "gbs_higher_order"]
EXTRA_BATCH_SEEDS = (43, 44, 45)  # seed 42 批次已由实验链路入库


def _restrict(df: pd.DataFrame, dates) -> pd.DataFrame:
    mask = df.index.get_level_values("date").isin(dates)
    return df[mask]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "experiments" / "q3_higher_order_csv.yaml")
    data_cfg = cfg["data"]

    # --- 数据与因子（与 run_experiment 同口径） ---
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

    # --- GBS 特征面板：原样本 vs 置换样本 ---
    backend, batch = _prepare_gbs_backend(cfg)
    panel_gbs = _build_gbs_panel(cfg, GBS_FAMILIES, feats_p, split0, backend, batch)
    batch_perm = PermutedBackend(backend, seed=123).permute(batch)
    panel_perm = _build_gbs_panel(cfg, GBS_FAMILIES, feats_p, split0, backend, batch_perm)

    gbs_v = _restrict(panel_gbs, split0.valid)
    perm_v = _restrict(panel_perm, split0.valid)

    # --- 诊断二：置换对照 ---
    r_perm = permutation_control(gbs_v, perm_v, y_valid)
    print("\n[诊断二 置换对照]", r_perm.verdict, flush=True)
    r_perm.table.to_csv(OUT / "diag2_permutation_table.csv")

    # --- 诊断三：低阶匹配经典随机特征对照 ---
    r_matched = matched_random_features_control(gbs_v, None, y_valid, seed=42)
    print("\n[诊断三 低阶匹配对照]", r_matched.verdict, flush=True)
    r_matched.table.to_csv(OUT / "diag3_matched_table.csv")

    # --- 诊断四：映射与批次稳定性 ---
    store = SampleStore(ROOT / cfg["gbs"]["sample_store"])
    sim_cfg = SimulatorConfig(
        n_temporal=int(cfg["gbs"]["n_temporal"]), n_spatial=int(cfg["gbs"]["n_spatial"]),
        seed=42, system_efficiency=float(cfg["gbs"]["system_efficiency"]),
        mean_click_rate=float(cfg["gbs"].get("mean_click_rate", 0.3)),
    )
    batches = [batch]
    for s in EXTRA_BATCH_SEEDS:
        b = GaussianSimulatorBackend(
            SimulatorConfig(**{**sim_cfg.__dict__, "seed": s})
        ).sample(int(cfg["gbs"]["n_shots"]))
        store.register_batch(b)  # 扩充仿真样本库（namespace=simulation）
        batches.append(b)
        print(f"[样本库] 入库批次 {b.meta.batch_id}", flush=True)

    cfg_p0 = copy.deepcopy(cfg)
    cfg_p0["encoder"]["mapping_id"] = "perm0"
    panel_p0 = _build_gbs_panel(cfg_p0, GBS_FAMILIES, feats_p, split0, backend, batch)
    r_stab = mapping_batch_stability(
        {"identity": gbs_v, "perm0": _restrict(panel_p0, split0.valid)}, batches
    )
    print("\n[诊断四 映射与批次稳定性]", r_stab.verdict, flush=True)
    r_stab.table.to_csv(OUT / "diag4_stability_table.csv", index=False)

    summary = {
        "diag2_permutation": {"verdict": r_perm.verdict, **r_perm.details},
        "diag3_matched": {"verdict": r_matched.verdict,
                          **{k: v for k, v in r_matched.details.items() if k != "batch_dispersion_head"}},
        "diag4_stability": {"verdict": r_stab.verdict,
                            "dispersion_curve": r_stab.details["dispersion_curve"]},
    }
    (OUT / "diagnostics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n产物已写入 {OUT}")


if __name__ == "__main__":
    main()
