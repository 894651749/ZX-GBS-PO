"""E7.2 汇总：四配置三指标对照表 + Q1−C1 配对 CI + 阈值建议。

用法：``PYTHONPATH=src .venv/bin/python scripts/e7_backfill_summarize.py``
产物：``artifacts/e7_metrics_backfill/compare.csv`` + ``compare.md``。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from qgbs.backtest.metrics import calmar, information_ratio, max_drawdown, sharpe
from qgbs.experiments.report import block_bootstrap_ci, paired_metric_bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "e7_metrics_backfill"
TAGS = ["C1", "C2", "Q1", "Q4"]


def _series(d: dict) -> pd.Series:
    s = pd.Series(d, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s


def main() -> None:
    rows = {}
    series = {}
    for tag in TAGS:
        r = json.load(open(OUT / tag / "result.json"))
        m = r["metrics"]
        rows[tag] = {
            "rank_ic": m["rank_ic_mean"], "icir": m["rank_icir"],
            "sharpe": m["sharpe"], "ir": m["ir"], "calmar": m["calmar"],
            "annual_return_arith": m["annual_return"],
            "annual_return_geom(旧口径)": m["annual_return_geometric"],
            "annual_vol": m["annual_volatility"], "mdd": m["max_drawdown"],
            "turnover": m["turnover_mean"],
        }
        series[tag] = {
            "returns": _series(r["extra"]["returns"]),
            "nav": _series(r["extra"]["nav"]),
            "ic": _series(r["extra"]["ic_per_date"]),
        }
    tab = pd.DataFrame(rows).T
    tab.to_csv(OUT / "compare.csv")
    print(tab.to_string(float_format=lambda v: f"{v:+.4f}"))

    # --- Q1−C1 配对 CI ---
    # 信号层：逐日 IC 配对（块 bootstrap）
    d_ic = (series["Q1"]["ic"] - series["C1"]["ic"]).dropna()
    lo, hi = block_bootstrap_ci(d_ic, statistic="mean", n_bootstrap=2000, block_length=5, seed=42)
    sig_ic = {"delta": float(d_ic.mean()), "ci95": [lo, hi]}

    # 组合层：Sharpe/IR/Calmar 的配对指标差 CI
    # IR 基准：000906.SH 日收益（与 runner 的 IR 口径同源）；
    # 配对 bootstrap 的重采样 Series 为位置索引，基准须先对齐交集日期并位置化
    msf = pd.read_csv(ROOT / "data_store/market_sector_factor.csv", parse_dates=["trade_date"])
    bench = (msf[(msf["kind"] == "index") & (msf["code"] == "000906.SH")]
             .set_index("trade_date")["value"].sort_index())
    common_dates = series["Q1"]["returns"].index.intersection(series["C1"]["returns"].index).intersection(bench.index)
    bench_pos = pd.Series(bench.loc[common_dates].to_numpy())
    fn_ir = lambda r: information_ratio(pd.Series(r.to_numpy()), bench_pos)
    rq, rc = series["Q1"]["returns"], series["C1"]["returns"]
    combo = {}
    for name, fn in (("sharpe", sharpe),
                     ("ir", fn_ir),
                     ("calmar", lambda r: calmar((1 + r).cumprod()))):
        point, lo, hi = paired_metric_bootstrap_ci(rq, rc, fn, n_bootstrap=2000, block_length=5, seed=42)
        combo[name] = {"delta": point, "ci95": [lo, hi]}
        print(f"Q1−C1 {name}: Δ={point:+.4f} CI=[{lo:+.4f},{hi:+.4f}]")
    print(f"Q1−C1 rank_ic: Δ={sig_ic['delta']:+.4f} CI=[{sig_ic['ci95'][0]:+.4f},{sig_ic['ci95'][1]:+.4f}]")
    print(f"Sharpe 提升: Q1/C1 = {rows['Q1']['sharpe']/rows['C1']['sharpe']:.3%} - 100% = {rows['Q1']['sharpe']/rows['C1']['sharpe']*100-100:+.2f}%（阈值 6%）")

    verdict = {"signal_layer": sig_ic, "portfolio_layer": combo,
               "sharpe_uplift_pct": float(rows["Q1"]["sharpe"] / rows["C1"]["sharpe"] * 100 - 100),
               "ir_uplift_pct": float(rows["Q1"]["ir"] / rows["C1"]["ir"] * 100 - 100),
               "calmar_uplift_pct": float(rows["Q1"]["calmar"] / rows["C1"]["calmar"] * 100 - 100)}
    (OUT / "q1_vs_c1_inference.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- compare.md ---
    md = ["# E7.2 四配置三指标对照（P1 协议 20 折，zz800，算术年化口径）\n",
          "## 指标对照（csv 同目录 compare.csv）\n\n```\n" + tab.to_string(float_format=lambda v: f"{v:+.4f}") + "\n```\n",
          "## Q1−C1 配对推断（块 bootstrap 2000/5/42）\n",
          f"- 信号层 rank IC：Δ={sig_ic['delta']:+.4f}，95% CI [{sig_ic['ci95'][0]:+.4f}, {sig_ic['ci95'][1]:+.4f}]",
          f"- Sharpe：Δ={combo['sharpe']['delta']:+.4f}，95% CI [{combo['sharpe']['ci95'][0]:+.4f}, {combo['sharpe']['ci95'][1]:+.4f}]",
          f"- IR：Δ={combo['ir']['delta']:+.4f}，95% CI [{combo['ir']['ci95'][0]:+.4f}, {combo['ir']['ci95'][1]:+.4f}]",
          f"- Calmar：Δ={combo['calmar']['delta']:+.4f}，95% CI [{combo['calmar']['ci95'][0]:+.4f}, {combo['calmar']['ci95'][1]:+.4f}]",
          f"- Sharpe 提升 {verdict['sharpe_uplift_pct']:+.2f}%（验收阈值 6%）；IR 提升 {verdict['ir_uplift_pct']:+.2f}%；Calmar 提升 {verdict['calmar_uplift_pct']:+.2f}%",
          "\n## 阈值建议（基于 Q1−C1 实测增量反推『已能达到』量级）",
          f"- IR：实测提升 {verdict['ir_uplift_pct']:+.1f}%，CI 下界为正 → 建议阈值 5%（留 2× 余量）" if combo['ir']['ci95'][0] > 0 else f"- IR：实测提升 {verdict['ir_uplift_pct']:+.1f}%，CI 含 0 → 暂不建议设阈值",
          f"- Calmar：实测提升 {verdict['calmar_uplift_pct']:+.1f}%，CI 下界{'为正' if combo['calmar']['ci95'][0] > 0 else '含 0'}",
          "\n口径：annual_return 为算术年化（客户文档）；几何口径见 annual_return_geometric 列。"]
    (OUT / "compare.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n产物已写入 {OUT}")


if __name__ == "__main__":
    main()
