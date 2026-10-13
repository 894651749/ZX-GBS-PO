"""业务资产池构建（E1.3）：pool29 / pool100 派生数据仓。

规则假设（**口径待中信确认**：中信"行业第一名/前四名"的准确排名规则未交付）：
以中信一级行业（30 个）分组，按**再平衡日当日可得的 float_mv** 行业内降序排名
（引擎为收盘后调仓口径，当日数据可得），pool29 = 每组第 1 名（30 股/期），
pool100 = 每组前 4 名（实际每期约 114 股：部分行业可交易股不足 4 只——
与方案书 28+1 / 99+1 的偏差来自行业分类版本差异，本项目数据为 30 个一级行业）。

PIT 正确性：排名仅用再平衡日 t 当日及以前数据（float_mv 与可交易状态均取 t 日）；
候选过滤沿用引擎 eligible 口径：buyable & sellable（trading_status 现有逻辑）。
成分随再平衡日逐期重建（monthly），输出区间表（effective_date 含、expiry_date 不含）。

生效机制：派生数据仓 ``data_store_derived/{pool}/`` 下 trading_status.csv 把
**非成分期**标记为 buyable=False（sellable 保持交付值），配合引擎"只减不增"
语义（buyable=False 的持仓上限为当前权重）：首期非成分股 w_prev=0 → 不可买入；
出池持仓股只允许减仓。派生仓其余文件硬链接 data_store/（零拷贝，不动交付原件）。

用法：``PYTHONPATH=src .venv/bin/python scripts/build_business_pools.py``
产物：``data_store_derived/{pool29,pool100}/`` + 构建摘要 build_summary.json。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from qgbs.data.calendar import rebalance_dates

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data_store"
DST = ROOT / "data_store_derived"
TOP_N = {"pool29": 1, "pool100": 4}
# 派生仓共享硬链接的文件（asset_pool/trading_status 为派生生成）
LINK_FILES = ["daily_market.csv", "market_sector_factor.csv", "constraints_costs.csv", "field_dict.csv", "MANIFEST.md"]


def build_compositions(
    pool: pd.DataFrame, fm: pd.DataFrame, tradable: pd.DataFrame, rebal: pd.DatetimeIndex
) -> dict[str, list[set[str]]]:
    """逐再平衡日 PIT 排名出各池成分集合序列。"""
    zz = pool[(pool["pool_id"] == "zz800") & (~pool["is_cash_asset"])]
    ind = zz.set_index("symbol")["citics_industry_l1"]
    info = zz.set_index("symbol")[["exchange", "citics_industry_l1", "citics_industry_l1_name", "citics_industry_l2"]]
    comps: dict[str, list[set[str]]] = {k: [] for k in TOP_N}
    for t in rebal:
        mv = fm.loc[t].dropna()
        ok = tradable.loc[t].reindex(mv.index).fillna(False)
        mv = mv[ok.astype(bool)]
        df = mv.to_frame("mv")
        df["ind"] = df.index.map(ind)
        df = df.dropna(subset=["ind"])
        r = df.groupby("ind")["mv"].rank(ascending=False, method="first")
        for name, n in TOP_N.items():
            comps[name].append(set(r[r <= n].index))
    return comps


def main() -> None:
    dm = pd.read_csv(SRC / "daily_market.csv", usecols=["symbol", "trade_date", "float_mv"], parse_dates=["trade_date"])
    pool = pd.read_csv(SRC / "asset_pool.csv")
    ts = pd.read_csv(SRC / "trading_status.csv", parse_dates=["trade_date"])
    ts["tradable"] = ts["buyable"] & ts["sellable"]

    dates = pd.DatetimeIndex(sorted(dm["trade_date"].unique()))
    rebal = rebalance_dates(dates, "1m")
    fm = dm.pivot(index="trade_date", columns="symbol", values="float_mv")
    tradable = ts.pivot(index="trade_date", columns="symbol", values="tradable")

    comps = build_compositions(pool, fm, tradable, rebal)
    base_info = pool[pool["pool_id"] == "zz800"].set_index("symbol")[
        ["exchange", "citics_industry_l1", "citics_industry_l1_name", "citics_industry_l2"]
    ]

    summary: dict[str, dict] = {}
    for name, comp_seq in comps.items():
        out_dir = DST / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # 成分区间表：[rebal[i], rebal[i+1])，末期 expiry 留空
        rows = []
        for i, comp in enumerate(comp_seq):
            eff = rebal[i]
            exp = rebal[i + 1] if i + 1 < len(comp_seq) else None
            for s in sorted(comp):
                info = base_info.loc[s]
                rows.append({
                    "pool_id": name, "symbol": s, "exchange": info["exchange"],
                    "effective_date": eff.strftime("%Y-%m-%d"),
                    "expiry_date": "" if exp is None else exp.strftime("%Y-%m-%d"),
                    "citics_industry_l1": info["citics_industry_l1"],
                    "citics_industry_l1_name": info["citics_industry_l1_name"],
                    "citics_industry_l2": info["citics_industry_l2"],
                    "is_cash_asset": False, "eligible": True,
                })
        ap = pd.DataFrame(rows)
        ap.to_csv(out_dir / "asset_pool.csv", index=False)

        # 派生 trading_status：非成分期 buyable=False（sellable 保持交付值）
        comp_arr = np.zeros((len(dates), len(tradable.columns)), dtype=bool)
        col_idx = {s: j for j, s in enumerate(tradable.columns)}
        for i, comp in enumerate(comp_seq):
            lo = dates.get_loc(rebal[i])
            hi = dates.get_loc(rebal[i + 1]) if i + 1 < len(comp_seq) else len(dates)
            for s in comp:
                if s in col_idx:
                    comp_arr[lo:hi, col_idx[s]] = True
        in_comp = pd.DataFrame(comp_arr, index=dates, columns=tradable.columns)
        ts_derived = ts.copy()
        member_flags = [
            bool(in_comp.at[d, s]) if (d in in_comp.index and s in in_comp.columns) else False
            for d, s in zip(ts_derived["trade_date"], ts_derived["symbol"])
        ]
        ts_derived["buyable"] = ts_derived["buyable"] & np.array(member_flags)
        ts_derived.to_csv(out_dir / "trading_status.csv", index=False)

        for f in LINK_FILES:
            src, dst = SRC / f, out_dir / f
            if dst.exists():
                dst.unlink()
            os.link(src, dst)

        sizes = [len(c) for c in comp_seq]
        turnover = [
            len(comp_seq[i] ^ comp_seq[i - 1]) / (2 * len(comp_seq[i]))
            for i in range(1, len(comp_seq))
        ]
        union = set().union(*comp_seq)
        summary[name] = {
            "n_periods": len(comp_seq),
            "size_per_period": {"min": min(sizes), "max": max(sizes), "mean": float(np.mean(sizes))},
            "union_size": len(union),
            "turnover_per_rebalance": {
                "mean": float(np.mean(turnover)), "max": float(np.max(turnover)),
                "min": float(np.min(turnover)),
            },
            "rank_rule": "citics_industry_l1 行业内 float_mv 降序（再平衡日当日，PIT）",
            "eligible_filter": "buyable & sellable（trading_status 现有口径）",
            "caliber_note": "行业第一名/前四名为假设口径，待中信确认；行业数 30（非方案书 28）",
        }
        print(f"[{name}] 期数={len(comp_seq)} 每期规模={min(sizes)}~{max(sizes)} "
              f"并集={len(union)} 逐期更替率 mean={np.mean(turnover):.3f} max={np.max(turnover):.3f}")

    (DST / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"产物已写入 {DST}/（asset_pool.csv / trading_status.csv 派生，其余硬链接）")


if __name__ == "__main__":
    main()
