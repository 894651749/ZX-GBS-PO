"""中信首批交付数据 → 框架七文件布局的 ETL（方案书 §9.4 口径）。

输入（data/ 下真实交付文件，只读）：
    静态数据_1.csv        —— 800 股静态档（中证800 的 2025 年末快照成分）
    行情数据_1.csv        —— 800 股 × 727 交易日行情（后复权价，逐日复权因子口径）
    指数相关时序数据_1.csv —— 沪深300/中证800 + 30 个中信一级行业指数日收益

输出（data_store/，与 configs/data/csv.yaml 的 files 映射一一对应）：
    asset_pool.csv / daily_market.csv / trading_status.csv /
    market_sector_factor.csv / constraints_costs.csv / field_dict.csv / MANIFEST.md
    （corporate_actions.csv、baseline_results.csv 交付缺失，按可选文件缺省处理）

关键口径决策（全部基于真实交付数据派生，无任何合成数据）：

1. 交付价格为后复权价，且除权日"复权前收盘"按新因子调整（2023-2025 共 772 行
   复权前收盘 ≠ 昨日复权收盘）。因此日收益只能取
   ``daily_return = 复权收盘 / 复权前收盘 − 1``；
2. 下游因子/标签管线使用 close 的比值与环比（finance.py、neutralization.py），
   要求 close 序列内部收益一致。故输出 close 为以 daily_return 连乘重构的
   **一致复权价**（基点 = 各股票首个交付日的交付复权收盘价），
   prev_close = close/(1+daily_return)，保证 close/prev_close − 1 ≡ daily_return；
3. open/high/low 交付中无真实值（high/low 未交付，open 与 close 因子口径在
   除权日不一致），统一以一致复权价 close 占位，当前管线无下游消费
   （引擎只消费 close/daily_return/amount，因子库只消费 close/volume/turnover）；
4. volume 输出为与一致复权价自洽的复权口径股数 ``amount / close``
   （成交额为交付真实值、单位元），保证 volume × close ≡ amount
   （finance.py 的 liquidity/amihud 以该恒等式消费）；
5. 单位换算：成交量 手→股（×100，经 成交额/(量×100)≈真实价 验证）、
   市值 万元→元（×1e4，经 总市值×1e4/总股本≈真实价 验证）、换手率为 % 原样；
6. citics_industry_l1 存放中信一级行业**指数代码**（CI005xxx.WI），
   因 JointFactorModel 以行业列值连接因子收益列名（neutralization.py）；
   中文行业名保留在 citics_industry_l1_name 供审计；
7. constraints_costs.csv 为**占位值（待中信确认）**，仅用于跑通链路，
   任何扣成本结论不得以该占位口径出具；占位事实同时记入 note 列、
   field_dict.csv 与 MANIFEST.md；
8. 资产池如实定义为中证800 全集快照（pool_id=zz800，成分时点 2025 年末，
   非 point-in-time——仅可用于方法学验证，正式结论须待历史成分交付）；
   29/100 业务池无真实交付来源，不予构造。

CLI：``PYTHONPATH=src python -m qgbs.data.etl_zx [--data-dir data] [--out-dir data_store]``
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger("qgbs.data.etl_zx")

SRC_STATIC = "静态数据_1.csv"
SRC_MARKET = "行情数据_1.csv"
SRC_INDEX = "指数相关时序数据_1.csv"

POOL_ID = "zz800"  # 中证800 全集（2025 年末快照，非 PIT）
BROAD_INDEX_KIND = {"000300.SH", "000906.SH"}  # 宽基；其余 CI005xxx.WI 为中信一级行业指数

# 占位约束与成本（待中信真实交付；当前仅用于链路跑通，见模块 docstring 第 7 条）
CONSTRAINTS_PLACEHOLDER = {
    "config_id": "default",
    "rebalance_freq": "1m",
    "commission_bps": 2.5,
    "stamp_tax_bps": 5.0,
    "slippage_bps": 5.0,
    "impact_coef": 0.0,
    "cash_lower": 0.0,
    "cash_upper": 0.05,
    "weight_lower_json": "",
    "weight_upper_json": "",
    "exposure_bounds_json": "",
    "max_turnover": None,
    "liquidity_limit": None,
    "blacklist": "",
    "note": "占位待中信确认：费率/滑点/现金区间/调仓规则均为假设值，禁止用于结论性产出",
}

FIELD_DICT: list[tuple[str, str]] = [
    # asset_pool.csv
    ("pool_id", "资产池标识（zz800=中证800全集，2025年末快照成分，非PIT）"),
    ("symbol", "证券代码（含交易所后缀，如 000001.SZ）"),
    ("exchange", "交易所：SH/SZ"),
    ("effective_date", "生效日期（取真实上市日期；池成分为2025年末快照）"),
    ("expiry_date", "失效日期（退市日；当前样本无退市，全为空）"),
    ("citics_industry_l1", "中信一级行业指数代码（CI005xxx.WI，对接行业因子收益列名）"),
    ("citics_industry_l1_name", "中信一级行业中文名（审计用冗余列）"),
    ("citics_industry_l2", "中信二级行业（中文名，交付无二级指数代码）"),
    ("is_cash_asset", "是否现金资产（当前交付无现金资产定义，全为 False）"),
    ("eligible", "当期是否允许入组合（当前全为 True）"),
    # daily_market.csv
    ("trade_date", "交易日期"),
    ("open", "开盘价（占位=一致复权收盘价；交付未含一致口径开盘价，无下游消费）"),
    ("high", "最高价（占位=一致复权收盘价；交付未含，待中信补充）"),
    ("low", "最低价（占位=一致复权收盘价；交付未含，待中信补充）"),
    ("close", "一致复权收盘价：以 daily_return 连乘重构，基点=首个交付日的交付复权收盘价"),
    ("prev_close", "一致复权前收盘 = close/(1+daily_return)，保证收益口径自洽"),
    ("volume", "成交量（股，复权口径）= 成交额/close，满足 volume×close≡amount"),
    ("amount", "成交额（元，交付真实值）"),
    ("turnover_rate", "换手率（%，交付真实值）"),
    ("total_mv", "总市值（元；交付万元×1e4 换算）"),
    ("float_mv", "流通市值（元；交付万元×1e4 换算）"),
    ("daily_return", "日收益率 = 交付复权收盘/复权前收盘 − 1（含分红除权修正）"),
    ("return_basis", "收益率口径：adjusted（复权价）"),
    # trading_status.csv
    ("is_suspended", "当日是否停牌（= 交付'可否买卖'为否；与成交量=0 完全重合）"),
    ("buyable", "当日可买（= 交付'可否买卖'为是）"),
    ("sellable", "当日可卖（口径同 buyable；交付未区分买卖方向）"),
    ("is_st", "是否 ST/*ST（交付 ST/*ST 标记非空）"),
    ("lot_size", "最小交易单位（股；A股制度常量 100）"),
    ("list_date", "上市日期（取自静态数据真实值）"),
    # market_sector_factor.csv
    ("code", "指数代码（宽基 000300.SH/000906.SH；行业 CI005xxx.WI）"),
    ("kind", "序列种类：index=宽基指数，industry=中信一级行业指数"),
    ("value", "日收益率（小数；交付 PCTCHANGE 原样）"),
    # constraints_costs.csv（占位）
    ("config_id", "约束成本配置标识（default=占位配置）"),
    ("rebalance_freq", "调仓频率（占位 1m，待中信）"),
    ("commission_bps", "佣金基点（占位 2.5，待中信）"),
    ("stamp_tax_bps", "印花税基点（占位 5.0，待中信）"),
    ("slippage_bps", "滑点基点（占位 5.0，待中信）"),
    ("impact_coef", "冲击成本系数（占位 0.0，待中信）"),
    ("cash_lower", "现金比例下限（占位 0.0，待中信）"),
    ("cash_upper", "现金比例上限（占位 0.05，待中信）"),
    ("note", "占位声明列：标注该配置为占位待确认"),
]


def industry_ci_code_map(index_df: pd.DataFrame) -> dict[str, str]:
    """由指数交付数据构建 中信一级行业中文名 → 行业指数代码 的映射。"""
    ci = index_df[index_df["SECU_ID"].astype(str).str.startswith("CI")]
    mapping = {
        str(name).removeprefix("中信"): str(code)
        for code, name in zip(ci["SECU_ID"], ci["SECU_NAME"])
    }
    if len(mapping) != ci["SECU_ID"].nunique():
        raise ValueError("中信行业指数名称去前缀后发生冲突，请检查指数交付数据")
    return mapping


def build_asset_pool(static: pd.DataFrame, industry_map: dict[str, str]) -> pd.DataFrame:
    """静态档 → asset_pool.csv（pool_id=zz800；行业一列存 CI 指数代码）。"""
    df = static.copy()
    missing_ind = sorted(set(df["中信一级行业"].dropna()) - set(industry_map))
    if missing_ind:
        raise ValueError(f"静态档中信一级行业无对应行业指数: {missing_ind}")
    out = pd.DataFrame(
        {
            "pool_id": POOL_ID,
            "symbol": df["证券代码"].astype(str).str.strip().str.upper(),
            "exchange": df["交易所"].astype(str).str.strip().str.upper(),
            "effective_date": pd.to_datetime(df["上市日期"]).dt.strftime("%Y-%m-%d"),
            "expiry_date": pd.to_datetime(df["退市日期"]).dt.strftime("%Y-%m-%d"),
            "citics_industry_l1": df["中信一级行业"].map(industry_map),
            "citics_industry_l1_name": df["中信一级行业"],
            "citics_industry_l2": df["中信二级行业"],
            "is_cash_asset": False,
            "eligible": True,
        }
    )
    return out.sort_values("symbol").reset_index(drop=True)


def build_daily_market(market: pd.DataFrame) -> pd.DataFrame:
    """行情 → daily_market.csv（一致复权价口径，见模块 docstring 第 1–5 条）。"""
    df = market.copy()
    df["symbol"] = df["SECU_ID"].astype(str).str.strip().str.upper()
    df["trade_date"] = pd.to_datetime(df["DT"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    daily_return = df["复权收盘价"] / df["复权前收盘价"] - 1.0
    df["daily_return"] = daily_return.astype(float)

    growth = (1.0 + df["daily_return"]).groupby(df["symbol"]).cumprod()
    first_close = df.groupby("symbol")["复权收盘价"].transform("first")
    close = (first_close * growth).astype(float)  # 基点=首个交付日交付复权收盘价
    df["close"] = close
    df["prev_close"] = close / (1.0 + df["daily_return"])
    for col in ("open", "high", "low"):  # 占位：交付无一致口径（high/low 完全未交付）
        df[col] = close
    df["volume"] = df["成交额"] / close  # 复权口径股数，保持 volume×close≡amount
    df["amount"] = df["成交额"].astype(float)
    df["turnover_rate"] = df["换手率"].astype(float)
    df["total_mv"] = df["总市值"] * 1e4  # 万元 → 元
    df["float_mv"] = df["流通市值"] * 1e4
    df["return_basis"] = "adjusted"

    cols = [
        "symbol", "trade_date", "open", "high", "low", "close", "prev_close",
        "volume", "amount", "turnover_rate", "total_mv", "float_mv",
        "daily_return", "return_basis",
    ]
    return df[cols].reset_index(drop=True)


def build_trading_status(market: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    """行情'可否买卖'/ST 标记 + 静态上市日期 → trading_status.csv。"""
    df = market.copy()
    df["symbol"] = df["SECU_ID"].astype(str).str.strip().str.upper()
    df["trade_date"] = pd.to_datetime(df["DT"]).dt.strftime("%Y-%m-%d")
    tradable = df["可否买卖"].astype(str).str.strip() == "是"
    list_date = static.set_index(static["证券代码"].astype(str).str.strip().str.upper())["上市日期"]
    out = pd.DataFrame(
        {
            "symbol": df["symbol"],
            "trade_date": df["trade_date"],
            "is_suspended": ~tradable,
            "buyable": tradable,
            "sellable": tradable,
            "is_st": df["ST/*ST"].notna(),
            "lot_size": 100,
            "list_date": pd.to_datetime(df["symbol"].map(list_date)).dt.strftime("%Y-%m-%d"),
        }
    )
    return out.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def build_market_sector_factor(index_df: pd.DataFrame) -> pd.DataFrame:
    """指数日收益 → market_sector_factor.csv 长表（trade_date, code, kind, value）。"""
    out = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(index_df["DT"]).dt.strftime("%Y-%m-%d"),
            "code": index_df["SECU_ID"].astype(str).str.strip().str.upper(),
            "kind": index_df["SECU_ID"].map(
                lambda c: "index" if str(c).upper() in BROAD_INDEX_KIND else "industry"
            ),
            "value": index_df["PCTCHANGE"].astype(float),
        }
    )
    return out.sort_values(["code", "trade_date"]).reset_index(drop=True)


def build_constraints_costs() -> pd.DataFrame:
    """占位约束成本表（待中信真实交付；见模块 docstring 第 7 条）。"""
    return pd.DataFrame([CONSTRAINTS_PLACEHOLDER])


def build_field_dict() -> pd.DataFrame:
    return pd.DataFrame(FIELD_DICT, columns=["column", "description"])


def _write_manifest(out_dir: Path, counts: dict[str, int], data_dir: Path) -> None:
    lines = [
        "# data_store 数据清单（ETL 产物）",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 生成方式：PYTHONPATH=src python -m qgbs.data.etl_zx --data-dir {data_dir} --out-dir {out_dir}",
        f"- 数据来源（只读真实交付）：{SRC_STATIC} / {SRC_MARKET} / {SRC_INDEX}",
        "- 数据区间：2023-01-03 ~ 2025-12-31（727 交易日 × 800 股 + 32 指数）",
        "",
        "## 文件与行数",
        "",
        *[f"- `{name}`：{n} 行" for name, n in counts.items()],
        "",
        "## 关键口径（详见 qgbs.data.etl_zx 模块 docstring 与 field_dict.csv）",
        "",
        "- daily_market.close 为 daily_return 连乘重构的一致复权价；open/high/low 为占位",
        "- daily_return = 交付复权收盘/复权前收盘 − 1（含分红除权修正）",
        "- volume = amount/close（复权口径股数）；市值万元→元；成交量单位经核验",
        "- 资产池 zz800 = 中证800 的 2025 年末快照成分（非 PIT，仅限方法学验证）",
        "- **constraints_costs.csv 为占位值（待中信确认），禁止用于结论性产出**",
        "- corporate_actions.csv / baseline_results.csv 交付缺失，按可选文件缺省",
        "- 29/100 业务池、行业排名规则、无风险利率序列：未交付，未构造",
    ]
    (out_dir / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_etl(data_dir: str | Path, out_dir: str | Path) -> dict[str, int]:
    """执行完整 ETL，返回各输出文件行数。"""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    static = pd.read_csv(data_dir / SRC_STATIC, encoding="utf-8-sig")
    market = pd.read_csv(data_dir / SRC_MARKET, encoding="utf-8-sig", low_memory=False)
    index = pd.read_csv(data_dir / SRC_INDEX, encoding="utf-8-sig")

    industry_map = industry_ci_code_map(index)
    outputs = {
        "asset_pool.csv": build_asset_pool(static, industry_map),
        "daily_market.csv": build_daily_market(market),
        "trading_status.csv": build_trading_status(market, static),
        "market_sector_factor.csv": build_market_sector_factor(index),
        "constraints_costs.csv": build_constraints_costs(),
        "field_dict.csv": build_field_dict(),
    }
    counts: dict[str, int] = {}
    for name, df in outputs.items():
        df.to_csv(out_dir / name, index=False)
        counts[name] = len(df)
        logger.info("写出 %s：%d 行", name, len(df))
    _write_manifest(out_dir, counts, data_dir)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="中信交付数据 → 框架七文件布局 ETL")
    parser.add_argument("--data-dir", default="data", help="交付数据目录（默认 data）")
    parser.add_argument("--out-dir", default="data_store", help="输出目录（默认 data_store）")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    counts = run_etl(args.data_dir, args.out_dir)
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
