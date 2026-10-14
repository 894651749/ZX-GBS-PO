"""中信格式 CSV/Parquet 数据适配器（方案书 §8 七类交付文件，§8.4 格式约定）。

约定：

- 行情/状态类：(证券代码, 交易日期) 联合主键长表；
- 区间类（成分/公司行动）：生效/失效日期列；
- 文件格式按扩展名识别（.csv / .parquet）；
- 证券代码保留交易所后缀（如 600000.SH），仅做去空白/大写规范化；
- 可选文件缺失：返回空容器并 warning；必需文件缺失：FileNotFoundError
  并列出全部期望文件清单；
- 字段字典（field_dict）：column → 中文说明，缺列报错时给出中文字段名提示。
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from qgbs.data.adapters.base import (
    AssetPoolHistory,
    BaselineResults,
    ConstraintsCosts,
    CorporateActions,
    DailyMarket,
    MarketSectorFactor,
    TradingStatus,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[4] / "configs" / "data" / "csv.yaml"

#: 各数据集的必需列（缺列即报错，附字段字典中文提示）
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "asset_pool": [
        "pool_id", "symbol", "exchange", "effective_date", "expiry_date",
        "citics_industry_l1", "citics_industry_l2", "is_cash_asset", "eligible",
    ],
    "daily_market": [
        "symbol", "trade_date", "open", "high", "low", "close", "prev_close",
        "volume", "amount", "turnover_rate", "total_mv", "float_mv", "daily_return",
    ],
    "trading_status": [
        "symbol", "trade_date", "is_suspended", "buyable", "sellable", "lot_size",
    ],
    "corporate_actions": ["symbol", "action_type", "ex_date", "adj_factor"],
    "market_sector_factor": ["trade_date", "code", "kind", "value"],
    "constraints_costs": [
        "config_id", "rebalance_freq", "commission_bps", "stamp_tax_bps",
        "slippage_bps", "impact_coef", "cash_lower", "cash_upper",
    ],
}

_BOOL_TRUE = {"1", "true", "t", "yes", "y", "1.0"}


class CsvAdapter:
    """中信交付文件适配器，实现 DataAdapter Protocol 全部 7 个方法。"""

    def __init__(self, root: str | Path, config_path: str | Path | None = None) -> None:
        self.root = Path(root)
        cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        self._files: dict[str, dict[str, Any]] = raw.get("files", {})
        self._field_dict = self._load_field_dict()
        self._cache: dict[str, pd.DataFrame | None] = {}

    # ------------------------------------------------------------------
    # 读取基础设施
    # ------------------------------------------------------------------

    def _load_field_dict(self) -> dict[str, str]:
        spec = self._files.get("field_dict") or {}
        path = self.root / spec.get("path", "")
        if not path.exists():
            return {}
        df = self._read_table(path)
        return dict(zip(df["column"].astype(str), df["description"].astype(str)))

    def _read_table(self, path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"不支持的文件格式: {path}（仅支持 .csv / .parquet）")

    def _expected_manifest(self) -> str:
        lines = []
        for name, spec in self._files.items():
            mark = "必需" if spec.get("required") else "可选"
            lines.append(f"  [{mark}] {name}: {self.root / spec.get('path', '')}")
        return "\n".join(lines)

    def _load(self, name: str) -> pd.DataFrame | None:
        """按配置读取数据集；必需缺失显式报错，可选缺失返回 None 并 warning。"""
        if name in self._cache:
            return self._cache[name]
        spec = self._files.get(name)
        if spec is None:
            raise KeyError(f"配置中无数据集 {name!r}")
        path = self.root / spec["path"]
        if not path.exists():
            if spec.get("required"):
                raise FileNotFoundError(
                    f"必需数据文件缺失: {path}\n期望文件清单（configs/data/csv.yaml）:\n"
                    + self._expected_manifest()
                )
            warnings.warn(f"可选数据文件缺失，返回空容器: {path}", stacklevel=3)
            self._cache[name] = None
            return None
        df = self._read_table(path)
        for col in spec.get("date_cols", []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        if "symbol" in df.columns:
            # 代码规范化：去空白、大写、保留交易所后缀
            df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        self._require_columns(df, name)
        self._cache[name] = df
        return df

    def _require_columns(self, df: pd.DataFrame, name: str) -> None:
        missing = [c for c in REQUIRED_COLUMNS.get(name, []) if c not in df.columns]
        if missing:
            hints = [f"{c}（{self._field_dict[c]}）" if c in self._field_dict else c for c in missing]
            raise ValueError(f"数据集 {name!r} 缺少必需列: {', '.join(hints)}")

    @staticmethod
    def _to_bool(s: pd.Series) -> pd.Series:
        if s.dtype == bool:
            return s
        return s.astype(str).str.strip().str.lower().isin(_BOOL_TRUE)

    @staticmethod
    def _slice(df: pd.DataFrame, symbols: list[str] | None, start: str, end: str) -> pd.DataFrame:
        out = df
        if symbols is not None:
            out = out[out["symbol"].isin([str(s).strip().upper() for s in symbols])]
        if "trade_date" in out.columns:
            out = out[(out["trade_date"] >= pd.Timestamp(start)) & (out["trade_date"] <= pd.Timestamp(end))]
        elif "ex_date" in out.columns:
            out = out[(out["ex_date"] >= pd.Timestamp(start)) & (out["ex_date"] <= pd.Timestamp(end))]
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # DataAdapter 接口
    # ------------------------------------------------------------------

    def load_asset_pool_history(self, pool_id: str) -> AssetPoolHistory:
        df = self._load("asset_pool")
        assert df is not None
        rows = df[df["pool_id"] == pool_id].drop(columns=["pool_id"]).reset_index(drop=True)
        if rows.empty:
            raise KeyError(f"资产池 {pool_id!r} 在成分文件中无记录")
        return AssetPoolHistory(pool_id=pool_id, records=rows)

    def load_daily_market(self, symbols: list[str], start: str, end: str) -> DailyMarket:
        df = self._load("daily_market")
        assert df is not None
        df = df.copy()
        if "adj_factor" not in df.columns:
            df["adj_factor"] = 1.0  # 复权因子缺省为 1
        if "return_basis" not in df.columns:
            df["return_basis"] = "raw"
        return DailyMarket(rows=self._slice(df, symbols, start, end))

    def load_trading_status(self, symbols: list[str], start: str, end: str) -> TradingStatus:
        df = self._load("trading_status")
        assert df is not None
        df = df.copy()
        for col in ("is_suspended", "buyable", "sellable", "is_st",
                    "is_limit_up_sealed", "is_limit_down_sealed"):
            if col in df.columns:
                df[col] = self._to_bool(df[col])
        return TradingStatus(rows=self._slice(df, symbols, start, end))

    def load_corporate_actions(self, symbols: list[str], start: str, end: str) -> CorporateActions:
        df = self._load("corporate_actions")
        if df is None:
            return CorporateActions(records=pd.DataFrame(columns=["symbol", "action_type", "ex_date"]))
        return CorporateActions(records=self._slice(df, symbols, start, end))

    def load_market_sector_factor(self, start: str, end: str) -> MarketSectorFactor:
        df = self._load("market_sector_factor")
        assert df is not None
        df = df[(df["trade_date"] >= pd.Timestamp(start)) & (df["trade_date"] <= pd.Timestamp(end))]

        def _pivot(kind: str) -> pd.DataFrame:
            sub = df[df["kind"] == kind]
            return sub.pivot(index="trade_date", columns="code", values="value").sort_index()

        def _series(kind: str, name: str) -> pd.Series:
            sub = df[df["kind"] == kind]
            s = sub.groupby("trade_date")["value"].first().sort_index()
            return s.rename(name)

        return MarketSectorFactor(
            index_returns=_pivot("index"),
            industry_index_returns=_pivot("industry"),
            risk_free_rate=_series("risk_free", "risk_free_rate"),
            cash_mgmt_yield=_series("cash_mgmt", "cash_mgmt_yield"),
        )

    def load_constraints_costs(self, config_id: str) -> ConstraintsCosts:
        df = self._load("constraints_costs")
        assert df is not None
        row = df[df["config_id"] == config_id]
        if row.empty:
            raise KeyError(
                f"约束成本配置 {config_id!r} 不存在，可用: {sorted(df['config_id'].unique())}"
            )
        r = row.iloc[0]

        def _json_dict(col: str) -> dict:
            v = r.get(col)
            return json.loads(v) if isinstance(v, str) and v else {}

        blacklist = r.get("blacklist")
        return ConstraintsCosts(
            config_id=config_id,
            rebalance_freq=str(r["rebalance_freq"]),
            commission_bps=float(r["commission_bps"]),
            stamp_tax_bps=float(r["stamp_tax_bps"]),
            slippage_bps=float(r["slippage_bps"]),
            impact_coef=float(r["impact_coef"]),
            weight_lower=_json_dict("weight_lower_json"),
            weight_upper=_json_dict("weight_upper_json"),
            cash_range=(float(r["cash_lower"]), float(r["cash_upper"])),
            exposure_bounds={
                k: tuple(v) for k, v in _json_dict("exposure_bounds_json").items()
            },
            max_turnover=None if pd.isna(r.get("max_turnover")) else float(r["max_turnover"]),
            liquidity_limit=None if pd.isna(r.get("liquidity_limit")) else float(r["liquidity_limit"]),
            blacklist=tuple(blacklist.split(",")) if isinstance(blacklist, str) and blacklist else (),
        )

    def load_baseline_results(self) -> BaselineResults:
        df = self._load("baseline_results")
        if df is None:
            return BaselineResults()  # 可选缺失 → 空容器
        items = dict(zip(df["item"].astype(str), df["payload"].astype(str)))
        matrix_items = {"return_matrix", "cov_matrix"} & set(items)
        if matrix_items:
            warnings.warn(f"baseline_results 复杂字段暂不解析，留空: {sorted(matrix_items)}", stacklevel=2)
        return BaselineResults(
            pool_29=items["pool_29"].split(",") if "pool_29" in items else None,
            pool_100=items["pool_100"].split(",") if "pool_100" in items else None,
            sharpe_convention=items.get("sharpe_convention"),
            solver_params=json.loads(items["solver_params"]) if "solver_params" in items else None,
        )
