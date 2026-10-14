"""数据适配器抽象接口（方案书 §8 交付文件口径）。

数据建模约定：

- 行情/状态类数据：以 (证券代码, 交易日期) 联合主键的长表为准，
  行级 schema 用 pydantic 模型定义（Field description 注明字段含义/单位/缺失规则）；
- 区间类数据（成分、公司行动）：用 生效日期/失效日期 表示；
- 所有返回数据均为 point-in-time 口径，不得包含未来信息。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

import pandas as pd
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 行级 schema（pydantic）：字段含义/单位/缺失规则的字典式定义
# ---------------------------------------------------------------------------


class AssetPoolRecord(BaseModel):
    """资产池成分区间记录：一行 = 一只证券的一段有效区间。"""

    symbol: str = Field(description="证券代码")
    exchange: str = Field(description="交易所：SH/SZ/BJ")
    effective_date: date = Field(description="生效日期（入池或属性变更生效日）")
    expiry_date: date | None = Field(default=None, description="失效日期；缺失(None)表示当前仍有效")
    citics_industry_l1: str = Field(description="中信一级行业代码")
    citics_industry_l2: str = Field(description="中信二级行业代码")
    industry_change_date: date | None = Field(default=None, description="行业分类变更日；无变更则缺失")
    industry_rank_rule: str = Field(default="", description="行业排名规则标识（如按市值/成交额）")
    industry_rank_date: date | None = Field(default=None, description="行业排名时点")
    is_cash_asset: bool = Field(default=False, description="是否现金资产（现金资产不走股票 alpha 模型）")
    cash_category: str | None = Field(default=None, description="现金类别（货币基金/逆回购等）；非现金资产缺失")
    eligible: bool = Field(default=True, description="当期是否允许入组合")


class DailyMarketRow(BaseModel):
    """日频行情长表行：联合主键 (symbol, trade_date)。价格为原始未复权价。"""

    symbol: str = Field(description="证券代码")
    trade_date: date = Field(description="交易日期")
    open: float = Field(description="开盘价（元，未复权）")
    high: float = Field(description="最高价（元，未复权）")
    low: float = Field(description="最低价（元，未复权）")
    close: float = Field(description="收盘价（元，未复权）")
    prev_close: float = Field(description="前收盘价（元，未复权）")
    volume: float = Field(description="成交量（股）")
    amount: float = Field(description="成交额（元）")
    turnover_rate: float = Field(description="换手率（%）")
    total_mv: float = Field(description="总市值（元）")
    float_mv: float = Field(description="流通市值（元）")
    adj_factor: float = Field(description="复权因子（相对基期累计值）")
    daily_return: float = Field(description="日收益率（小数）")
    return_basis: str = Field(description="收益率口径：raw（未复权价）/ adjusted（复权价）")


class TradingStatusRow(BaseModel):
    """日频交易状态长表行：联合主键 (symbol, trade_date)。"""

    symbol: str = Field(description="证券代码")
    trade_date: date = Field(description="交易日期")
    is_suspended: bool = Field(description="当日是否停牌")
    resume_date: date | None = Field(default=None, description="复牌日期；未停牌缺失")
    limit_up_price: float | None = Field(default=None, description="涨停价（元）；停牌日缺失")
    limit_down_price: float | None = Field(default=None, description="跌停价（元）；停牌日缺失")
    is_limit_up_sealed: bool = Field(default=False, description="涨停是否封板")
    is_limit_down_sealed: bool = Field(default=False, description="跌停是否封板")
    is_st: bool = Field(default=False, description="是否 ST/*ST")
    list_date: date = Field(description="上市日期")
    delist_date: date | None = Field(default=None, description="退市日期；未退市缺失")
    buyable: bool = Field(description="当日可买")
    sellable: bool = Field(description="当日可卖")
    lot_size: int = Field(description="最小交易单位（股，A股通常 100）")


class CorporateActionRecord(BaseModel):
    """公司行动区间/事件记录。"""

    symbol: str = Field(description="证券代码")
    action_type: str = Field(description="类型：dividend（分红送转）/ allotment（配股）/ split（拆并）")
    ex_date: date = Field(description="除权除息日")
    effective_date: date = Field(description="生效日期")
    expiry_date: date | None = Field(default=None, description="失效日期；事件型记录缺失")
    cash_dividend: float = Field(default=0.0, description="每股现金分红（元）")
    split_ratio: float = Field(default=1.0, description="送转/拆并比例")
    allotment_ratio: float = Field(default=0.0, description="配股比例")
    allotment_price: float | None = Field(default=None, description="配股价（元）；非配股缺失")
    adj_factor: float = Field(description="本次行动对应的复权因子")
    adj_factor_effective_date: date = Field(description="复权因子生效日")


# ---------------------------------------------------------------------------
# 返回容器（dataclass）：长表 DataFrame + 行级 schema 引用
# ---------------------------------------------------------------------------


@dataclass
class AssetPoolHistory:
    """PIT 资产池历史成分（区间表）。行结构见 :class:`AssetPoolRecord`。"""

    pool_id: str
    records: pd.DataFrame  # 长表：每行一条 AssetPoolRecord


@dataclass
class DailyMarket:
    """日频行情（长表，(symbol, trade_date) 联合主键）。行结构见 :class:`DailyMarketRow`。"""

    rows: pd.DataFrame  # 价格为原始未复权价；复权由下游用 adj_factor 计算


@dataclass
class TradingStatus:
    """日频交易状态（长表，(symbol, trade_date) 联合主键）。行结构见 :class:`TradingStatusRow`。"""

    rows: pd.DataFrame


@dataclass
class CorporateActions:
    """公司行动表（区间/事件）。行结构见 :class:`CorporateActionRecord`。"""

    records: pd.DataFrame


@dataclass
class MarketSectorFactor:
    """市场/行业/因子收益与风险数据。"""

    index_returns: pd.DataFrame  # 宽基指数日收益：date × index_code
    industry_index_returns: pd.DataFrame  # 中信一级行业指数日收益：date × industry_code
    risk_free_rate: pd.Series  # 无风险利率（日化，索引为 date）
    cash_mgmt_yield: pd.Series  # 现金管理收益率（日化，索引为 date）
    factor_returns: pd.DataFrame | None = None  # 可选：风格因子日收益（date × factor）
    factor_exposure: pd.DataFrame | None = None  # 可选：股票因子暴露 (date, symbol) × factor
    specific_risk: pd.DataFrame | None = None  # 可选：特质风险 (date, symbol)
    factor_cov: pd.DataFrame | None = None  # 可选：因子协方差（逐期）


@dataclass(frozen=True)
class ConstraintsCosts:
    """约束与成本配置（方案书 §8.2(6)）。不同场景以 config_id 区分。"""

    config_id: str
    rebalance_freq: str  # 调仓频率，如 "1m"
    commission_bps: float  # 佣金（双边，基点）
    stamp_tax_bps: float  # 印花税（卖出单边，基点）
    slippage_bps: float  # 滑点（基点）
    impact_coef: float  # 冲击成本系数
    weight_lower: dict[str, float] = field(default_factory=dict)  # 个股权重下限
    weight_upper: dict[str, float] = field(default_factory=dict)  # 个股权重上限
    cash_range: tuple[float, float] = (0.0, 0.05)  # 现金比例区间
    exposure_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)  # 行业/风格暴露区间
    max_turnover: float | None = None  # 单期最大换手（单边）
    liquidity_limit: float | None = None  # 流动性限制（如单日成交额占比上限）
    blacklist: tuple[str, ...] = ()  # 禁止投资名单


@dataclass
class BaselineResults:
    """原模型基线交付物。全部字段可选，允许部分缺失。"""

    return_matrix: pd.DataFrame | None = None  # 原模型收益率矩阵（date × symbol）
    mean_vector: pd.Series | None = None  # 预期收益均值
    cov_matrix: pd.DataFrame | None = None  # 协方差矩阵
    pool_29: list[str] | None = None  # 29 池名单
    pool_100: list[str] | None = None  # 100 池名单
    target_return_grid: list[float] | None = None  # 目标收益网格
    final_weights: pd.DataFrame | None = None  # 最终权重（date × symbol）
    sharpe_convention: str | None = None  # Sharpe 口径（年化方式/无风险利率约定）
    solver_params: dict[str, Any] | None = None  # 求解器参数
    perf_2026_01_04: pd.DataFrame | None = None  # 2026.1-4 绩效明细


# ---------------------------------------------------------------------------
# 适配器 Protocol
# ---------------------------------------------------------------------------


class DataAdapter(Protocol):
    """行情/基本面数据适配器抽象（对应中信交付文件 7 类数据）。

    所有数据源（mock 合成数据、真实行情库）统一实现此接口，
    上层（因子计算、回测引擎）只依赖 Protocol，不关心数据来源。
    """

    def load_asset_pool_history(self, pool_id: str) -> AssetPoolHistory:
        """PIT 成分：代码/交易所、生效/失效日期、中信一二级行业及变更日、
        行业排名规则与时点、是否现金资产及现金类别、当期是否允许入组合。"""
        ...

    def load_daily_market(self, symbols: list[str], start: str, end: str) -> DailyMarket:
        """日频行情：OHLC+前收、成交量/额、换手率、总市值/流通市值、
        复权因子、日收益率及口径；保留原始未复权价。"""
        ...

    def load_trading_status(self, symbols: list[str], start: str, end: str) -> TradingStatus:
        """交易状态：停牌/复牌、涨跌停价及封板、ST、上市/退市日期、
        可买/可卖、最小交易单位。"""
        ...

    def load_corporate_actions(self, symbols: list[str], start: str, end: str) -> CorporateActions:
        """公司行动：分红送转/配股/拆并、除权除息日、复权因子及生效日。"""
        ...

    def load_market_sector_factor(self, start: str, end: str) -> MarketSectorFactor:
        """市场/行业/因子：宽基指数日收益、中信一级行业指数日收益、
        无风险利率/现金管理收益率、可选因子收益与股票因子暴露/特质风险/因子协方差。"""
        ...

    def load_constraints_costs(self, config_id: str) -> ConstraintsCosts:
        """约束与成本：调仓频率、费率、滑点/冲击成本、个股权重上下限、
        现金比例区间、行业/风格暴露区间、最大换手、流动性限制、禁止投资名单。"""
        ...

    def load_baseline_results(self) -> BaselineResults:
        """原模型基线交付物（全部可选，允许部分缺失）。"""
        ...
