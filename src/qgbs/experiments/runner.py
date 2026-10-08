"""配置驱动的实验执行器。

滚动协议（每个滚动折）：

1. train 窗：拟合因子模型（标签构建）、预处理流水线、特征变换、
   特征筛选，训练读出模型；
2. valid 窗：超参搜索评分、（可选）ρ 估计、（可选）优化器参数网格搜索，
   随后并入训练集重训；
3. test 窗：样本外预测 → alpha 面板 → 引擎统一回测。

防泄漏：train 标签前视窗 [t+1, t+h] 不进入 valid 窗，train/valid 标签
前视窗不进入 test 窗（purge/embargo 之外的硬约束）。

C0 在 mock 下以"样本均值 + 样本协方差经典 Markowitz"占位实现；
真正的 C0 复现需中信 baseline_results（load_baseline_results 交付物）。
Q1-Q4 已实现：GBS 特征族（gbs_*）统一走仿真样本库 cached 后端
（近似仿真口径，不替代真机；真机结论需 hardware 后端 + 方案书 §7.1 诊断）。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from qgbs.backtest.engine import BacktestEngine, BacktestResult
from qgbs.backtest.metrics import (
    annual_return,
    annual_volatility,
    direction_accuracy,
    icir,
    max_drawdown,
    rank_ic,
    sharpe,
    sortino,
)
from qgbs.backtest.rolling import WindowSpec, rolling_splits
from qgbs.data.adapters.csv_adapter import CsvAdapter
from qgbs.data.adapters.mock import MockAdapter, MockDataConfig
from qgbs.data.calendar import rebalance_dates
from qgbs.factors.finance import build_finance_features
from qgbs.factors.neutralization import MARKET_COL, JointFactorModel, build_alpha_label
from qgbs.factors.pipeline import FactorPipeline
from qgbs.gbs.backends.base import SampleBatch
from qgbs.gbs.backends.cached import CachedBackend
from qgbs.gbs.backends.simulator import GaussianSimulatorBackend, SimulatorConfig
from qgbs.gbs.basis import BasisSelectionParams, select_higher_order_basis
from qgbs.gbs.compression import (
    NystromConfig,
    NystromFeatures,
    QuantileExplicitConfig,
    QuantileExplicitFeatures,
    RFFConfig,
    RFFeatures,
)
from qgbs.gbs.encoder import (
    EncoderParams,
    ModeAddress,
    QuantileEncoder,
    ensure_default_mappings,
    resolve_mapping_id,
)
from qgbs.gbs.feature_service import GBSFeatureConfig, GBSFeatureService
from qgbs.gbs.sample_store import SampleStore
from qgbs.models.blending import estimate_rho, estimate_uncertainty_by_method
from qgbs.models.readout import GBMReadout, ReadoutModel, RidgeReadout
from qgbs.models.selection import search_hyperparams, select_features
from qgbs.portfolio.costs import CostModel
from qgbs.portfolio.optimizer import (
    DEFAULT_UNCERTAINTY_KAPPA,
    OptimizerParams,
    PortfolioConstraints,
)
from qgbs.portfolio.risk import FactorRiskConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"

EXPERIMENT_CONFIG_FILES = {
    "C0": "c0_legacy_replication.yaml",
    "C1": "c1_classical_baseline.yaml",
    "C2": "c2_ridge_baseline.yaml",
    "C3": "c3_gbm_baseline.yaml",
    "C4": "c4_quantile_explicit.yaml",
    "C5": "c5_random_features.yaml",
    "Q1": "q1_marginal.yaml",
    "Q2": "q2_second_order.yaml",
    "Q3": "q3_higher_order.yaml",
    "Q4": "q4_full_enhanced.yaml",
}


@dataclass(frozen=True)
class ExperimentResult:
    """一次实验的完整记录，保证可复现。"""

    experiment_id: str
    metrics: dict[str, float]  # 指标名 → 数值
    git_hash: str  # 代码版本
    seed: int  # 随机种子
    data_version: str  # 数据版本
    artifacts_dir: str = ""  # 产物目录（净值曲线、权重、日志）
    extra: dict[str, Any] = field(default_factory=dict)


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: Path) -> dict[str, Any]:
    """加载实验 yaml 并与 ``configs/base.yaml`` 深合并。"""
    config_path = Path(config_path)
    base_path = CONFIG_DIR / "base.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8")) if base_path.exists() else {}
    over = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg = _merge(base, over)
    cfg["_config_path"] = str(config_path)
    return cfg


def _git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "nogit"
    except Exception:
        return "nogit"


def _build_mock_adapter(cfg: dict[str, Any]) -> MockAdapter:
    mock_yaml = CONFIG_DIR / "data" / "mock.yaml"
    raw: dict[str, Any] = {}
    if mock_yaml.exists():
        raw = yaml.safe_load(mock_yaml.read_text(encoding="utf-8")) or {}
    # 种子优先级：configs/data/mock.yaml < 顶层 seed（全局种子）< data.mock.seed
    if cfg.get("seed") is not None:
        raw = _merge(raw, {"seed": cfg["seed"]})
    raw = _merge(raw, cfg.get("data", {}).get("mock", {}))
    known = {f for f in MockDataConfig.__dataclass_fields__}
    kwargs = {k: v for k, v in raw.items() if k in known}
    if "cash_range" in kwargs:
        kwargs["cash_range"] = tuple(kwargs["cash_range"])
    return MockAdapter(MockDataConfig(**kwargs))


def _build_adapter(cfg: dict[str, Any]) -> tuple[Any, str, str, int, float]:
    """按 ``data.adapter`` 构建数据适配器，并归一化数据运行参数。

    Returns:
        (adapter, start, end, seed, rf_annual)：数据区间、随机种子与
        无风险年化利率——mock 取自 MockDataConfig，csv 取自 ``data.*`` 配置键。
    """
    data_cfg = cfg.get("data", {})
    name = str(data_cfg.get("adapter", "mock"))
    if name == "mock":
        adapter = _build_mock_adapter(cfg)
        mc = adapter.config
        return adapter, mc.start, mc.end, int(mc.seed), float(mc.rf_annual)
    if name == "csv":
        store_dir = data_cfg.get("store_dir")
        if not store_dir:
            raise ValueError("data.adapter='csv' 需要配置 data.store_dir（中信交付文件根目录）")
        root = Path(store_dir)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        if not root.is_dir():
            raise FileNotFoundError(
                f"csv 数据目录不存在: {root}（data.store_dir 应指向中信交付文件根目录，"
                f"必需文件清单见 configs/data/csv.yaml）"
            )
        missing = [k for k in ("start", "end") if not data_cfg.get(k)]
        if missing:
            raise ValueError(f"data.adapter='csv' 需要配置数据区间键: {['data.' + k for k in missing]}")
        return (
            CsvAdapter(root),
            str(data_cfg["start"]),
            str(data_cfg["end"]),
            int(data_cfg.get("seed", cfg.get("seed", 42))),
            float(data_cfg.get("rf_annual", 0.02)),
        )
    raise ValueError(f"未知数据适配器: {name!r}（支持 mock / csv）")


def _build_readout(cfg: dict[str, Any]) -> ReadoutModel | None:
    name = cfg.get("readout")
    if name is None:
        return None
    rp = cfg.get("readout_params") or {}
    if name == "ridge":
        alpha = rp.get("alpha", (rp.get("params") or {}).get("alpha", 1.0))
        return RidgeReadout(alpha=float(alpha))
    if name == "gbm":
        return GBMReadout(params=dict(rp.get("params") or {}))
    raise ValueError(f"未知读出模型: {name!r}")


def _gbs_feature_dim(ref_cfg: dict[str, Any], n_finance: int) -> int:
    """解析参照实验的 GBS 特征块维度（不含 finance_base），用于 C5 等维对齐。"""
    fams = [f for f in (ref_cfg.get("features") or []) if str(f).startswith("gbs_")]
    dim = 0
    if "gbs_marginal" in fams:
        dim += n_finance
    if "gbs_second_order" in fams:
        dim += n_finance * (n_finance - 1) // 2
    if "gbs_higher_order" in fams:
        dim += int((ref_cfg.get("basis") or {}).get("N", 3000))
    if dim == 0:
        raise ValueError("参照实验无 GBS 特征族，无法 auto 推导等维维度")
    return dim


def _resolve_target_dim(spec: object, n_finance: int) -> int:
    """解析 random_features.target_dim：整数或 "auto:<experiment_id>"。"""
    if isinstance(spec, int):
        return spec
    if isinstance(spec, str) and spec.startswith("auto:"):
        ref_id = spec[len("auto:"):]
        path = CONFIG_DIR / "experiments" / EXPERIMENT_CONFIG_FILES.get(ref_id, "")
        if not path.exists():
            raise ValueError(f"等维参照实验 {ref_id!r} 未注册或无配置文件: {path}")
        ref_cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return _gbs_feature_dim(ref_cfg, n_finance)
    raise ValueError(f"无法解析 target_dim: {spec!r}（支持整数或 'auto:<experiment_id>'）")


def _assemble_features(
    family_cfg: dict[str, Any],
    feats: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """按特征族组装特征面板（变换器在 train 窗 fit，返回全样本变换结果）。"""
    parts: list[pd.DataFrame] = []
    families = family_cfg["features"]
    if "finance_base" in families or not families:
        parts.append(feats)
    train_mask = feats.index.get_level_values("date").isin(train_dates)
    if "quantile_explicit" in families:
        qcfg = QuantileExplicitConfig(**(family_cfg.get("quantile_explicit") or {}))
        qe = QuantileExplicitFeatures(qcfg).fit(feats[train_mask])
        parts.append(qe.transform(feats))
    if "random_features" in families:
        rcfg = dict(family_cfg.get("random_features") or {})
        kind = rcfg.pop("kind", "rff")
        rcfg["target_dim"] = _resolve_target_dim(rcfg.get("target_dim"), feats.shape[1])
        if kind == "rff":
            rf = RFFeatures(RFFConfig(**rcfg)).fit(feats[train_mask].dropna())
        elif kind == "nystrom":
            rf = NystromFeatures(NystromConfig(**rcfg)).fit(feats[train_mask].dropna())
        else:
            raise ValueError(f"未知随机特征类型: {kind!r}")
        arr = rf.transform(feats)
        cols = [f"rf_{i}" for i in range(arr.shape[1])]
        parts.append(pd.DataFrame(arr, index=feats.index, columns=cols))
    return pd.concat(parts, axis=1)


def _prepare_gbs_backend(cfg: dict[str, Any]) -> tuple[CachedBackend, SampleBatch]:
    """准备 GBS 后端：检查样本库，不存在则用 simulator 生成入库（可复现）。"""
    gbs_cfg = dict(cfg.get("gbs") or {})
    sim_cfg = SimulatorConfig(
        n_temporal=int(gbs_cfg.get("n_temporal", 32)),
        n_spatial=int(gbs_cfg.get("n_spatial", 9)),
        seed=int(gbs_cfg.get("seed", cfg.get("seed", 42))),
        system_efficiency=float(gbs_cfg.get("system_efficiency", 0.7)),
        mean_click_rate=float(gbs_cfg.get("mean_click_rate", 0.3)),
    )
    n_shots = int(gbs_cfg.get("n_shots", 20_000))
    # batch_id 由配置字符串确定性派生（规则与 GaussianSimulatorBackend.sample
    # 的 BatchMeta 构造一致，见 simulator.py），样本库命中时无需实际采样
    batch_id = (
        f"sim-{sim_cfg.method}-{sim_cfg.n_temporal}x{sim_cfg.n_spatial}"
        f"-s{sim_cfg.seed}-n{n_shots}-eta{sim_cfg.system_efficiency}"
    )
    store_root = gbs_cfg.get("sample_store")
    store = SampleStore(PROJECT_ROOT / store_root) if store_root else SampleStore()
    existing = store.list_batches(namespace="simulation")
    if len(existing) == 0 or batch_id not in set(existing["batch_id"]):
        # 样本缺失时才用 simulator 实际采样入库
        store.register_batch(GaussianSimulatorBackend(sim_cfg).sample(n_shots))
    backend = CachedBackend(store, batch_id)
    batch = backend.sample(n_shots)
    # 样本版本管理正确性：命中批次的模式结构必须与当前配置一致，
    # 不一致说明 batch_id 规则漂移或库存被污染，显式报错而非静默复用
    mc = batch.meta.mode_config
    if (
        int(mc.get("n_temporal", -1)) != sim_cfg.n_temporal
        or int(mc.get("n_spatial", -1)) != sim_cfg.n_spatial
    ):
        raise ValueError(
            f"样本库批次 {batch_id!r} 模式结构不一致：库存为 "
            f"{mc.get('n_temporal')}×{mc.get('n_spatial')}，"
            f"当前配置 {sim_cfg.n_temporal}×{sim_cfg.n_spatial}；请检查样本库"
        )
    return backend, batch


def _slice_addresses(addr: ModeAddress, row_mask: np.ndarray) -> ModeAddress:
    """按行掩码切分 ModeAddress（用于只在训练行上 fit 压缩等）。"""
    return ModeAddress(
        asset_index=addr.asset_index[row_mask],
        feature_index=addr.feature_index[row_mask],
        bin_index=addr.bin_index[row_mask],
        spatial_index=addr.spatial_index[row_mask],
        temporal_index=addr.temporal_index[row_mask],
        partition=addr.partition[row_mask],
        mode_index=addr.mode_index[row_mask],
    )


def _build_gbs_panel(
    cfg: dict[str, Any],
    gbs_families: list[str],
    feats_p: pd.DataFrame,
    split,
    backend: CachedBackend,
    batch: SampleBatch,
) -> pd.DataFrame:
    """构建 GBS 特征面板：encoder 训练窗 fit/freeze → 地址 → 特征服务。"""
    enc_cfg = dict(cfg.get("encoder") or {})
    gbs_cfg = dict(cfg.get("gbs") or {})
    n_temporal = int(gbs_cfg.get("n_temporal", 32))
    n_spatial = int(gbs_cfg.get("n_spatial", 9))
    ensure_default_mappings(feats_p.shape[1], n_spatial, n_temporal)
    mapping_id = resolve_mapping_id(
        str(enc_cfg.get("mapping_id", "identity")), feats_p.shape[1], n_spatial, n_temporal
    )
    params = EncoderParams(
        n_bins_a=int(enc_cfg.get("n_bins_a", 256)),
        n_bins_b=int(enc_cfg.get("n_bins_b", 255)),
        partition_boundary=int(enc_cfg.get("partition_boundary", 256)),
    )
    train_mask = feats_p.index.get_level_values("date").isin(split.train)
    enc = QuantileEncoder(params, mapping_id=mapping_id)
    enc.fit(feats_p[train_mask].dropna())
    enc = enc.freeze()

    stat_kinds: list[str] = []
    if "gbs_marginal" in gbs_families:
        stat_kinds.append("marginal")
    if "gbs_second_order" in gbs_families:
        stat_kinds.append("second_order")
    basis = None
    if "gbs_higher_order" in gbs_families:
        stat_kinds.append("higher_order")
        known = set(BasisSelectionParams.__dataclass_fields__)
        bcfg = {k: v for k, v in (cfg.get("basis") or {}).items() if k in known}
        basis = select_higher_order_basis(
            batch, tuple(range(backend.n_modes)), BasisSelectionParams(**bcfg)
        )

    addresses = enc.transform(feats_p)
    service = GBSFeatureService(
        GBSFeatureConfig(
            stat_kinds=tuple(stat_kinds),
            activation=str(gbs_cfg.get("activation", "soft")),
        ),
    )
    service.fit(_slice_addresses(addresses, train_mask), [batch], basis=basis)
    service.precompute_cache(addresses)
    service = service.freeze()
    return service.build_features(addresses, feats_p.index)


def _optimizer_param_grid(opt_cfg: dict[str, Any]) -> list[dict[str, float]]:
    """解析优化器参数网格（方案书 §6.2：λ/τ/γ/κ 经 train/valid 选择）。

    ``optimizer.search`` 未配置时返回单点（= ``optimizer`` 固定值，行为与
    历史完全一致）；配置后按键名字典序做笛卡尔积，网格顺序确定。
    """
    base = {
        "risk_aversion": float(opt_cfg.get("risk_aversion", 5.0)),
        "turnover_penalty": float(opt_cfg.get("turnover_penalty", 0.001)),
        "l2_regularization": float(opt_cfg.get("l2_regularization", 0.0001)),
        "uncertainty_kappa": float(opt_cfg.get("uncertainty_kappa", DEFAULT_UNCERTAINTY_KAPPA)),
    }
    search = opt_cfg.get("search") or {}
    unknown = set(search) - set(base)
    if unknown:
        raise ValueError(f"optimizer.search 含未知参数: {sorted(unknown)}（支持 {sorted(base)}）")
    keys = sorted(search)
    grid: list[dict[str, float]] = []
    for vals in product(*(search[k] for k in keys)):
        grid.append({**base, **dict(zip(keys, (float(v) for v in vals)))})
    return grid or [base]


def _window_net_return(bt: BacktestResult, window: pd.DatetimeIndex) -> float:
    """窗口内扣成本净收益（成本已由引擎扣减）；窗口无收益记录返回 -inf。"""
    r = bt.returns[bt.returns.index.isin(window)]
    if len(r) == 0:
        return float("-inf")
    return float((1.0 + r).prod() - 1.0)


def run_experiment(config_path: Path) -> ExperimentResult:
    """执行一个对照实验（详见模块 docstring 的滚动协议）。

    可复现性保证：所有对照实验共用同一测试期、资产池、成本模型；
    记录 git hash、随机种子、数据版本到 ExperimentResult。
    """
    cfg = load_config(config_path)
    exp_id = cfg["experiment_id"]
    data_cfg = cfg.get("data", {})
    # features 支持两种形态：列表（特征族）或映射 {families: [...], finance_set: base|extended}
    features_spec = cfg.get("features") or []
    if isinstance(features_spec, dict):
        families = list(features_spec.get("families") or [])
        finance_set = str(features_spec.get("finance_set", "base"))
    else:
        families = list(features_spec)
        finance_set = "base"
    cfg["features"] = families  # 归一化为列表，下游（_assemble_features 等）统一消费
    gbs_families = [f for f in families if f.startswith("gbs_")]
    backend: CachedBackend | None = None
    gbs_batch: SampleBatch | None = None
    if gbs_families:
        backend, gbs_batch = _prepare_gbs_backend(cfg)

    # --- 数据（data.adapter 分发：mock / csv）---
    adapter, data_start, data_end, data_seed, rf_annual = _build_adapter(cfg)
    pool = adapter.load_asset_pool_history(data_cfg.get("pool", "pool100"))
    stock_records = pool.records[~pool.records["is_cash_asset"]]
    symbols = sorted(stock_records["symbol"].unique())
    industry = stock_records.set_index("symbol")["citics_industry_l1"]
    market = adapter.load_daily_market(symbols, data_start, data_end)
    status = adapter.load_trading_status(symbols, data_start, data_end)
    msf = adapter.load_market_sector_factor(data_start, data_end)
    cc = adapter.load_constraints_costs(str(data_cfg.get("constraints_config", "default")))

    close = market.rows.pivot(index="trade_date", columns="symbol", values="close").sort_index()
    ret_wide = market.rows.pivot(index="trade_date", columns="symbol", values="daily_return").sort_index()
    dates = close.index
    mkt_code = str(data_cfg.get("market_index", "IDX_MKT"))
    if mkt_code not in msf.index_returns.columns:
        raise KeyError(
            f"市场指数 {mkt_code!r}（data.market_index）不在 index_returns 中"
            f"（可用: {sorted(msf.index_returns.columns)}）"
        )
    factor_df = pd.concat([msf.index_returns[mkt_code].rename(MARKET_COL), msf.industry_index_returns], axis=1)

    # 因果特征，全样本计算安全；finance_set=extended 时市场状态特征用市场指数收益
    feats = build_finance_features(
        market, feature_set=finance_set, market_returns=msf.index_returns[mkt_code]
    )
    spec = WindowSpec(**{k: cfg["rolling"][k] for k in ("train", "valid", "test", "purge", "embargo")})
    horizon = int(cfg.get("label_horizon", 21))
    rebal_all = rebalance_dates(dates, spec.test)
    test_set_dates = [d for split in rolling_splits(dates, spec) for d in split.test]
    test_rebal = [d for d in rebal_all if d in set(test_set_dates)]

    # --- 逐折训练与样本外预测 ---
    model_name = cfg.get("readout")
    blend_cfg = cfg.get("blending") or {}
    blend_enabled = bool(blend_cfg.get("enabled", False)) and model_name is not None
    # 优化器参数网格（方案书 §6.2）：默认单点 = optimizer 固定值，行为不变；
    # 多点网格时逐折在 valid 窗用 BacktestEngine 扣成本净收益选优，测试期用选中值
    opt_cfg = cfg.get("optimizer") or {}
    opt_grid = _optimizer_param_grid(opt_cfg)
    opt_search_active = len(opt_grid) > 1 and model_name is not None
    risk_cfg = cfg.get("risk") or {}
    cov_method = cfg.get("cov_method", risk_cfg.get("cov_method", "ledoit_wolf"))
    factor_risk_cfg = FactorRiskConfig(
        factor_cov_method=str(risk_cfg.get("factor_cov_method", "ewma")),
        ewma_halflife=float(risk_cfg.get("ewma_halflife", 60.0)),
    )
    # 约束与成本统一取自适配器的 constraints_costs 交付（全折共用）
    constraints_obj = PortfolioConstraints(
        weight_cap={s: cc.weight_upper.get(s, opt_cfg.get("weight_cap", 0.10)) for s in symbols},
        cash_range=tuple(opt_cfg.get("cash_range", cc.cash_range)),
        exposure_bounds=dict(cc.exposure_bounds),
        max_turnover=cc.max_turnover,
        liquidity_limit=cc.liquidity_limit,
        blacklist=cc.blacklist,
    )
    cost_model_obj = CostModel(
        commission_bps=cc.commission_bps,
        stamp_tax_bps=cc.stamp_tax_bps,
        slippage_bps=cc.slippage_bps,
        impact_coef=cc.impact_coef,
    )
    rf_daily = rf_annual / 252.0
    # 行业暴露哑变量（mock 行业静态；真实数据须用 point-in-time 暴露）
    exposure_matrix = pd.get_dummies(industry).astype(float)

    predictions: dict[pd.Timestamp, pd.Series] = {}
    ic_rows: dict[pd.Timestamp, float] = {}
    dir_rows: dict[pd.Timestamp, float] = {}
    unc_rows: dict[pd.Timestamp, pd.Series] = {}
    rho_per_fold: list[float] = []
    opt_sel_folds: list[dict[str, Any]] = []
    fold_rets: list[pd.Series] = []
    fold_turns: list[pd.Series] = []
    fold_costs: list[pd.Series] = []
    fold_weights: list[pd.DataFrame] = []
    fold_ledgers: list[pd.DataFrame] = []
    n_folds = 0
    for split in rolling_splits(dates, spec):
        fold_rebal = [d for d in test_rebal if d in set(split.test)]
        if not fold_rebal or model_name is None:
            continue
        n_folds += 1
        jfm = JointFactorModel().fit(ret_wide.loc[split.train], industry, factor_df.loc[split.train])
        labels = build_alpha_label(close, industry, horizon, factor_df, model=jfm)

        pipe = FactorPipeline()
        train_mask = feats.index.get_level_values("date").isin(split.train)
        pipe.fit(feats[train_mask].dropna())
        feats_p = pipe.transform(feats)
        X_all = _assemble_features(cfg, feats_p, split.train)
        if gbs_families:
            assert backend is not None and gbs_batch is not None
            gbs_panel = _build_gbs_panel(cfg, gbs_families, feats_p, split, backend, gbs_batch)
            X_all = pd.concat([X_all, gbs_panel], axis=1)

        # 防泄漏：train 标签前视窗不得伸入 valid 窗，train/valid 标签前视窗
        # 不得伸入 test 窗（避免验证评分与测试评估的乐观偏差）
        label_dates = labels.index.get_level_values("date")
        label_pos = dates.get_indexer(pd.DatetimeIndex(label_dates))
        first_test_pos = dates.get_loc(split.test[0])
        first_valid_pos = dates.get_loc(split.valid[0]) if len(split.valid) else first_test_pos
        in_train = label_dates.isin(split.train)
        in_valid = label_dates.isin(split.valid)
        allowed = (in_train & (label_pos + horizon < first_valid_pos)) | (
            in_valid & (label_pos + horizon < first_test_pos)
        )
        labels_tv = labels[allowed]

        X_tv = X_all.reindex(labels_tv.index)
        valid_mask = labels_tv.index.get_level_values("date").isin(split.valid)
        X_tr, y_tr = X_tv[~valid_mask], labels_tv[~valid_mask]
        X_va, y_va = X_tv[valid_mask], labels_tv[valid_mask]

        # 特征筛选（train 内）
        sel_cfg = cfg.get("selection") or {}
        if sel_cfg.get("max_features"):
            sel = select_features(
                X_tr.dropna(), y_tr.reindex(X_tr.dropna().index),
                method=sel_cfg.get("method", "ic_mean"),
                max_features=int(sel_cfg["max_features"]),
            )
            X_tr, X_va, X_all = X_tr[sel.selected], X_va[sel.selected], X_all[sel.selected]

        # 超参搜索（train 拟合 valid 评分），随后 train+valid 重训
        # best = 基础参数（readout_params.params）与搜索最优的合并，保证
        # random_state 等固定参数不在搜索后丢失
        best = dict((cfg.get("readout_params") or {}).get("params") or {})
        search_cfg = cfg.get("search") or {}
        if search_cfg.get("param_grid") and len(y_va.dropna()) > 0:
            grid = search_cfg["param_grid"]
            res = search_hyperparams(model_name, X_tr, y_tr, X_va, y_va, grid, fixed_params=best)
            best = {**best, **res.best_params}
        if model_name == "ridge":
            model: ReadoutModel = RidgeReadout(alpha=float(best.get("alpha", 1.0)))
        else:
            model = GBMReadout(params=dict(best))

        train_all = X_tv.dropna()
        y_all = labels_tv.reindex(train_all.index)
        fit_mask = y_all.notna()

        # ρ 收缩混合（方案书 §5.4/§6.3）：基准 = 同折内不含 GBS 特征的 C2 口径 ridge
        # train-only 模型：ρ 估计与优化器参数 valid 评分共用（避免重训模型的乐观偏差）
        rho = float("nan")
        base_model: RidgeReadout | None = None
        finance_cols: list[str] = []
        model_tr: ReadoutModel | None = None
        base_tr: RidgeReadout | None = None
        unc_series: pd.Series | None = None
        if blend_enabled or opt_search_active:
            model_tr = (
                RidgeReadout(alpha=float(best.get("alpha", 1.0)))
                if model_name == "ridge"
                else GBMReadout(params=dict(best))
            ).fit(X_tr, y_tr)
        if blend_enabled:
            assert model_tr is not None
            finance_cols = [c for c in feats_p.columns if c in X_all.columns]
            base_grid = dict(blend_cfg.get("base_grid") or {"alpha": [0.1, 1.0, 10.0]})
            base_res = search_hyperparams(
                "ridge", X_tr[finance_cols], y_tr, X_va[finance_cols], y_va, base_grid
            )
            base_alpha = float(base_res.best_params.get("alpha", 1.0))
            # valid 窗上用 train-only 模型估计 ρ（避免用重训后模型造成乐观偏差）
            base_tr = RidgeReadout(alpha=base_alpha).fit(X_tr[finance_cols], y_tr)
            rho = estimate_rho(model_tr.predict(X_va), base_tr.predict(X_va[finance_cols]), y_va)
            rho_per_fold.append(rho)
            base_model = RidgeReadout(alpha=base_alpha).fit(
                train_all[finance_cols][fit_mask], y_all[fit_mask]
            )
            # 可选：逐资产不确定性（口径由 blending.uncertainty.method 配置）→ 优化器保守 alpha
            if blend_cfg.get("uncertainty_enabled"):
                pred_blend_va = (1 - rho) * base_tr.predict(X_va[finance_cols]) + rho * model_tr.predict(X_va)
                resid = (y_va - pred_blend_va).dropna()
                unc_cfg = (cfg.get("blending") or {}).get("uncertainty") or {}
                unc_method = str(unc_cfg.get("method", "residual"))
                if unc_method == "quantile":
                    # 条件分位数模型：train 窗拟合，valid 窗评分宽度
                    unc_series = estimate_uncertainty_by_method(
                        "quantile", X_train=X_tr, y_train=y_tr, X_score=X_va, seed=data_seed
                    )
                else:
                    unc_series = estimate_uncertainty_by_method(
                        unc_method,
                        residuals=resid.unstack("symbol"),
                        block_length=int(unc_cfg.get("block_length", 21)),
                        n_bootstrap=int(unc_cfg.get("n_bootstrap", 200)),
                        seed=data_seed,
                    )
                for d in fold_rebal:
                    unc_rows[d] = unc_series

        model.fit(train_all[fit_mask], y_all[fit_mask])
        for d in fold_rebal:
            if d not in X_all.index.get_level_values("date"):
                continue
            X_d = X_all.loc[d].dropna()
            if X_d.empty:
                continue
            pred = model.predict(X_d)
            if base_model is not None:
                # α^final = (1−ρ)·α^base + ρ·α^GBS
                pred = (1.0 - rho) * base_model.predict(X_d[finance_cols]) + rho * pred
            predictions[d] = pred
            y_d = labels.loc[d].reindex(pred.index) if d in labels.index.get_level_values("date") else None
            if y_d is not None:
                ic_rows[d] = rank_ic(pred, y_d)
                dir_rows[d] = direction_accuracy(pred, y_d)

        # --- 优化器参数验证期选择（方案书 §6.2）---
        # 用 train-only 信号在 valid 窗逐候选跑 BacktestEngine，按扣成本净收益
        # 选最优组合；本折 test 窗用选中参数单独回测，最后跨折拼接。
        if opt_search_active:
            assert model_tr is not None
            va_rebal = [d for d in rebal_all if d in set(split.valid)]
            alpha_va: dict[pd.Timestamp, pd.Series] = {}
            for d in va_rebal:
                if d not in X_all.index.get_level_values("date"):
                    continue
                X_d = X_all.loc[d].dropna()
                if X_d.empty:
                    continue
                pred_va = model_tr.predict(X_d)
                if base_tr is not None and np.isfinite(rho):
                    pred_va = (1.0 - rho) * base_tr.predict(X_d[finance_cols]) + rho * pred_va
                alpha_va[d] = pred_va
            alpha_va_df = pd.DataFrame(alpha_va).T.sort_index()
            unc_va_df = (
                pd.DataFrame({d: unc_series for d in alpha_va}).T if unc_series is not None else None
            )
            cand_logs: list[dict[str, Any]] = []
            for cand in opt_grid:
                eng_v = BacktestEngine(
                    window_spec=spec,
                    constraints=constraints_obj,
                    optimizer_params=OptimizerParams(**cand),
                    cost_model=cost_model_obj,
                    cov_method=cov_method,
                    factor_risk_config=factor_risk_cfg,
                )
                bt_v = eng_v.run(
                    market,
                    status,
                    risk_free_daily=rf_daily,
                    alpha_panel=alpha_va_df,
                    exposure_matrix=exposure_matrix,
                    alpha_uncertainty_panel=unc_va_df,
                )
                cand_logs.append(
                    {"params": dict(cand), "valid_score": _window_net_return(bt_v, split.valid)}
                )
            # 确定性平局规则：取网格序最先者；valid 无法评分时回退首候选
            best_idx = max(range(len(opt_grid)), key=lambda i: cand_logs[i]["valid_score"])
            sel_params = opt_grid[best_idx]
            if not np.isfinite(cand_logs[best_idx]["valid_score"]):
                sel_params = opt_grid[0]
            fold_log: dict[str, Any] = {
                "split_id": split.split_id,
                "params": dict(sel_params),
                "candidates": cand_logs,
            }
            fold_alpha = {d: predictions[d] for d in fold_rebal if d in predictions}
            unc_f = (
                pd.DataFrame({d: unc_rows[d] for d in fold_rebal if d in unc_rows}).T
                if unc_rows
                else None
            )
            eng_f = BacktestEngine(
                window_spec=spec,
                constraints=constraints_obj,
                optimizer_params=OptimizerParams(**sel_params),
                cost_model=cost_model_obj,
                cov_method=cov_method,
                factor_risk_config=factor_risk_cfg,
            )
            bt_f = eng_f.run(
                market,
                status,
                risk_free_daily=rf_daily,
                alpha_panel=pd.DataFrame(fold_alpha).T.sort_index(),
                exposure_matrix=exposure_matrix,
                alpha_uncertainty_panel=unc_f,
            )
            fold_log["test_turnovers"] = [float(v) for v in bt_f.turnover]
            opt_sel_folds.append(fold_log)
            ret_f = bt_f.returns[bt_f.returns.index.isin(split.test)]
            if len(ret_f):
                fold_rets.append(ret_f)
                fold_turns.append(bt_f.turnover)
                fold_costs.append(bt_f.costs)
                fold_weights.append(bt_f.weights)
                fold_ledgers.append(bt_f.ledger)

    alpha_panel = (
        pd.DataFrame(predictions).T.sort_index() if predictions else None
    )
    unc_panel = (
        pd.DataFrame(unc_rows).T.sort_index()
        if unc_rows and blend_cfg.get("uncertainty_enabled")
        else None
    )

    # --- 统一回测 ---
    if opt_search_active and fold_rets:
        # 逐折 test 窗已用该折选中参数分别回测：拼接收益/换手/成本/权重流水。
        # 注：与单次连续回测相比，每折 test 起点从全现金建仓，开启多点搜索后
        # 结果口径与默认单点（连续回测）不直接可比。
        rets = pd.concat(fold_rets).sort_index()
        bt = BacktestResult(
            nav=pd.Series((1.0 + rets).cumprod().to_numpy(), index=rets.index, name="nav"),
            weights=pd.concat(fold_weights),
            turnover=pd.concat(fold_turns).sort_index(),
            costs=pd.concat(fold_costs).sort_index(),
            returns=rets,
            ledger=pd.concat(fold_ledgers, ignore_index=True),
        )
    else:
        engine = BacktestEngine(
            window_spec=spec,
            constraints=constraints_obj,
            optimizer_params=OptimizerParams(**opt_grid[0]),
            cost_model=cost_model_obj,
            cov_method=cov_method,
            factor_risk_config=factor_risk_cfg,
        )
        bt = engine.run(
            market,
            status,
            risk_free_daily=rf_daily,
            alpha_panel=alpha_panel,
            rebalance_override=None if alpha_panel is not None else pd.DatetimeIndex(test_rebal),
            exposure_matrix=exposure_matrix,
            alpha_uncertainty_panel=unc_panel,
        )

    ic_s = pd.Series(ic_rows).dropna()
    metrics = {
        "rank_ic_mean": float(ic_s.mean()) if len(ic_s) else float("nan"),
        "rank_icir": icir(ic_s),
        "direction_accuracy": float(pd.Series(dir_rows).mean()) if dir_rows else float("nan"),
        "sharpe": sharpe(bt.returns),
        "sortino": sortino(bt.returns),
        "annual_return": annual_return(bt.returns),
        "annual_volatility": annual_volatility(bt.returns),
        "max_drawdown": max_drawdown(bt.nav),
        "turnover_mean": float(bt.turnover.mean()) if len(bt.turnover) else float("nan"),
        "total_cost": float(bt.costs.sum()) if len(bt.costs) else 0.0,
        "n_rebalances": float(len(bt.weights)),
        "n_folds": float(n_folds),
        "rho_mean": float(np.mean(rho_per_fold)) if rho_per_fold else float("nan"),
        "rho_std": float(np.std(rho_per_fold)) if rho_per_fold else float("nan"),
    }
    # 数据版本：配置标识（data.data_version）+ 适配器自动指纹
    auto_dv = f"{data_cfg.get('adapter', 'mock')}-seed{data_seed}-{data_start}_{data_end}"
    dv_prefix = str(data_cfg.get("data_version", "")).strip()
    return ExperimentResult(
        experiment_id=exp_id,
        metrics=metrics,
        git_hash=_git_hash(),
        seed=int(data_seed),
        data_version=f"{dv_prefix}-{auto_dv}" if dv_prefix else auto_dv,
        extra={
            "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
            "rho_per_fold": rho_per_fold,
            "optimizer_selection": {
                "search_active": bool(opt_search_active),
                "score": "valid 窗扣成本净收益（BacktestEngine 口径）",
                "fixed_params": dict(opt_grid[0]),
                "folds": opt_sel_folds,
            },
        },
    )


def run_all(registry_ids: list[str] | None = None) -> pd.DataFrame:
    """批量执行对照矩阵，汇总各实验核心指标为对比表。

    默认执行经典五实验 C1-C5；C0（mock 占位）与 Q1-Q4（仿真 GBS 样本）
    均可经 ``registry_ids`` 显式指定执行。
    """
    ids = registry_ids or ["C1", "C2", "C3", "C4", "C5"]
    rows: dict[str, dict[str, float]] = {}
    for exp_id in ids:
        cfg_path = CONFIG_DIR / "experiments" / EXPERIMENT_CONFIG_FILES[exp_id]
        result = run_experiment(cfg_path)
        rows[exp_id] = result.metrics
    table = pd.DataFrame(rows).T
    table.index.name = "experiment"
    return table
