"""E8.1 单折验证：fold 0 上 R0/R1/R2/R3' 四配置跑通 + sanity（不作结论）。

配置：alpha 引擎固定 Q1（26 extended_plus + 真机边际 + ridge）；风险引擎
分别为经典 LW（R0）/ GBS 时变先验（R1）/ 随机先验（R2）/ 静态 GBS 先验
（R3'）。δ ∈ {0, 0.25, 0.5, 0.75, 1.0} 在 valid 窗择优（净收益口径）。

用法：``PYTHONPATH=src .venv/bin/python scripts/e8_phase1_single_fold.py``
产物：``artifacts/e8_risk_prior/phase1/``。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qgbs.backtest.engine import BacktestEngine
from qgbs.backtest.metrics import annual_volatility, calmar, information_ratio, sharpe
from qgbs.backtest.rolling import WindowSpec, rolling_splits
from qgbs.data.adapters.csv_adapter import CsvAdapter
from qgbs.experiments.runner import _build_gbs_panel, _prepare_gbs_backend, load_config
from qgbs.factors.finance import build_finance_features
from qgbs.factors.neutralization import MARKET_COL, JointFactorModel, build_alpha_label
from qgbs.factors.pipeline import FactorPipeline
from qgbs.gbs.risk_prior import PriorProvider, build_static_corr
from qgbs.gbs.statistics import second_order_stats
from qgbs.models.readout import RidgeReadout
from qgbs.portfolio.costs import CostModel
from qgbs.portfolio.optimizer import OptimizerParams, PortfolioConstraints

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "e8_risk_prior" / "phase1"
DELTA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    cfg = load_config(ROOT / "configs/experiments" / "q1_marginal_hardware_p1extplus.yaml")
    data_cfg = cfg["data"]
    adapter = CsvAdapter(ROOT / data_cfg["store_dir"])
    pool = adapter.load_asset_pool_history(data_cfg["pool"])
    stock_records = pool.records[~pool.records["is_cash_asset"]]
    symbols = sorted(stock_records["symbol"].unique())
    industry = stock_records.drop_duplicates("symbol").set_index("symbol")["citics_industry_l1"]
    market = adapter.load_daily_market(symbols, data_cfg["start"], data_cfg["end"])
    status = adapter.load_trading_status(symbols, data_cfg["start"], data_cfg["end"])
    msf = adapter.load_market_sector_factor(data_cfg["start"], data_cfg["end"])
    close = market.rows.pivot(index="trade_date", columns="symbol", values="close").sort_index()
    ret_wide = market.rows.pivot(index="trade_date", columns="symbol", values="daily_return").sort_index()
    mkt_code = data_cfg["market_index"]
    factor_df = pd.concat([msf.index_returns[mkt_code].rename(MARKET_COL), msf.industry_index_returns], axis=1)
    feats = build_finance_features(market, feature_set="extended_plus",
                                   market_returns=msf.index_returns[mkt_code],
                                   industry_index_returns=msf.industry_index_returns,
                                   industry_map=industry, status=status)
    spec = WindowSpec(train="1y", valid="3m", test="1m", purge="5d", embargo="2d")
    split = next(iter(rolling_splits(close.index, spec)))
    print(f"fold 0: train {split.train[0].date()}~{split.train[-1].date()} "
          f"valid {split.valid[0].date()}~{split.valid[-1].date()} test {split.test[0].date()}~{split.test[-1].date()}", flush=True)

    # --- Q1 alpha 链（train 拟合、全期预测） ---
    jfm = JointFactorModel().fit(ret_wide.loc[split.train], industry, factor_df.loc[split.train])
    labels = build_alpha_label(close, industry, 21, factor_df, model=jfm)
    pipe = FactorPipeline()
    train_mask = feats.index.get_level_values("date").isin(split.train)
    pipe.fit(feats[train_mask].dropna())
    feats_p = pipe.transform(feats)
    backend, batch = _prepare_gbs_backend(cfg)
    gbs_panel = _build_gbs_panel(cfg, ["gbs_marginal"], feats_p, split, backend, batch)
    X_all = pd.concat([feats_p, gbs_panel], axis=1)
    X_tr = X_all[train_mask].dropna()
    ridge = RidgeReadout(alpha=1.0).fit(X_tr, labels.reindex(X_tr.index))
    alpha_all = ridge.predict(X_all)
    alpha_panel = pd.DataFrame(alpha_all).T
    alpha_panel.columns = [str(c) for c in alpha_all.index]  # 引擎内部重排，保持 symbol 列
    alpha_panel = alpha_all.unstack("symbol")  # date × symbol
    print(f"Q1 alpha 链 done（{time.time()-t_start:.0f}s）", flush=True)

    # --- ρ̂ 词表（真机冻结 200k，与金融无关） ---
    t0 = time.time()
    rho_hat = second_order_stats(batch, "correlation")
    print(f"ρ̂ 词表 {time.time()-t0:.0f}s（{rho_hat.shape}）", flush=True)

    # --- 寻址（Q1 口径 QuantileEncoder，train 窗拟合冻结） ---
    from qgbs.gbs.encoder import EncoderParams, QuantileEncoder, ensure_default_mappings, resolve_mapping_id
    enc_cfg = dict(cfg.get("encoder") or {})
    ensure_default_mappings(26, 16, 511)
    mid = resolve_mapping_id(str(enc_cfg.get("mapping_id", "identity")), 26, 16, 511)
    enc = QuantileEncoder(
        EncoderParams(n_bins_a=int(enc_cfg.get("n_bins_a", 255)),
                      n_bins_b=int(enc_cfg.get("n_bins_b", 256)),
                      partition_boundary=int(enc_cfg.get("partition_boundary", 255))),
        mapping_id=mid,
    )
    enc.fit(feats_p[train_mask].dropna())
    addr = enc.freeze().transform(feats_p)  # 全期地址
    # 按日整理 symbol × 26 模式表
    mode_df = pd.DataFrame(addr.mode_index,
                           index=feats_p.index).rename_axis(["date", "symbol"])
    addr_by_date = {d: mode_df.loc[d] for d in mode_df.index.get_level_values("date").unique()}
    print(f"寻址 done（{len(addr_by_date)} 日）", flush=True)

    # --- 引擎公共件 ---
    cc = adapter.load_constraints_costs("default")
    constraints = PortfolioConstraints(
        weight_cap={s: 0.10 for s in symbols}, cash_range=(0.0, 0.05),
        exposure_bounds={}, max_turnover=None, liquidity_limit=None, blacklist=(),
    )
    cost_model = CostModel(commission_bps=cc.commission_bps, stamp_tax_bps=cc.stamp_tax_bps,
                           slippage_bps=cc.slippage_bps, impact_coef=cc.impact_coef)
    opt = OptimizerParams(risk_aversion=5.0, turnover_penalty=0.001, l2_regularization=0.0001)
    rf_daily = 0.02 / 252.0
    bench = msf.index_returns[mkt_code]
    va_dates = pd.DatetimeIndex(split.valid)
    te_dates = pd.DatetimeIndex(split.test)
    exposure = pd.get_dummies(industry).astype(float)

    def run_window(delta: float, provider, dates) -> dict:
        # alpha_panel 只保留目标窗日期行（引擎在 alpha_panel 非空时按其索引调仓）
        panel = alpha_panel.loc[alpha_panel.index.isin(dates)]
        eng = BacktestEngine(window_spec=spec, constraints=constraints, optimizer_params=opt,
                             cost_model=cost_model, risk_prior_delta=delta)
        bt = eng.run(market, status, risk_free_daily=rf_daily, alpha_panel=panel,
                     exposure_matrix=exposure,
                     risk_prior_provider=provider)
        r = bt.returns
        return {"sharpe": sharpe(r), "ir": information_ratio(r, bench),
                "calmar": calmar(bt.nav), "realized_vol": annual_volatility(r),
                "turnover": float(bt.turnover.mean()), "n_rebalances": len(bt.weights)}

    static_K = build_static_corr(addr_by_date, rho_hat, list(va_dates))  # 防泄漏：仅 valid 调仓日
    providers = {
        "R0_lw": None,
        "R1_gbs": PriorProvider("gbs_prior", addr_by_date=addr_by_date, rho_hat=rho_hat),
        "R2_random": PriorProvider("random_prior", seed=42),
        "R3_static": PriorProvider("static_gbs_prior", static_corr=static_K),
    }
    results: dict = {}
    for name, prov in providers.items():
        t0 = time.time()
        if prov is None:  # R0：固定 LW（δ=1）
            res = run_window(1.0, None, te_dates)
            results[name] = {"delta": 1.0, **res}
        else:
            # δ 在 valid 窗择优（净收益口径）
            scores = {}
            for delta in DELTA_GRID:
                r = run_window(delta, prov, va_dates)
                scores[delta] = r["sharpe"]
            best = max(scores, key=lambda d: scores[d])
            res = run_window(best, prov, te_dates)
            results[name] = {"delta": best, "valid_scores": scores, **res}
        print(f"{name}: δ={results[name]['delta']} test sharpe={res['sharpe']:.3f} "
              f"ir={res['ir']:+.3f} calmar={res['calmar']:+.3f} vol={res['realized_vol']:.3f} "
              f"（{time.time()-t0:.0f}s）", flush=True)
        # 逐配置即时落盘
        (OUT / f"{name}.json").write_text(json.dumps(results[name], ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    report = {"fold": "fold 0", "delta_grid": DELTA_GRID, "results": results,
              "total_s": time.time() - t_start,
              "note": "单折 sanity（不作结论）；静态先验 K 仅用 valid 调仓日平均（防泄漏）"}
    (OUT / "phase1_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n总耗时 {report['total_s']:.0f}s；产物已写入 {OUT}", flush=True)


if __name__ == "__main__":
    main()
