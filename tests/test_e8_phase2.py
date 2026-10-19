"""E8.2 主脚本测试：CLI 解析、断点续跑、汇总逻辑。"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from scripts.e8_phase2_run import _done_units, _parse_args, _parse_folds, _write_summary


def test_parse_args_defaults() -> None:
    args = _parse_args([])
    assert args.processes == 16 and args.seed == 42
    assert args.outdir == "artifacts/e8_risk_prior/phase2"


def test_parse_folds() -> None:
    assert _parse_folds("all") is None
    assert _parse_folds("0,2,5") == {0, 2, 5}
    assert _parse_folds("1-3") == {1, 2, 3}


def test_done_units_resume(tmp_path) -> None:
    j = tmp_path / "fold_results.jsonl"
    j.write_text('{"split_id": 0, "config": "R0_lw"}\n{"split_id": 0, "config": "R1_gbs"}\n',
                 encoding="utf-8")
    assert _done_units(j) == {(0, "R0_lw"), (0, "R1_gbs")}
    assert _done_units(tmp_path / "missing.jsonl") == set()


def test_write_summary(tmp_path) -> None:
    """汇总：四配置指标均值 + 配对 CI 结构（合成两折数据）。"""
    rng = np.random.default_rng(0)
    recs = []
    for fold in range(10):
        for cfg_name, shift in (("R0_lw", 0.0), ("R1_gbs", 0.002), ("R2_random", 0.0005)):
            rets = rng.normal(shift, 0.01, 20)
            dates = pd.bdate_range("2024-01-01", periods=20) + pd.Timedelta(days=fold * 30)
            recs.append({"split_id": fold, "config": cfg_name, "delta": 0.5,
                         "valid_scores": {}, "rank_ic_test": 0.01,
                         "sharpe": 1.0 + shift * 100, "ir": 0.5, "calmar": 1.5,
                         "realized_vol": 0.2, "turnover": 0.5, "n_rebalances": 1,
                         "returns": {str(d.date()): float(v) for d, v in zip(dates, rets)},
                         "nav": {str(d.date()): float(v) for d, v in zip(dates, (1 + rets).cumprod())},
                         "elapsed_s": 1.0})
    j = tmp_path / "fold_results.jsonl"
    j.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    from scripts import e8_phase2_run
    e8_phase2_run._SHARED["bench"] = None  # 无基准 → IR 跳过
    _write_summary(tmp_path, j)
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["n_units"] == 30
    assert s["configs"]["R1_gbs"]["sharpe"] > s["configs"]["R0_lw"]["sharpe"]
    assert "R1−R0_lw" in s["paired"]
    assert s["paired"]["R1−R0_lw"]["sharpe"]["delta"] > 0
