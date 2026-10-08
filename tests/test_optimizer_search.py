"""optimizer.search 验证期网格搜索：valid 扣成本净收益选优 + 测试期使用选中值。"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import yaml

from qgbs.backtest.engine import BacktestEngine
from qgbs.experiments.runner import run_experiment
from tests.test_runner import metrics_equal_nan_safe


def _tiny_cfg(optimizer: dict, mock_overrides: dict | None = None) -> dict:
    return {
        "experiment_id": "C2",
        "features": ["finance_base"],
        "readout": "ridge",
        "label_horizon": 21,
        "search": {"param_grid": {"alpha": [0.1, 1.0]}},
        "optimizer": optimizer,
        "data": {
            "pool": "pool100",
            "mock": {
                "n_assets": 20, "start": "2020-01-01", "end": "2021-06-30",
                "n_industries": 4, "n_sub_industries": 8, "seed": 9,
                **(mock_overrides or {}),
            },
        },
        # train 需覆盖最长特征 warmup（mom_120）+ horizon 的 valid 边界 purge 余量
        "rolling": {"train": "10m", "valid": "2m", "test": "1m", "purge": "5d", "embargo": "2d"},
    }


def _write(path: Path, cfg: dict) -> Path:
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_optimizer_search_selects_on_valid_and_uses_on_test(tmp_path, monkeypatch) -> None:
    """2 点网格：极端交易成本下 τ=1e6（不换手）在 valid 必然优于 τ=0。"""
    # 记录每次 BacktestEngine.run 实际使用的优化器参数（valid 评分 + 测试回测）
    used_params: list[dict] = []
    orig_run = BacktestEngine.run

    def spy_run(self, *args, **kwargs):
        used_params.append(dataclasses.asdict(self.optimizer_params))
        return orig_run(self, *args, **kwargs)

    monkeypatch.setattr(BacktestEngine, "run", spy_run)

    cfg = _tiny_cfg(
        {"search": {"turnover_penalty": [0.0, 1e6]}},
        # 佣金 50%/换手：任何调仓都巨额亏损 → valid 上必然选中不换手的一端
        mock_overrides={"commission_bps": 5000.0},
    )
    r = run_experiment(_write(tmp_path / "opt_search.yaml", cfg))
    sel = r.extra["optimizer_selection"]
    assert sel["search_active"] is True
    folds = sel["folds"]
    assert len(folds) >= 3
    n_grid = 2
    # 每折恰好 len(grid) 次 valid 评分 + 1 次测试回测
    assert len(used_params) == len(folds) * (n_grid + 1)
    for k, fold in enumerate(folds):
        cands = fold["candidates"]
        assert [c["params"]["turnover_penalty"] for c in cands] == [0.0, 1e6]
        scores = [c["valid_score"] for c in cands]
        assert all(np.isfinite(s) for s in scores)
        assert scores[0] != scores[1]  # 网格确实产生区分度
        best = max(cands, key=lambda c: c["valid_score"])
        # 选中点 = valid 扣成本净收益最优者（此处必然是不换手的一端）
        assert fold["params"]["turnover_penalty"] == best["params"]["turnover_penalty"]
        assert fold["params"]["turnover_penalty"] == 1e6
        # valid 评分用候选参数、测试回测用选中参数
        base = k * (n_grid + 1)
        assert used_params[base]["turnover_penalty"] == 0.0
        assert used_params[base + 1]["turnover_penalty"] == 1e6
        assert used_params[base + 2] == fold["params"]
    assert np.isfinite(r.metrics["sharpe"])


def test_single_point_search_matches_fixed(tmp_path) -> None:
    """单点 search = 现有固定值：行为与不配置 search 完全一致。"""
    base_opt = {"risk_aversion": 5.0, "turnover_penalty": 0.001, "l2_regularization": 0.0001}
    r_fixed = run_experiment(_write(tmp_path / "fixed.yaml", _tiny_cfg(base_opt)))
    r_single = run_experiment(
        _write(tmp_path / "single.yaml", _tiny_cfg({**base_opt, "search": {"risk_aversion": [5.0]}}))
    )
    assert metrics_equal_nan_safe(r_fixed.metrics, r_single.metrics)
    sel = r_single.extra["optimizer_selection"]
    assert sel["search_active"] is False
    assert sel["fixed_params"]["risk_aversion"] == 5.0


def test_optimizer_search_unknown_param_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="未知参数"):
        run_experiment(_write(tmp_path / "bogus.yaml", _tiny_cfg({"search": {"bogus": [1.0, 2.0]}})))
