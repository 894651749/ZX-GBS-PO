"""修正 1/2 测试：ρ 混合链路、C5 等维 auto 解析、不确定性接线。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from qgbs.experiments.runner import (
    EXPERIMENT_CONFIG_FILES,
    CONFIG_DIR,
    _resolve_target_dim,
    run_experiment,
)
from qgbs.models.blending import estimate_rho, shrinkage_blend, uncertainty_residual
from qgbs.experiments.runner import _gbs_feature_dim
import qgbs.experiments.runner as runner_mod


def _panel_index(n_dates: int = 20, n_sym: int = 8) -> pd.MultiIndex:
    dates = pd.bdate_range("2021-01-01", periods=n_dates)
    return pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(n_sym)]], names=["date", "symbol"])


def test_estimate_rho_prefers_better_model() -> None:
    idx = _panel_index()
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
    pred_good = y + rng.normal(0, 0.01, len(idx))  # 近完美
    pred_bad = pd.Series(rng.normal(0, 1, len(idx)), index=idx)  # 纯噪声
    assert estimate_rho(pred_good, pred_bad, y) == pytest.approx(1.0)
    assert estimate_rho(pred_bad, pred_good, y) == pytest.approx(0.0)


def test_shrinkage_blend_math() -> None:
    idx = _panel_index(2, 3)
    pg = pd.Series([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], index=idx)
    pc = pd.Series([0.0, 0.0, 0.0, 2.0, 2.0, 2.0], index=idx)
    out = shrinkage_blend(pg, pc, rho=0.5)
    assert np.allclose(out.prediction.to_numpy(), [0.5, 1.0, 1.5, 1.5, 2.0, 2.5])
    assert out.rho == 0.5
    # 不确定性高的资产向经典端额外收缩
    unc = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=idx)
    out2 = shrinkage_blend(pg, pc, rho=1.0, uncertainty=unc)
    assert out2.prediction.iloc[0] == pytest.approx(0.0)  # u 最大 → 完全收缩到经典端
    assert out2.prediction.iloc[1] == pytest.approx(2.0)


def test_uncertainty_residual() -> None:
    resid = pd.DataFrame(
        {"a": [0.01, -0.01, 0.01], "b": [0.1, -0.1, 0.1]},
        index=pd.bdate_range("2021-01-01", periods=3),
    )
    unc = uncertainty_residual(resid)
    assert unc["b"] > unc["a"]


def test_auto_target_dim_resolution() -> None:
    # Q2: gbs_marginal(11) + gbs_second_order(C(11,2)=55) = 66
    assert _resolve_target_dim("auto:Q2", 11) == 66
    assert _resolve_target_dim(64, 11) == 64
    with pytest.raises(ValueError, match="无法解析"):
        _resolve_target_dim("bogus", 11)
    with pytest.raises(ValueError, match="无 GBS 特征族"):
        _resolve_target_dim("auto:C0", 11)  # C0 无 GBS 特征族
    ref_cfg = yaml.safe_load(
        (CONFIG_DIR / "experiments" / EXPERIMENT_CONFIG_FILES["Q3"]).read_text(encoding="utf-8")
    )
    assert _gbs_feature_dim(ref_cfg, 11) == 11 + 55 + 50  # 含 basis.N=50


def _tiny_q_config(tmp_path, blending: dict) -> "object":
    cfg = {
        "experiment_id": "Q1",
        "features": ["finance_base", "gbs_marginal"],
        "readout": "ridge",
        "label_horizon": 21,
        "search": {"param_grid": {"alpha": [0.1, 1.0]}},
        "gbs": {
            "backend": "cached",
            "sample_store": str(tmp_path / "store"),
            "n_temporal": 8,
            "n_spatial": 4,
            "n_shots": 2000,
            "system_efficiency": 0.7,
            "seed": 1,
        },
        "encoder": {"n_bins_a": 4, "n_bins_b": 4, "partition_boundary": 4, "mapping_id": "identity"},
        "blending": blending,
        "data": {
            "pool": "pool100",
            "mock": {"n_assets": 20, "start": "2020-01-01", "end": "2021-06-30",
                     "n_industries": 4, "n_sub_industries": 8, "seed": 9},
        },
        # train 需覆盖最长特征 warmup（mom_120）+ horizon 的 valid 边界 purge 余量
        "rolling": {"train": "10m", "valid": "2m", "test": "1m", "purge": "5d", "embargo": "2d"},
    }
    path = tmp_path / "q1_blend.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_rho_recorded_when_blending_enabled(tmp_path) -> None:
    r = run_experiment(_tiny_q_config(tmp_path, {"enabled": True, "base": "c2_ridge"}))
    assert 0.0 <= r.metrics["rho_mean"] <= 1.0
    assert len(r.extra["rho_per_fold"]) == int(r.metrics["n_folds"])
    assert np.isfinite(r.metrics["sharpe"])


def test_no_rho_when_blending_disabled(tmp_path) -> None:
    r = run_experiment(_tiny_q_config(tmp_path, {"enabled": False}))
    assert np.isnan(r.metrics["rho_mean"])
    assert r.extra["rho_per_fold"] == []


def test_uncertainty_link_runs(tmp_path) -> None:
    """不确定性链路接通（开启时结果有限且不报错）。"""
    r = run_experiment(
        _tiny_q_config(tmp_path, {"enabled": True, "base": "c2_ridge", "uncertainty_enabled": True})
    )
    assert np.isfinite(r.metrics["sharpe"])
