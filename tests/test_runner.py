"""runner 端到端测试：小参数快速跑通、可复现、Q 实验正确拒绝。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from qgbs.experiments.runner import CONFIG_DIR, ExperimentResult, run_experiment

TINY_CFG = {
    "experiment_id": "C2",
    "features": ["finance_base"],
    "readout": "ridge",
    "cov_method": "ledoit_wolf",
    "label_horizon": 21,
    "search": {"param_grid": {"alpha": [0.1, 1.0]}},
    "data": {
        "pool": "pool100",
        "mock": {"n_assets": 20, "start": "2020-01-01", "end": "2021-06-30",
                 "n_industries": 4, "n_sub_industries": 8, "seed": 9},
    },
    # train 需覆盖最长特征 warmup（mom_120）+ horizon 的 valid 边界 purge 余量
    "rolling": {"train": "10m", "valid": "2m", "test": "1m", "purge": "5d", "embargo": "2d"},
}


@pytest.fixture(scope="module")
def tiny_config_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("cfg") / "c2_tiny.yaml"
    path.write_text(yaml.safe_dump(TINY_CFG), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def tiny_result(tiny_config_path: Path) -> ExperimentResult:
    return run_experiment(tiny_config_path)


def test_runner_end_to_end(tiny_result: ExperimentResult) -> None:
    r = tiny_result
    assert r.experiment_id == "C2"
    assert isinstance(r.git_hash, str) and len(r.git_hash) > 0
    assert r.seed == 9
    assert r.data_version.startswith("v0-placeholder-mock-seed9")
    for name in ("sharpe", "annual_return", "max_drawdown", "turnover_mean"):
        assert np.isfinite(r.metrics[name]), f"{name} 不可计算"
    assert r.metrics["n_rebalances"] >= 3
    assert r.metrics["n_folds"] >= 3
    # 预测层指标应有值（mock 上数值不重要，但必须非 NaN）
    assert np.isfinite(r.metrics["rank_ic_mean"])
    assert np.isfinite(r.metrics["direction_accuracy"])


def metrics_equal_nan_safe(m1: dict, m2: dict) -> bool:
    """NaN 安全的 metrics 字典比较（nan != nan 的陷阱规避）。"""
    if m1.keys() != m2.keys():
        return False
    for k in m1:
        a, b = m1[k], m2[k]
        if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
            continue
        if a != b:
            return False
    return True


def test_runner_reproducible(tiny_config_path: Path) -> None:
    r1 = run_experiment(tiny_config_path)
    r2 = run_experiment(tiny_config_path)
    assert metrics_equal_nan_safe(r1.metrics, r2.metrics)


def test_q1_tiny_end_to_end(tmp_path) -> None:
    """Q1 小参数端到端（仿真 GBS 样本入库 → cached 后端 → 特征 → 读出）。"""
    cfg = dict(TINY_CFG)
    cfg["experiment_id"] = "Q1"
    cfg["features"] = ["finance_base", "gbs_marginal"]
    cfg["gbs"] = {
        "backend": "cached",
        "sample_store": str(tmp_path / "store"),
        "n_temporal": 8,
        "n_spatial": 4,
        "n_shots": 2000,
        "system_efficiency": 0.7,
        "seed": 1,
    }
    cfg["encoder"] = {"n_bins_a": 4, "n_bins_b": 4, "partition_boundary": 4, "mapping_id": "identity"}
    path = tmp_path / "q1_tiny.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    r1 = run_experiment(path)
    r2 = run_experiment(path)  # 第二次应命中样本库
    assert np.isfinite(r1.metrics["sharpe"])
    assert np.isfinite(r1.metrics["rank_ic_mean"])
    assert metrics_equal_nan_safe(r1.metrics, r2.metrics)
    # 样本已落盘
    assert (tmp_path / "store" / "simulation").exists()


def test_c1_no_readout_path() -> None:
    r = run_experiment(CONFIG_DIR / "experiments" / "c1_classical_baseline.yaml")
    assert np.isfinite(r.metrics["sharpe"])
    assert np.isnan(r.metrics["rank_ic_mean"])  # 无监督模型 → 无 IC
    assert r.metrics["n_rebalances"] >= 3


def test_adapter_dispatch_errors(tmp_path) -> None:
    """data.adapter 分发：未知适配器 / csv 缺 store_dir / 目录不存在 → 清晰报错。"""
    cfg = dict(TINY_CFG)
    cfg["data"] = {"adapter": "oracle"}
    path = tmp_path / "bad_adapter.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="未知数据适配器"):
        run_experiment(path)

    cfg["data"] = {"adapter": "csv", "store_dir": None, "start": "2020-01-01", "end": "2021-06-30"}
    path = tmp_path / "csv_no_store.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="data.store_dir"):
        run_experiment(path)

    cfg["data"] = {
        "adapter": "csv", "store_dir": str(tmp_path / "nope"),
        "start": "2020-01-01", "end": "2021-06-30",
    }
    path = tmp_path / "csv_missing_dir.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="csv 数据目录不存在"):
        run_experiment(path)


def test_finance_set_extended_runs(tmp_path) -> None:
    """features.finance_set=extended 接线（默认 base 行为不变）。"""
    cfg = dict(TINY_CFG)
    cfg["features"] = {"families": ["finance_base"], "finance_set": "extended"}
    path = tmp_path / "c2_extended.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    r = run_experiment(path)
    assert np.isfinite(r.metrics["sharpe"])
    assert np.isfinite(r.metrics["rank_ic_mean"])
