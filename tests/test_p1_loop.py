"""P1 端到端：C1-C5 经典对照实验在统一滚动协议下批量跑通并输出对比表。

可直接运行查看对比表：``PYTHONPATH=src .venv/bin/python -m tests.test_p1_loop``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qgbs.experiments.runner import CONFIG_DIR, EXPERIMENT_CONFIG_FILES, run_experiment

CLASSICAL_IDS = ["C1", "C2", "C3", "C4", "C5"]
DISPLAY_COLS = ["rank_ic_mean", "rank_icir", "sharpe", "annual_return", "max_drawdown", "turnover_mean"]


def run_classical_matrix() -> pd.DataFrame:
    rows = {}
    for exp_id in CLASSICAL_IDS:
        result = run_experiment(CONFIG_DIR / "experiments" / EXPERIMENT_CONFIG_FILES[exp_id])
        rows[exp_id] = result.metrics
    table = pd.DataFrame(rows).T
    table.index.name = "experiment"
    return table


@pytest.fixture(scope="module")
def comparison() -> pd.DataFrame:
    return run_classical_matrix()


def test_all_experiments_produce_finite_metrics(comparison: pd.DataFrame) -> None:
    assert list(comparison.index) == CLASSICAL_IDS
    for col in ("sharpe", "annual_return", "annual_volatility", "max_drawdown", "turnover_mean"):
        assert np.isfinite(comparison[col]).all(), f"{col} 存在 NaN/inf"
    # C2-C5 有监督读出 → 预测层指标必须有值
    assert np.isfinite(comparison.loc[["C2", "C3", "C4", "C5"], "rank_ic_mean"]).all()
    assert (comparison["n_rebalances"] >= 3).all()
    # 同一滚动协议：各实验调仓期数一致
    assert comparison["n_rebalances"].nunique() == 1


def test_c2_reproducible() -> None:
    from tests.test_runner import metrics_equal_nan_safe

    path = CONFIG_DIR / "experiments" / EXPERIMENT_CONFIG_FILES["C2"]
    r1 = run_experiment(path)
    r2 = run_experiment(path)
    assert metrics_equal_nan_safe(r1.metrics, r2.metrics)
    assert r1.data_version == r2.data_version


def test_c3_gbm_reproducible() -> None:
    """GBM 实验同 seed 复现（锁定 fixed_params 合入搜索阶段的修复）。"""
    from tests.test_runner import metrics_equal_nan_safe

    path = CONFIG_DIR / "experiments" / EXPERIMENT_CONFIG_FILES["C3"]
    r1 = run_experiment(path)
    r2 = run_experiment(path)
    assert metrics_equal_nan_safe(r1.metrics, r2.metrics)


if __name__ == "__main__":
    table = run_classical_matrix()
    pd.set_option("display.width", 160)
    print("=== P1 经典对照矩阵（mock 数据，统一滚动协议 train=1y/valid=3m/test=1m）===")
    print(table[DISPLAY_COLS].to_string(float_format=lambda v: f"{v: .4f}"))
    print()
    print("组合层补充：")
    print(table[["sortino", "annual_volatility", "total_cost", "n_rebalances"]].to_string(
        float_format=lambda v: f"{v: .4f}"
    ))
