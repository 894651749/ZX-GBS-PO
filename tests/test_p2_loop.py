"""P2 端到端：C2/C5 与 Q1-Q4（仿真 GBS 样本）在统一滚动协议下对比。

可直接运行查看对比表：``PYTHONPATH=src .venv/bin/python -m tests.test_p2_loop``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qgbs.experiments.runner import CONFIG_DIR, EXPERIMENT_CONFIG_FILES, run_experiment

P2_IDS = ["C2", "C5", "Q1", "Q2", "Q3", "Q4"]
DISPLAY_COLS = ["rank_ic_mean", "rank_icir", "rho_mean", "sharpe", "annual_return", "max_drawdown", "turnover_mean"]


def run_p2_matrix() -> pd.DataFrame:
    rows = {}
    for exp_id in P2_IDS:
        result = run_experiment(CONFIG_DIR / "experiments" / EXPERIMENT_CONFIG_FILES[exp_id])
        rows[exp_id] = result.metrics
    table = pd.DataFrame(rows).T
    table.index.name = "experiment"
    return table


@pytest.fixture(scope="module")
def comparison() -> pd.DataFrame:
    return run_p2_matrix()


def test_all_p2_experiments_finite(comparison: pd.DataFrame) -> None:
    assert list(comparison.index) == P2_IDS
    for col in ("sharpe", "annual_return", "max_drawdown", "turnover_mean"):
        assert np.isfinite(comparison[col]).all(), f"{col} 存在 NaN/inf"
    # 全部有监督读出 → 预测层指标必须有值
    assert np.isfinite(comparison["rank_ic_mean"]).all()
    # 统一滚动协议：调仓期数一致
    assert comparison["n_rebalances"].nunique() == 1
    assert (comparison["n_folds"] >= 3).all()


def test_q2_reproducible() -> None:
    from tests.test_runner import metrics_equal_nan_safe

    path = CONFIG_DIR / "experiments" / EXPERIMENT_CONFIG_FILES["Q2"]
    r1 = run_experiment(path)
    r2 = run_experiment(path)  # 第二次命中样本库缓存
    assert metrics_equal_nan_safe(r1.metrics, r2.metrics)


if __name__ == "__main__":
    table = run_p2_matrix()
    pd.set_option("display.width", 160)
    print("=== P2 对照矩阵（mock 金融数据 + 仿真 GBS 样本，统一滚动协议）===")
    print("注：mock + 近似仿真器下 GBS 特征无物理增量理由，IC 数值不重要，")
    print("    链路正确、口径一致、可复现才是验收点。")
    print(table[DISPLAY_COLS].to_string(float_format=lambda v: f"{v: .4f}"))
    print()
    print("组合层补充：")
    print(table[["sortino", "annual_volatility", "total_cost", "n_rebalances"]].to_string(
        float_format=lambda v: f"{v: .4f}"
    ))
