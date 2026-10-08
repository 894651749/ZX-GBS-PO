"""统计严谨性测试（B3）：DSR/BH 手算核对、分块 bootstrap CI、Reality Check 方向。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from qgbs.backtest.metrics import sharpe
from qgbs.experiments.report import (
    ablation_table,
    block_bootstrap_ci,
    deflated_sharpe,
    multiple_testing_correction,
    white_reality_check,
)

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# deflated_sharpe
# ---------------------------------------------------------------------------


def test_deflated_sharpe_handcheck() -> None:
    """小样例按 Bailey & López de Prado (2014) 公式独立复算核对。"""
    trials = pd.Series([0.5, 1.0, 1.5, 2.0])
    sr_obs, n_obs = 1.5, 100
    n = len(trials)
    v = float(trials.var(ddof=1))
    sr0 = np.sqrt(v) * (
        (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n) + EULER_GAMMA * norm.ppf(1 - 1 / (n * np.e))
    )
    var_term = 1.0 + (3.0 - 1.0) / 4.0 * sr_obs**2  # 1−γ₃·SR̂+(γ₄−1)/4·SR̂²，γ₃=0、γ₄=3
    expected = float(norm.cdf((sr_obs - sr0) * np.sqrt(n_obs - 1) / np.sqrt(var_term)))
    assert deflated_sharpe(sr_obs, trials, n_obs) == pytest.approx(expected, rel=1e-12)


def test_deflated_sharpe_penalizes_more_trials() -> None:
    """试算次数越多 → 期望最大 Sharpe 基准越高 → DSR 越低（选择偏差惩罚）。"""
    rng = np.random.default_rng(0)
    few = pd.Series([0.5, 1.0, 1.5])
    many = pd.Series(rng.normal(1.0, 0.3, 50))
    sr_obs, n_obs = 1.0, 50
    dsr_few = deflated_sharpe(sr_obs, few, n_obs)
    dsr_many = deflated_sharpe(sr_obs, many, n_obs)
    assert 0.0 < dsr_many < dsr_few < 1.0


def test_deflated_sharpe_skew_kurt_effect() -> None:
    """厚尾（高峰度）压低 DSR：其他条件相同时 kurt 越大 p 值越小。"""
    trials = pd.Series([0.2, 0.5, 0.8])
    dsr_normal = deflated_sharpe(1.0, trials, 252, skew=0.0, kurt=3.0)
    dsr_fat = deflated_sharpe(1.0, trials, 252, skew=0.0, kurt=9.0)
    assert dsr_fat < dsr_normal


def test_deflated_sharpe_edges() -> None:
    # 单次试算：SR0 退化为该次试算值
    val = deflated_sharpe(2.0, pd.Series([1.0]), 100)
    var_term = 1.0 + (3.0 - 1.0) / 4.0 * 2.0**2
    expected = float(norm.cdf((2.0 - 1.0) * np.sqrt(99) / np.sqrt(var_term)))
    assert val == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError, match="sharpe_trials"):
        deflated_sharpe(1.0, pd.Series(dtype=float), 100)


# ---------------------------------------------------------------------------
# multiple_testing_correction
# ---------------------------------------------------------------------------


def test_benjamini_hochberg_handcheck() -> None:
    p = pd.Series([0.01, 0.02, 0.03])
    adj = multiple_testing_correction(p)
    # m=3：adj = min_{j≥i} (3/j)·p_j → [0.03, 0.03, 0.03]
    assert adj.tolist() == pytest.approx([0.03, 0.03, 0.03], rel=1e-12)
    p2 = pd.Series([0.001, 0.5, 0.9])
    adj2 = multiple_testing_correction(p2)
    # [3×0.001, min(1.5×0.5, 0.9), 0.9] = [0.003, 0.75, 0.9]
    assert adj2.tolist() == pytest.approx([0.003, 0.75, 0.9], rel=1e-12)


def test_bh_preserves_input_order() -> None:
    p = pd.Series([0.9, 0.001, 0.5], index=["c", "a", "b"])
    adj = multiple_testing_correction(p)
    assert list(adj.index) == ["c", "a", "b"]
    assert adj.loc["a"] == pytest.approx(0.003, rel=1e-12)


def test_bonferroni_handcheck() -> None:
    p = pd.Series([0.01, 0.5])
    adj = multiple_testing_correction(p, method="bonferroni")
    assert adj.tolist() == pytest.approx([0.02, 1.0], rel=1e-12)
    with pytest.raises(ValueError, match="未知校正方法"):
        multiple_testing_correction(p, method="holm")


# ---------------------------------------------------------------------------
# block_bootstrap_ci
# ---------------------------------------------------------------------------


def test_block_bootstrap_ci_positive_sharpe() -> None:
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.002, 0.01, 300))
    lo, hi = block_bootstrap_ci(r, statistic="sharpe", n_bootstrap=500, seed=7)
    assert lo > 0.0  # 日收益均值显著为正 → Sharpe CI 整体为正
    assert lo <= sharpe(r) <= hi


def test_block_bootstrap_ci_noise_contains_zero() -> None:
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0, 0.01, 300))
    lo, hi = block_bootstrap_ci(r, statistic="sharpe", n_bootstrap=500, seed=7)
    assert lo < 0.0 < hi


def test_block_bootstrap_ci_mean_and_block_clip() -> None:
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.001, 0.01, 50))
    lo, hi = block_bootstrap_ci(r, statistic="mean", n_bootstrap=300, block_length=21, seed=7)
    assert lo <= r.mean() <= hi  # block_length < n 正常
    lo2, hi2 = block_bootstrap_ci(r, statistic="mean", n_bootstrap=100, block_length=999, seed=7)
    assert np.isfinite(lo2) and np.isfinite(hi2)  # 超长块自动截断不报错
    with pytest.raises(ValueError, match="太短"):
        block_bootstrap_ci(pd.Series([0.01, 0.02]), n_bootstrap=10)
    with pytest.raises(ValueError, match="未知统计量"):
        block_bootstrap_ci(r, statistic="cvar")


# ---------------------------------------------------------------------------
# white_reality_check
# ---------------------------------------------------------------------------


def test_white_reality_check_direction() -> None:
    """构造已知优势策略（恒定正超额）→ p 值应很小；纯噪声策略 → p 值大。"""
    n = 250
    idx = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(10)
    bench = pd.Series(rng.normal(0.0, 0.01, n), index=idx)
    good = bench + 0.002  # 恒定日均超额 20bp
    noise = bench + rng.normal(0.0, 0.01, n)
    table = white_reality_check(
        bench, {"good": good, "noise": noise}, n_bootstrap=500, block_length=10, seed=5
    )
    assert list(table.index) == ["good", "noise"]
    row_good = table.loc["good"]
    assert bool(row_good["is_best"])
    assert row_good["mean_excess"] == pytest.approx(0.002, rel=1e-9)
    assert row_good["p_individual"] < 0.05
    assert row_good["p_reality_check"] < 0.05
    # 纯噪声策略的单策略 p 值应明显更大
    assert table.loc["noise", "p_individual"] > 0.2


def test_white_reality_check_noise_not_significant() -> None:
    n = 200
    rng = np.random.default_rng(11)
    bench = pd.Series(rng.normal(0.0, 0.01, n))
    strats = {f"s{i}": bench + rng.normal(0.0, 0.01, n) for i in range(5)}
    table = white_reality_check(bench, strats, n_bootstrap=500, block_length=10, seed=6)
    assert table["p_reality_check"].iloc[0] > 0.05  # 无真实优势 → 整体不显著
    assert (table["p_reality_check"] == table["p_reality_check"].iloc[0]).all()


# ---------------------------------------------------------------------------
# ablation_table
# ---------------------------------------------------------------------------


def test_ablation_table_delta_and_ci() -> None:
    n = 300
    idx = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(20)
    base = pd.Series(rng.normal(0.0005, 0.01, n), index=idx)
    better = base + 0.001  # 恒定 10bp 日超额 → 增量显著为正
    table = ablation_table(
        {"C2": base, "QX": better}, baseline_id="C2",
        metrics=("sharpe",), n_bootstrap=400, block_length=21, seed=3,
    )
    assert list(table.index) == ["C2", "QX"]
    row_base, row_qx = table.loc["C2"], table.loc["QX"]
    assert row_base["sharpe_delta"] == 0.0
    assert row_base["sharpe_ci_lo"] == 0.0 and row_base["sharpe_ci_hi"] == 0.0
    expected_delta = sharpe(better) - sharpe(base)
    assert row_qx["sharpe_delta"] == pytest.approx(expected_delta, rel=1e-12)
    assert row_qx["sharpe_ci_lo"] > 0.0  # 恒定正漂移 → 增量 CI 下界为正
    assert row_qx["sharpe_ci_lo"] <= row_qx["sharpe_delta"] <= row_qx["sharpe_ci_hi"]


def test_ablation_table_accepts_experiment_result_like() -> None:
    """鸭子类型：extra 含 'returns' 的 ExperimentResult 可直接传入。"""
    rng = np.random.default_rng(21)
    base = pd.Series(rng.normal(0.0, 0.01, 120))
    exp = pd.Series(rng.normal(0.0008, 0.01, 120))
    stub = SimpleNamespace(extra={"returns": exp})
    table = ablation_table({"C2": base, "Q2": stub}, baseline_id="C2",
                           metrics=("sharpe",), n_bootstrap=100, seed=3)
    assert table.loc["Q2", "sharpe"] == pytest.approx(sharpe(exp), rel=1e-12)
    with pytest.raises(KeyError, match="基线"):
        ablation_table({"C2": base}, baseline_id="C9")
    with pytest.raises(TypeError, match="ablation_table"):
        ablation_table({"C2": base, "Q2": 1.23}, baseline_id="C2")
