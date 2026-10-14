"""E6 阶段 2：臂 B 贪心筛选规则、断点续跑、CLI 解析测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.e6_phase2_run import _done_folds, _parse_args, _parse_folds, greedy_select

pytestmark = pytest.mark.usefixtures()


def _mk_corr_data(seed: int = 0, n: int = 400):
    """构造：候选 c0 与经典 f0 高度相关（应被跳过）、c1 独立强信号、其余噪声。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n // 10)
    idx = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(10)]], names=["date", "symbol"])
    f0 = pd.Series(rng.normal(size=len(idx)), index=idx)
    c0 = f0 + rng.normal(0, 0.01, len(idx))  # 与 f0 几乎相同
    y = pd.Series(rng.normal(size=len(idx)), index=idx)
    c1 = y + rng.normal(0, 0.3, len(idx))  # 强信号
    cand = pd.DataFrame({"c0": c0, "c1": c1, "c2": rng.normal(size=len(idx)), "c3": rng.normal(size=len(idx))}, index=idx)
    classic = pd.DataFrame({"f0": f0}, index=idx)
    return cand, classic, y, idx


def test_greedy_select_corr_skip_and_signal() -> None:
    cand, classic, y, idx = _mk_corr_data()
    corr_ref = pd.concat([classic, cand], axis=1)
    sel = greedy_select(cand, classic, y, corr_ref, k_max=3)
    assert "c1" in sel  # 强信号入选
    assert "c0" not in sel  # 与已选 f0 |corr|≈1 → 跳过
    assert len(sel) <= 3


def test_greedy_select_kmax_cap() -> None:
    cand, classic, y, idx = _mk_corr_data()
    corr_ref = pd.concat([classic, cand], axis=1)
    sel = greedy_select(cand, classic, y, corr_ref, k_max=1)
    assert len(sel) == 1


def test_parse_folds() -> None:
    assert _parse_folds("all") is None
    assert _parse_folds("0,2,5") == {0, 2, 5}
    assert _parse_folds("1-3") == {1, 2, 3}
    assert _parse_folds("0,2-3") == {0, 2, 3}


def test_cli_defaults() -> None:
    args = _parse_args([])
    assert args.shots == 1_000_000 and args.processes == 8 and args.seed == 42
    assert args.save_features == "none"
    assert args.outdir == "artifacts/e6_paper_replica/phase2"


def test_done_folds_resume(tmp_path) -> None:
    j = tmp_path / "fold_results.jsonl"
    j.write_text('{"split_id": 0, "x": 1}\n{"split_id": 2, "x": 1}\n', encoding="utf-8")
    assert _done_folds(j) == {0, 2}
    assert _done_folds(tmp_path / "missing.jsonl") == set()
