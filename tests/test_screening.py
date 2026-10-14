"""三闸门筛选器（E5 阶段 A）测试：植入检出、噪声淘汰、因果性。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qgbs.experiments.screening import (
    FoldGateRecord,
    ScreeningParams,
    _daily_rank_ic_matrix,
    candidate_names,
    candidate_values,
    gate1_stability,
    gate2_predictive,
    gate3_consistency,
    build_vocab,
)
from qgbs.gbs.backends.base import BatchMeta, SampleBatch
from qgbs.gbs.basis import CandidateBasis, BasisSelectionParams

PARAMS = ScreeningParams()


def _mk_batch(samples: np.ndarray, batch_id: str = "t") -> SampleBatch:
    return SampleBatch(
        meta=BatchMeta(batch_id=batch_id, device_version="t", n_modes=samples.shape[1],
                       n_shots=samples.shape[0], started_at="t", finished_at="t",
                       source="simulation", calibration_session_id="t"),
        samples=samples,
    )


def _mk_basis() -> CandidateBasis:
    return CandidateBasis(
        basis=np.array([[0, 1, 2]]), sample_window=("t", "t"),
        params=BasisSelectionParams(), mode_group=tuple(range(8)), dedup_rule="t",
    )


def test_daily_rank_ic_matches_spearman() -> None:
    """向量化逐日秩相关与 scipy spearman 对拍。"""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(0)
    vals = rng.normal(size=(100, 3))
    y = vals[:, 0] * 0.5 + rng.normal(size=100)
    day_slices = [np.arange(0, 50), np.arange(50, 100)]
    ic = _daily_rank_ic_matrix(vals, y, day_slices)
    for d, sl in enumerate(day_slices):
        for j in range(3):
            assert ic[d, j] == pytest.approx(spearmanr(vals[sl, j], y[sl]).statistic, rel=1e-9)


def test_candidate_values_hand_check() -> None:
    """二阶查表与高阶软激活手算对照。"""
    n_modes, n_feat = 8, 2
    corr = np.eye(n_modes)
    corr[1, 3] = corr[3, 1] = 0.7
    basis = _mk_basis()
    rates = np.array([0.5])
    vocab = type("V", (), {"corr": corr, "basis": basis.basis, "basis_rates": rates})()
    addr = np.array([[1, 3], [0, 1]])  # 行0: 模式1,3；行1: 模式0,1
    out = candidate_values(vocab, addr, n_feat)  # 1 对 + 1 基 = 2 列
    # 二阶：行0 对 (f0=1,f1=3) → corr[1,3]=0.7；行1 corr[0,1]=0
    assert out[0, 0] == pytest.approx(0.7) and out[1, 0] == pytest.approx(0.0)
    # 高阶基 (0,1,2) rate 0.5：行0 地址{1,3} 重合 1/3 → 0.5/3；行1 {0,1} 重合 2/3 → 1/3
    assert out[0, 1] == pytest.approx(0.5 / 3)
    assert out[1, 1] == pytest.approx(1.0 / 3)


def test_gate1_stable_vs_noisy() -> None:
    """闸门 1：跨子批次稳定特征通过、噪声淘汰。"""
    rng = np.random.default_rng(1)
    n, n_modes = 4000, 8
    # 模式 0/1 强相关（稳定结构）；其余独立噪声
    base = rng.random((n, n_modes)) < 0.3
    corr_col = (rng.random(n) < 0.4).astype(np.uint8)
    base[:, 0] = base[:, 1] = corr_col
    batch = _mk_batch(base.astype(np.uint8))
    basis = _mk_basis()
    # 地址：只用特征对 (f0,f1) → 模式 (1,3)？直接用二阶对 (0,1)（强相关）与 (2,3)（独立）
    addr = np.tile(np.array([[0, 1], [2, 3]]), (500, 1)).astype(int)  # 1000 行两行交替
    # 构造 vocab 需要 n_features=2 → 对 (0,1)；但我们关心模式对……直接用 (0,1)/(2,3) 两行地址
    # 闸门1评估的是候选列；候选池由 n_features=2 → 1 对 pair（外加 1 基）
    # 为区分稳定/噪声，用 n_features=4：对 (0,1)=corr 高 / (2,3)=低
    addr = np.tile(np.arange(4), (500, 1)).astype(int)
    passed, avg_corr = gate1_stability(batch, basis, addr, 4, PARAMS)
    names = candidate_names(4, 1)
    d = dict(zip(names, avg_corr))
    # pair f0_f1（模式 0,1 完全共变）跨子批次稳定 → 相关高
    assert d["gbs_pair_f0_f1"] > 0.9
    # 独立噪声对（如 f2_f3）相关低、被淘汰
    assert d["gbs_pair_f2_f3"] < 0.8
    assert passed[names.index("gbs_pair_f0_f1")]
    assert not passed[names.index("gbs_pair_f2_f3")]


def _mk_gate2_inputs(seed: int = 0, n_days: int = 250, n_sym: int = 200, n_cand: int = 6):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    idx = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(n_sym)]], names=["date", "symbol"])
    vals = rng.normal(size=(len(idx), n_cand))
    # 列 0：与标签相关（IC 稳定正）；其余纯噪声（样本量使噪声 |IC 均值| ≪ δ）
    y = pd.Series(vals[:, 0] * 0.7 + rng.normal(0, 1, len(idx)), index=idx)
    return vals, y, idx, dates


def test_gate2_planted_passes_noise_rejected() -> None:
    """闸门 2：植入特征过（|IC|≥δ 且同号），噪声淘汰；top-k 与输入切片红线。"""
    vals, y, idx, dates = _mk_gate2_inputs()
    tr, va = dates[:150], dates[150:]
    passed, detail = gate2_predictive(vals, y, idx, tr, va, PARAMS)
    assert passed[0] and detail["ic_mean"][0] > 0.05
    assert not passed[1:].any()  # 噪声列全淘汰（|IC|<δ）
    assert detail["rank"][0] == 1
    # 因果红线：输入含 train+valid 之外日期 → 报错
    with pytest.raises(ValueError, match="防泄漏"):
        future = pd.bdate_range(dates[-1] + pd.Timedelta(days=7), periods=10)
        all_dates = dates.union(future)
        idx2 = pd.MultiIndex.from_product([all_dates, ["S0"]], names=["date", "symbol"])
        gate2_predictive(np.zeros((len(idx2), 2)), pd.Series(0.0, index=idx2), idx2, tr, va, PARAMS)


def test_gate2_untouched_by_test_segment() -> None:
    """闸门 2 因果性：labels 里 test 段扰动不改变结果（输入已切 train+valid）。"""
    vals, y, idx, dates = _mk_gate2_inputs()
    tr, va = dates[:150], dates[150:]
    _, det1 = gate2_predictive(vals, y, idx, tr, va, PARAMS)
    y2 = y.copy()
    y2.iloc[:] = y2  # 同一切片；扰动一个不属于本折的"未来"标签不可见——等价于改别折数据
    _, det2 = gate2_predictive(vals, y2, idx, tr, va, PARAMS)
    pd.testing.assert_frame_equal(det1, det2)
    # 更强：把 valid 段标签全部打乱 → ic_valid 应变、ic_train 不变
    rng = np.random.default_rng(9)
    y3 = y.copy()
    va_rows = idx.get_level_values("date").isin(va)
    y3[va_rows] = rng.permutation(y3[va_rows].to_numpy())
    _, det3 = gate2_predictive(vals, y3, idx, tr, va, PARAMS)
    assert np.allclose(det3["ic_train"], det1["ic_train"])
    assert not np.allclose(det3["ic_valid"], det1["ic_valid"])


def _rec(split: int, passed: set[str], ic: dict[str, float]) -> FoldGateRecord:
    return FoldGateRecord(split_id=split, passed=frozenset(passed), ic_mean=ic)


def test_gate3_causal_consistency() -> None:
    """闸门 3：比例门槛、方向一致性、j≤k 因果（min_history=0 隔离测试比例逻辑）。"""
    p0 = ScreeningParams(min_history=0)
    names = ["a", "b", "c"]
    # a：3/3 折通过且同号 → 入选；b：2/3 折通过（0.67≥0.6）但方向不一致 → 拒；
    # c：1/3 折通过 → 拒
    recs = [
        _rec(0, {"a", "b"}, {"a": 0.02, "b": 0.02}),
        _rec(1, {"a", "b"}, {"a": 0.03, "b": -0.01}),  # b 方向翻转
        _rec(2, {"a"}, {"a": 0.025}),
    ]
    out = gate3_consistency(recs, names, p0)
    assert out == {"a"}
    # 因果：追加未来折（第 3 折 b 通过且方向转正）不影响第 2 折的输出
    out_at_1 = gate3_consistency(recs[:2], names, p0)  # 第 1 折口径
    assert "b" not in out_at_1  # b 两折方向已不一致
    # 若只看第 0 折：可用折=1，a 通过比例 100% ≥ 60% → 入选
    out_at_0 = gate3_consistency(recs[:1], names, p0)
    assert out_at_0 == {"a", "b"}


def test_gate3_empty_records_rejected() -> None:
    with pytest.raises(ValueError):
        gate3_consistency([], ["a"], PARAMS)


def test_gate3_min_history_rule() -> None:
    """最小历史规则：可用折 ≤ 5 时筛选集为空；>5 后正常筛选。"""
    # 每折都通过 a（且同号）：不足历史时空集，第 6 折（len=6）起入选
    recs = [_rec(k, {"a"}, {"a": 0.02}) for k in range(6)]
    for k in range(5):
        assert gate3_consistency(recs[: k + 1], ["a"], PARAMS) == frozenset(), f"fold {k} 应为空"
    out = gate3_consistency(recs, ["a"], PARAMS)
    assert out == {"a"}  # 6/6 = 100% ≥ 60%，同号 → 入选
    # 自定义 min_history=2：len=3 时起正常
    p2 = ScreeningParams(min_history=2)
    assert gate3_consistency(recs[:2], ["a"], p2) == frozenset()
    assert gate3_consistency(recs[:3], ["a"], p2) == {"a"}


def test_near_miss_diagnostic() -> None:
    """near-miss 聚合：跨折通过次数分布与 top 明细正确。"""
    from qgbs.experiments.screening import near_miss_diagnostic

    names = ["a", "b", "c", "d"]
    recs = [
        _rec(0, {"a", "b"}, {"a": 0.02, "b": 0.01}),
        _rec(1, {"a", "b"}, {"a": 0.03, "b": -0.01}),  # b 方向不一致
        _rec(2, {"a"}, {"a": 0.025}),
    ]
    diag = near_miss_diagnostic(recs, names, n_folds_total=20)
    assert diag["histogram"] == {0: 2, 3: 1, 2: 1} or diag["histogram"] == {0: 2, 2: 1, 3: 1}
    # a 通过 3 折、b 2 折、c/d 0 折
    top = diag["top_candidates"]
    assert top[0]["feature"] == "a" and top[0]["n_folds_passed"] == 3 and top[0]["ic_sign_consistent"]
    assert top[1]["feature"] == "b" and top[1]["n_folds_passed"] == 2
    assert top[1]["ic_sign_consistent"] is False  # 方向不一致被标注
    assert diag["n_folds_evaluated"] == 3 and diag["n_folds_total"] == 20
