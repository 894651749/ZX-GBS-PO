"""诊断三测试（B4）：低阶匹配随机特征生成器与 matched_random_features_control 判读方向。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qgbs.experiments.diagnostics import (
    GaussianCopulaMatchedFeatures,
    mapping_batch_stability,
    matched_random_features_control,
)

N_DATES = 60
N_ASSETS = 10


def _panel_index(n_dates: int = N_DATES, n_assets: int = N_ASSETS) -> pd.MultiIndex:
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    syms = [f"S{i:02d}" for i in range(n_assets)]
    return pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"])


def _reference_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """已知低阶统计的参照面板：c0/c1 相关 0.7，c2 为指数化正态（非正态边际）。"""
    rng = np.random.default_rng(seed)
    z0 = rng.normal(size=n)
    z1 = 0.7 * z0 + np.sqrt(1 - 0.49) * rng.normal(size=n)
    c2 = np.exp(rng.normal(size=n))
    c3 = rng.normal(size=n)
    return pd.DataFrame({"c0": z0, "c1": z1, "c2": c2, "c3": c3})


# ---------------------------------------------------------------------------
# GaussianCopulaMatchedFeatures 生成器
# ---------------------------------------------------------------------------


def test_generator_matches_low_order_moments() -> None:
    ref = _reference_frame()
    gen = GaussianCopulaMatchedFeatures(seed=1).fit(ref)
    sample = gen.sample(ref.index)
    assert sample.shape == ref.shape
    assert list(sample.columns) == list(ref.columns)
    assert gen.output_dim == ref.shape[1]
    # 逐列均值/标准差匹配（容差 10% 标准差量级）
    for col in ref.columns:
        assert abs(sample[col].mean() - ref[col].mean()) < 0.1 * ref[col].std()
        assert sample[col].std() == pytest.approx(ref[col].std(), rel=0.1)
    # 相关结构近似保留：corr(c0, c1) ≈ 0.7
    ref_corr = float(ref["c0"].corr(ref["c1"]))
    got_corr = float(sample["c0"].corr(sample["c1"]))
    assert got_corr == pytest.approx(ref_corr, abs=0.1)


def test_generator_reproducible_and_requires_fit() -> None:
    ref = _reference_frame()
    s1 = GaussianCopulaMatchedFeatures(seed=5).fit(ref).sample(ref.index)
    s2 = GaussianCopulaMatchedFeatures(seed=5).fit(ref).sample(ref.index)
    pd.testing.assert_frame_equal(s1, s2)
    with pytest.raises(RuntimeError, match="未 fit"):
        GaussianCopulaMatchedFeatures().sample(ref.index)
    with pytest.raises(ValueError, match="有效行不足"):
        GaussianCopulaMatchedFeatures().fit(ref.iloc[:1])


# ---------------------------------------------------------------------------
# matched_random_features_control
# ---------------------------------------------------------------------------


def _signal_panel(seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """GBS 面板含真实信号列 sig：y = 5·sig + 小噪声。"""
    index = _panel_index()
    rng = np.random.default_rng(seed)
    sig = rng.normal(size=len(index))
    cols = {"sig": sig}
    cols.update({f"n{i}": rng.normal(size=len(index)) for i in range(4)})
    X = pd.DataFrame(cols, index=index)
    y = pd.Series(5.0 * sig + 0.3 * rng.normal(size=len(index)), index=index, name="y")
    return X, y


def test_control_supports_gbs_when_signal() -> None:
    """GBS 面板含真实信号 → IC 差显著为正 → 判读支持独特增量。"""
    X, y = _signal_panel()
    result = matched_random_features_control(X, None, y, seed=9, n_bootstrap=300)
    assert result.diagnostic == "matched_random_features_control"
    assert list(result.table.columns) == ["ic_gbs", "ic_matched", "ic_diff"]
    assert result.details["matched_source"] == "auto_gaussian_copula"
    assert result.details["n_eval_dates"] >= 8
    assert result.details["ic_diff_mean"] > 0.3
    lo, hi = result.details["ic_diff_ci"]
    assert lo > 0.0
    assert "支持 GBS 特征含有超出低阶统计匹配的增量信息" in result.verdict
    assert "未支持" not in result.verdict


def test_control_null_when_pure_noise() -> None:
    """GBS 面板为纯噪声且与 y 独立 → IC 差 CI 含 0 → 判读未支持。"""
    index = _panel_index()
    rng = np.random.default_rng(1)
    X = pd.DataFrame({f"g{i}": rng.normal(size=len(index)) for i in range(5)}, index=index)
    y = pd.Series(rng.normal(size=len(index)), index=index, name="y")
    result = matched_random_features_control(X, None, y, seed=9, n_bootstrap=300)
    lo, hi = result.details["ic_diff_ci"]
    assert lo <= 0.0 <= hi
    assert "未支持独特结构贡献" in result.verdict


def test_control_accepts_provided_panel() -> None:
    """显式传入匹配面板（如 RFF/Nyström 产出）时按给定面板对比。"""
    X, y = _signal_panel()
    rng = np.random.default_rng(2)
    provided = pd.DataFrame(
        rng.normal(size=(len(X), X.shape[1])), index=X.index, columns=X.columns
    )
    result = matched_random_features_control(X, provided, y, seed=9, n_bootstrap=200)
    assert result.details["matched_source"] == "provided"
    assert result.details["ic_diff_mean"] > 0.3


def test_mapping_batch_stability_rejects_empty_batches() -> None:
    """空 batches 输入应在入口抛出清晰 ValueError，而非在后续统计处崩溃。"""
    with pytest.raises(ValueError, match="batches 为空"):
        mapping_batch_stability({}, [])
