"""特征筛选策略对比测试（B2）：相关性剪枝、特征族配额与策略对比表。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qgbs.models.selection import compare_selection_strategies, select_features

N_DATES = 30
N_ASSETS = 6


def _make_panel(seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """构造含已知信号的 (date, symbol) 面板：y = 2·fa_1 + fb_1 + 噪声。

    fa_2 与 fa_1 近乎完全相关（用于相关性剪枝测试）；fb_2 为纯噪声。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=N_DATES)
    syms = [f"S{i:02d}" for i in range(N_ASSETS)]
    index = pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"])
    n = len(index)
    fa_1 = rng.normal(size=n)
    fa_2 = fa_1 + rng.normal(scale=0.05, size=n)
    fb_1 = rng.normal(size=n)
    fb_2 = rng.normal(size=n)
    X = pd.DataFrame({"fa_1": fa_1, "fa_2": fa_2, "fb_1": fb_1, "fb_2": fb_2}, index=index)
    y = pd.Series(2.0 * fa_1 + 1.0 * fb_1 + rng.normal(scale=0.5, size=n), index=index, name="y")
    return X, y


@pytest.fixture(scope="module")
def train_panel() -> tuple[pd.DataFrame, pd.Series]:
    return _make_panel(seed=0)


def test_corr_pruned_drops_redundant(train_panel) -> None:
    """|ρ|>阈值的冗余特征被剔除，保留 IC 更高者（fa_1 留、fa_2 剪）。"""
    X, y = train_panel
    sel = select_features(X, y, method="corr_pruned", max_features=3, corr_threshold=0.9)
    assert "fa_1" in sel.selected
    assert "fa_2" not in sel.selected  # 与 fa_1 相关性 ≈1，贪心剪枝
    assert "fb_1" in sel.selected
    assert sel.method == "corr_pruned"
    assert len(sel.scores) == X.shape[1]


def test_family_quota_balanced(train_panel) -> None:
    """均衡配额：max_features=2、两族 → 各族取最优（fa_1 与 fb_1）。"""
    X, y = train_panel
    sel = select_features(X, y, method="family_quota", max_features=2)
    assert set(sel.selected) == {"fa_1", "fb_1"}


def test_family_quota_explicit(train_panel) -> None:
    """显式配额：fa 族 2 个、fb 族 0 个 → 选 fa_1/fa_2。"""
    X, y = train_panel
    sel = select_features(
        X, y, method="family_quota", max_features=2, family_quota={"fa": 2, "fb": 0}
    )
    assert set(sel.selected) == {"fa_1", "fa_2"}


def test_family_quota_fill_when_family_short(train_panel) -> None:
    """族内特征不足配额时按全局分数补足到 max_features。"""
    X, y = train_panel
    sel = select_features(
        X, y, method="family_quota", max_features=3, family_quota={"fa": 3, "fb": 0}
    )
    assert len(sel.selected) == 3
    assert sel.selected[:2] == ["fa_1", "fa_2"]  # fa 族仅 2 个，第 3 名全局补入 fb_1
    assert "fb_1" in sel.selected


def test_base_methods_unchanged(train_panel) -> None:
    """既有 ic_mean/ic_stability 行为不变；未知方法报错。"""
    X, y = train_panel
    sel = select_features(X, y, method="ic_mean", max_features=2)
    assert len(sel.selected) == 2
    assert "fa_1" in sel.selected  # 最强信号
    with pytest.raises(ValueError, match="未知筛选方法"):
        select_features(X, y, method="lasso")


def test_compare_selection_strategies_table() -> None:
    """对比表结构：行 = 策略 × 特征数（all 只评一次），列齐全且数值有效。"""
    X_tr, y_tr = _make_panel(seed=0)
    X_va, y_va = _make_panel(seed=1)
    table = compare_selection_strategies(
        X_tr, y_tr, X_va, y_va,
        strategies=["ic_mean", {"method": "family_quota"}, "all"],
        max_features_grid=(2, 3),
    )
    assert list(table.columns) == [
        "strategy", "max_features", "n_selected", "valid_rank_ic", "valid_rank_icir"
    ]
    assert len(table) == 2 * 2 + 1
    row_all = table[table["strategy"] == "all"]
    assert len(row_all) == 1
    assert int(row_all["n_selected"].iloc[0]) == X_tr.shape[1]
    row_ic2 = table[(table["strategy"] == "ic_mean") & (table["max_features"] == 2)]
    # 训练/验证同一生成律且信号强 → 选到信号特征后 valid IC 显著为正
    assert row_ic2["valid_rank_ic"].iloc[0] > 0.5
    assert np.isfinite(table["valid_rank_ic"]).all()
