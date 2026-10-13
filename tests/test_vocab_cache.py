"""跨折词表缓存测试：命中跳过重算、误命中防护、面板口径不变。"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qgbs.experiments import runner
from qgbs.gbs.backends.simulator import GaussianSimulatorBackend, SimulatorConfig
from qgbs.gbs.feature_service import GBSFeatureService


def _mini_cfg() -> dict:
    return {
        "gbs": {"n_temporal": 8, "n_spatial": 4, "activation": "soft"},
        "encoder": {"n_bins_a": 4, "n_bins_b": 4, "partition_boundary": 4},
    }


def _mini_panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    idx = pd.MultiIndex.from_product(
        [dates, [f"S{i}" for i in range(5)]], names=["date", "symbol"]
    )
    return pd.DataFrame(rng.normal(size=(len(idx), 3)), index=idx, columns=["f0", "f1", "f2"])


@pytest.fixture(autouse=True)
def _clear_vocab_cache():
    runner._VOCAB_CACHE.clear()
    yield
    runner._VOCAB_CACHE.clear()


def test_vocab_cache_skips_refit_across_folds(monkeypatch) -> None:
    """同一批次两折调用：GBSFeatureService.fit 只执行一次，两折共享词表。"""
    batch = GaussianSimulatorBackend(
        SimulatorConfig(n_temporal=8, n_spatial=4, seed=1)
    ).sample(500)
    backend = SimpleNamespace(n_modes=32)  # 无 higher_order 时仅 n_modes 被读取
    feats_p = _mini_panel()
    dates = feats_p.index.get_level_values("date").unique()
    split1 = SimpleNamespace(train=dates[:15])
    split2 = SimpleNamespace(train=dates[5:20])

    calls = {"fit": 0}
    orig_fit = GBSFeatureService.fit

    def counted_fit(self, *a, **k):
        calls["fit"] += 1
        return orig_fit(self, *a, **k)

    monkeypatch.setattr(GBSFeatureService, "fit", counted_fit)
    p1 = runner._build_gbs_panel(_mini_cfg(), ["gbs_marginal", "gbs_second_order"], feats_p, split1, backend, batch)
    p2 = runner._build_gbs_panel(_mini_cfg(), ["gbs_marginal", "gbs_second_order"], feats_p, split2, backend, batch)
    assert calls["fit"] == 1
    assert list(p1.columns) == list(p2.columns)
    assert p1.shape == p2.shape
    # 词表本体（_corr）进程内只存一份
    assert len(runner._VOCAB_CACHE) == 1


def test_vocab_cache_key_separates_batch_and_config() -> None:
    """不同 batch_id / stat_kinds / n_shots 不误命中：各算各的词表。"""
    sim = GaussianSimulatorBackend(SimulatorConfig(n_temporal=8, n_spatial=4, seed=1))
    batch = sim.sample(500)
    backend = SimpleNamespace(n_modes=32)
    feats_p = _mini_panel()
    dates = feats_p.index.get_level_values("date").unique()
    split = SimpleNamespace(train=dates[:15])

    runner._build_gbs_panel(_mini_cfg(), ["gbs_marginal", "gbs_second_order"], feats_p, split, backend, batch)
    # 置换对照路径：batch_id 不同 → 独立词表
    permuted = dataclasses.replace(
        batch, meta=dataclasses.replace(batch.meta, batch_id=batch.meta.batch_id + "-permuted")
    )
    runner._build_gbs_panel(_mini_cfg(), ["gbs_marginal", "gbs_second_order"], feats_p, split, backend, permuted)
    # stat_kinds 不同 → 独立词表
    runner._build_gbs_panel(_mini_cfg(), ["gbs_marginal"], feats_p, split, backend, batch)
    assert len(runner._VOCAB_CACHE) == 3
