"""冒烟测试：所有子包可导入。"""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "qgbs",
    "qgbs.data",
    "qgbs.data.adapters",
    "qgbs.data.adapters.base",
    "qgbs.data.adapters.mock",
    "qgbs.data.adapters.csv_adapter",
    "qgbs.data.calendar",
    "qgbs.data.pool",
    "qgbs.data.store",
    "qgbs.factors",
    "qgbs.factors.finance",
    "qgbs.factors.neutralization",
    "qgbs.factors.pipeline",
    "qgbs.gbs",
    "qgbs.gbs.backends",
    "qgbs.gbs.backends.base",
    "qgbs.gbs.backends.simulator",
    "qgbs.gbs.backends.hardware",
    "qgbs.gbs.backends.cached",
    "qgbs.gbs.backends.permuted",
    "qgbs.gbs.backends.exact_gbs",
    "qgbs.gbs.sample_store",
    "qgbs.gbs.encoder",
    "qgbs.gbs.basis",
    "qgbs.gbs.compression",
    "qgbs.gbs.distribution_compare",
    "qgbs.gbs.statistics",
    "qgbs.gbs.feature_service",
    "qgbs.models",
    "qgbs.models.readout",
    "qgbs.models.blending",
    "qgbs.models.selection",
    "qgbs.portfolio",
    "qgbs.portfolio.risk",
    "qgbs.portfolio.optimizer",
    "qgbs.portfolio.costs",
    "qgbs.backtest",
    "qgbs.backtest.rolling",
    "qgbs.backtest.engine",
    "qgbs.backtest.metrics",
    "qgbs.experiments",
    "qgbs.experiments.registry",
    "qgbs.experiments.runner",
    "qgbs.experiments.report",
    "qgbs.experiments.diagnostics",
]


@pytest.mark.parametrize("module", MODULES)
def test_importable(module: str) -> None:
    importlib.import_module(module)


def test_version() -> None:
    import qgbs

    assert isinstance(qgbs.__version__, str)
