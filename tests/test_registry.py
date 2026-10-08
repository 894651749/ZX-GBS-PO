"""registry 与 configs/experiments/*.yaml 双源一致性（防漂移）。"""

from __future__ import annotations

import yaml

from qgbs.experiments.registry import REGISTRY
from qgbs.experiments.runner import CONFIG_DIR, EXPERIMENT_CONFIG_FILES


def _yaml_features(raw: dict) -> list[str]:
    """yaml 的 features 支持列表或 {families, finance_set} 映射两种形态。"""
    feats = raw.get("features") or []
    if isinstance(feats, dict):
        return list(feats.get("families") or [])
    return list(feats)


def test_registry_covers_all_config_files() -> None:
    assert set(REGISTRY) == set(EXPERIMENT_CONFIG_FILES)


def test_registry_matches_yaml() -> None:
    for exp_id, fname in EXPERIMENT_CONFIG_FILES.items():
        path = CONFIG_DIR / "experiments" / fname
        assert path.exists(), f"{exp_id} 缺配置文件 {fname}"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        reg = REGISTRY[exp_id]
        assert raw["experiment_id"] == exp_id == reg.experiment_id
        assert _yaml_features(raw) == reg.feature_families, f"{exp_id} 特征族漂移"
        assert raw.get("readout") == reg.readout, f"{exp_id} 读出模型漂移"
        assert (raw.get("gbs") or {}).get("backend") == reg.gbs_backend, f"{exp_id} GBS 后端漂移"


def test_registry_descriptions_grounded() -> None:
    """Q 系当前均为仿真样本（cached/simulation），描述不得宣称真机。"""
    for exp_id in ("Q1", "Q2", "Q3", "Q4"):
        assert "真机 GBS" not in REGISTRY[exp_id].description
