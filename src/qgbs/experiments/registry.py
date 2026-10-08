"""C0-C5 / Q1-Q4 对照实验注册表。

与 ``configs/experiments/*.yaml`` 一一对应；编号、特征族、读出模型、
GBS 后端与用途说明以方案书实验矩阵为准。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentConfig:
    """一个对照实验的完整配置声明。"""

    experiment_id: str  # "C0"-"C5" / "Q1"-"Q4"
    feature_families: list[str]  # 特征族列表，如 ["finance_base", "gbs_marginal"]
    readout: str | None  # 读出模型："ridge" / "gbm" / None（纯组合流程）
    gbs_backend: str | None  # GBS 后端："simulator" / "hardware" / "cached" / None
    description: str  # 用途说明
    overrides: dict[str, object] = field(default_factory=dict)  # 相对 base.yaml 的覆盖项


REGISTRY: dict[str, ExperimentConfig] = {
    "C0": ExperimentConfig(
        experiment_id="C0",
        feature_families=[],
        readout=None,
        gbs_backend=None,
        description="原经典均值—方差流程，复现客户结果",
    ),
    "C1": ExperimentConfig(
        experiment_id="C1",
        feature_families=[],
        readout=None,
        gbs_backend=None,
        description="修正口径 + 收缩风险 + 稳健约束，建立可比经典基线（无监督，引擎内置历史均值信号）",
    ),
    "C2": ExperimentConfig(
        experiment_id="C2",
        feature_families=["finance_base"],
        readout="ridge",
        gbs_backend=None,
        description="原始金融特征 + 岭回归，线性监督学习基线",
    ),
    "C3": ExperimentConfig(
        experiment_id="C3",
        feature_families=["finance_base"],
        readout="gbm",
        gbs_backend=None,
        description="原始金融特征 + 梯度提升树，经典非线性基线",
    ),
    "C4": ExperimentConfig(
        experiment_id="C4",
        feature_families=["finance_base", "quantile_explicit"],
        readout="ridge",
        gbs_backend=None,
        description="分位数 one-hot/显式交互，检验离散编码贡献（由 gbs/compression.py 的编码类特征承接）",
    ),
    "C5": ExperimentConfig(
        experiment_id="C5",
        feature_families=["finance_base", "random_features"],
        readout="ridge",
        gbs_backend=None,
        description="等维 RFF/Nyström 等随机特征，检验普通随机升维贡献（由 gbs/compression.py 的 ClassicalRandomFeatures 承接，与 GBS 特征等维对齐）",
    ),
    "Q1": ExperimentConfig(
        experiment_id="Q1",
        feature_families=["finance_base", "gbs_marginal"],
        readout="ridge",
        gbs_backend="cached",
        description="原始特征 + 仿真 GBS 边际特征，检验单模式信息（统一走质检样本库 cached，保证与 Q2-Q4 对照口径一致；当前为近似仿真样本，非真机）",
    ),
    "Q2": ExperimentConfig(
        experiment_id="Q2",
        feature_families=["finance_base", "gbs_marginal", "gbs_second_order"],
        readout="ridge",
        gbs_backend="cached",
        description="Q1 + 仿真 GBS 二阶联合特征，检验联合关联增量",
    ),
    "Q3": ExperimentConfig(
        experiment_id="Q3",
        feature_families=["finance_base", "gbs_marginal", "gbs_second_order", "gbs_higher_order"],
        readout="ridge",
        gbs_backend="cached",
        description="Q2 + 经筛选的局部高阶基，检验高阶交互增量",
    ),
    "Q4": ExperimentConfig(
        experiment_id="Q4",
        feature_families=["finance_base", "gbs_marginal", "gbs_second_order", "gbs_higher_order"],
        readout="gbm",
        gbs_backend="cached",
        description="最优 GBS 特征 + 经典非线性读出模型，评估完整增强配置",
    ),
}


def get_experiment(experiment_id: str) -> ExperimentConfig:
    """按编号查询实验配置。"""
    try:
        return REGISTRY[experiment_id]
    except KeyError:
        raise KeyError(f"未知实验编号 {experiment_id!r}，已注册: {sorted(REGISTRY)}") from None


def list_experiments() -> list[str]:
    """返回全部已注册实验编号。"""
    return list(REGISTRY)
