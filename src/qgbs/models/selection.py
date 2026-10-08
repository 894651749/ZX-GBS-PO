"""训练期内的特征筛选与超参搜索（只允许使用 train/valid 窗口）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import pandas as pd

from qgbs.models.readout import GBMReadout, ReadoutModel, RidgeReadout


@dataclass(frozen=True)
class SelectionResult:
    """特征筛选结果。"""

    selected: list[str]  # 入选特征名
    scores: pd.Series  # 各特征评分（如 IC 均值/稳定性）
    method: str


@dataclass(frozen=True)
class HyperParamResult:
    """超参搜索结果。"""

    best_params: dict[str, Any]
    valid_score: float
    search_log: pd.DataFrame = field(default_factory=pd.DataFrame)


def _per_date_rank_ic(x: pd.Series, y: pd.Series) -> pd.Series:
    """逐日横截面 Spearman IC（x、y 均为 (date, symbol) 索引）。"""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    ics = df.groupby(level="date").apply(
        lambda g: g["x"].corr(g["y"], method="spearman") if len(g) >= 3 else float("nan")
    )
    return ics.dropna()


def _ic_score_series(X_train: pd.DataFrame, y_train: pd.Series, method: str) -> pd.Series:
    """逐特征打分（|IC 均值| 或 |IC 均值|/IC 标准差），按分数降序返回。"""
    scores: dict[str, float] = {}
    for col in X_train.columns:
        ics = _per_date_rank_ic(X_train[col], y_train)
        if len(ics) == 0:
            scores[col] = 0.0
        elif method == "ic_mean":
            scores[col] = abs(float(ics.mean()))
        elif method == "ic_stability":
            std = float(ics.std())
            scores[col] = abs(float(ics.mean())) / (std if std > 0 else 1.0)
        else:
            raise ValueError(f"未知打分方法: {method!r}")
    return pd.Series(scores).sort_values(ascending=False)


def _default_feature_family(name: str) -> str:
    """默认特征族：去掉结尾的数字段（mom_5→mom、rf_12→rf、mkt_vol_20→mkt_vol）。"""
    return re.sub(r"(_\d+)+$", "", name)


def _corr_pruned_select(
    X_train: pd.DataFrame, scores: pd.Series, max_features: int, corr_threshold: float
) -> list[str]:
    """相关性剪枝：按分数降序贪心入选，与已选特征 |ρ|>阈值的候选剔除。"""
    corr = X_train[scores.index].corr().abs()
    selected: list[str] = []
    for feat in scores.index:
        if len(selected) >= max_features:
            break
        if any(float(corr.loc[feat, s]) > corr_threshold for s in selected):
            continue
        selected.append(feat)
    return selected


def _family_quota_select(
    scores: pd.Series,
    max_features: int,
    family_map: dict[str, str] | None,
    family_quota: dict[str, int] | None,
) -> list[str]:
    """按特征族配额取 top-k：各族内按分数降序取配额个，不足时按全局分数补足。"""
    families = {
        f: (family_map.get(f) if family_map else None) or _default_feature_family(f)
        for f in scores.index
    }
    fam_best: dict[str, float] = {}
    for f, s in scores.items():
        g = families[f]
        fam_best[g] = max(fam_best.get(g, float("-inf")), float(s))
    fams_ordered = sorted(fam_best, key=lambda g: -fam_best[g])
    if family_quota is None:
        q, rem = divmod(max_features, len(fams_ordered))
        quota = {g: q + (1 if i < rem else 0) for i, g in enumerate(fams_ordered)}
    else:
        quota = {g: int(family_quota.get(g, 0)) for g in fams_ordered}
    selected: list[str] = []
    for g in fams_ordered:
        members = [f for f in scores.index if families[f] == g]
        selected.extend(members[: quota[g]])
    if len(selected) < max_features:  # 某族特征数不足配额时按全局分数补足
        chosen = set(selected)
        selected.extend(f for f in scores.index if f not in chosen)
        selected = selected[:max_features]
    return selected


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "ic_stability",
    max_features: int | None = None,
    *,
    corr_threshold: float = 0.95,
    base_method: str = "ic_mean",
    family_map: dict[str, str] | None = None,
    family_quota: dict[str, int] | None = None,
) -> SelectionResult:
    """在训练窗口内做特征筛选（不触碰测试期）。

    method:
        - ``"ic_mean"`` / ``"ic_stability"``：逐特征 IC 打分取 top-k（原行为）；
        - ``"corr_pruned"``：先按 ``base_method`` 打分，再贪心相关性剪枝
          （与已选特征 |ρ|>``corr_threshold`` 者剔除，保留 IC 更高者）；
        - ``"family_quota"``：按特征族配额 top-k（族由 ``family_map`` 指定或
          按名称前缀规则推导；配额缺省为均衡分配、余额按各族最优分补齐，
          某族入选不足配额时按全局分数补足）。
    """
    score_methods = {"ic_mean", "ic_stability", "corr_pruned", "family_quota"}
    if method not in score_methods:
        raise ValueError(f"未知筛选方法: {method!r}（支持 {sorted(score_methods)}）")
    base = method if method in ("ic_mean", "ic_stability") else base_method
    score_series = _ic_score_series(X_train, y_train, base)
    if method in ("ic_mean", "ic_stability") or max_features is None:
        selected = list(score_series.index[:max_features]) if max_features else list(score_series.index)
    elif method == "corr_pruned":
        selected = _corr_pruned_select(X_train, score_series, max_features, corr_threshold)
    else:
        selected = _family_quota_select(score_series, max_features, family_map, family_quota)
    return SelectionResult(selected=selected, scores=score_series, method=method)


def _build_model(model_name: str, params: dict[str, Any]) -> ReadoutModel:
    if model_name == "ridge":
        return RidgeReadout(alpha=float(params.get("alpha", 1.0)))
    if model_name == "gbm":
        return GBMReadout(params=dict(params))
    raise ValueError(f"未知读出模型: {model_name!r}")


def search_hyperparams(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    param_grid: dict[str, list[Any]],
    fixed_params: dict[str, Any] | None = None,
) -> HyperParamResult:
    """train 拟合、valid 评分（逐日 rank IC 均值）的小网格超参搜索。

    ``fixed_params`` 为不参与搜索的固定参数（如 random_state），
    与网格参数合并后传入模型，保证搜索阶段的可复现性。
    """
    keys = list(param_grid)
    fixed = dict(fixed_params or {})
    rows: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_params: dict[str, Any] = {}
    for combo in product(*(param_grid[k] for k in keys)):
        params = {**fixed, **dict(zip(keys, combo))}
        model = _build_model(model_name, params)
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        ics = _per_date_rank_ic(pred, y_valid)
        score = float(ics.mean()) if len(ics) else float("nan")
        rows.append({**params, "valid_rank_ic": score})
        if pd.notna(score) and score > best_score:
            best_score, best_params = score, params
    return HyperParamResult(
        best_params=best_params,
        valid_score=best_score,
        search_log=pd.DataFrame(rows),
    )


def _strategy_label(spec: str | dict[str, Any]) -> str:
    """策略标签：字符串原样；dict 取 method 并附关键参数。"""
    if isinstance(spec, str):
        return spec
    params = dict(spec)
    method = str(params.pop("method", "ic_mean"))
    rest = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"{method}({rest})" if rest else method


def compare_selection_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    strategies: list[str | dict[str, Any]] | None = None,
    max_features_grid: tuple[int, ...] = (5, 10, 20),
    model_name: str = "ridge",
    model_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """特征筛选策略对比：策略 × 特征数 × valid rank IC（防泄漏：仅用 train/valid）。

    Args:
        strategies: 元素为 :func:`select_features` 的方法名字符串或 kwargs dict
            （如 ``{"method": "corr_pruned", "corr_threshold": 0.8}``）；
            特殊值 ``"all"`` 表示不筛选的全量基线。缺省为 ic_mean /
            ic_stability / corr_pruned / family_quota / all。
        max_features_grid: 特征数网格（``"all"`` 与网格无关，只评一次）。
        model_name: 读出模型（与超参搜索同一构造入口）。

    Returns:
        DataFrame，列为 strategy / max_features / n_selected /
        valid_rank_ic / valid_rank_icir。
    """
    if strategies is None:
        strategies = [
            "ic_mean",
            "ic_stability",
            {"method": "corr_pruned"},
            {"method": "family_quota"},
            "all",
        ]
    rows: list[dict[str, Any]] = []
    for spec in strategies:
        label = _strategy_label(spec)
        kwargs = dict(spec) if isinstance(spec, dict) else {"method": spec}
        if kwargs.get("method") == "all":
            grid: list[int | None] = [None]
        else:
            grid = list(max_features_grid)
        for k in grid:
            if k is None:
                selected = list(X_train.columns)
            else:
                sel = select_features(X_train, y_train, max_features=k, **kwargs)
                selected = sel.selected
            if not selected:
                continue
            model = _build_model(model_name, dict(model_params or {}))
            model.fit(X_train[selected], y_train)
            ics = _per_date_rank_ic(model.predict(X_valid[selected]), y_valid)
            std = float(ics.std()) if len(ics) else float("nan")
            mean = float(ics.mean()) if len(ics) else float("nan")
            rows.append(
                {
                    "strategy": label,
                    "max_features": len(selected) if k is None else k,
                    "n_selected": len(selected),
                    "valid_rank_ic": mean,
                    "valid_rank_icir": mean / std if std and std > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)
