"""QuantileEncoder：金融特征 → GBS 时空模式地址的分位数编码器（论文二维寻址）。

二维寻址（论文 arXiv:2506.19707 机制）：

- 特征索引 → 空间模式（spatial mode，由 ModeMapping 指定）；
- 量化值（bin）→ 时间模式（temporal mode）：bin < n_bins_a → A 分区，
  否则 B 分区，基准时间模式 = bin 或 partition_boundary + (bin − n_bins_a)，
  再经映射的时间置换得到最终时间模式；
- 分区边界作为参数传入（论文默认 256），不硬编码。

防泄漏状态机：

- ``fit`` 只能在训练窗口数据上调用；
- ``freeze`` 返回冻结副本，bin 边界与模式映射一并锁定；
- 未冻结实例调用 ``transform``、已冻结实例调用 ``fit``，均抛 RuntimeError；
- 超出训练期分布范围的值截断到端点 bin；
- bin → 基准时间模式的地址预算在 fit 时对照所选映射校验，越界配置抛
  ValueError（禁止静默截断）；
- 模式映射须在 fit 时按 ``mapping_id`` 选定，**进入测试阶段不得重新选择**。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModeAddress:
    """模式地址：资产 × 特征 × bin → GBS 时空模式的映射结果。

    各字段为同形状数组 (n_rows, n_features)（一次 transform 覆盖的行 × 特征网格）：

    - ``asset_index``：行在调用面板中的位置（0..n_rows-1）；
    - ``feature_index`` / ``bin_index``：特征编号与分箱编号；
    - ``spatial_index`` / ``temporal_index``：二维时空模式坐标；
    - ``partition``：时间模式所属分区（"A" / "B"）；
    - ``mode_index``：展平全局模式编号 = temporal × n_spatial + spatial。
    """

    asset_index: np.ndarray
    feature_index: np.ndarray
    bin_index: np.ndarray
    spatial_index: np.ndarray
    temporal_index: np.ndarray
    partition: np.ndarray  # 取值 "A" / "B"
    mode_index: np.ndarray


@dataclass(frozen=True)
class ModeMapping:
    """一组预先固定的模式映射（特征→空间模式、基准时间模式→时间模式）。

    映射集合在实验设计阶段固定若干组并注册；进入测试阶段不得重新选择。
    """

    mapping_id: str
    n_spatial_modes: int
    n_temporal_modes: int
    spatial_assignment: np.ndarray  # feature_index → 空间模式编号
    temporal_assignment: np.ndarray  # 基准时间模式 → 时间模式编号（置换）
    description: str = ""


MAPPING_REGISTRY: dict[str, ModeMapping] = {}


def register_mapping(mapping: ModeMapping) -> None:
    """注册一组预先固定的模式映射（重复注册同一 ID 抛错）。"""
    if mapping.mapping_id in MAPPING_REGISTRY:
        raise ValueError(f"映射 {mapping.mapping_id!r} 已注册")
    MAPPING_REGISTRY[mapping.mapping_id] = mapping


def get_mapping(mapping_id: str) -> ModeMapping:
    """按 ID 取回已注册映射；未注册抛 KeyError。"""
    try:
        return MAPPING_REGISTRY[mapping_id]
    except KeyError:
        raise KeyError(f"映射 {mapping_id!r} 未注册，可用: {sorted(MAPPING_REGISTRY)}") from None


def list_mappings() -> list[str]:
    """列出全部已注册映射 ID。"""
    return list(MAPPING_REGISTRY)


def _compatible(m: ModeMapping, n_features: int, n_spatial: int, n_temporal: int) -> bool:
    return (
        len(m.spatial_assignment) >= n_features
        and m.n_spatial_modes == n_spatial
        and m.n_temporal_modes == n_temporal
    )


def _dim_suffix(n_spatial: int, n_temporal: int) -> str:
    return f"__{n_spatial}s{n_temporal}t"


def ensure_default_mappings(
    n_features: int,
    n_spatial: int,
    n_temporal: int,
    n_variants: int = 3,
    seed: int = 0,
) -> list[str]:
    """注册默认映射组："identity" + 若干确定性置换变体，返回可用的 mapping_id 列表。

    注册表是进程级全局状态；同名但维度不兼容的已有映射不会被覆盖，
    而是注册维度后缀变体（如 "identity__4s8t"）。
    """
    def _ensure(mid: str, spatial: np.ndarray, temporal: np.ndarray, desc: str) -> str:
        existing = MAPPING_REGISTRY.get(mid)
        if existing is not None:
            if _compatible(existing, n_features, n_spatial, n_temporal):
                return mid
            mid = mid + _dim_suffix(n_spatial, n_temporal)
            if mid in MAPPING_REGISTRY:
                return mid
        register_mapping(
            ModeMapping(
                mapping_id=mid,
                n_spatial_modes=n_spatial,
                n_temporal_modes=n_temporal,
                spatial_assignment=spatial,
                temporal_assignment=temporal,
                description=desc,
            )
        )
        return mid

    ids = [
        _ensure(
            "identity",
            np.arange(n_features) % n_spatial,
            np.arange(n_temporal),
            "恒等映射",
        )
    ]
    rng = np.random.default_rng(seed)
    for k in range(n_variants):
        ids.append(
            _ensure(
                f"perm{k}",
                rng.permutation(n_spatial)[np.arange(n_features) % n_spatial],
                rng.permutation(n_temporal),
                f"确定性置换变体 {k}",
            )
        )
    return ids


def resolve_mapping_id(requested: str, n_features: int, n_spatial: int, n_temporal: int) -> str:
    """解析请求的 mapping_id：兼容则原样返回，否则尝试维度后缀变体。"""
    existing = MAPPING_REGISTRY.get(requested)
    if existing is not None and _compatible(existing, n_features, n_spatial, n_temporal):
        return requested
    suffixed = requested + _dim_suffix(n_spatial, n_temporal)
    if suffixed in MAPPING_REGISTRY:
        return suffixed
    raise KeyError(
        f"映射 {requested!r} 与当前模式结构 ({n_spatial}×{n_temporal}) 不兼容，"
        f"可用: {sorted(MAPPING_REGISTRY)}"
    )


@dataclass(frozen=True)
class EncoderParams:
    """编码参数（论文口径）：A 区 256 箱（特征值映射到 [0,255]）、B 区 255 箱
    （映射到 [0,254]），合计 511 个 bin，对应 511 个时间模式。"""

    n_bins_a: int = 256  # A 分区 bin 数（论文值 256）
    n_bins_b: int = 255  # B 分区 bin 数（论文值 255）
    partition_boundary: int = 256  # B 分区时间模式起点（论文默认 256，须按 click 剖面传入）


class QuantileEncoder:
    """分位数编码器（防泄漏状态机 + 二维时空寻址）。"""

    def __init__(
        self,
        params: EncoderParams | None = None,
        mapping_id: str | None = None,
    ) -> None:
        self.params = params or EncoderParams()
        self.mapping_id = mapping_id
        self._fitted: bool = False
        self._frozen: bool = False
        self._edges: dict[str, np.ndarray] = {}
        self._fill: pd.Series | None = None
        self._features: list[str] = []
        self._mapping: ModeMapping | None = None

    @property
    def n_bins_total(self) -> int:
        return self.params.n_bins_a + self.params.n_bins_b

    def fit(self, X: pd.DataFrame, mapping_id: str | None = None) -> "QuantileEncoder":
        """在训练窗口上估计各特征的分位数边界，并选定模式映射。

        对已冻结实例调用抛 RuntimeError。``mapping_id`` 在 fit 时确定，
        freeze 后映射锁定，进入测试阶段不得重新选择。
        """
        if self._frozen:
            raise RuntimeError("QuantileEncoder 已冻结：不允许再 fit")
        mid = mapping_id or self.mapping_id
        if mid is None:
            raise ValueError("必须指定 mapping_id（构造参数或 fit 参数）")
        self._mapping = get_mapping(mid)
        self.mapping_id = mid
        self._validate_address_budget()
        self._features = list(X.columns)
        if len(self._features) > len(self._mapping.spatial_assignment):
            raise ValueError("特征数超过映射的空间模式分配长度")
        qs = np.linspace(0.0, 1.0, self.n_bins_total + 1)[1:-1]
        Xn = X.fillna(X.median())
        self._fill = X.median()
        for col in self._features:
            self._edges[col] = np.unique(np.quantile(Xn[col].to_numpy(dtype=float), qs))
        self._fitted = True
        return self

    def _validate_address_budget(self) -> None:
        """对照映射的空间/时间模式预算校验编码参数与映射一致性。

        任何 bin 可能落到的基准时间模式都必须落在 [0, n_temporal_modes) 内，
        且 A/B 分区地址区间不得重叠；映射数组长度与取值须与声明的模式数一致。
        违规时抛带参数信息的 ValueError——禁止 transform 阶段静默截断。
        """
        p = self.params
        mapping = self._mapping
        if mapping is None:  # 内部不变量：fit 已选定映射
            raise RuntimeError("QuantileEncoder 内部状态不完整：缺少模式映射")
        if p.n_bins_a < 0 or p.n_bins_b < 0 or p.n_bins_a + p.n_bins_b < 1:
            raise ValueError(
                f"非法 bin 配置: n_bins_a={p.n_bins_a}, n_bins_b={p.n_bins_b}（合计须 ≥ 1）"
            )
        if p.n_bins_a > p.partition_boundary:
            raise ValueError(
                f"A 区 bin 数 n_bins_a={p.n_bins_a} 越过分区边界 "
                f"partition_boundary={p.partition_boundary}：A 区地址会侵入 B 区"
            )
        n_t = mapping.n_temporal_modes
        budget = n_t - 1
        if p.n_bins_a > 0 and p.n_bins_a - 1 > budget:
            raise ValueError(
                f"A 区最高基准时间模式 {p.n_bins_a - 1} 超出映射时间模式预算 [0, {budget}]"
                f"（n_bins_a={p.n_bins_a}, n_temporal_modes={n_t}）"
            )
        max_b = p.partition_boundary + p.n_bins_b - 1
        if p.n_bins_b > 0 and max_b > budget:
            raise ValueError(
                f"B 区最高基准时间模式 {max_b} 超出映射时间模式预算 [0, {budget}]"
                f"（n_bins_b={p.n_bins_b}, partition_boundary={p.partition_boundary}, "
                f"n_temporal_modes={n_t}）"
            )
        if len(mapping.temporal_assignment) != n_t:
            raise ValueError(
                f"映射 {mapping.mapping_id!r} 的 temporal_assignment 长度 "
                f"{len(mapping.temporal_assignment)} 与声明的时间模式数 {n_t} 不一致"
            )
        if len(mapping.spatial_assignment) and (
            int(mapping.spatial_assignment.min()) < 0
            or int(mapping.spatial_assignment.max()) >= mapping.n_spatial_modes
        ):
            raise ValueError(
                f"映射 {mapping.mapping_id!r} 的 spatial_assignment 取值超出 "
                f"[0, {mapping.n_spatial_modes})"
            )
        if n_t and (
            int(mapping.temporal_assignment.min()) < 0
            or int(mapping.temporal_assignment.max()) >= n_t
        ):
            raise ValueError(
                f"映射 {mapping.mapping_id!r} 的 temporal_assignment 取值超出 [0, {n_t})"
            )

    def freeze(self) -> "QuantileEncoder":
        """冻结编码器，返回不可变副本（bin 边界与模式映射锁定）。"""
        if not self._fitted:
            raise RuntimeError("QuantileEncoder 未 fit，无法 freeze")
        frozen = copy.deepcopy(self)
        frozen._frozen = True
        return frozen

    def transform(self, X: pd.DataFrame) -> ModeAddress:
        """将特征值映射为二维时空模式地址。

        仅允许在冻结副本上调用；越界值按训练期分位边界截断到端点 bin；
        NaN 用训练期中位数填充。bin → 时间模式的地址预算已在 fit 时对照
        映射校验（越界配置在 fit 即抛 ValueError），此处不做静默截断。
        """
        if not self._frozen:
            raise RuntimeError("QuantileEncoder 未冻结：仅允许在 freeze() 副本上 transform")
        if self._mapping is None or self._fill is None:
            raise RuntimeError("QuantileEncoder 内部状态不完整：须先 fit 再 freeze")
        X2 = X.reindex(columns=self._features).fillna(self._fill)
        n_rows, n_feat = X2.shape
        p = self.params
        mapping = self._mapping

        bins = np.empty((n_rows, n_feat), dtype=int)
        for j, col in enumerate(self._features):
            edges = self._edges[col]
            # digitize 天然把越界值截到端点 bin
            bins[:, j] = np.digitize(X2[col].to_numpy(dtype=float), edges)
        bins = np.clip(bins, 0, self.n_bins_total - 1)

        in_a = bins < p.n_bins_a
        base_temporal = np.where(in_a, bins, p.partition_boundary + (bins - p.n_bins_a))
        # fit 阶段已完成地址预算校验，此处直接索引，不做静默截断
        temporal = mapping.temporal_assignment[base_temporal]
        spatial = np.broadcast_to(mapping.spatial_assignment[:n_feat], (n_rows, n_feat)).copy()
        partition = np.where(in_a, "A", "B")
        mode_index = temporal * mapping.n_spatial_modes + spatial

        return ModeAddress(
            asset_index=np.broadcast_to(np.arange(n_rows)[:, None], (n_rows, n_feat)).copy(),
            feature_index=np.broadcast_to(np.arange(n_feat), (n_rows, n_feat)).copy(),
            bin_index=bins,
            spatial_index=spatial,
            temporal_index=temporal,
            partition=partition,
            mode_index=mode_index,
        )

    @property
    def is_frozen(self) -> bool:
        return self._frozen
