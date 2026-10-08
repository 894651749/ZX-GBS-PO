"""压缩表征骨架与经典特征变换实现（C4/C5 对照用）。

- C4 ``quantile_explicit``：分位数分箱 one-hot + 两两显式交互
  （训练期 fit 分位边界，冻结语义）；
- C5 ``random_features``：RFF / Nyström 等维经典随机特征（真实实现）；
- 压缩表征（PCA/随机投影/稀疏筛选）仍为 P2 骨架。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Protocol

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 压缩表征（P2 骨架，供 GBS 统计量降维）
# ---------------------------------------------------------------------------


class CompressionTransform(Protocol):
    """压缩表征抽象：高维 GBS 统计量 → 低维特征。"""

    def fit(self, X_train: pd.DataFrame) -> "CompressionTransform":
        """在训练窗口上拟合压缩变换。"""
        ...

    def freeze(self) -> "CompressionTransform":
        """冻结变换参数，返回不可变副本。"""
        ...

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """应用冻结的压缩变换；未冻结实例调用应抛异常。"""
        ...

    @property
    def is_frozen(self) -> bool:
        ...

    @property
    def output_dim(self) -> int:
        """压缩后维度。"""
        ...


class PCACompression:
    """PCA 线性压缩。fit 仅训练期；freeze 后 transform 只读。"""

    def __init__(self, output_dim: int) -> None:
        self._output_dim = output_dim
        self._fitted: bool = False
        self._frozen: bool = False
        self._mean: np.ndarray | None = None
        self._components: np.ndarray | None = None

    def fit(self, X_train: pd.DataFrame) -> "PCACompression":
        if self._frozen:
            raise RuntimeError("PCACompression 已冻结：不允许再 fit")
        X = X_train.dropna().to_numpy(dtype=float)
        self._mean = X.mean(axis=0)
        _, _, vt = np.linalg.svd(X - self._mean, full_matrices=False)
        self._components = vt[: self._output_dim].T  # (in_dim, output_dim)
        self._fitted = True
        return self

    def freeze(self) -> "PCACompression":
        if not self._fitted:
            raise RuntimeError("PCACompression 未 fit，无法 freeze")
        import copy

        frozen = copy.deepcopy(self)
        frozen._frozen = True
        return frozen

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._frozen or self._components is None or self._mean is None:
            raise RuntimeError("PCACompression 未冻结：仅允许在 freeze() 后 transform")
        Z = (X.fillna(0.0).to_numpy(dtype=float) - self._mean) @ self._components
        return pd.DataFrame(Z, index=X.index, columns=[f"pca_{i}" for i in range(self._output_dim)])

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def output_dim(self) -> int:
        return self._output_dim


class RandomProjectionCompression:
    """随机投影压缩（高斯随机矩阵，维度有保证的近似保距映射）。"""

    def __init__(self, output_dim: int, seed: int = 42) -> None:
        self._output_dim = output_dim
        self.seed = seed
        self._fitted: bool = False
        self._frozen: bool = False
        self._matrix: np.ndarray | None = None

    def fit(self, X_train: pd.DataFrame) -> "RandomProjectionCompression":
        if self._frozen:
            raise RuntimeError("RandomProjectionCompression 已冻结：不允许再 fit")
        rng = np.random.default_rng(self.seed)
        in_dim = X_train.shape[1]
        self._matrix = rng.normal(0.0, 1.0 / np.sqrt(self._output_dim), (in_dim, self._output_dim))
        self._fitted = True
        return self

    def freeze(self) -> "RandomProjectionCompression":
        if not self._fitted:
            raise RuntimeError("RandomProjectionCompression 未 fit，无法 freeze")
        import copy

        frozen = copy.deepcopy(self)
        frozen._frozen = True
        return frozen

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._frozen or self._matrix is None:
            raise RuntimeError("RandomProjectionCompression 未冻结：仅允许在 freeze() 后 transform")
        Z = X.fillna(0.0).to_numpy(dtype=float) @ self._matrix
        return pd.DataFrame(Z, index=X.index, columns=[f"rp_{i}" for i in range(self._output_dim)])

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def output_dim(self) -> int:
        return self._output_dim


class SparseSelectionCompression:
    """稀疏筛选压缩骨架（按稳定性/信息量筛选统计量坐标子集）。"""

    def __init__(self, output_dim: int, criterion: str = "stability") -> None:
        self._output_dim = output_dim
        self.criterion = criterion
        self._frozen: bool = False

    def fit(self, X_train: pd.DataFrame) -> "SparseSelectionCompression":
        raise NotImplementedError  # P2

    def freeze(self) -> "SparseSelectionCompression":
        raise NotImplementedError  # P2

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError  # P2

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def output_dim(self) -> int:
        return self._output_dim


# ---------------------------------------------------------------------------
# C4：分位数 one-hot + 显式交互
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantileExplicitConfig:
    """C4 显式离散编码配置。"""

    n_bins: int = 5  # 每特征分箱数
    max_interactions: int = 5  # 两两交互项数量上限（按训练期方差选 top 特征对）


class QuantileExplicitFeatures:
    """分位数分箱 one-hot + 两两显式交互（C4 对照特征变换）。

    冻结语义：``fit`` 只允许在训练窗口调用一次（重复 fit 抛异常），
    ``transform`` 仅允许在 fit 之后调用；超出训练期分布的值截断到端点 bin。
    """

    def __init__(self, config: QuantileExplicitConfig | None = None) -> None:
        self.config = config or QuantileExplicitConfig()
        self._fitted: bool = False
        self._edges: dict[str, np.ndarray] = {}
        self._pairs: list[tuple[str, str]] = []
        self._features: list[str] = []

    def fit(self, X_train: pd.DataFrame) -> "QuantileExplicitFeatures":
        if self._fitted:
            raise RuntimeError("QuantileExplicitFeatures 已冻结：不允许重复 fit")
        X = X_train.dropna()
        self._features = list(X.columns)
        qs = np.linspace(0.0, 1.0, self.config.n_bins + 1)[1:-1]
        for col in self._features:
            edges = np.quantile(X[col].to_numpy(), qs)
            self._edges[col] = np.unique(edges)  # 去重防止退化分布
        # 交互对：按训练期方差排序的特征依次取对，不超过上限
        ranked = list(X.var().sort_values(ascending=False).index)
        self._pairs = list(combinations(ranked, 2))[: self.config.max_interactions]
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("QuantileExplicitFeatures 未 fit")
        X2 = X.reindex(columns=self._features)
        n = len(X2)
        cols: dict[str, np.ndarray] = {}
        bin_idx: dict[str, np.ndarray] = {}
        for col in self._features:
            edges = self._edges[col]
            idx = np.digitize(X2[col].fillna(X2[col].median()).to_numpy(), edges)  # 越界自动落到端点 bin
            bin_idx[col] = idx
            for k in range(len(edges) + 1):
                cols[f"{col}__bin{k}"] = (idx == k).astype(float)
        center = self.config.n_bins / 2.0
        for f, g in self._pairs:
            cols[f"{f}__x__{g}"] = (bin_idx[f] - center) * (bin_idx[g] - center)
        out = pd.DataFrame(cols, index=X2.index)
        assert len(out) == n
        return out

    @property
    def is_frozen(self) -> bool:
        return self._fitted


# ---------------------------------------------------------------------------
# C5：等维经典随机特征（RFF / Nyström）
# ---------------------------------------------------------------------------


class ClassicalRandomFeatures(Protocol):
    """等维经典随机特征抽象（C5 对照用）。

    ``target_dim`` 必须与对照的 GBS 特征维度一致（等维对齐），
    以隔离"随机升维本身"的贡献。fit 仅训练期。
    """

    def fit(self, X_train: pd.DataFrame) -> "ClassicalRandomFeatures":
        ...

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        ...

    @property
    def target_dim(self) -> int:
        ...


@dataclass(frozen=True)
class RFFConfig:
    """Random Fourier Features 配置。"""

    target_dim: int  # 输出维度（与 GBS 特征等维对齐）
    gamma: float = 1.0  # RBF 核带宽参数
    seed: int = 42


class RFFeatures:
    """Random Fourier Features（C5）：z(x) = √(2/D)·cos(Wx + b)，W~N(0, 2γ)。"""

    def __init__(self, config: RFFConfig) -> None:
        self.config = config
        self._W: np.ndarray | None = None
        self._b: np.ndarray | None = None
        self._features: list[str] = []

    def fit(self, X_train: pd.DataFrame) -> "RFFeatures":
        rng = np.random.default_rng(self.config.seed)
        d = X_train.shape[1]
        dim = self.config.target_dim
        self._W = rng.normal(0.0, np.sqrt(2.0 * self.config.gamma), (d, dim))
        self._b = rng.uniform(0.0, 2.0 * np.pi, dim)
        self._features = list(X_train.columns)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self._W is None or self._b is None:
            raise RuntimeError("RFFeatures 未 fit")
        Xv = X.reindex(columns=self._features).fillna(0.0).to_numpy(dtype=float)
        return np.sqrt(2.0 / self.config.target_dim) * np.cos(Xv @ self._W + self._b)

    @property
    def target_dim(self) -> int:
        return self.config.target_dim


@dataclass(frozen=True)
class NystromConfig:
    """Nyström 随机特征配置。"""

    target_dim: int  # 输出维度（与 GBS 特征等维对齐）
    n_landmarks: int = 200  # 地标点数
    kernel: str = "rbf"
    gamma: float = 1.0  # RBF 核带宽
    seed: int = 42


class NystromFeatures:
    """Nyström 随机特征（C5）：z(x) = k(x, L)·K_LL^{-1/2} 的前 target_dim 个主成分。"""

    def __init__(self, config: NystromConfig) -> None:
        self.config = config
        self._landmarks: np.ndarray | None = None
        self._basis: np.ndarray | None = None
        self._features: list[str] = []

    def _kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.config.kernel != "rbf":
            raise NotImplementedError(f"P1 仅支持 rbf 核，收到 {self.config.kernel!r}")
        d2 = (a**2).sum(axis=1)[:, None] + (b**2).sum(axis=1)[None, :] - 2.0 * a @ b.T
        return np.exp(-self.config.gamma * np.maximum(d2, 0.0))

    def fit(self, X_train: pd.DataFrame) -> "NystromFeatures":
        X = X_train.dropna()
        if len(X) < self.config.n_landmarks:
            raise ValueError("训练样本数小于地标点数，无法 Nyström 拟合")
        landmarks = X.sample(self.config.n_landmarks, random_state=self.config.seed)
        self._landmarks = landmarks.to_numpy(dtype=float)
        self._features = list(X.columns)
        K = self._kernel(self._landmarks, self._landmarks)
        eigvals, eigvecs = np.linalg.eigh((K + K.T) / 2.0)
        order = np.argsort(-eigvals)[: self.config.target_dim]
        eigvals = np.clip(eigvals[order], 1e-12, None)
        self._basis = eigvecs[:, order] / np.sqrt(eigvals)  # K^{-1/2} 的前 target_dim 列
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self._landmarks is None or self._basis is None:
            raise RuntimeError("NystromFeatures 未 fit")
        Xv = X.reindex(columns=self._features).fillna(0.0).to_numpy(dtype=float)
        return self._kernel(Xv, self._landmarks) @ self._basis

    @property
    def target_dim(self) -> int:
        return self.config.target_dim
