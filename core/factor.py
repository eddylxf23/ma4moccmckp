"""
factor.py — Factor 类（item 数据封装）

从原始 original/src/utils/factor.py 迁移并精简，增加初始化统计量的便捷方法。
"""

from __future__ import annotations
import numpy as np

_MAX_K = 10


class Factor:
    """
    表示 MO-CCMCKP 中一个 item 的全部信息：
    - 成本 (cost)
    - 价值 (value = max_cost - cost)
    - 随机权重样本 (samples)
    - 权重 (weight = sample_mean + λ·sample_std)
    - 效用 (utility = value / weight)
    """

    def __init__(
        self,
        factor_id: int,
        node_id: int,
        order_id: int,
        cost: float,
        value: float,
        sample_filepath: str,
        big_sample_filepath: str | None = None,
    ):
        self.factor_id = factor_id
        self.node_id = node_id
        self.order_id = order_id
        self.cost = cost
        self.value = value
        self.sample_filepath = sample_filepath
        self.big_sample_filepath = big_sample_filepath

        # 懒加载
        self._samples: np.ndarray | None = None
        self._big_samples: np.ndarray | None = None

        # 缓存的统计量
        self._sample_min: float | None = None
        self._sample_max: float | None = None
        self._sample_mean: float | None = None
        self._sample_var: float | None = None
        self._sample_std: float | None = None
        self._moments_min: np.ndarray | None = None
        self._moments_origin: np.ndarray | None = None

        self.weight: float | None = None
        self.utility: float | None = None
        self.increment: float | None = None

    # ── 样本访问 ──────────────────────────────────────────────────────────────

    @property
    def samples(self) -> np.ndarray:
        if self._samples is None:
            self._samples = np.loadtxt(self.sample_filepath)
        return self._samples

    @property
    def big_samples(self) -> np.ndarray:
        if self._big_samples is None:
            if self.big_sample_filepath is None:
                raise ValueError(f"Factor {self.factor_id} 没有大规模样本文件。")
            self._big_samples = np.loadtxt(self.big_sample_filepath)
        return self._big_samples

    # ── 统计量属性 ────────────────────────────────────────────────────────────

    @property
    def sample_min(self) -> float:
        if self._sample_min is None:
            self._sample_min = float(np.min(self.samples))
        return self._sample_min

    @property
    def sample_max(self) -> float:
        if self._sample_max is None:
            self._sample_max = float(np.max(self.samples))
        return self._sample_max

    @property
    def sample_mean(self) -> float:
        if self._sample_mean is None:
            self._sample_mean = float(np.mean(self.samples))
        return self._sample_mean

    @property
    def sample_var(self) -> float:
        if self._sample_var is None:
            self._sample_var = float(np.var(self.samples))
        return self._sample_var

    @property
    def sample_std(self) -> float:
        if self._sample_std is None:
            self._sample_std = float(np.std(self.samples, ddof=1))
        return self._sample_std

    @property
    def moments_min(self) -> np.ndarray:
        """减去最小值后的 1~K 阶矩向量。"""
        if self._moments_min is None:
            self._moments_min = self._compute_moments_min(_MAX_K)
        return self._moments_min

    @property
    def moments_origin(self) -> np.ndarray:
        """1~K 阶原点矩向量。"""
        if self._moments_origin is None:
            self._moments_origin = self._compute_moments_origin(_MAX_K)
        return self._moments_origin

    # ── 采样方法 ──────────────────────────────────────────────────────────────

    def resample_once(self) -> np.ndarray:
        return self.resample_multi(1)

    def resample_multi(self, n: int) -> np.ndarray:
        idx = np.random.randint(0, len(self.samples), n)
        return self.samples[idx]

    def get_index_sample(self, l: int) -> float:
        """返回样本中第 l 大的值（0 为最大）。"""
        return float(np.sort(self.samples)[::-1][l])

    # ── Setter ────────────────────────────────────────────────────────────────

    def set_weight(self, weight: float) -> None:
        self.weight = weight

    def set_utility(self, utility: float) -> None:
        self.utility = utility

    def set_increment(self, increment: float) -> None:
        self.increment = increment

    # ── 便捷初始化 ────────────────────────────────────────────────────────────

    def _init_stats(self, lambda_param: float = 3.0) -> None:
        """预计算权重和效用，避免求解时重复 I/O。"""
        self.set_weight(self.sample_mean + lambda_param * self.sample_std)
        if self.weight and self.weight > 0:
            self.set_utility(self.value / self.weight)
        else:
            self.set_utility(0.0)

    # ── 矩计算私有方法 ────────────────────────────────────────────────────────

    def _compute_moments_min(self, K: int) -> np.ndarray:
        s = self.samples
        aux = np.ones((2, len(s)))
        moments = np.ones(K + 1)
        for k in range(1, K + 1):
            cur = k % 2
            prev = 1 - cur
            aux[cur] = (s - self.sample_min) * aux[prev]
            moments[k] = float(np.mean(aux[cur]))
        return moments

    def _compute_moments_origin(self, K: int) -> np.ndarray:
        s = self.samples
        aux = np.ones((2, len(s)))
        moments = np.ones(K + 1)
        for k in range(1, K + 1):
            cur = k % 2
            prev = 1 - cur
            aux[cur] = s * aux[prev]
            moments[k] = float(np.mean(aux[cur]))
        return moments

    def __repr__(self) -> str:
        return (
            f"Factor(id={self.factor_id}, node={self.node_id}, order={self.order_id}, "
            f"cost={self.cost:.3f}, value={self.value:.3f})"
        )
