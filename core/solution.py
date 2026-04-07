"""
solution.py — MO-CCMCKP 解表示

Solution 封装解向量、目标值、约束满足情况，以及与 parameter_set 的绑定。
"""

from __future__ import annotations

import copy
import time
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple

import numpy as np


class SolutionStatus(str, Enum):
    """解的状态枚举。"""
    UNEVALUATED = "unevaluated"   # 尚未评估
    FEASIBLE = "feasible"          # 可行（满足置信度约束）
    INFEASIBLE = "infeasible"      # 不可行（违反置信度约束）


class Solution:
    """
    MO-CCMCKP 解对象。

    Attributes
    ----------
    x : np.ndarray, shape (m,)
        解向量，``x[i]`` 为第 i 个节点选中 item 的 order_id（0-based）。
    cost : float | None
        确定性总成本 f₁（最小化）。
    confidence : float | None
        置信度概率 f₂（最大化），即满足容量约束的概率。
    constraint_violation : float
        约束违反量 max(0, CL - confidence)。
    status : SolutionStatus
        解的评估状态。
    created_at : float
        创建时间戳。
    metadata : dict
        附加元数据（来源 Agent、迭代次数等）。
    """

    def __init__(
        self,
        x: np.ndarray,
        cost: Optional[float] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.x: np.ndarray = np.asarray(x, dtype=int)
        self.cost: Optional[float] = cost
        self.confidence: Optional[float] = confidence
        self.constraint_violation: float = 0.0
        self.status: SolutionStatus = SolutionStatus.UNEVALUATED
        self.created_at: float = time.time()
        self.metadata: Dict[str, Any] = metadata or {}

        if cost is not None and confidence is not None:
            self._update_status()

    # ── 基础属性 ──────────────────────────────────────────────────────────────

    @property
    def m(self) -> int:
        """节点（类别）数量。"""
        return len(self.x)

    @property
    def objectives(self) -> Tuple[float, float]:
        """返回目标值对 (cost, -confidence)，均为最小化形式。"""
        if self.cost is None or self.confidence is None:
            raise ValueError("解尚未评估，无法获取目标值。")
        return (self.cost, -self.confidence)

    @property
    def is_feasible(self) -> bool:
        """是否满足置信度约束。"""
        return self.status == SolutionStatus.FEASIBLE

    # ── 评估结果更新 ──────────────────────────────────────────────────────────

    def update_evaluation(
        self,
        cost: float,
        confidence: float,
        confidence_level: float,
    ) -> None:
        """更新评估结果并自动计算约束违反量和状态。"""
        self.cost = cost
        self.confidence = confidence
        self.constraint_violation = max(0.0, confidence_level - confidence)
        self._update_status()

    def _update_status(self, confidence_level: float = 0.0) -> None:
        if self.confidence is None:
            self.status = SolutionStatus.UNEVALUATED
        elif self.constraint_violation <= 0.0:
            self.status = SolutionStatus.FEASIBLE
        else:
            self.status = SolutionStatus.INFEASIBLE

    # ── 支配关系 ──────────────────────────────────────────────────────────────

    def dominates(self, other: "Solution") -> bool:
        """
        Pareto 支配：self 支配 other。

        规则（两个目标均最小化）：
        - cost 更小或相等 AND confidence 更大或相等
        - 至少一个严格更好
        """
        if self.cost is None or other.cost is None:
            return False
        return (
            self.cost <= other.cost
            and self.confidence >= other.confidence  # type: ignore[operator]
            and (self.cost < other.cost or self.confidence > other.confidence)  # type: ignore[operator]
        )

    # ── 复制与序列化 ──────────────────────────────────────────────────────────

    def copy(self) -> "Solution":
        """深拷贝，保留评估结果但重置时间戳。"""
        new = Solution(
            x=self.x.copy(),
            cost=self.cost,
            confidence=self.confidence,
            metadata=copy.deepcopy(self.metadata),
        )
        new.constraint_violation = self.constraint_violation
        new.status = self.status
        return new

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典，方便 Agent 间消息传递。"""
        return {
            "x": self.x.tolist(),
            "cost": self.cost,
            "confidence": self.confidence,
            "constraint_violation": self.constraint_violation,
            "status": self.status.value,
            "is_feasible": self.is_feasible,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Solution":
        """从字典反序列化。"""
        sol = cls(
            x=np.array(d["x"], dtype=int),
            cost=d.get("cost"),
            confidence=d.get("confidence"),
            metadata=d.get("metadata", {}),
        )
        sol.constraint_violation = d.get("constraint_violation", 0.0)
        sol.status = SolutionStatus(d.get("status", "unevaluated"))
        return sol

    def __repr__(self) -> str:
        return (
            f"Solution(m={self.m}, cost={self.cost:.4f if self.cost else 'N/A'}, "
            f"conf={self.confidence:.4f if self.confidence else 'N/A'}, "
            f"status={self.status.value})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Solution):
            return NotImplemented
        return np.array_equal(self.x, other.x)

    def __hash__(self) -> int:
        return hash(tuple(self.x.tolist()))


# ── 工厂函数 ──────────────────────────────────────────────────────────────────

def random_solution(m: int, n: int) -> Solution:
    """生成一个随机解向量。"""
    x = np.random.randint(0, n, size=m, dtype=int)
    return Solution(x=x, metadata={"source": "random"})


def greedy_solution(parameter_set: dict, key: str = "value") -> Solution:
    """
    贪心初始化：每个节点选择按指定键排序第一的 item。

    Parameters
    ----------
    key : str
        排序键，可选 ``"value"``、``"weight"``、``"utility"``。
    """
    from core.sorting import resort_factor
    nr = resort_factor(parameter_set)
    m = parameter_set["m"]
    x = np.array([nr[key][node_id][0] for node_id in range(m)], dtype=int)
    return Solution(x=x, metadata={"source": f"greedy_{key}"})
