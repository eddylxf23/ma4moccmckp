"""
pareto_manager.py — Pareto 维护 Agent

职责：
  - 维护当前非支配解集（Pareto 前沿）
  - 增量式更新：接收新解，过滤支配关系，保持前沿最新
  - 计算多样性指标（拥挤距离）
  - 响应前沿查询请求
  - 记录每轮迭代的前沿演变历史
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from agents.base_agent import BaseAgent, AgentMessage, MessageType
from core.solution import Solution


class ParetoManagerAgent(BaseAgent):
    """
    Pareto 维护 Agent。

    内部维护一个非支配解集合，提供增量更新和查询功能。

    响应消息：
    - INITIAL_POPULATION   → 初始化前沿 → 广播 PARETO_FRONT
    - POPULATION_RESULT    → 增量更新前沿 → 广播 PARETO_UPDATE
    - REPAIRED_POPULATION  → 同 POPULATION_RESULT
    - SCHEDULE_REQUEST (op="query_front") → 返回当前前沿
    """

    NAME = "ParetoManager"

    def __init__(
        self,
        problem: Any = None,
        shared_memory: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(self.NAME, problem, shared_memory, config)
        self._front: List[Solution] = []          # 当前 Pareto 前沿
        self._history: List[Dict[str, Any]] = []  # 迭代历史
        self._update_count: int = 0

    # ── 核心接口实现 ──────────────────────────────────────────────────────────

    def process_message(self, msg: AgentMessage) -> List[AgentMessage]:
        if msg.msg_type == MessageType.INITIAL_POPULATION:
            raw_pop = msg.content.get("population", [])
            solutions = [Solution.from_dict(d) for d in raw_pop]
            self.initialize(solutions)
            return [self.broadcast(
                MessageType.PARETO_FRONT,
                content=self._front_to_dict(),
                reply_to=msg.id,
            )]

        if msg.msg_type in (
            MessageType.POPULATION_RESULT,
            MessageType.REPAIRED_POPULATION,
            MessageType.EVALUATION_RESULT,
        ):
            raw_pop = msg.content.get("population", [])
            solutions = [Solution.from_dict(d) for d in raw_pop]
            added, removed = self.update(solutions)
            return [self.broadcast(
                MessageType.PARETO_UPDATE,
                content={
                    **self._front_to_dict(),
                    "added": added,
                    "removed": removed,
                },
                reply_to=msg.id,
            )]

        if msg.msg_type == MessageType.SCHEDULE_REQUEST:
            if msg.content.get("op") == "query_front":
                return [self.send(
                    MessageType.PARETO_FRONT,
                    receiver=msg.sender,
                    content=self._front_to_dict(),
                    reply_to=msg.id,
                )]

        return []

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": self.NAME,
            "description": "维护 Pareto 前沿，增量更新非支配解集，提供前沿查询接口。",
            "supported_message_types": [
                MessageType.INITIAL_POPULATION,
                MessageType.POPULATION_RESULT,
                MessageType.REPAIRED_POPULATION,
                MessageType.SCHEDULE_REQUEST,
            ],
            "output_types": [MessageType.PARETO_FRONT, MessageType.PARETO_UPDATE],
        }

    # ── 前沿管理方法 ──────────────────────────────────────────────────────────

    def initialize(self, solutions: List[Solution]) -> None:
        """用初始种群初始化 Pareto 前沿。"""
        self._front = []
        self._update_count = 0
        self._history.clear()
        self.update(solutions)
        self._log(f"Pareto 前沿初始化：front_size={len(self._front)}")

    def update(
        self, new_solutions: List[Solution]
    ) -> tuple[int, int]:
        """
        增量更新 Pareto 前沿。

        Parameters
        ----------
        new_solutions : List[Solution]
            新候选解列表（可包含不可行解）。

        Returns
        -------
        (added, removed) : Tuple[int, int]
            加入前沿的解数 / 被淘汰的解数。
        """
        # 仅处理已评估的解
        candidates = [s for s in new_solutions if s.cost is not None]
        if not candidates:
            return 0, 0

        old_size = len(self._front)
        combined = self._front + candidates
        new_front = self._compute_nondominated(combined)

        added = sum(1 for s in new_front if any(np.array_equal(s.x, c.x) for c in candidates))
        removed = old_size + len(candidates) - len(new_front) - (len(candidates) - added)

        self._front = new_front
        self._update_count += 1

        # 记录历史快照
        self._history.append({
            "update": self._update_count,
            "front_size": len(self._front),
            "added": added,
        })

        if added > 0:
            self._log(
                f"Pareto 前沿更新 #{self._update_count}："
                f"size={len(self._front)}, +{added}, -{removed}"
            )
        return added, removed

    @property
    def front(self) -> List[Solution]:
        """当前 Pareto 前沿（只读副本）。"""
        return list(self._front)

    @property
    def history(self) -> List[Dict[str, Any]]:
        """迭代历史记录（只读副本）。"""
        return list(self._history)

    @property
    def front_size(self) -> int:
        return len(self._front)

    @property
    def feasible_front(self) -> List[Solution]:
        """只返回可行解组成的前沿。"""
        return [s for s in self._front if s.is_feasible]

    # ── 前沿分析方法 ──────────────────────────────────────────────────────────

    def crowding_distances(self) -> List[float]:
        """计算前沿中每个解的拥挤距离。"""
        if len(self._front) <= 2:
            return [float("inf")] * len(self._front)
        return self._compute_crowding_distance(self._front)

    def best_feasible(self) -> Optional[Solution]:
        """返回前沿中成本最低的可行解。"""
        feasible = self.feasible_front
        if not feasible:
            return None
        return min(feasible, key=lambda s: s.cost)  # type: ignore[arg-type]

    def spread(self) -> float:
        """计算前沿的 Spread 指标（分布均匀性）。"""
        feasible = self.feasible_front
        if len(feasible) < 2:
            return 0.0
        costs = sorted(s.cost for s in feasible)  # type: ignore[misc]
        confs = sorted(s.confidence for s in feasible)  # type: ignore[misc]
        cost_range = costs[-1] - costs[0] if costs else 1.0
        conf_range = confs[-1] - confs[0] if confs else 1.0
        return float((cost_range ** 2 + conf_range ** 2) ** 0.5)

    # ── 私有方法 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_nondominated(solutions: List[Solution]) -> List[Solution]:
        """计算非支配解集（O(n²) 实现，适用于小到中等规模）。"""
        if not solutions:
            return []
        # 过滤未评估的解
        evaluated = [s for s in solutions if s.cost is not None and s.confidence is not None]
        if not evaluated:
            return []

        front: List[Solution] = []
        for s in evaluated:
            dominated = False
            new_front = []
            for f in front:
                if f.dominates(s):
                    dominated = True
                    new_front.append(f)
                elif not s.dominates(f):
                    new_front.append(f)
                # s 支配 f → f 从 front 中移除
            if not dominated:
                new_front.append(s)
            front = new_front
        return front

    @staticmethod
    def _compute_crowding_distance(solutions: List[Solution]) -> List[float]:
        """NSGA-II 拥挤距离计算。"""
        n = len(solutions)
        distances = [0.0] * n

        for obj_idx, key in enumerate(["cost", "confidence"]):
            values = [getattr(s, key) for s in solutions]
            sorted_idx = np.argsort(values)
            distances[sorted_idx[0]] = float("inf")
            distances[sorted_idx[-1]] = float("inf")
            vmin, vmax = values[sorted_idx[0]], values[sorted_idx[-1]]
            if vmax - vmin < 1e-10:
                continue
            for i in range(1, n - 1):
                distances[sorted_idx[i]] += (
                    (values[sorted_idx[i + 1]] - values[sorted_idx[i - 1]])
                    / (vmax - vmin)
                )
        return distances

    def _front_to_dict(self) -> Dict[str, Any]:
        """将 Pareto 前沿序列化为字典。"""
        return {
            "front": [s.to_dict() for s in self._front],
            "front_size": len(self._front),
            "feasible_count": len(self.feasible_front),
            "update_count": self._update_count,
            "spread": self.spread(),
        }
