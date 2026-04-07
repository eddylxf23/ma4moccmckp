"""
solution_repairer.py — 解修复 Agent

职责：
  - 修复不可行解（置信度不满足约束的解）
  - 实现 DDALS 中的 local_swap_search 和 further_swap_search
  - 在修复过程中保持或改善成本目标
  - 支持策略切换：贪心修复 / 局部搜索修复

核心算法（基于 DDALS）：
  local_swap_search:  单节点邻近 item 替换，同时优化成本和置信度
  further_swap_search: 双节点组合搜索，4个改进方向
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agents.base_agent import BaseAgent, AgentMessage, MessageType
from core.solution import Solution
from core.evaluator import Evaluator


class SolutionRepairerAgent(BaseAgent):
    """
    解修复 Agent。

    响应消息：
    - PARETO_UPDATE / PARETO_FRONT → 触发对不可行解的修复 → 发出 REPAIRED_POPULATION
    - SCHEDULE_REQUEST (op="local_search") → 执行局部搜索
    - SCHEDULE_REQUEST (op="deep_search")  → 执行深度搜索
    """

    NAME = "SolutionRepairer"

    def __init__(
        self,
        problem: Any = None,
        shared_memory: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(self.NAME, problem, shared_memory, config)
        self.local_search_prob: float = self.config.get("local_search_prob", 0.1)
        self.deep_search_prob: float = self.config.get("deep_search_prob", 0.1)

    # ── 核心接口实现 ──────────────────────────────────────────────────────────

    def process_message(self, msg: AgentMessage) -> List[AgentMessage]:
        if msg.msg_type == MessageType.CONSTRAINT_REPORT:
            # 更新搜索参数
            rec = msg.content.get("recommended", {})
            self.local_search_prob = rec.get("local_search_prob", self.local_search_prob)
            self.deep_search_prob = rec.get("deep_search_prob", self.deep_search_prob)
            return []

        if msg.msg_type in (MessageType.POPULATION_RESULT, MessageType.INITIAL_POPULATION):
            raw_pop = msg.content.get("population", [])
            solutions = [Solution.from_dict(d) for d in raw_pop]
            repaired = self.repair_population(solutions)
            return [self.broadcast(
                MessageType.REPAIRED_POPULATION,
                content={"population": [s.to_dict() for s in repaired]},
                reply_to=msg.id,
            )]

        if msg.msg_type == MessageType.SCHEDULE_REQUEST:
            op = msg.content.get("op")
            if op == "local_search":
                return self._handle_local_search(msg)
            if op == "deep_search":
                return self._handle_deep_search(msg)
            if op == "repair":
                return self._handle_repair(msg)

        return []

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": self.NAME,
            "description": "修复不可行解，执行 local_swap_search 和 further_swap_search 局部搜索。",
            "supported_message_types": [
                MessageType.POPULATION_RESULT,
                MessageType.INITIAL_POPULATION,
                MessageType.SCHEDULE_REQUEST,
            ],
            "output_types": [MessageType.REPAIRED_POPULATION],
            "supported_ops": ["local_search", "deep_search", "repair"],
        }

    # ── 修复方法 ──────────────────────────────────────────────────────────────

    def repair_population(
        self,
        solutions: List[Solution],
    ) -> List[Solution]:
        """
        对种群中的不可行解执行修复。

        Strategy:
        - 不可行解 → local_swap_search（向置信度改善方向移动）
        - 可行解 → local_swap_search（向成本改善方向移动）

        Returns
        -------
        List[Solution]
            修复后的所有新解（包括原解修复产生的改进解）。
        """
        all_new: List[Solution] = []
        for sol in solutions:
            if not sol.is_feasible:
                new_sols = self.local_swap_search(sol, prefer_confidence=True)
                all_new.extend(new_sols)
        return all_new

    def local_swap_search(
        self,
        sol: Solution,
        prefer_confidence: bool = False,
    ) -> List[Solution]:
        """
        单节点邻近 item 局部搜索（DDALS local_swap_search）。

        策略：遍历每个节点，尝试替换为邻近的 item：
          - cost 改善方向：选价值更高（成本更低）的 item
          - confidence 改善方向：选权重更小的 item

        Parameters
        ----------
        prefer_confidence : bool
            True 时优先改善置信度（适合修复不可行解）。

        Returns
        -------
        List[Solution]
            找到的所有改进解（已评估）。
        """
        if self.problem is None:
            return []

        ps = self.problem.ps
        nr = self.problem.nr
        CL = self.problem.CL
        m = self.problem.m

        current_sol = sol.x.copy()
        current_cost = sol.cost if sol.cost is not None else Evaluator.compute_cost(current_sol, ps)
        current_conf = sol.confidence if sol.confidence is not None else 0.0

        improved: List[Solution] = []
        class_order = list(range(m))
        random.shuffle(class_order)

        for cls in class_order:
            original_item = current_sol[cls]

            # ── 成本优化方向（价值降序排列中，往高价值移动）──────────────────
            cost_order = nr["value"][cls]
            if original_item in cost_order:
                ci = cost_order.index(original_item)
                if ci > 0:
                    candidate = cost_order[ci - 1]
                    new_x = current_sol.copy()
                    new_x[cls] = candidate
                    new_cost = Evaluator.compute_cost(new_x, ps)
                    new_conf = ps["eval_func"](new_x, ps)
                    if self._is_better(new_cost, new_conf, current_cost, current_conf, CL):
                        new_sol = Solution(x=new_x, cost=new_cost, confidence=new_conf,
                                           metadata={"source": "local_swap_cost"})
                        new_sol.update_evaluation(new_cost, new_conf, CL)
                        improved.append(new_sol)

            # ── 置信度优化方向（权重升序排列中，往低权重移动）───────────────
            conf_order = nr["weight"][cls]
            if original_item in conf_order:
                wi = conf_order.index(original_item)
                if wi < len(conf_order) - 1:
                    candidate = conf_order[wi + 1]
                    new_x = current_sol.copy()
                    new_x[cls] = candidate
                    new_cost = Evaluator.compute_cost(new_x, ps)
                    new_conf = ps["eval_func"](new_x, ps)
                    if prefer_confidence:
                        better = new_conf > current_conf or (
                            new_conf >= CL and current_conf < CL
                        )
                    else:
                        better = self._is_better(new_cost, new_conf, current_cost, current_conf, CL)
                    if better:
                        new_sol = Solution(x=new_x, cost=new_cost, confidence=new_conf,
                                           metadata={"source": "local_swap_conf"})
                        new_sol.update_evaluation(new_cost, new_conf, CL)
                        improved.append(new_sol)

        return improved

    def further_swap_search(
        self,
        sol: Solution,
    ) -> List[Solution]:
        """
        双节点组合搜索（DDALS further_swap_search）。

        尝试同时替换两个节点的 item，探索 4 个方向：
          cost↓cost↓ / conf↑conf↑ / cost↓conf↑ / conf↑cost↓

        Returns
        -------
        List[Solution]
            找到的所有改进解（已评估）。
        """
        if self.problem is None:
            return []

        ps = self.problem.ps
        nr = self.problem.nr
        CL = self.problem.CL
        m = self.problem.m

        current_sol = sol.x.copy()
        current_cost = sol.cost if sol.cost is not None else Evaluator.compute_cost(current_sol, ps)
        current_conf = sol.confidence if sol.confidence is not None else 0.0

        improved: List[Solution] = []
        nodes = list(range(m))
        random.shuffle(nodes)

        # 取前 min(m, 6) 个节点进行双节点组合（控制计算量）
        for i_idx in range(min(len(nodes), 6)):
            cls_i = nodes[i_idx]
            for j_idx in range(i_idx + 1, min(len(nodes), 6)):
                cls_j = nodes[j_idx]

                # 4 个方向
                directions: List[Tuple[int, int, str, str]] = []
                ci = nr["value"][cls_i].index(current_sol[cls_i]) if current_sol[cls_i] in nr["value"][cls_i] else -1
                cj = nr["value"][cls_j].index(current_sol[cls_j]) if current_sol[cls_j] in nr["value"][cls_j] else -1
                wi = nr["weight"][cls_i].index(current_sol[cls_i]) if current_sol[cls_i] in nr["weight"][cls_i] else -1
                wj = nr["weight"][cls_j].index(current_sol[cls_j]) if current_sol[cls_j] in nr["weight"][cls_j] else -1

                candidates = []
                if ci > 0 and cj > 0:
                    candidates.append((nr["value"][cls_i][ci - 1], nr["value"][cls_j][cj - 1]))
                if wi < len(nr["weight"][cls_i]) - 1 and wj < len(nr["weight"][cls_j]) - 1:
                    candidates.append((nr["weight"][cls_i][wi + 1], nr["weight"][cls_j][wj + 1]))
                if ci > 0 and wj < len(nr["weight"][cls_j]) - 1:
                    candidates.append((nr["value"][cls_i][ci - 1], nr["weight"][cls_j][wj + 1]))
                if wi < len(nr["weight"][cls_i]) - 1 and cj > 0:
                    candidates.append((nr["weight"][cls_i][wi + 1], nr["value"][cls_j][cj - 1]))

                for item_i, item_j in candidates:
                    new_x = current_sol.copy()
                    new_x[cls_i] = item_i
                    new_x[cls_j] = item_j
                    new_cost = Evaluator.compute_cost(new_x, ps)
                    new_conf = ps["eval_func"](new_x, ps)
                    if self._is_better(new_cost, new_conf, current_cost, current_conf, CL):
                        new_sol = Solution(x=new_x, cost=new_cost, confidence=new_conf,
                                           metadata={"source": "further_swap"})
                        new_sol.update_evaluation(new_cost, new_conf, CL)
                        improved.append(new_sol)

        return improved

    # ── 私有方法 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _is_better(
        new_cost: float, new_conf: float,
        cur_cost: float, cur_conf: float,
        CL: float,
    ) -> bool:
        """多目标解比较：支配或约束改善。"""
        new_feasible = new_conf >= CL
        cur_feasible = cur_conf >= CL
        if new_feasible and not cur_feasible:
            return True
        if not new_feasible and cur_feasible:
            return False
        if new_cost <= cur_cost and new_conf > cur_conf:
            return True
        if new_cost < cur_cost and new_conf >= cur_conf:
            return True
        return False

    def _handle_local_search(self, msg: AgentMessage) -> List[AgentMessage]:
        raw_pop = msg.content.get("population", [])
        solutions = [Solution.from_dict(d) for d in raw_pop]
        all_new: List[Solution] = []
        for sol in solutions:
            if np.random.rand() < self.local_search_prob:
                all_new.extend(self.local_swap_search(sol))
        return [self.send(
            MessageType.REPAIRED_POPULATION,
            receiver=msg.sender,
            content={"population": [s.to_dict() for s in all_new]},
            reply_to=msg.id,
        )]

    def _handle_deep_search(self, msg: AgentMessage) -> List[AgentMessage]:
        raw_pop = msg.content.get("population", [])
        solutions = [Solution.from_dict(d) for d in raw_pop]
        # 只对精英解执行深度搜索
        elite_size = max(2, len(solutions) // 10)
        elites = sorted(
            [s for s in solutions if s.cost is not None],
            key=lambda s: s.cost  # type: ignore[arg-type]
        )[:elite_size]
        all_new: List[Solution] = []
        for sol in elites:
            if np.random.rand() < self.deep_search_prob:
                all_new.extend(self.further_swap_search(sol))
        return [self.send(
            MessageType.REPAIRED_POPULATION,
            receiver=msg.sender,
            content={"population": [s.to_dict() for s in all_new]},
            reply_to=msg.id,
        )]

    def _handle_repair(self, msg: AgentMessage) -> List[AgentMessage]:
        raw_pop = msg.content.get("population", [])
        solutions = [Solution.from_dict(d) for d in raw_pop]
        repaired = self.repair_population(solutions)
        return [self.send(
            MessageType.REPAIRED_POPULATION,
            receiver=msg.sender,
            content={"population": [s.to_dict() for s in repaired]},
            reply_to=msg.id,
        )]
