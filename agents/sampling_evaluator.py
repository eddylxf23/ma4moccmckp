"""
sampling_evaluator.py — 采样评估 Agent

职责：
  - 生成多样化的初始种群（贪心 + 随机扰动策略）
  - 批量评估候选解的目标值（成本和置信度）
  - 执行 DDALS 采样扰动操作（degrade）
  - 支持并行采样（可扩展）

核心算法：construct_diverse_initial_population
  - 先生成多种贪心初始解（按 value/weight/utility 排序）
  - 再随机扰动生成多样性种群
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agents.base_agent import BaseAgent, AgentMessage, MessageType
from core.solution import Solution


class SamplingEvaluatorAgent(BaseAgent):
    """
    采样评估 Agent。

    响应消息：
    - TASK_START → 生成初始种群 → 发出 INITIAL_POPULATION
    - SCHEDULE_REQUEST (op="sample") → 采样新解 → 发出 POPULATION_RESULT
    - SCHEDULE_REQUEST (op="evaluate") → 批量评估 → 发出 EVALUATION_RESULT
    """

    NAME = "SamplingEvaluator"

    def __init__(
        self,
        problem: Any = None,
        shared_memory: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(self.NAME, problem, shared_memory, config)
        # 从配置中读取默认参数
        self.degrade_prob: float = self.config.get("degrade_prob", 0.1)

    # ── 核心接口实现 ──────────────────────────────────────────────────────────

    def process_message(self, msg: AgentMessage) -> List[AgentMessage]:
        if msg.msg_type == MessageType.TASK_START:
            pop_size = msg.content.get("pop_size", 50)
            constraint_report = msg.content.get("constraint_report", {})
            # 从约束报告中读取建议的种群大小
            if "recommended" in constraint_report:
                pop_size = constraint_report["recommended"].get("pop_size", pop_size)

            pop = self.generate_initial_population(pop_size)
            return [self.broadcast(
                MessageType.INITIAL_POPULATION,
                content={
                    "population": [s.to_dict() for s in pop],
                    "pop_size": len(pop),
                },
                reply_to=msg.id,
            )]

        if msg.msg_type == MessageType.SCHEDULE_REQUEST:
            op = msg.content.get("op")
            if op == "sample":
                return self._handle_sample_request(msg)
            if op == "evaluate":
                return self._handle_evaluate_request(msg)
            if op == "degrade":
                return self._handle_degrade_request(msg)

        return []

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": self.NAME,
            "description": "生成初始种群、批量评估候选解、执行 degrade 扰动操作。",
            "supported_message_types": [
                MessageType.TASK_START,
                MessageType.SCHEDULE_REQUEST,
            ],
            "output_types": [
                MessageType.INITIAL_POPULATION,
                MessageType.POPULATION_RESULT,
                MessageType.EVALUATION_RESULT,
            ],
            "supported_ops": ["sample", "evaluate", "degrade"],
        }

    # ── 核心方法 ──────────────────────────────────────────────────────────────

    def generate_initial_population(
        self,
        pop_size: int = 50,
        strategy: str = "hybrid",
    ) -> List[Solution]:
        """
        生成多样化初始种群。

        Parameters
        ----------
        pop_size : int
            种群大小。
        strategy : str
            ``"hybrid"``（贪心+随机扰动）| ``"random"``（纯随机）。

        Returns
        -------
        List[Solution]
            已评估的初始种群。
        """
        if self.problem is None:
            raise ValueError("SamplingEvaluatorAgent: 未绑定 problem 对象。")

        population: List[Solution] = []

        if strategy == "hybrid":
            # 每种排序策略生成若干贪心解并评估
            greedy_per_key = max(1, pop_size // 8)
            for key in ["value", "weight", "utility"]:
                for _ in range(greedy_per_key):
                    sol = self.problem.greedy_solution(key)
                    # 随机扰动以增加多样性
                    sol = self._apply_random_degrade(sol, degrade_rate=0.3)
                    self.problem.evaluate(sol)
                    sol.metadata["source"] = f"greedy_{key}_degrade"
                    population.append(sol)

        # 补充纯随机解
        while len(population) < pop_size:
            sol = self.problem.random_solution()
            self.problem.evaluate(sol)
            sol.metadata["source"] = "random"
            population.append(sol)

        self._log(
            f"初始种群生成：size={len(population)}, "
            f"feasible={sum(1 for s in population if s.is_feasible)}"
        )
        return population

    def evaluate_solution(self, x: np.ndarray) -> Solution:
        """评估单个解向量，返回完整 Solution 对象。"""
        sol = Solution(x=x.copy())
        self.problem.evaluate(sol)
        return sol

    def evaluate_population(
        self, population: List[np.ndarray]
    ) -> List[Solution]:
        """批量评估解向量列表。"""
        solutions = [Solution(x=x.copy()) for x in population]
        return self.problem.evaluate_batch(solutions)

    def apply_degrade(
        self,
        solutions: List[Solution],
        degrade_prob: Optional[float] = None,
    ) -> List[Solution]:
        """
        对种群应用 degrade 扰动（随机改变某节点的 item 选择）。

        Parameters
        ----------
        degrade_prob : float
            每个解被扰动的概率。

        Returns
        -------
        List[Solution]
            扰动后的新解列表（已评估）。
        """
        prob = degrade_prob or self.degrade_prob
        new_solutions: List[Solution] = []
        for sol in solutions:
            if np.random.rand() < prob:
                new_sol = self._apply_random_degrade(sol)
                self.problem.evaluate(new_sol)
                new_solutions.append(new_sol)
        return new_solutions

    # ── 私有方法 ──────────────────────────────────────────────────────────────

    def _apply_random_degrade(
        self,
        sol: Solution,
        degrade_rate: float = 1.0,
    ) -> Solution:
        """
        随机扰动解：以 degrade_rate 概率替换一个节点的 item。

        Parameters
        ----------
        degrade_rate : float
            扰动程度，0~1（0=不扰动，1=必然扰动一个节点）。
        """
        new_x = sol.x.copy()
        if np.random.rand() < max(degrade_rate, 0.1):
            idx = np.random.randint(0, self.problem.m)
            new_item = np.random.randint(0, self.problem.n)
            new_x[idx] = new_item
        new_sol = Solution(x=new_x, metadata={"source": "degrade"})
        return new_sol

    def _handle_sample_request(self, msg: AgentMessage) -> List[AgentMessage]:
        count = msg.content.get("count", 10)
        pop = self.generate_initial_population(count, strategy="random")
        return [self.send(
            MessageType.POPULATION_RESULT,
            receiver=msg.sender,
            content={"population": [s.to_dict() for s in pop]},
            reply_to=msg.id,
        )]

    def _handle_evaluate_request(self, msg: AgentMessage) -> List[AgentMessage]:
        raw_pop = msg.content.get("population", [])
        solutions = [Solution.from_dict(d) for d in raw_pop]
        self.problem.evaluate_batch(solutions)
        return [self.send(
            MessageType.EVALUATION_RESULT,
            receiver=msg.sender,
            content={"population": [s.to_dict() for s in solutions]},
            reply_to=msg.id,
        )]

    def _handle_degrade_request(self, msg: AgentMessage) -> List[AgentMessage]:
        raw_pop = msg.content.get("population", [])
        solutions = [Solution.from_dict(d) for d in raw_pop]
        new_pop = self.apply_degrade(solutions)
        return [self.send(
            MessageType.POPULATION_RESULT,
            receiver=msg.sender,
            content={"population": [s.to_dict() for s in new_pop]},
            reply_to=msg.id,
        )]
