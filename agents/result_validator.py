"""
result_validator.py — 结果验证 Agent

职责：
  - 验证最终 Pareto 前沿中每个解的可行性（置信度 >= CL）
  - 重新精确评估所有解（使用最高精度置信度方法）
  - 计算性能指标：HV（Hypervolume）、Spread、GD 等
  - 生成最终结果报告并广播给协调器
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agents.base_agent import BaseAgent, AgentMessage, MessageType
from core.solution import Solution
from core.evaluator import Evaluator


class ResultValidatorAgent(BaseAgent):
    """
    结果验证 Agent。

    在优化结束后对最终 Pareto 前沿进行全面的质量检验。

    响应消息：
    - SCHEDULE_REQUEST (op="validate")  → 对指定解集进行验证 → VALIDATION_RESULT
    - PARETO_FRONT                      → 被动接收并缓存最新前沿
    """

    NAME = "ResultValidator"

    def __init__(
        self,
        problem: Any = None,
        shared_memory: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(self.NAME, problem, shared_memory, config)
        self._cached_front: List[Solution] = []

    # ── 核心接口实现 ──────────────────────────────────────────────────────────

    def process_message(self, msg: AgentMessage) -> List[AgentMessage]:
        """分发消息到对应处理函数。"""
        if msg.msg_type == MessageType.PARETO_FRONT:
            return self._handle_front_update(msg)
        elif msg.msg_type == MessageType.SCHEDULE_REQUEST:
            op = msg.content.get("op", "")
            if op == "validate":
                return self._handle_validate_request(msg)
        return []

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": self.NAME,
            "description": "验证 Pareto 前沿中解的可行性，计算性能指标（HV、Spacing 等），生成最终报告",
            "supported_message_types": [
                MessageType.PARETO_FRONT,
                MessageType.SCHEDULE_REQUEST,
            ],
        }

    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行独立验证任务（不依赖消息传递时使用）。"""
        solutions = task.get("solutions", [])
        return self.validate_front(solutions)

    # ── 内部处理函数 ──────────────────────────────────────────────────────────

    def _handle_front_update(self, msg: AgentMessage) -> List[AgentMessage]:
        """缓存最新 Pareto 前沿。"""
        front_data = msg.content.get("front", [])
        if front_data and isinstance(front_data[0], dict):
            self._cached_front = [Solution.from_dict(d) for d in front_data]
        elif front_data and isinstance(front_data[0], Solution):
            self._cached_front = list(front_data)
        self._log(f"缓存 Pareto 前沿，共 {len(self._cached_front)} 个解")
        return []

    def _handle_validate_request(self, msg: AgentMessage) -> List[AgentMessage]:
        """验证指定解集，返回验证报告。"""
        raw_solutions = msg.content.get("solutions", self._cached_front)
        if not raw_solutions:
            self._log("没有可验证的解", level="warning")
            return []

        if raw_solutions and isinstance(raw_solutions[0], dict):
            solutions = [Solution.from_dict(d) for d in raw_solutions]
        else:
            solutions = list(raw_solutions)

        report = self.validate_front(solutions)

        response = self.send(
            msg_type=MessageType.VALIDATION_RESULT,
            receiver=msg.sender,
            content=report,
            reply_to=msg.id,
        )
        return [response]

    # ── 公开验证方法 ──────────────────────────────────────────────────────────

    def validate_front(self, solutions: List[Solution]) -> Dict[str, Any]:
        """
        对一组解进行全面验证和指标计算。

        Parameters
        ----------
        solutions : list of Solution
            待验证的解集合。

        Returns
        -------
        dict
            包含验证结果、统计信息和性能指标的报告。
        """
        if not solutions:
            return {"status": "empty", "message": "解集为空"}

        t0 = time.time()
        evaluator: Optional[Evaluator] = (
            self.problem.evaluator if self.problem is not None else None
        )
        cl = self.problem.cl if self.problem is not None else None

        # ── 逐解验证 ──────────────────────────────────────────────────────────
        validated: List[Dict[str, Any]] = []
        feasible_count = 0
        infeasible_count = 0

        for sol in solutions:
            entry: Dict[str, Any] = {
                "solution": sol.x.tolist(),
                "cost": sol.cost,
                "confidence": sol.confidence,
                "is_feasible_cached": sol.is_feasible,
            }

            # 若有评估器，重新精确评估
            if evaluator is not None:
                cost_recalc = evaluator.evaluate_cost(sol.x)
                conf_recalc = evaluator.evaluate_confidence(sol.x)
                is_feasible = (cl is None) or (conf_recalc >= cl)
                entry.update(
                    {
                        "cost_recalculated": cost_recalc,
                        "confidence_recalculated": conf_recalc,
                        "is_feasible_recalculated": is_feasible,
                        "cost_diff": abs(sol.cost - cost_recalc),
                        "conf_diff": abs(sol.confidence - conf_recalc),
                    }
                )
                if is_feasible:
                    feasible_count += 1
                else:
                    infeasible_count += 1
            else:
                if sol.is_feasible:
                    feasible_count += 1
                else:
                    infeasible_count += 1

            validated.append(entry)

        # ── 计算性能指标 ───────────────────────────────────────────────────────
        costs = np.array([s.cost for s in solutions])
        confidences = np.array([s.confidence for s in solutions])
        metrics = self._compute_metrics(costs, confidences)

        elapsed = time.time() - t0
        self._log(
            f"验证完成：{len(solutions)} 个解，"
            f"可行 {feasible_count}，不可行 {infeasible_count}，"
            f"耗时 {elapsed:.3f}s"
        )

        return {
            "status": "ok",
            "total": len(solutions),
            "feasible_count": feasible_count,
            "infeasible_count": infeasible_count,
            "solutions": validated,
            "metrics": metrics,
            "elapsed": elapsed,
        }

    # ── 性能指标计算 ──────────────────────────────────────────────────────────

    def _compute_metrics(
        self, costs: np.ndarray, confidences: np.ndarray
    ) -> Dict[str, float]:
        """
        计算 Pareto 前沿的多样性和分布性指标。

        当前实现的指标：
        - front_size:   前沿中解的数量
        - cost_range:   成本目标的范围
        - conf_range:   置信度目标的范围
        - spread:       前沿跨度（归一化对角线距离）
        - spacing:      解的分布均匀性（越小越均匀）
        - hv_approx:    超体积近似（使用 2D 解析公式）
        """
        n = len(costs)
        if n == 0:
            return {}

        cost_min, cost_max = float(costs.min()), float(costs.max())
        conf_min, conf_max = float(confidences.min()), float(confidences.max())

        metrics: Dict[str, float] = {
            "front_size": float(n),
            "cost_min": cost_min,
            "cost_max": cost_max,
            "conf_min": conf_min,
            "conf_max": conf_max,
            "cost_range": cost_max - cost_min,
            "conf_range": conf_max - conf_min,
        }

        # Spread（归一化跨度）
        if (cost_max - cost_min) > 0 and (conf_max - conf_min) > 0:
            spread = np.sqrt(
                ((cost_max - cost_min) / (cost_max + 1e-12)) ** 2
                + ((conf_max - conf_min) ** 2)
            )
            metrics["spread"] = float(spread)

        # Spacing（解间距的标准差）
        if n >= 2:
            points = np.column_stack([costs, -confidences])  # 最小化成本，最大化置信度
            dists = []
            for i in range(n):
                d = np.min(
                    [np.linalg.norm(points[i] - points[j]) for j in range(n) if j != i]
                )
                dists.append(d)
            dists = np.array(dists)
            metrics["spacing"] = float(np.std(dists))
            metrics["spacing_mean"] = float(np.mean(dists))

        # Hypervolume 近似（2D 前沿，参考点：最大成本 * 1.1，最小置信度 * 0.9）
        if n >= 1:
            hv = self._hypervolume_2d(costs, confidences)
            metrics["hypervolume"] = float(hv)

        return metrics

    @staticmethod
    def _hypervolume_2d(
        costs: np.ndarray,
        confidences: np.ndarray,
        ref_cost_factor: float = 1.1,
        ref_conf_factor: float = 0.9,
    ) -> float:
        """
        计算 2D Pareto 前沿的超体积（HV）。

        目标方向：最小化 cost，最大化 confidence。
        参考点设置为稍差于所有解的点。
        """
        ref_cost = costs.max() * ref_cost_factor
        ref_conf = confidences.min() * ref_conf_factor

        # 按 cost 升序排序（同时 confidence 应降序，才是非支配前沿）
        idx = np.argsort(costs)
        sorted_costs = costs[idx]
        sorted_confs = confidences[idx]

        hv = 0.0
        prev_conf = sorted_confs.max() if len(sorted_confs) > 0 else ref_conf
        # 扫描线法计算 HV
        prev_cost = ref_cost
        for i in range(len(sorted_costs) - 1, -1, -1):
            width = prev_cost - sorted_costs[i]
            height = sorted_confs[i] - ref_conf
            if width > 0 and height > 0:
                hv += width * height
            prev_cost = sorted_costs[i]

        return hv
