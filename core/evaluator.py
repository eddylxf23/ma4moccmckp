"""
evaluator.py — MO-CCMCKP 评估器

封装成本计算、置信度评估两项核心计算，供所有 Agent 调用。
支持精确评估（小样本）和蒙特卡洛评估（大样本）两种模式，
并提供批量评估接口以支持种群级操作。
"""

from __future__ import annotations

from typing import List, Callable, Dict, Any, Optional, Tuple

import numpy as np

from core._confidence_utils import advanced_exact_evaluation, advanced_monte_carlo
from core.solution import Solution


# ── 类型别名 ──────────────────────────────────────────────────────────────────
EvalFunc = Callable[[np.ndarray, dict], float]
_EVAL_REGISTRY: Dict[str, EvalFunc] = {
    "advanced_exact_evaluation": advanced_exact_evaluation,
    "advanced_monte_carlo": advanced_monte_carlo,
}


class Evaluator:
    """
    MO-CCMCKP 评估器。

    Parameters
    ----------
    eval_type : str
        评估方法类型：``"advanced_exact_evaluation"`` 或 ``"advanced_monte_carlo"``。
    parameter_set : dict, optional
        直接传入 parameter_set，避免重复加载。

    Examples
    --------
    >>> evaluator = Evaluator(eval_type="advanced_monte_carlo", parameter_set=ps)
    >>> sol = evaluator.evaluate(solution)
    >>> print(sol.cost, sol.confidence)
    """

    def __init__(
        self,
        eval_type: str = "advanced_monte_carlo",
        parameter_set: Optional[dict] = None,
    ):
        if eval_type not in _EVAL_REGISTRY:
            raise ValueError(
                f"未知评估类型 '{eval_type}'。"
                f"可选：{list(_EVAL_REGISTRY)}"
            )
        self.eval_type = eval_type
        self._eval_func: EvalFunc = _EVAL_REGISTRY[eval_type]
        self.parameter_set: Optional[dict] = parameter_set

    # ── 获取评估函数 ──────────────────────────────────────────────────────────

    def get_eval_func(self) -> EvalFunc:
        """返回底层评估函数（供 parameter_set["eval_func"] 使用）。"""
        return self._eval_func

    def bind(self, parameter_set: dict) -> "Evaluator":
        """绑定 parameter_set，返回 self 以支持链式调用。"""
        self.parameter_set = parameter_set
        return self

    # ── 单解评估 ──────────────────────────────────────────────────────────────

    def evaluate(
        self,
        solution: Solution,
        parameter_set: Optional[dict] = None,
    ) -> Solution:
        """
        就地评估 ``solution``，填写 cost / confidence / status。

        Parameters
        ----------
        solution : Solution
            解对象（会被原地修改并返回）。
        parameter_set : dict, optional
            若未在构造器中绑定，可在此传入。

        Returns
        -------
        Solution
            已评估的解（同一对象）。
        """
        ps = parameter_set or self.parameter_set
        if ps is None:
            raise ValueError("必须提供 parameter_set。")

        x = solution.x
        cost = self.compute_cost(x, ps)
        confidence = self._eval_func(x, ps)
        solution.update_evaluation(cost, confidence, ps["CL"])
        return solution

    def evaluate_array(
        self,
        x: np.ndarray,
        parameter_set: Optional[dict] = None,
    ) -> Tuple[float, float]:
        """
        评估裸解向量，返回 (cost, confidence)。

        Parameters
        ----------
        x : np.ndarray, shape (m,)
            解向量。

        Returns
        -------
        Tuple[float, float]
            (cost, confidence)
        """
        ps = parameter_set or self.parameter_set
        if ps is None:
            raise ValueError("必须提供 parameter_set。")
        cost = self.compute_cost(x, ps)
        confidence = self._eval_func(x, ps)
        return cost, confidence

    # ── 批量评估 ──────────────────────────────────────────────────────────────

    def evaluate_batch(
        self,
        solutions: List[Solution],
        parameter_set: Optional[dict] = None,
    ) -> List[Solution]:
        """
        批量评估解列表（顺序评估，可扩展为并行）。

        Parameters
        ----------
        solutions : List[Solution]
            解列表，每个解都会被原地更新。

        Returns
        -------
        List[Solution]
            已评估的解列表。
        """
        ps = parameter_set or self.parameter_set
        for sol in solutions:
            self.evaluate(sol, ps)
        return solutions

    def evaluate_population(
        self,
        population: List[np.ndarray],
        parameter_set: Optional[dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量评估种群（裸解向量列表）。

        Returns
        -------
        List[Dict]
            每个元素含 ``{"x", "cost", "confidence", "is_feasible"}``。
        """
        ps = parameter_set or self.parameter_set
        if ps is None:
            raise ValueError("必须提供 parameter_set。")
        results = []
        for x in population:
            cost, conf = self.evaluate_array(x, ps)
            results.append({
                "x": x,
                "cost": cost,
                "confidence": conf,
                "is_feasible": conf >= ps["CL"],
            })
        return results

    # ── 静态方法 ──────────────────────────────────────────────────────────────

    @staticmethod
    def compute_cost(x: np.ndarray, parameter_set: dict) -> float:
        """计算解 x 的确定性总成本 f₁。"""
        fto = parameter_set["fto"]
        return float(sum(fto[node_id][x[node_id]].cost for node_id in range(len(x))))

    @staticmethod
    def compute_cost_vector(population: np.ndarray, parameter_set: dict) -> np.ndarray:
        """向量化计算种群成本，population.shape = (pop_size, m)。"""
        fto = parameter_set["fto"]
        pop_size, m = population.shape
        costs = np.zeros(pop_size)
        for i in range(pop_size):
            costs[i] = sum(fto[node_id][population[i, node_id]].cost for node_id in range(m))
        return costs

    @staticmethod
    def is_feasible(solution: Solution, confidence_level: float) -> bool:
        """检查解是否满足置信度约束。"""
        return solution.confidence is not None and solution.confidence >= confidence_level

    @staticmethod
    def build_from_parameter_set(ps: dict) -> "Evaluator":
        """从 parameter_set 自动推断评估类型并创建 Evaluator。"""
        # 逆向查找 eval_func
        ef = ps.get("eval_func")
        for name, func in _EVAL_REGISTRY.items():
            if ef is func:
                return Evaluator(eval_type=name, parameter_set=ps)
        # 根据样本数量自动选择
        sample_num = ps.get("param", (0, 0, 500))[2]
        eval_type = "advanced_exact_evaluation" if sample_num <= 30 else "advanced_monte_carlo"
        return Evaluator(eval_type=eval_type, parameter_set=ps)
