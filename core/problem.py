"""
problem.py — MO-CCMCKP 问题定义类

封装问题实例的完整状态：参数、评估器、排序索引。
是所有 Agent 访问问题数据的统一入口。
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List

import numpy as np

from core.solution import Solution, random_solution
from core.evaluator import Evaluator
from core.sorting import resort_factor


class MOCCMCKPProblem:
    """
    多目标机会约束多选择背包问题（MO-CCMCKP）。

    Attributes
    ----------
    ps : dict
        原始 ``parameter_set`` 字典（向后兼容）。
    m : int
        节点（类别）数量。
    n : int
        每节点 item 数量。
    CL : float
        机会约束置信度阈值。
    Wmax : float
        随机权重上限。
    evaluator : Evaluator
        绑定到本问题的评估器实例。
    nr : dict
        节点内 item 的排序索引，键为 ``"default/value/weight/utility"``。

    Examples
    --------
    >>> from core.data_loader import DataLoader
    >>> loader = DataLoader()
    >>> ps = loader.load("APP_10_10_500_")
    >>> problem = MOCCMCKPProblem(ps)
    >>> sol = problem.random_solution()
    >>> sol = problem.evaluate(sol)
    >>> print(sol.cost, sol.confidence, sol.is_feasible)
    """

    def __init__(self, parameter_set: dict, instance_name: Optional[str] = None):
        self.ps: dict = parameter_set
        self.m: int = parameter_set["m"]
        self.n: int = parameter_set["n"]
        self.CL: float = float(parameter_set["CL"])
        self.Wmax: float = float(parameter_set["Wmax"])
        # instance_name 优先使用参数，其次从 parameter_set 中取
        self.instance_name: str = instance_name or parameter_set.get("folder", "unknown")

        self.evaluator: Evaluator = Evaluator.build_from_parameter_set(parameter_set)
        self.nr: Dict[str, Dict[int, List[int]]] = resort_factor(parameter_set)

    @property
    def cl(self) -> float:
        """置信度约束阈值别名（与 CL 等价）。"""
        return self.CL

    # ── 核心评估接口 ──────────────────────────────────────────────────────────

    def evaluate(self, solution: Solution) -> Solution:
        """评估解，就地修改并返回。"""
        return self.evaluator.evaluate(solution, self.ps)

    def evaluate_x(self, x: np.ndarray) -> Tuple[float, float]:
        """评估裸解向量，返回 (cost, confidence)。"""
        return self.evaluator.evaluate_array(x, self.ps)

    def evaluate_batch(self, solutions: List[Solution]) -> List[Solution]:
        """批量评估解列表。"""
        return self.evaluator.evaluate_batch(solutions, self.ps)

    def compute_cost(self, x: np.ndarray) -> float:
        """仅计算成本 f₁（无需评估置信度）。"""
        return Evaluator.compute_cost(x, self.ps)

    # ── 解生成接口 ────────────────────────────────────────────────────────────

    def random_solution(self) -> Solution:
        """随机生成一个解（未评估）。"""
        return random_solution(self.m, self.n)

    def greedy_solution(self, key: str = "value") -> Solution:
        """贪心生成初始解（每节点选第一个最优 item）。"""
        x = np.array([self.nr[key][node_id][0] for node_id in range(self.m)], dtype=int)
        return Solution(x=x, metadata={"source": f"greedy_{key}"})

    def random_feasible_solution(self, max_trials: int = 1000) -> Optional[Solution]:
        """
        生成一个随机可行解（满足置信度约束）。
        若 max_trials 次内未找到可行解，返回 None。
        """
        for _ in range(max_trials):
            sol = self.random_solution()
            self.evaluate(sol)
            if sol.is_feasible:
                return sol
        return None

    # ── 问题信息查询 ──────────────────────────────────────────────────────────

    def item_cost(self, node_id: int, order_id: int) -> float:
        """查询指定节点、item 的成本。"""
        return self.ps["fto"][node_id][order_id].cost

    def item_factor(self, node_id: int, order_id: int):
        """返回 Factor 对象。"""
        return self.ps["fto"][node_id][order_id]

    def get_sorted_items(self, node_id: int, key: str = "value") -> List[int]:
        """返回指定节点按键排序的 order_id 列表。"""
        return self.nr.get(key, self.nr["default"])[node_id]

    @property
    def num_classes(self) -> int:
        return self.m

    @property
    def items_per_class(self) -> int:
        return self.n

    @property
    def sample_num(self) -> int:
        return self.ps.get("sample_num", self.ps["param"][2])

    @property
    def instance_folder(self) -> str:
        return self.instance_name

    # ── 辅助方法 ──────────────────────────────────────────────────────────────

    def is_feasible(self, solution: Solution) -> bool:
        """检查解是否满足置信度约束（需已评估）。"""
        return solution.is_feasible

    def dominates(self, sol_a: Solution, sol_b: Solution) -> bool:
        """判断 sol_a 是否 Pareto 支配 sol_b。"""
        return sol_a.dominates(sol_b)

    def __repr__(self) -> str:
        return (
            f"MOCCMCKPProblem(instance='{self.instance_folder}', "
            f"m={self.m}, n={self.n}, CL={self.CL}, Wmax={self.Wmax})"
        )

    def summary(self) -> Dict[str, Any]:
        """返回问题摘要字典。"""
        return {
            "instance": self.instance_folder,
            "m": self.m,
            "n": self.n,
            "CL": self.CL,
            "Wmax": self.Wmax,
            "sample_num": self.sample_num,
            "eval_type": self.evaluator.eval_type,
        }
