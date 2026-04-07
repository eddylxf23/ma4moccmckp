"""
constraint_tools.py — 约束处理工具

提供 MO-CCMCKP 约束相关的辅助函数：
  - 置信度快速估计（用于筛选阶段）
  - 贪心修复：将不可行解提升至可行
  - 约束紧度分析
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def greedy_repair(
    solution: np.ndarray,
    problem_ps: Dict[str, Any],
    direction: str = "confidence",
) -> np.ndarray:
    """
    贪心修复不可行解。

    通过逐节点替换 item，优化置信度（或成本），直到解变为可行。

    Parameters
    ----------
    solution : np.ndarray, shape (m,)
        当前解向量（每个元素是该节点选中 item 的局部索引）。
    problem_ps : dict
        parameter_set 字典。
    direction : str
        修复方向：``"confidence"`` 优先提升置信度；
                   ``"cost"`` 优先降低成本。

    Returns
    -------
    np.ndarray
        修复后的解向量（不保证可行，但尽力改善）。
    """
    fto = problem_ps["fto"]      # list[list[Factor]]
    m = problem_ps["m"]
    n = problem_ps["n"]
    Wmax = problem_ps["Wmax"]
    CL = problem_ps["CL"]
    eval_func = problem_ps.get("eval_func")
    nr = problem_ps.get("nr", {})

    sol = solution.copy()

    # 选择排序键
    if direction == "confidence":
        # 按 weight（均值+λ·std）升序 → 有助于降低容量消耗 → 提升置信度
        order_key = "weight"
    else:
        # 按 cost 升序
        order_key = "cost"

    # 逐节点贪心替换
    for i in range(m):
        items = fto[i]
        # 获取排序后的 item 顺序
        if nr and order_key in nr:
            order = nr[order_key][i]  # 排序后的 item 局部索引列表
        else:
            # 按属性排序
            if order_key == "weight":
                order = sorted(range(len(items)), key=lambda j: items[j].mean + items[j].std)
            else:
                order = sorted(range(len(items)), key=lambda j: items[j].cost)

        for item_idx in order:
            sol[i] = item_idx
            # 快速估计：使用均值评估是否有改善
            if _quick_feasible_check(sol, problem_ps):
                break

    return sol


def _quick_feasible_check(solution: np.ndarray, ps: Dict[str, Any]) -> bool:
    """
    基于均值的快速可行性检查（非精确，用于贪心修复内循环）。

    仅检查确定性约束：sum(mean_i) <= Wmax * 0.95（保留安全余量）。
    """
    fto = ps["fto"]
    Wmax = ps["Wmax"]
    total_mean = sum(fto[i][solution[i]].mean for i in range(ps["m"]))
    return total_mean <= Wmax * 0.95


def compute_constraint_tightness(
    ps: Dict[str, Any],
    n_sample: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    估计约束紧度：随机采样解，统计满足约束的比例。

    Parameters
    ----------
    ps : dict
        parameter_set 字典。
    n_sample : int
        随机解数量。
    rng : np.random.Generator, optional
        随机数生成器（None 时使用默认）。

    Returns
    -------
    dict
        包含约束紧度指标的字典：
        - ``"feasible_rate"``: 可行解比例（均值法估计）
        - ``"mean_total_weight"``: 随机解的平均总权重
        - ``"wmax"``: 容量上限
        - ``"tightness_ratio"``: mean_total_weight / Wmax
    """
    if rng is None:
        rng = np.random.default_rng()

    fto = ps["fto"]
    m = ps["m"]
    n = ps["n"]
    Wmax = ps["Wmax"]

    feasible_count = 0
    total_weights = []

    for _ in range(n_sample):
        # 随机解
        sol = rng.integers(0, n, size=m)
        total_mean = sum(fto[i][sol[i]].mean for i in range(m))
        total_weights.append(total_mean)
        if total_mean <= Wmax:
            feasible_count += 1

    mean_w = float(np.mean(total_weights))
    return {
        "feasible_rate": feasible_count / n_sample,
        "mean_total_weight": mean_w,
        "wmax": float(Wmax),
        "tightness_ratio": mean_w / Wmax if Wmax > 0 else float("inf"),
    }


def estimate_difficulty(ps: Dict[str, Any]) -> str:
    """
    根据问题结构快速估计求解难度级别。

    Returns
    -------
    str
        ``"easy"`` / ``"medium"`` / ``"hard"`` / ``"very_hard"``
    """
    m = ps["m"]
    n = ps["n"]
    CL = ps["CL"]
    sample_num = ps.get("sample_num", 30)

    # 大规模 + 高置信度要求 = 更难
    scale_score = m * n / 50.0  # 规模得分
    cl_score = CL / 0.95        # 置信度严格程度
    sample_score = 1.0 if sample_num <= 30 else 2.0  # 小样本更难精确评估

    total = scale_score * cl_score * sample_score

    if total < 1.0:
        return "easy"
    elif total < 3.0:
        return "medium"
    elif total < 8.0:
        return "hard"
    else:
        return "very_hard"
