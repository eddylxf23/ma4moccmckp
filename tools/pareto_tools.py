"""
pareto_tools.py — Pareto 前沿操作工具

提供：
  - is_dominated: 判断解 a 是否被解 b 支配
  - non_dominated_sort: 快速非支配排序（NSGA-II 风格）
  - crowding_distance: 拥挤距离计算
  - update_pareto_front: 增量式更新非支配解集
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


def is_dominated(a: np.ndarray, b: np.ndarray) -> bool:
    """
    判断解 a 是否被解 b 弱支配（MO-CCMCKP 双目标）。

    目标方向：f1=cost（最小化），f2=confidence（最大化）。
    b 支配 a 的条件：
      - b 在所有目标上不比 a 差（成本≤a的成本 且 置信度≥a的置信度）
      - b 在至少一个目标上严格优于 a

    Parameters
    ----------
    a, b : np.ndarray, shape (2,)
        两个解的目标向量 [cost, confidence]。

    Returns
    -------
    bool
        True 表示 a 被 b 支配。
    """
    cost_a, conf_a = a[0], a[1]
    cost_b, conf_b = b[0], b[1]
    # b 不劣于 a（成本更小/等，置信度更大/等）
    not_worse = (cost_b <= cost_a) and (conf_b >= conf_a)
    # b 至少在一个目标上严格更好
    strictly_better = (cost_b < cost_a) or (conf_b > conf_a)
    return not_worse and strictly_better


def update_pareto_front(
    front: List[np.ndarray],
    new_solution: np.ndarray,
) -> Tuple[List[np.ndarray], bool]:
    """
    增量式更新非支配解集。

    Parameters
    ----------
    front : list of np.ndarray
        当前 Pareto 前沿（每个元素是目标向量 [cost, confidence]）。
    new_solution : np.ndarray, shape (2,)
        新解的目标向量。

    Returns
    -------
    updated_front : list of np.ndarray
        更新后的 Pareto 前沿。
    is_accepted : bool
        新解是否被接受（加入前沿）。
    """
    # 检查新解是否被当前前沿中的某个解支配
    for existing in front:
        if is_dominated(new_solution, existing):
            return front, False  # 新解被支配，直接丢弃

    # 移除被新解支配的旧解
    new_front = [s for s in front if not is_dominated(s, new_solution)]
    new_front.append(new_solution)
    return new_front, True


def batch_update_pareto_front(
    front: List[np.ndarray],
    candidates: List[np.ndarray],
) -> Tuple[List[np.ndarray], int]:
    """
    批量更新 Pareto 前沿。

    Parameters
    ----------
    front : list of np.ndarray
        当前前沿。
    candidates : list of np.ndarray
        候选解批次。

    Returns
    -------
    updated_front : list of np.ndarray
        更新后的前沿。
    accepted_count : int
        实际加入前沿的解数量。
    """
    accepted = 0
    for sol in candidates:
        front, is_accepted = update_pareto_front(front, sol)
        if is_accepted:
            accepted += 1
    return front, accepted


def crowding_distance(front: List[np.ndarray]) -> np.ndarray:
    """
    计算 Pareto 前沿中每个解的拥挤距离（NSGA-II 公式）。

    目标方向：f1=cost（最小化），f2=confidence（最大化）。

    Parameters
    ----------
    front : list of np.ndarray
        Pareto 前沿解集，每个元素是 [cost, confidence]。

    Returns
    -------
    np.ndarray, shape (n,)
        每个解的拥挤距离（两端解的距离为 inf）。
    """
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)

    objectives = np.array(front)  # shape (n, 2)
    distances = np.zeros(n)
    num_obj = objectives.shape[1]

    for obj_idx in range(num_obj):
        # 按当前目标排序
        sorted_idx = np.argsort(objectives[:, obj_idx])
        obj_values = objectives[sorted_idx, obj_idx]
        obj_range = obj_values[-1] - obj_values[0]

        # 边界解距离设为无穷
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf

        if obj_range == 0:
            continue

        for i in range(1, n - 1):
            distances[sorted_idx[i]] += (
                (obj_values[i + 1] - obj_values[i - 1]) / obj_range
            )

    return distances


def select_by_crowding(
    front: List[np.ndarray],
    k: int,
) -> List[int]:
    """
    从 Pareto 前沿按拥挤距离选择 k 个解（多样性选择）。

    Parameters
    ----------
    front : list of np.ndarray
    k : int
        选择数量。

    Returns
    -------
    list of int
        选中解的索引列表。
    """
    n = len(front)
    if k >= n:
        return list(range(n))

    cd = crowding_distance(front)
    # 拥挤距离降序选择（选最稀疏的解）
    return list(np.argsort(cd)[::-1][:k])
