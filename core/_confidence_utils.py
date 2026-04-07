"""
_confidence_utils.py — 置信度评估底层工具函数

从 original/src/functions/func_confidencelevel_estimator.py 迁移，
去除对原项目路径的依赖，改为接受 Factor 对象列表作为输入。
"""

from __future__ import annotations

import math
import heapq
import random
from collections import defaultdict
from typing import List, TYPE_CHECKING

import numpy as np

try:
    import numba
    _USE_NUMBA = True
except ImportError:
    _USE_NUMBA = False

if TYPE_CHECKING:
    from core.factor import Factor


# ── 快速剪枝列表 ─────────────────────────────────────────────────────────────

def factorization_list(
    L: int, m: int, P0: float, list_size: int, K: int
) -> List[List[int]]:
    """
    生成用于置信度快速剪枝的整数列表（质因数分解组合）。

    Parameters
    ----------
    L : int
        样本数量上限。
    m : int
        节点数量。
    P0 : float
        置信度阈值。
    list_size : int
        需要生成的列表数量。
    K : int
        每个列表的排列数量。

    Returns
    -------
    List[List[int]]
        快速检查用的下标组合列表。
    """
    def list_product(nums: List[int]) -> int:
        p = 1
        for num in nums:
            p *= num
        return p

    def factorization(n: int) -> List[int]:
        factors: List[int] = []
        while n % 2 == 0:
            factors.append(2)
            n //= 2
        f = 3
        while f * f <= n:
            if n % f == 0:
                factors.append(f)
                n //= f
            else:
                f += 2
        if n > 1:
            factors.append(n)
        return factors

    def generate_permutations(lst: List[int], k: int) -> List[List[int]]:
        if len(lst) <= 1:
            return [lst.copy() for _ in range(k)]
        permutations = set()
        max_iter = k * 100
        it = 0
        while len(permutations) < k and it < max_iter:
            perm = random.sample(lst, len(lst))
            permutations.add(tuple(perm))
            it += 1
        return [list(p) for p in permutations]

    def divide_into_n_groups(elements: List[int], n: int) -> List[List[int]]:
        groups: List[List[int]] = [[] for _ in range(n)]
        for i, elem in enumerate(elements):
            groups[i % n].append(elem)
        return groups

    if L <= 0 or m <= 0 or list_size <= 0 or K <= 0:
        return []

    base = 100 if L >= 100 else 10
    L_base = L // base
    num_L = L_base ** m
    denominator = round(1 / (1 - P0))
    num_b = (10 ** m) // denominator
    num_ = 10 ** m

    f_list = factorization(num_L) + factorization(num_b) + factorization(num_)

    if not f_list or max(f_list) > L:
        return []

    seen: defaultdict = defaultdict(int)
    product_list: List[List[int]] = []
    max_iterations = min(10000, list_size * 100)

    for _ in range(max_iterations):
        if len(product_list) >= list_size:
            break
        random.shuffle(f_list)
        groups = divide_into_n_groups(f_list, m)
        products = [list_product(g) - 1 for g in groups]
        if products and max(products) >= L:
            continue
        sorted_products = tuple(sorted(products))
        if seen[sorted_products] < K:
            product_list.append(list(sorted_products))
            seen[sorted_products] += 1

    final: List[List[int]] = []
    for products in product_list:
        final.extend(generate_permutations(products, K))
    return final


# ── 蒙特卡洛评估 ─────────────────────────────────────────────────────────────

def _mc_inner(sum_samples: np.ndarray, Wmax: float) -> float:
    """计算 sum_samples < Wmax 的比例。"""
    return float(np.sum(sum_samples < Wmax) / len(sum_samples))


def advanced_monte_carlo(
    solution: np.ndarray,
    parameter_dict: dict,
) -> float:
    """
    自适应蒙特卡洛置信度评估（10k → 100k → 1M）。

    Parameters
    ----------
    solution : np.ndarray, shape (m,)
        解向量。
    parameter_dict : dict
        ``parameter_set`` 字典（含 fto, Wmax, CL, qcl, MCtimes, param）。

    Returns
    -------
    float
        估计的置信度概率 P(Σw ≤ Wmax)。
    """
    fto = parameter_dict["fto"]
    factor_list = [fto[node_id][solution[node_id]] for node_id in range(len(solution))]

    Wmax = parameter_dict["Wmax"]
    CL = parameter_dict["CL"]
    MC_max = parameter_dict["MCtimes"]
    qcl = parameter_dict.get("qcl", [])
    L = parameter_dict["param"][2]
    m = parameter_dict["param"][0]

    _prob = 1 - CL
    _data_l = math.ceil((L ** m * _prob) ** (1 / m))

    # 快速不可行剪枝
    data_sum = sum(f.get_index_sample(_data_l) for f in factor_list)
    if data_sum >= Wmax:
        return 0.0
    for check in qcl:
        if len(check) != m:
            continue
        data_sum = sum(factor_list[j].get_index_sample(check[j]) for j in range(m))
        if data_sum >= Wmax:
            return 0.0

    # 阶段 1：10k 次采样
    n1 = min(10000, MC_max)
    ss = _vectorized_sample_sum(factor_list, n1)
    conf = _mc_inner(ss, Wmax)
    if conf <= 0.999:
        return conf

    # 阶段 2：100k 次采样
    n2 = 100000
    ss = _vectorized_sample_sum(factor_list, n2)
    conf = _mc_inner(ss, Wmax)
    if conf <= 0.9999:
        return conf

    # 阶段 3：1M 次采样
    ss = _vectorized_sample_sum(factor_list, 1_000_000)
    return _mc_inner(ss, Wmax)


def _vectorized_sample_sum(factor_list: List["Factor"], n: int) -> np.ndarray:
    """向量化采样并求和。"""
    total = np.zeros(n)
    for f in factor_list:
        total += f.resample_multi(n)
    return total


# ── 精确评估 ─────────────────────────────────────────────────────────────────

def advanced_exact_evaluation(
    solution: np.ndarray,
    parameter_dict: dict,
) -> float:
    """
    精确置信度评估（基于堆排序枚举违约组合数）。
    适用于小样本量（sample_num=30）。

    Returns
    -------
    float
        精确置信度概率。
    """
    fto = parameter_dict["fto"]
    factor_list = [fto[node_id][solution[node_id]] for node_id in range(len(solution))]

    Wmax = parameter_dict["Wmax"]
    CL = parameter_dict["CL"]
    qcl = parameter_dict.get("qcl", [])
    L = parameter_dict["param"][2]
    m = parameter_dict["param"][0]

    _prob = round(1 - CL, 3)
    L_power_m = L ** m
    max_eval = math.ceil(L_power_m * _prob)
    _data_lz = math.ceil((L_power_m * _prob) ** (1 / m))

    # 快速剪枝
    data_sum = sum(f.get_index_sample(_data_lz) for f in factor_list)
    if data_sum >= Wmax:
        return 0.0
    for check in qcl:
        if len(check) != m:
            continue
        data_sum = sum(factor_list[j].get_index_sample(check[j]) for j in range(m))
        if data_sum >= Wmax:
            return 0.0

    # 二分查找精确边界
    low, high = 0, _data_lz
    while low <= high:
        mid = (low + high) // 2
        data_sum = sum(f.get_index_sample(mid) for f in factor_list)
        if data_sum < Wmax:
            low = mid + 1
        else:
            high = mid - 1
    _data_lz = max(high, 0)

    visited: set = set()
    queue: list = []
    count = 0

    def push_q(D: List[int]) -> None:
        state = tuple(D)
        if state in visited:
            return
        if any(v >= L for v in D):
            return
        s = sum(factor_list[i].get_index_sample(D[i]) for i in range(m))
        visited.add(state)
        heapq.heappush(queue, (-s, state))

    for i in range(m):
        D = [_data_lz] * m
        D[i] = _data_lz + 1
        push_q(D)

    while queue:
        neg_s, state = heapq.heappop(queue)
        s = -neg_s
        count += 1

        if count > max_eval:
            return 0.0

        if s < Wmax:
            conf = 1.0 - count / L_power_m
            return conf

        for i in range(m):
            D_new = list(state)
            D_new[i] += 1
            push_q(D_new)

    return 1.0 - count / L_power_m
