"""
metrics.py — 多目标优化性能指标计算

提供以下性能指标：
  - HV（Hypervolume / 超体积）：前沿质量的综合度量
  - IGD（Inverted Generational Distance）：与参考前沿的距离
  - Spread / Delta：前沿多样性度量
  - Spacing：前沿点分布均匀性
  - GD（Generational Distance）：与真实前沿的接近度

所有函数均接受 numpy 数组，返回标量值。
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional


def hypervolume_2d(
    front: np.ndarray,
    ref_point: Optional[np.ndarray] = None,
    obj_directions: Optional[List[str]] = None,
) -> float:
    """
    计算 2D Pareto 前沿的精确超体积（HV）。

    Parameters
    ----------
    front : np.ndarray, shape (n, 2)
        Pareto 前沿点集，每行 [cost, confidence]。
    ref_point : np.ndarray, shape (2,), optional
        参考点（超体积的上界）。若为 None，自动设置为
        [max_cost * 1.1, min_conf * 0.9]。
    obj_directions : list of str, optional
        每个目标的优化方向：``"min"`` 或 ``"max"``。
        默认 ``["min", "max"]``（最小化成本，最大化置信度）。

    Returns
    -------
    float
        超体积值（越大越好）。
    """
    if len(front) == 0:
        return 0.0

    front = np.asarray(front, dtype=float)
    directions = obj_directions or ["min", "max"]

    # 统一转换为最小化问题
    normalized = np.copy(front)
    for i, d in enumerate(directions):
        if d == "max":
            normalized[:, i] = -normalized[:, i]

    if ref_point is None:
        ref = np.array([normalized[:, i].max() * (1.1 if normalized[:, i].max() > 0 else 0.9)
                        for i in range(normalized.shape[1])])
    else:
        ref = np.asarray(ref_point, dtype=float)
        for i, d in enumerate(directions):
            if d == "max":
                ref[i] = -ref[i]

    # 按第一个目标升序排序
    idx = np.argsort(normalized[:, 0])
    pts = normalized[idx]

    hv = 0.0
    prev_x = ref[0]
    # 扫描线（从右到左）
    for i in range(len(pts) - 1, -1, -1):
        width = prev_x - pts[i, 0]
        height = ref[1] - pts[i, 1]
        if width > 0 and height > 0:
            hv += width * height
        prev_x = pts[i, 0]

    return float(hv)


def spacing(front: np.ndarray) -> float:
    """
    计算 Pareto 前沿的 Spacing 指标（分布均匀性）。

    值越小表示前沿点分布越均匀（0 = 完全均匀）。

    Parameters
    ----------
    front : np.ndarray, shape (n, d)
        Pareto 前沿点集。

    Returns
    -------
    float
        Spacing 值。
    """
    n = len(front)
    if n <= 1:
        return 0.0

    front = np.asarray(front, dtype=float)
    dists = []
    for i in range(n):
        d = min(np.linalg.norm(front[i] - front[j]) for j in range(n) if j != i)
        dists.append(d)
    dists = np.array(dists)
    return float(np.std(dists))


def igd(front: np.ndarray, reference_front: np.ndarray) -> float:
    """
    计算 IGD（Inverted Generational Distance）。

    衡量参考前沿上每个点到当前前沿的最短距离的均值。

    Parameters
    ----------
    front : np.ndarray, shape (n, d)
        当前近似 Pareto 前沿。
    reference_front : np.ndarray, shape (m, d)
        真实或高质量参考前沿。

    Returns
    -------
    float
        IGD 值（越小越好）。
    """
    if len(front) == 0 or len(reference_front) == 0:
        return float("inf")

    front = np.asarray(front, dtype=float)
    ref = np.asarray(reference_front, dtype=float)
    total = 0.0
    for r in ref:
        d = np.min(np.linalg.norm(front - r, axis=1))
        total += d
    return float(total / len(ref))


def generational_distance(front: np.ndarray, reference_front: np.ndarray) -> float:
    """
    计算 GD（Generational Distance）。

    衡量当前前沿上每个点到参考前沿的最短距离的均值。

    Parameters
    ----------
    front : np.ndarray, shape (n, d)
        当前近似 Pareto 前沿。
    reference_front : np.ndarray, shape (m, d)
        参考前沿。

    Returns
    -------
    float
        GD 值（越小越好）。
    """
    if len(front) == 0 or len(reference_front) == 0:
        return float("inf")

    front = np.asarray(front, dtype=float)
    ref = np.asarray(reference_front, dtype=float)
    total = 0.0
    for p in front:
        d = np.min(np.linalg.norm(ref - p, axis=1))
        total += d
    return float(total / len(front))


def compute_all_metrics(
    front: np.ndarray,
    reference_front: Optional[np.ndarray] = None,
    ref_point: Optional[np.ndarray] = None,
) -> dict:
    """
    一次性计算所有可用指标。

    Parameters
    ----------
    front : np.ndarray, shape (n, 2)
        近似 Pareto 前沿，列顺序：[cost, confidence]。
    reference_front : np.ndarray, optional
        参考前沿（用于 IGD/GD 计算）。
    ref_point : np.ndarray, optional
        超体积参考点。

    Returns
    -------
    dict
        包含所有计算出的指标的字典。
    """
    front = np.asarray(front, dtype=float)
    metrics: dict = {
        "front_size": len(front),
    }

    if len(front) > 0:
        metrics["cost_min"] = float(front[:, 0].min())
        metrics["cost_max"] = float(front[:, 0].max())
        metrics["conf_min"] = float(front[:, 1].min())
        metrics["conf_max"] = float(front[:, 1].max())
        metrics["hypervolume"] = hypervolume_2d(front, ref_point=ref_point)
        metrics["spacing"] = spacing(front)

    if reference_front is not None and len(reference_front) > 0:
        ref = np.asarray(reference_front, dtype=float)
        metrics["igd"] = igd(front, ref)
        metrics["gd"] = generational_distance(front, ref)

    return metrics
