"""
sorting.py — Factor 排序工具（节点内 item 按不同属性排序）

对应原始代码中的 resort_factor 函数，解耦为独立模块。
"""

from __future__ import annotations

from typing import Dict, List


def resort_factor(parameter_set: dict) -> Dict[str, Dict[int, List[int]]]:
    """
    计算不同排序标准下每个节点的 item order_id 排列。

    Parameters
    ----------
    parameter_set : dict
        包含 ``fto``、``lambda``、``param`` 字段的问题参数字典。

    Returns
    -------
    Dict[str, Dict[int, List[int]]]
        ``nr[sort_key][node_id]`` = 按 sort_key 排序后的 order_id 列表。
        sort_key 包括 ``"default"``、``"value"``、``"weight"``、``"utility"``。

    Notes
    -----
    - ``"value"``   按价值降序（对应成本升序）
    - ``"weight"``  按权重升序（置信度潜力降序）
    - ``"utility"`` 按效用降序
    """
    fto = parameter_set["fto"]
    _lambda = parameter_set["lambda"]
    m = parameter_set["m"]
    n = parameter_set["n"]

    # 确保权重和效用已初始化
    for node_id in range(m):
        for factor in fto[node_id]:
            if factor.weight is None:
                factor._init_stats(_lambda)

    nr: Dict[str, Dict[int, List[int]]] = {"default": {}}

    # 默认顺序
    for node_id in range(m):
        nr["default"][node_id] = list(range(n))

    # 按属性排序
    sort_configs = {
        "value":   ("value",   True),   # 价值 降序
        "weight":  ("weight",  False),  # 权重 升序
        "utility": ("utility", True),   # 效用 降序
    }

    for key, (attr, reverse) in sort_configs.items():
        nr[key] = {}
        for node_id in range(m):
            factors = fto[node_id]
            sorted_factors = sorted(
                factors, key=lambda f: getattr(f, attr) or 0.0, reverse=reverse
            )
            nr[key][node_id] = [f.order_id for f in sorted_factors]

    return nr
