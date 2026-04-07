"""
visualization_tools.py — Pareto 前沿可视化工具

提供：
  - plot_pareto_front: 绘制 Pareto 前沿散点图
  - plot_convergence: 绘制迭代收敛曲线（HV 或前沿大小 vs 迭代次数）
  - plot_solution_heatmap: 解决方案热图（各节点选择分布）
  - save_figure: 保存图表到文件
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def plot_pareto_front(
    fronts: Dict[str, List[Tuple[float, float]]],
    title: str = "Pareto Front",
    xlabel: str = "Cost (minimize)",
    ylabel: str = "Confidence (maximize)",
    confidence_level: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """
    绘制一个或多个 Pareto 前沿的散点图。

    Parameters
    ----------
    fronts : dict
        键为前沿名称（字符串），值为 [(cost, confidence), ...] 列表。
    title : str
        图表标题。
    xlabel, ylabel : str
        坐标轴标签。
    confidence_level : float, optional
        在 y 轴绘制置信度约束水平线。
    save_path : str, optional
        若指定则保存图表到文件。
    show : bool
        是否调用 plt.show() 显示图表。

    Returns
    -------
    matplotlib.figure.Figure
        生成的图表对象。
    """
    import matplotlib
    matplotlib.use("Agg" if not show else "TkAgg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    markers = ["o", "s", "^", "D", "v", "*", "P"]
    colors = plt.cm.tab10.colors  # type: ignore

    for i, (name, pts) in enumerate(fronts.items()):
        if not pts:
            continue
        costs = [p[0] for p in pts]
        confs = [p[1] for p in pts]
        idx = np.argsort(costs)
        sorted_c = [costs[j] for j in idx]
        sorted_f = [confs[j] for j in idx]
        ax.scatter(
            costs, confs,
            label=name,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            s=60, zorder=3,
        )
        ax.plot(sorted_c, sorted_f, "--", color=colors[i % len(colors)], alpha=0.5)

    if confidence_level is not None:
        ax.axhline(
            y=confidence_level, color="red", linestyle="--", linewidth=1.5,
            label=f"CL = {confidence_level:.2f}"
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_convergence(
    history: List[Dict[str, Any]],
    metric: str = "hypervolume",
    title: str = "Convergence",
    save_path: Optional[str] = None,
    show: bool = True,
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """
    绘制迭代收敛曲线。

    Parameters
    ----------
    history : list of dict
        每轮迭代的记录，每个 dict 包含 ``"iter"``、``"front_size"``、
        ``"hypervolume"`` 等字段。
    metric : str
        绘制的指标：``"hypervolume"``、``"front_size"`` 等。
    title : str
        图表标题。
    save_path : str, optional
        保存路径。
    show : bool
        是否显示。

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use("Agg" if not show else "TkAgg")
    import matplotlib.pyplot as plt

    iters = [h.get("iter", i) for i, h in enumerate(history)]
    values = [h.get(metric, h.get("front_size", 0)) for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, values, "b-o", markersize=3, linewidth=1.5)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def save_figure(fig: Any, path: str, dpi: int = 150) -> None:
    """将 matplotlib Figure 保存到文件。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
