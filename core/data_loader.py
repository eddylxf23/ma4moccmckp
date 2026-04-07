"""
data_loader.py — MO-CCMCKP benchmark 数据加载器

支持从原始 benchmark 文件夹加载问题实例，返回结构化的参数字典。
Benchmark 目录格式（每个实例一个子目录）：
  {instance_folder}/
    parameter.txt   — 问题元数据（m, n, sample_num, Wmax, eval_func, ...）
    cost.txt        — m*n 行，各 item 的确定性成本
    {idx}_sample.txt  — 第 idx 个 item 的随机权重样本（sample_num 行）
"""

from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import List, Optional

from core.factor import Factor


# ── 默认 benchmark 目录（相对于项目根） ───────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
BENCHMARK_DIR = _REPO_ROOT / "original" / "benchmark"


def _resolve_benchmark_path(instance_folder: str, *sub_paths: str) -> Path:
    """拼接 benchmark 目录路径。"""
    return BENCHMARK_DIR / instance_folder / Path(*sub_paths) if sub_paths else BENCHMARK_DIR / instance_folder


def load_instance(
    instance_folder: str,
    confidence_level: float = 0.9,
    lambda_param: float = 3.0,
    benchmark_dir: Optional[str] = None,
) -> dict:
    """
    加载单个 MO-CCMCKP 问题实例。

    Parameters
    ----------
    instance_folder : str
        benchmark 实例子目录名称，例如 ``"APP_10_10_500_"``。
    confidence_level : float
        机会约束置信度阈值 CL（0~1）。
    lambda_param : float
        权重计算参数 λ，影响 Factor.weight = mean + λ·std。
    benchmark_dir : str, optional
        benchmark 根目录，默认使用 ``original/benchmark``。

    Returns
    -------
    dict
        ``parameter_set`` 字典，字段含义见下方注释。
    """
    bdir = Path(benchmark_dir) if benchmark_dir else BENCHMARK_DIR
    inst_dir = bdir / instance_folder

    if not inst_dir.exists():
        raise FileNotFoundError(f"实例目录不存在：{inst_dir}")

    # ── 读取元数据 ────────────────────────────────────────────────────────────
    meta = _read_parameter_txt(inst_dir / "parameter.txt")
    m = int(meta["m"])            # 节点（类别）数量
    n = int(meta["n"])            # 每节点 item 数量
    sample_num = int(meta["sample_num"])
    Wmax = float(meta["Wmax"])
    mc_times = int(meta["MonteCarlotimes"])
    eval_type = meta.get("eval_func", "advanced_monte_carlo")

    # ── 读取成本 ──────────────────────────────────────────────────────────────
    cost_list = np.loadtxt(str(inst_dir / "cost.txt"))
    if cost_list.ndim == 0:
        cost_list = np.array([float(cost_list)])

    # ── 构建 Factor 二维列表 fto[node_id][order_id] ───────────────────────────
    fto: List[List[Factor]] = []
    max_cost = float(np.max(cost_list))

    for node_id in range(m):
        row: List[Factor] = []
        for order_id in range(n):
            global_idx = node_id * n + order_id
            cost = float(cost_list[global_idx])
            value = max_cost - cost  # 价值 = 最大成本 - 自身成本

            sample_path = str(inst_dir / f"{global_idx}_sample.txt")
            factor = Factor(
                factor_id=global_idx,
                node_id=node_id,
                order_id=order_id,
                cost=cost,
                value=value,
                sample_filepath=sample_path,
            )
            # 预计算统计量
            factor._init_stats(lambda_param)
            row.append(factor)
        fto.append(row)

    # ── 构建快速验证列表 qcl（置信度快速剪枝用）────────────────────────────────
    from core._confidence_utils import factorization_list
    qcl = factorization_list(sample_num, m, confidence_level, 10, 50)

    # ── 选择评估函数 ──────────────────────────────────────────────────────────
    from core.evaluator import Evaluator
    evaluator = Evaluator(eval_type=eval_type)
    eval_func = evaluator.get_eval_func()

    parameter_set = {
        "folder": instance_folder,
        "CL": confidence_level,
        "eval_func": eval_func,
        "param": (m, n, sample_num),
        "MCtimes": mc_times,
        "qcl": qcl,
        "itype": meta.get("type", "APP"),
        "fto": fto,
        "Wmax": Wmax,
        "lambda": lambda_param,
        # 便于访问的扁平化字段
        "m": m,
        "n": n,
        "sample_num": sample_num,
    }
    return parameter_set


def list_instances(benchmark_dir: Optional[str] = None) -> List[str]:
    """列出所有可用的 benchmark 实例目录名称。"""
    bdir = Path(benchmark_dir) if benchmark_dir else BENCHMARK_DIR
    if not bdir.exists():
        return []
    return sorted(
        d.name for d in bdir.iterdir()
        if d.is_dir() and (d / "parameter.txt").exists()
    )


# ── 内部辅助 ──────────────────────────────────────────────────────────────────

def _read_parameter_txt(path: Path) -> dict:
    """解析 ``parameter.txt`` 文件为 dict。"""
    meta = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and ": " in line:
                key, val = line.split(": ", 1)
                meta[key.strip()] = val.strip()
    return meta


class DataLoader:
    """
    MO-CCMCKP 数据加载器（对象接口）。

    Examples
    --------
    >>> loader = DataLoader(benchmark_dir="original/benchmark")
    >>> ps = loader.load("APP_10_10_500_", confidence_level=0.9)
    >>> print(ps["m"], ps["n"])  # 10 10
    """

    def __init__(
        self,
        benchmark_dir: Optional[str] = None,
        confidence_level: float = 0.9,
        lambda_param: float = 3.0,
    ):
        self.benchmark_dir = benchmark_dir
        self.confidence_level = confidence_level
        self.lambda_param = lambda_param

    def load(
        self,
        instance_folder: str,
        confidence_level: Optional[float] = None,
        lambda_param: Optional[float] = None,
    ) -> dict:
        """加载一个 benchmark 实例，返回 parameter_set 字典。"""
        return load_instance(
            instance_folder,
            confidence_level=confidence_level or self.confidence_level,
            lambda_param=lambda_param or self.lambda_param,
            benchmark_dir=self.benchmark_dir,
        )

    def list_instances(self) -> List[str]:
        """列出所有可用实例。"""
        return list_instances(self.benchmark_dir)
