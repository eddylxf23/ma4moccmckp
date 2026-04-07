"""
run_solver.py — MO-CCMCKP 多 Agent 求解器主入口

用法示例
--------
# 使用默认配置求解单个实例
python run_solver.py --instance APP_10_10_500_ --max-iter 200

# 使用自定义置信度阈值
python run_solver.py --instance LAB_5_10_30_ --cl 0.85 --pop-size 30

# 求解所有 benchmark 实例
python run_solver.py --all --max-iter 100 --output results/

命令行参数
----------
--instance   实例目录名（benchmark/ 下的子目录）
--cl         置信度阈值（默认 0.9）
--pop-size   种群大小（默认 50）
--max-iter   最大迭代次数（默认 200）
--output     结果输出目录（默认 results/）
--verbose    显示详细日志
--all        批量求解所有 benchmark 实例
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 确保项目根目录在 Python 路径中
_PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import setup_logger
from core.data_loader import load_instance, list_instances
from core.problem import MOCCMCKPProblem
from agents.coordinator import MOCCMCKPCoordinator


logger = setup_logger("moccmckp.solver", level="INFO")


def solve_instance(
    instance_name: str,
    cl: float = 0.9,
    pop_size: int = 50,
    max_iter: int = 200,
    output_dir: str = "results",
    verbose: bool = True,
) -> dict:
    """
    求解单个 MO-CCMCKP 实例。

    Parameters
    ----------
    instance_name : str
        benchmark 实例名称。
    cl : float
        置信度约束阈值。
    pop_size : int
        种群大小。
    max_iter : int
        最大迭代次数。
    output_dir : str
        结果保存目录。
    verbose : bool
        是否显示详细日志。

    Returns
    -------
    dict
        求解结果字典。
    """
    logger.info(f"正在加载实例: {instance_name}")
    ps = load_instance(instance_name, confidence_level=cl)

    problem = MOCCMCKPProblem(ps, instance_name=instance_name)

    config = {
        "pop_size": pop_size,
        "max_iter": max_iter,
        "stagnation_limit": max(20, max_iter // 10),
    }

    coordinator = MOCCMCKPCoordinator(problem, config=config)
    result = coordinator.solve(verbose=verbose)

    # 保存结果
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{instance_name}_result.json"

    # 序列化 Pareto 前沿
    front_data = [
        {"cost": float(s.cost), "confidence": float(s.confidence), "solution": s.x.tolist()}
        for s in result.get("pareto_front", [])
    ]

    summary = {
        "instance": instance_name,
        "config": config,
        "pareto_front_size": len(front_data),
        "pareto_front": front_data,
        "metrics": result.get("validation_report", {}).get("metrics", {}),
        "elapsed": result.get("elapsed", 0),
        "total_messages": result.get("total_messages", 0),
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"结果已保存到: {out_file}")
    logger.info(
        f"实例 {instance_name}: "
        f"前沿大小={len(front_data)}, "
        f"耗时={summary['elapsed']:.2f}s"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MO-CCMCKP 多 Agent 求解器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--instance", "-i", type=str, help="实例目录名")
    parser.add_argument("--cl", type=float, default=0.9, help="置信度阈值（默认 0.9）")
    parser.add_argument("--pop-size", type=int, default=50, help="种群大小（默认 50）")
    parser.add_argument("--max-iter", type=int, default=200, help="最大迭代次数（默认 200）")
    parser.add_argument("--output", "-o", type=str, default="results", help="结果输出目录")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志")
    parser.add_argument("--all", action="store_true", dest="all_instances", help="求解所有实例")
    parser.add_argument("--list", action="store_true", help="列出所有可用实例")

    args = parser.parse_args()

    if args.list:
        instances = list_instances()
        print("可用 benchmark 实例：")
        for inst in sorted(instances):
            print(f"  {inst}")
        return

    if args.all_instances:
        instances = list_instances()
        if not instances:
            logger.error("未找到任何 benchmark 实例，请检查 original/benchmark/ 目录")
            sys.exit(1)

        logger.info(f"批量求解 {len(instances)} 个实例")
        results = []
        for inst in sorted(instances):
            try:
                r = solve_instance(
                    inst, args.cl, args.pop_size, args.max_iter, args.output, args.verbose
                )
                results.append(r)
            except Exception as e:
                logger.error(f"实例 {inst} 求解失败: {e}")

        # 汇总报告
        summary_file = Path(args.output) / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"批量求解完成，汇总报告: {summary_file}")
        return

    if not args.instance:
        parser.print_help()
        print("\n错误：请指定 --instance 或使用 --all 求解所有实例")
        sys.exit(1)

    solve_instance(
        args.instance, args.cl, args.pop_size, args.max_iter, args.output, args.verbose
    )


if __name__ == "__main__":
    main()
