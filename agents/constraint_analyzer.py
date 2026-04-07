"""
constraint_analyzer.py — 约束分析 Agent

职责：
  - 分析 MO-CCMCKP 问题结构（m, n, CL, Wmax, 样本分布特征）
  - 计算约束难度指标（紧度、分布特征、初始可行解比例估计）
  - 推荐合适的求解策略参数（种群大小、最大迭代次数等）
  - 向其他 Agent 广播问题摘要和分析报告
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from agents.base_agent import BaseAgent, AgentMessage, MessageType


class ConstraintAnalyzerAgent(BaseAgent):
    """
    约束分析 Agent。

    收到 TASK_START 消息后，自动分析问题约束结构，
    生成 CONSTRAINT_REPORT 广播给所有 Agent。
    """

    NAME = "ConstraintAnalyzer"

    def __init__(
        self,
        problem: Any = None,
        shared_memory: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(self.NAME, problem, shared_memory, config)

    # ── 核心接口实现 ──────────────────────────────────────────────────────────

    def process_message(self, msg: AgentMessage) -> List[AgentMessage]:
        """处理消息：TASK_START → 触发分析并广播报告。"""
        if msg.msg_type == MessageType.TASK_START:
            report = self.analyze()
            out_msg = self.broadcast(
                msg_type=MessageType.CONSTRAINT_REPORT,
                content=report,
                reply_to=msg.id,
            )
            return [out_msg]
        return []

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": self.NAME,
            "description": "分析 MO-CCMCKP 问题约束结构，生成约束难度报告和求解策略建议。",
            "supported_message_types": [MessageType.TASK_START],
            "output_types": [MessageType.CONSTRAINT_REPORT],
        }

    # ── 分析方法 ──────────────────────────────────────────────────────────────

    def analyze(self, problem: Any = None) -> Dict[str, Any]:
        """
        执行完整的问题约束分析。

        Returns
        -------
        dict
            包含问题统计量、约束难度评估和策略建议的报告。
        """
        p = problem or self.problem
        if p is None:
            raise ValueError("ConstraintAnalyzerAgent: 未绑定 problem 对象。")

        ps = p.ps
        m, n = p.m, p.n
        CL = p.CL
        Wmax = p.Wmax
        fto = ps["fto"]

        # ── 统计各 item 成本分布 ──────────────────────────────────────────────
        all_costs = [fto[i][j].cost for i in range(m) for j in range(n)]
        cost_stats = {
            "min": float(np.min(all_costs)),
            "max": float(np.max(all_costs)),
            "mean": float(np.mean(all_costs)),
            "std": float(np.std(all_costs)),
        }

        # ── 统计各 item 随机权重分布 ─────────────────────────────────────────
        all_means = [fto[i][j].sample_mean for i in range(m) for j in range(n)]
        all_stds = [fto[i][j].sample_std for i in range(m) for j in range(n)]
        weight_stats = {
            "mean_of_means": float(np.mean(all_means)),
            "std_of_means": float(np.std(all_means)),
            "mean_of_stds": float(np.mean(all_stds)),
        }

        # ── 约束紧度估计 ─────────────────────────────────────────────────────
        # 最小权重组合（最宽松场景）的均值之和
        min_weight_sum = sum(
            min(fto[i][j].sample_mean for j in range(n)) for i in range(m)
        )
        # 最大权重组合的均值之和
        max_weight_sum = sum(
            max(fto[i][j].sample_mean for j in range(m) for j in range(n))
            for i in range(m)
        )
        tightness = (Wmax - min_weight_sum) / max(max_weight_sum - min_weight_sum, 1e-6)
        tightness = float(np.clip(tightness, 0, 1))

        # ── 约束难度评级 ─────────────────────────────────────────────────────
        if tightness > 0.7:
            difficulty = "low"       # 约束宽松，容易找到可行解
            feasibility_ratio = "high (>50%)"
        elif tightness > 0.3:
            difficulty = "medium"
            feasibility_ratio = "medium (10%~50%)"
        else:
            difficulty = "high"      # 约束紧，可行解稀少
            feasibility_ratio = "low (<10%)"

        # ── 策略建议 ─────────────────────────────────────────────────────────
        recommended_pop_size = max(50, min(200, m * n * 2))
        recommended_max_iter = max(100, min(500, m * 10))

        report = {
            # 问题基本参数
            "instance": p.instance_folder,
            "m": m,
            "n": n,
            "CL": CL,
            "Wmax": Wmax,
            "sample_num": p.sample_num,
            "eval_type": p.evaluator.eval_type,

            # 统计信息
            "cost_stats": cost_stats,
            "weight_stats": weight_stats,

            # 约束分析
            "constraint_tightness": tightness,
            "difficulty": difficulty,
            "feasibility_ratio_estimate": feasibility_ratio,

            # 策略建议
            "recommended": {
                "pop_size": recommended_pop_size,
                "max_iterations": recommended_max_iter,
                "repair_strategy": "greedy_repair" if difficulty == "high" else "random_repair",
                "init_strategy": "hybrid" if difficulty == "high" else "random",
                "local_search_prob": 0.2 if difficulty == "high" else 0.1,
                "deep_search_prob": 0.1,
                "degrade_prob": 0.05 if difficulty == "high" else 0.1,
            },
        }

        self._log(
            f"约束分析完成：m={m}, n={n}, CL={CL}, "
            f"紧度={tightness:.3f}, 难度={difficulty}"
        )
        return report

    def estimate_feasibility_ratio(
        self,
        n_samples: int = 200,
    ) -> float:
        """
        通过随机采样估算可行解比例。

        Parameters
        ----------
        n_samples : int
            随机采样次数。

        Returns
        -------
        float
            估算的可行解比例（0~1）。
        """
        if self.problem is None:
            return 0.0
        feasible_count = 0
        for _ in range(n_samples):
            sol = self.problem.random_solution()
            self.problem.evaluate(sol)
            if sol.is_feasible:
                feasible_count += 1
        ratio = feasible_count / n_samples
        self._log(f"可行解比例估计（{n_samples} 次采样）：{ratio:.3f}")
        return ratio
