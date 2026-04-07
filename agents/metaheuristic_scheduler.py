"""
metaheuristic_scheduler.py — 元启发式调度 Agent

职责：
  - 根据约束分析报告动态选择优化策略
  - 管理迭代流程：决定何时调用 degrade / local_search / deep_search
  - 监控收敛状态，适时停止或调整策略
  - 协调其他 Agent 的执行顺序（作为 Autogen 中的任务编排者）

迭代策略（OPERA-MC 的 Agent 化版本）：
  每轮迭代：
    1. 选择父代（基于拥挤距离排序）
    2. degrade 扰动（SamplingEvaluator）
    3. local_swap_search（SolutionRepairer）
    4. further_swap_search（SolutionRepairer，仅精英解）
    5. 更新 Pareto 前沿（ParetoManager）
  停止条件：
    - 达到最大迭代次数
    - 前沿连续 K 代无改进
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent, AgentMessage, MessageType


class MetaheuristicSchedulerAgent(BaseAgent):
    """
    元启发式调度 Agent。

    控制主优化循环，向其他 Agent 发送操作请求，并根据进度调整策略。

    响应消息：
    - TASK_START + CONSTRAINT_REPORT → 初始化调度参数
    - PARETO_FRONT / PARETO_UPDATE   → 检查收敛并决定下一步操作
    - INITIAL_POPULATION             → 触发第一轮迭代
    """

    NAME = "MetaheuristicScheduler"

    def __init__(
        self,
        problem: Any = None,
        shared_memory: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(self.NAME, problem, shared_memory, config)

        # ── 默认超参数 ────────────────────────────────────────────────────────
        self.max_iterations: int = self.config.get("max_iterations", 200)
        self.pop_size: int = self.config.get("pop_size", 100)
        self.stagnation_limit: int = self.config.get("stagnation_limit", 30)
        self.local_search_prob: float = self.config.get("local_search_prob", 0.1)
        self.deep_search_prob: float = self.config.get("deep_search_prob", 0.1)
        self.degrade_prob: float = self.config.get("degrade_prob", 0.1)

        # ── 迭代状态 ──────────────────────────────────────────────────────────
        self._iteration: int = 0
        self._best_front_size: int = 0
        self._stagnation_count: int = 0
        self._start_time: float = 0.0
        self._current_front: List[Dict] = []
        self._running: bool = False

        # ── 阶段记录 ──────────────────────────────────────────────────────────
        self._phase: str = "init"  # init → search → converged

    # ── 核心接口实现 ──────────────────────────────────────────────────────────

    def process_message(self, msg: AgentMessage) -> List[AgentMessage]:
        if msg.msg_type == MessageType.CONSTRAINT_REPORT:
            self._apply_constraint_report(msg.content)
            return []

        if msg.msg_type == MessageType.TASK_START:
            return self._handle_task_start(msg)

        if msg.msg_type == MessageType.INITIAL_POPULATION:
            self._phase = "search"
            self._running = True
            self._start_time = time.time()
            self._iteration = 0
            return self._schedule_next_iteration(msg)

        if msg.msg_type in (MessageType.PARETO_UPDATE, MessageType.PARETO_FRONT):
            return self._handle_pareto_update(msg)

        return []

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": self.NAME,
            "description": "管理优化迭代流程，动态调度 degrade/local_search/deep_search 操作。",
            "supported_message_types": [
                MessageType.TASK_START,
                MessageType.CONSTRAINT_REPORT,
                MessageType.INITIAL_POPULATION,
                MessageType.PARETO_UPDATE,
                MessageType.PARETO_FRONT,
            ],
            "output_types": [
                MessageType.SCHEDULE_REQUEST,
                MessageType.ITERATION_PROGRESS,
                MessageType.TASK_COMPLETE,
            ],
        }

    # ── 调度策略 ──────────────────────────────────────────────────────────────

    def _handle_task_start(self, msg: AgentMessage) -> List[AgentMessage]:
        """启动任务：发送初始化参数给采样评估 Agent。"""
        constraint_report = self.shared_memory.get("constraint_report", {}) \
            if self.shared_memory else {}
        return [self.send(
            MessageType.TASK_START,
            receiver="SamplingEvaluator",
            content={
                "pop_size": self.pop_size,
                "constraint_report": constraint_report,
            },
            reply_to=msg.id,
        )]

    def _schedule_next_iteration(
        self,
        trigger_msg: AgentMessage,
    ) -> List[AgentMessage]:
        """
        调度下一轮迭代操作。

        根据当前阶段和收敛状态，选择操作序列并发送 SCHEDULE_REQUEST。
        """
        if not self._running:
            return []

        self._iteration += 1
        messages: List[AgentMessage] = []

        # ── 进度广播 ─────────────────────────────────────────────────────────
        elapsed = time.time() - self._start_time
        progress_msg = self.broadcast(
            MessageType.ITERATION_PROGRESS,
            content={
                "iteration": self._iteration,
                "max_iterations": self.max_iterations,
                "front_size": len(self._current_front),
                "stagnation": self._stagnation_count,
                "phase": self._phase,
                "elapsed_seconds": round(elapsed, 2),
            },
        )
        messages.append(progress_msg)

        # ── 检查停止条件 ──────────────────────────────────────────────────────
        if self._should_stop():
            messages.append(self._build_complete_message(trigger_msg.id))
            self._running = False
            return messages

        # ── 自适应调整策略 ────────────────────────────────────────────────────
        if self._stagnation_count > self.stagnation_limit // 2:
            # 停滞时提高 degrade 概率，增加多样性
            degrade_prob = min(self.degrade_prob * 2, 0.3)
            deep_prob = min(self.deep_search_prob * 1.5, 0.25)
        else:
            degrade_prob = self.degrade_prob
            deep_prob = self.deep_search_prob

        # ── 操作序列 ─────────────────────────────────────────────────────────
        # 1. degrade 扰动
        messages.append(self.send(
            MessageType.SCHEDULE_REQUEST,
            receiver="SamplingEvaluator",
            content={
                "op": "degrade",
                "population": self._current_front,
                "degrade_prob": degrade_prob,
            },
        ))

        # 2. local_swap_search
        messages.append(self.send(
            MessageType.SCHEDULE_REQUEST,
            receiver="SolutionRepairer",
            content={
                "op": "local_search",
                "population": self._current_front,
                "local_search_prob": self.local_search_prob,
            },
        ))

        # 3. further_swap_search（奇数轮执行）
        if self._iteration % 2 == 1:
            messages.append(self.send(
                MessageType.SCHEDULE_REQUEST,
                receiver="SolutionRepairer",
                content={
                    "op": "deep_search",
                    "population": self._current_front,
                    "deep_search_prob": deep_prob,
                },
            ))

        return messages

    def _handle_pareto_update(self, msg: AgentMessage) -> List[AgentMessage]:
        """接收 Pareto 更新，更新状态并调度下一步。"""
        front = msg.content.get("front", [])
        new_size = msg.content.get("front_size", 0)
        added = msg.content.get("added", 0)

        self._current_front = front

        # 更新停滞计数
        if added > 0 and new_size >= self._best_front_size:
            self._best_front_size = new_size
            self._stagnation_count = 0
        else:
            self._stagnation_count += 1

        # 写入共享内存
        if self.shared_memory:
            self.shared_memory.set("current_front", front)
            self.shared_memory.set("iteration", self._iteration)

        return self._schedule_next_iteration(msg)

    def _apply_constraint_report(self, report: Dict[str, Any]) -> None:
        """从约束分析报告更新调度参数。"""
        rec = report.get("recommended", {})
        self.pop_size = rec.get("pop_size", self.pop_size)
        self.max_iterations = rec.get("max_iterations", self.max_iterations)
        self.local_search_prob = rec.get("local_search_prob", self.local_search_prob)
        self.deep_search_prob = rec.get("deep_search_prob", self.deep_search_prob)
        self._log(
            f"调度参数更新：pop={self.pop_size}, "
            f"max_iter={self.max_iterations}, "
            f"ls_prob={self.local_search_prob:.2f}"
        )

    def _should_stop(self) -> bool:
        """判断是否满足停止条件。"""
        if self._iteration >= self.max_iterations:
            self._log(f"达到最大迭代次数 {self.max_iterations}，停止。")
            return True
        if self._stagnation_count >= self.stagnation_limit:
            self._log(f"前沿连续 {self.stagnation_limit} 代无改进，停止。")
            return True
        return False

    def _build_complete_message(self, reply_to: Optional[str] = None) -> AgentMessage:
        elapsed = time.time() - self._start_time
        return self.broadcast(
            MessageType.TASK_COMPLETE,
            content={
                "total_iterations": self._iteration,
                "front_size": len(self._current_front),
                "elapsed_seconds": round(elapsed, 2),
                "stop_reason": (
                    "max_iterations" if self._iteration >= self.max_iterations
                    else "stagnation"
                ),
                "front": self._current_front,
            },
            reply_to=reply_to,
        )

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def is_running(self) -> bool:
        return self._running
