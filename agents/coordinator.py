"""
coordinator.py — MO-CCMCKP 多 Agent 协调器

职责：
  - 管理所有 Agent 的生命周期（注册、启动、停止）
  - 实现消息路由：将 Agent 发出的消息分发给目标接收者
  - 驱动主优化循环（Autogen GroupChat 风格，但轻量化实现）
  - 记录完整的消息历史，支持回放和调试
  - 提供简洁的 ``solve()`` 入口，一键启动多 Agent 协作求解

架构说明：
  本协调器采用"轻量级消息总线"模式，不依赖 Autogen 运行时，
  保持与 Autogen 接口的兼容性（Agent 消息格式可序列化为 Autogen 文本）。
  若需接入 Autogen GroupChat，只需将 ``run_loop`` 替换为 Autogen 调度器即可。
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional

from agents.base_agent import AgentMessage, BaseAgent, MessageType
from agents.constraint_analyzer import ConstraintAnalyzerAgent
from agents.sampling_evaluator import SamplingEvaluatorAgent
from agents.pareto_manager import ParetoManagerAgent
from agents.solution_repairer import SolutionRepairerAgent
from agents.metaheuristic_scheduler import MetaheuristicSchedulerAgent
from agents.result_validator import ResultValidatorAgent
from core.problem import MOCCMCKPProblem
from utils.shared_memory import SharedMemory


logger = logging.getLogger(__name__)


class MOCCMCKPCoordinator:
    """
    MO-CCMCKP 多 Agent 协调器。

    负责：
    1. 注册和管理所有 Agent
    2. 提供消息总线，路由 Agent 间的消息
    3. 驱动异步优化循环
    4. 收集并返回最终结果

    用法示例
    --------
    >>> from core.data_loader import DataLoader
    >>> from agents.coordinator import MOCCMCKPCoordinator
    >>>
    >>> problem = DataLoader.load("path/to/instance/")
    >>> coordinator = MOCCMCKPCoordinator(problem, config={"max_iter": 200})
    >>> result = coordinator.solve()
    >>> print(result["pareto_front"])
    """

    def __init__(
        self,
        problem: MOCCMCKPProblem,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.problem = problem
        self.config: Dict[str, Any] = config or {}
        self.shared_memory = SharedMemory()

        # 消息队列（待分发的消息）
        self._message_queue: deque[AgentMessage] = deque()
        # 完整消息历史（用于调试/回放）
        self._message_history: List[AgentMessage] = []
        # Agent 字典：name → Agent 实例
        self._agents: Dict[str, BaseAgent] = {}
        # 回调钩子
        self._on_message_callbacks: List[Callable[[AgentMessage], None]] = []
        # 优化结果
        self._result: Optional[Dict[str, Any]] = None

        self._setup_agents()

    # ── Agent 初始化 ───────────────────────────────────────────────────────────

    def _setup_agents(self) -> None:
        """实例化并注册所有 Agent。"""
        agent_config = self.config.get("agent_config", {})

        agents_to_register: List[BaseAgent] = [
            ConstraintAnalyzerAgent(
                problem=self.problem,
                shared_memory=self.shared_memory,
                config=agent_config.get("constraint_analyzer", {}),
            ),
            SamplingEvaluatorAgent(
                problem=self.problem,
                shared_memory=self.shared_memory,
                config=agent_config.get("sampling_evaluator", {
                    "pop_size": self.config.get("pop_size", 50),
                }),
            ),
            ParetoManagerAgent(
                problem=self.problem,
                shared_memory=self.shared_memory,
                config=agent_config.get("pareto_manager", {}),
            ),
            SolutionRepairerAgent(
                problem=self.problem,
                shared_memory=self.shared_memory,
                config=agent_config.get("solution_repairer", {}),
            ),
            MetaheuristicSchedulerAgent(
                problem=self.problem,
                shared_memory=self.shared_memory,
                config=agent_config.get("metaheuristic_scheduler", {
                    "max_iter": self.config.get("max_iter", 200),
                    "stagnation_limit": self.config.get("stagnation_limit", 30),
                }),
            ),
            ResultValidatorAgent(
                problem=self.problem,
                shared_memory=self.shared_memory,
                config=agent_config.get("result_validator", {}),
            ),
        ]

        for agent in agents_to_register:
            self.register_agent(agent)

        logger.info(f"已注册 {len(self._agents)} 个 Agent: {list(self._agents.keys())}")

    # ── Agent 管理 ─────────────────────────────────────────────────────────────

    def register_agent(self, agent: BaseAgent) -> None:
        """注册一个 Agent。"""
        self._agents[agent.name] = agent
        logger.debug(f"注册 Agent: {agent.name}")

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """按名称获取 Agent。"""
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        """返回所有已注册 Agent 的名称列表。"""
        return list(self._agents.keys())

    # ── 消息路由 ───────────────────────────────────────────────────────────────

    def post_message(self, msg: AgentMessage) -> None:
        """将消息放入队列。"""
        self._message_queue.append(msg)

    def _dispatch(self, msg: AgentMessage) -> List[AgentMessage]:
        """
        将消息分发给目标 Agent，收集并返回所有回复消息。

        广播消息（receiver == "broadcast"）将发给除发送者外的所有 Agent。
        """
        self._message_history.append(msg)

        # 触发回调
        for cb in self._on_message_callbacks:
            try:
                cb(msg)
            except Exception as exc:
                logger.warning(f"消息回调异常: {exc}")

        replies: List[AgentMessage] = []
        receivers: List[str] = []

        if msg.receiver == "broadcast":
            receivers = [name for name in self._agents if name != msg.sender]
        elif msg.receiver in self._agents:
            receivers = [msg.receiver]
        else:
            logger.warning(f"目标 Agent '{msg.receiver}' 未注册，消息丢弃: {msg.msg_type}")
            return []

        for receiver_name in receivers:
            agent = self._agents[receiver_name]
            try:
                agent_replies = agent.process_message(msg)
                replies.extend(agent_replies)
            except Exception as exc:
                logger.error(f"Agent [{receiver_name}] 处理消息 {msg.msg_type} 异常: {exc}", exc_info=True)

        return replies

    def _process_queue(self, max_steps: int = 10000) -> None:
        """
        消费消息队列，直到队列清空或达到 max_steps 上限。

        每处理一条消息，将回复消息重新入队，形成 Agent 间自主协作链。
        """
        steps = 0
        while self._message_queue and steps < max_steps:
            msg = self._message_queue.popleft()
            replies = self._dispatch(msg)
            for reply in replies:
                self._message_queue.append(reply)
            steps += 1

        if steps >= max_steps:
            logger.warning(f"消息处理达到上限 {max_steps} 步，强制终止。")

    # ── 主求解入口 ─────────────────────────────────────────────────────────────

    def solve(self, verbose: bool = True) -> Dict[str, Any]:
        """
        启动多 Agent 协作求解 MO-CCMCKP 问题。

        Returns
        -------
        dict
            包含以下字段：
            - ``pareto_front``: 最终 Pareto 前沿（Solution 列表）
            - ``validation_report``: 结果验证报告
            - ``history``: 每轮迭代的前沿大小和超体积
            - ``elapsed``: 总耗时（秒）
            - ``total_messages``: 处理的消息总数
        """
        if verbose:
            logger.info("=" * 60)
            logger.info("MO-CCMCKP 多 Agent 求解启动")
            logger.info(f"  问题规模: m={self.problem.m}, n={self.problem.n}")
            logger.info(f"  置信度约束: CL={self.problem.cl:.4f}")
            logger.info(f"  配置: {self.config}")
            logger.info("=" * 60)

        t_start = time.time()

        # ── 通知所有 Agent 任务开始 ────────────────────────────────────────────
        for agent in self._agents.values():
            agent.on_start()
            agent.on_task_start(self.problem, self.shared_memory)

        # ── 第一步：触发约束分析 ───────────────────────────────────────────────
        start_msg = AgentMessage(
            msg_type=MessageType.TASK_START,
            sender="coordinator",
            receiver=ConstraintAnalyzerAgent.NAME,
            content={"problem_id": getattr(self.problem, "instance_name", "unknown")},
        )
        self.post_message(start_msg)
        self._process_queue()

        # ── 第二步：触发初始种群采样（发给 MetaheuristicScheduler 以统一控制） ──
        # MetaheuristicScheduler 在收到 CONSTRAINT_REPORT 后会自动触发
        # 若未自动触发，手动推送一次种群采样请求
        if not self._has_message_type(MessageType.INITIAL_POPULATION):
            init_msg = AgentMessage(
                msg_type=MessageType.TASK_START,
                sender="coordinator",
                receiver=SamplingEvaluatorAgent.NAME,
                content={"op": "init_population"},
            )
            self.post_message(init_msg)
            self._process_queue()

        # ── 主循环：由 MetaheuristicScheduler 驱动，协调器负责处理消息队列 ──────
        max_total_steps = self.config.get("max_total_steps", 50000)
        self._process_queue(max_steps=max_total_steps)

        # ── 收集结果 ───────────────────────────────────────────────────────────
        pareto_agent: ParetoManagerAgent = self._agents.get(ParetoManagerAgent.NAME)  # type: ignore
        validator: ResultValidatorAgent = self._agents.get(ResultValidatorAgent.NAME)  # type: ignore

        final_front = pareto_agent.front if pareto_agent else []
        front_history = pareto_agent.history if pareto_agent else []

        # 触发最终验证
        validation_report = {}
        if validator and final_front:
            validation_report = validator.validate_front(final_front)

        elapsed = time.time() - t_start

        self._result = {
            "pareto_front": final_front,
            "validation_report": validation_report,
            "history": front_history,
            "elapsed": elapsed,
            "total_messages": len(self._message_history),
        }

        if verbose:
            logger.info("=" * 60)
            logger.info(f"求解完成！耗时 {elapsed:.2f}s")
            logger.info(f"  Pareto 前沿大小: {len(final_front)}")
            logger.info(f"  消息总数: {len(self._message_history)}")
            if validation_report.get("metrics"):
                m = validation_report["metrics"]
                logger.info(f"  超体积 (HV): {m.get('hypervolume', 'N/A'):.4f}")
            logger.info("=" * 60)

        return self._result

    # ── 辅助方法 ───────────────────────────────────────────────────────────────

    def _has_message_type(self, msg_type: MessageType) -> bool:
        """检查消息历史中是否出现过指定类型的消息。"""
        return any(m.msg_type == msg_type for m in self._message_history)

    def add_message_callback(self, callback: Callable[[AgentMessage], None]) -> None:
        """注册消息监听回调（用于实时日志或可视化）。"""
        self._on_message_callbacks.append(callback)

    def get_message_history(self) -> List[AgentMessage]:
        """返回完整消息历史副本。"""
        return list(self._message_history)

    @property
    def result(self) -> Optional[Dict[str, Any]]:
        """返回最近一次求解结果。"""
        return self._result

    def __repr__(self) -> str:
        return (
            f"MOCCMCKPCoordinator("
            f"agents={list(self._agents.keys())}, "
            f"problem=m={self.problem.m},n={self.problem.n})"
        )
