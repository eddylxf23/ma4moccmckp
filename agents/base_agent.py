"""
base_agent.py — Agent 基础抽象类和消息协议

定义所有 Agent 共享的接口规范：
  - AgentMessage: Agent 间消息结构
  - MessageType:  消息类型枚举
  - BaseAgent:    抽象基类（所有 Agent 继承自此）

设计原则：
  - Agent 之间通过 AgentMessage 通信，不直接调用彼此方法
  - 共享状态通过 SharedMemory 传递
  - 每个 Agent 实现 process_message() 处理输入
"""

from __future__ import annotations

import time
import uuid
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


# ── 消息类型枚举 ──────────────────────────────────────────────────────────────

class MessageType(str, Enum):
    """Agent 间消息类型。"""
    # 约束分析 Agent 发出
    CONSTRAINT_REPORT = "constraint_report"        # 约束结构分析报告
    PARAMETER_SUMMARY = "parameter_summary"        # 问题参数摘要

    # 采样评估 Agent 发出
    EVALUATION_RESULT = "evaluation_result"        # 单解评估结果
    POPULATION_RESULT = "population_result"        # 种群评估结果
    INITIAL_POPULATION = "initial_population"      # 初始种群

    # Pareto 维护 Agent 发出
    PARETO_UPDATE = "pareto_update"                # Pareto 前沿更新
    PARETO_FRONT = "pareto_front"                  # 完整 Pareto 前沿

    # 解修复 Agent 发出
    REPAIRED_SOLUTION = "repaired_solution"        # 修复后的解
    REPAIRED_POPULATION = "repaired_population"    # 修复后的种群

    # 元启发式调度 Agent 发出
    SCHEDULE_REQUEST = "schedule_request"          # 请求执行优化操作
    SCHEDULE_RESULT = "schedule_result"            # 优化操作结果
    ITERATION_PROGRESS = "iteration_progress"      # 迭代进度报告

    # 结果验证 Agent 发出
    VALIDATION_RESULT = "validation_result"        # 验证结果（单次请求回复）
    VALIDATION_REPORT = "validation_report"        # 完整验证报告
    FINAL_RESULT = "final_result"                  # 最终结果

    # 系统消息
    TASK_START = "task_start"                      # 任务启动
    TASK_COMPLETE = "task_complete"                # 任务完成
    ERROR = "error"                                # 错误
    STATUS_UPDATE = "status_update"                # 状态更新


# ── Agent 消息结构 ────────────────────────────────────────────────────────────

class AgentMessage(BaseModel):
    """
    Agent 间通信消息格式。

    Attributes
    ----------
    id : str
        消息唯一标识符（UUID）。
    msg_type : MessageType
        消息类型。
    sender : str
        发送方 Agent 名称。
    receiver : str
        接收方 Agent 名称，``"broadcast"`` 表示广播。
    content : Dict[str, Any]
        消息内容（任意结构化数据）。
    reply_to : str, optional
        回复的消息 ID。
    timestamp : float
        消息创建时间戳（Unix 时间）。
    metadata : Dict[str, Any]
        附加元数据（如优先级、标签等）。
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    msg_type: MessageType
    sender: str
    receiver: str = "broadcast"
    content: Dict[str, Any] = Field(default_factory=dict)
    reply_to: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True

    def reply(
        self,
        msg_type: MessageType,
        sender: str,
        content: Dict[str, Any],
        **kwargs: Any,
    ) -> "AgentMessage":
        """快速构建回复消息。"""
        return AgentMessage(
            msg_type=msg_type,
            sender=sender,
            receiver=self.sender,
            content=content,
            reply_to=self.id,
            **kwargs,
        )

    def to_text(self) -> str:
        """序列化为供 Autogen 传递的文本格式。"""
        import json
        return json.dumps({
            "id": self.id,
            "type": self.msg_type,
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
        }, ensure_ascii=False, indent=2)

    @classmethod
    def from_text(cls, text: str) -> "AgentMessage":
        """从 Autogen 文本消息反序列化。"""
        import json
        d = json.loads(text)
        return cls(
            msg_type=MessageType(d["type"]),
            sender=d["sender"],
            receiver=d.get("receiver", "broadcast"),
            content=d.get("content", {}),
        )


# ── 基础 Agent 抽象类 ─────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    所有 MO-CCMCKP Agent 的基础抽象类。

    每个具体 Agent 必须实现：
    - ``process_message(msg)``  — 处理收到的消息，返回回复消息列表
    - ``get_capabilities()``    — 返回该 Agent 的能力描述

    Agent 内部可以：
    - 访问 ``self.problem`` 获取问题实例
    - 访问 ``self.shared_memory`` 读写共享状态
    - 调用 ``self._log()`` 记录日志
    """

    def __init__(
        self,
        name: str,
        problem: Any = None,   # MOCCMCKPProblem
        shared_memory: Any = None,  # SharedMemory
        config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.problem = problem
        self.shared_memory = shared_memory
        self.config: Dict[str, Any] = config or {}
        self._message_history: List[AgentMessage] = []
        self._logger = logging.getLogger(f"agent.{name}")

    # ── 必须实现的接口 ────────────────────────────────────────────────────────

    @abstractmethod
    def process_message(self, msg: AgentMessage) -> List[AgentMessage]:
        """
        处理收到的消息，返回需要发出的回复消息列表。

        Parameters
        ----------
        msg : AgentMessage
            收到的消息。

        Returns
        -------
        List[AgentMessage]
            需要发出的回复消息（可以为空列表）。
        """
        ...

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        返回该 Agent 的能力描述（供协调器调度决策使用）。

        Returns
        -------
        Dict[str, Any]
            包含 ``"name"``、``"description"``、``"supported_message_types"`` 等字段。
        """
        ...

    # ── 生命周期方法（可选 override） ─────────────────────────────────────────

    def on_start(self) -> None:
        """Agent 启动时调用（初始化资源）。"""
        self._log(f"Agent [{self.name}] 已启动。")

    def on_stop(self) -> None:
        """Agent 停止时调用（清理资源）。"""
        self._log(f"Agent [{self.name}] 已停止。")

    def on_task_start(self, problem: Any, shared_memory: Any) -> None:
        """接收新任务时的初始化。"""
        self.problem = problem
        self.shared_memory = shared_memory
        self._message_history.clear()

    # ── 工具方法 ──────────────────────────────────────────────────────────────

    def send(
        self,
        msg_type: MessageType,
        receiver: str,
        content: Dict[str, Any],
        reply_to: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentMessage:
        """构建并记录一条待发出的消息。"""
        msg = AgentMessage(
            msg_type=msg_type,
            sender=self.name,
            receiver=receiver,
            content=content,
            reply_to=reply_to,
            **kwargs,
        )
        return msg

    def broadcast(
        self,
        msg_type: MessageType,
        content: Dict[str, Any],
        **kwargs: Any,
    ) -> AgentMessage:
        """广播消息给所有 Agent。"""
        return self.send(msg_type, "broadcast", content, **kwargs)

    def _log(self, msg: str, level: str = "info") -> None:
        """结构化日志记录。"""
        getattr(self._logger, level)(msg)

    def record_message(self, msg: AgentMessage) -> None:
        """记录消息到历史。"""
        self._message_history.append(msg)

    @property
    def message_history(self) -> List[AgentMessage]:
        return list(self._message_history)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
