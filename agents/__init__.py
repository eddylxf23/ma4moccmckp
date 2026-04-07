"""
agents — MO-CCMCKP 多 Agent 协作求解系统

包含 6 个专业化 Agent 及协调器：
  - ConstraintAnalyzerAgent   约束分析
  - SamplingEvaluatorAgent    采样评估
  - ParetoManagerAgent        Pareto 维护
  - SolutionRepairerAgent     解修复
  - MetaheuristicSchedulerAgent  元启发式调度
  - ResultValidatorAgent      结果验证
  - Coordinator               协调器（Autogen GroupChat 封装）
"""

from agents.base_agent import BaseAgent, AgentMessage, MessageType
from agents.constraint_analyzer import ConstraintAnalyzerAgent
from agents.sampling_evaluator import SamplingEvaluatorAgent
from agents.pareto_manager import ParetoManagerAgent
from agents.solution_repairer import SolutionRepairerAgent
from agents.metaheuristic_scheduler import MetaheuristicSchedulerAgent
from agents.result_validator import ResultValidatorAgent
from agents.coordinator import MOCCMCKPCoordinator

__all__ = [
    "BaseAgent",
    "AgentMessage",
    "MessageType",
    "ConstraintAnalyzerAgent",
    "SamplingEvaluatorAgent",
    "ParetoManagerAgent",
    "SolutionRepairerAgent",
    "MetaheuristicSchedulerAgent",
    "ResultValidatorAgent",
    "MOCCMCKPCoordinator",
]
