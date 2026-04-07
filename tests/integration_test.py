"""
integration_test.py — 快速集成测试

验证：
1. 数据加载器能读取 benchmark 实例
2. MOCCMCKPProblem 能正确初始化
3. 解评估正常（成本计算 + 置信度计算）
4. 所有 6 个 Agent 能正确导入并实例化
5. Coordinator 能完成初始化
6. 约束分析 Agent 能处理 TASK_START 消息
7. Pareto 工具正确工作
8. 指标计算正确
"""

import sys
sys.path.insert(0, 'e:/ma4moccmckp')

import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np

print("=" * 55)
print("MO-CCMCKP Multi-Agent Framework — 集成测试")
print("=" * 55)

passed = 0
failed = 0


def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        failed += 1


# ── 测试 1：数据加载 ────────────────────────────────────────────────────
def t1():
    from core.data_loader import load_instance, list_instances
    instances = list_instances()
    assert len(instances) == 20, f"应有20个实例，实际有 {len(instances)}"
    ps = load_instance('APP_3_5_30_', confidence_level=0.9)
    assert ps['m'] == 3 and ps['n'] == 5

test("数据加载 - 列举实例 + 加载 APP_3_5_30_", t1)


# ── 测试 2：问题初始化 ─────────────────────────────────────────────────
def t2():
    from core.data_loader import load_instance
    from core.problem import MOCCMCKPProblem
    ps = load_instance('APP_3_5_30_')
    problem = MOCCMCKPProblem(ps, instance_name='APP_3_5_30_')
    assert problem.m == 3 and problem.n == 5
    assert problem.cl == problem.CL == 0.9
    assert problem.instance_name == 'APP_3_5_30_'

test("问题初始化 - MOCCMCKPProblem", t2)


# ── 测试 3：解生成与评估 ───────────────────────────────────────────────
def t3():
    from core.data_loader import load_instance
    from core.problem import MOCCMCKPProblem
    ps = load_instance('APP_3_5_30_')
    problem = MOCCMCKPProblem(ps)
    sol = problem.random_solution()
    assert sol.x.shape == (3,)
    assert sol.cost is None  # 未评估
    sol = problem.evaluate(sol)
    assert sol.cost is not None and sol.confidence is not None
    assert 0 <= sol.confidence <= 1

test("解生成与评估 - 随机解", t3)


# ── 测试 4：解 Pareto 支配关系 ─────────────────────────────────────────
def t4():
    from core.solution import Solution
    import numpy as np
    # sol_a: 低成本高置信度 → 支配 sol_b
    sol_a = Solution(x=np.array([0, 1, 2]), cost=10.0, confidence=0.95)
    sol_b = Solution(x=np.array([1, 2, 0]), cost=15.0, confidence=0.90)
    assert sol_a.dominates(sol_b), "sol_a 应支配 sol_b"
    assert not sol_b.dominates(sol_a), "sol_b 不应支配 sol_a"

test("解 Pareto 支配关系", t4)


# ── 测试 5：Pareto 工具 ────────────────────────────────────────────────
def t5():
    from tools.pareto_tools import update_pareto_front, is_dominated, crowding_distance
    front = [np.array([10.0, 0.95]), np.array([20.0, 0.99])]
    # 新解被支配
    new_dom = np.array([15.0, 0.92])
    front2, accepted = update_pareto_front(front, new_dom)
    assert not accepted
    # 新解支配旧解
    new_good = np.array([8.0, 0.99])
    front3, accepted2 = update_pareto_front(front, new_good)
    assert accepted2

test("Pareto 工具 - 前沿更新", t5)


# ── 测试 6：指标计算 ───────────────────────────────────────────────────
def t6():
    from utils.metrics import hypervolume_2d, spacing, compute_all_metrics
    front = np.array([[10.0, 0.99], [15.0, 0.97], [20.0, 0.95]])
    hv = hypervolume_2d(front)
    assert hv > 0, "超体积应 > 0"
    sp = spacing(front)
    assert sp >= 0
    m = compute_all_metrics(front)
    assert 'hypervolume' in m and 'spacing' in m

test("指标计算 - HV + Spacing", t6)


# ── 测试 7：全 Agent 导入 ──────────────────────────────────────────────
def t7():
    from agents.constraint_analyzer import ConstraintAnalyzerAgent
    from agents.sampling_evaluator import SamplingEvaluatorAgent
    from agents.pareto_manager import ParetoManagerAgent
    from agents.solution_repairer import SolutionRepairerAgent
    from agents.metaheuristic_scheduler import MetaheuristicSchedulerAgent
    from agents.result_validator import ResultValidatorAgent

test("全 Agent 模块导入", t7)


# ── 测试 8：Coordinator 初始化 ─────────────────────────────────────────
def t8():
    from core.data_loader import load_instance
    from core.problem import MOCCMCKPProblem
    from agents.coordinator import MOCCMCKPCoordinator
    ps = load_instance('APP_3_5_30_')
    problem = MOCCMCKPProblem(ps)
    coord = MOCCMCKPCoordinator(problem, config={'pop_size': 5, 'max_iter': 5})
    agents = coord.list_agents()
    assert len(agents) == 6, f"应有6个Agent，实际 {len(agents)}"
    assert 'ConstraintAnalyzer' in agents
    assert 'ParetoManager' in agents

test("Coordinator 初始化 + 6 Agent 注册", t8)


# ── 测试 9：约束分析 Agent 消息处理 ───────────────────────────────────
def t9():
    from core.data_loader import load_instance
    from core.problem import MOCCMCKPProblem
    from agents.constraint_analyzer import ConstraintAnalyzerAgent
    from agents.base_agent import AgentMessage, MessageType
    ps = load_instance('APP_3_5_30_')
    problem = MOCCMCKPProblem(ps)
    agent = ConstraintAnalyzerAgent(problem=problem)
    msg = AgentMessage(
        msg_type=MessageType.TASK_START,
        sender='coordinator',
        content={}
    )
    replies = agent.process_message(msg)
    assert len(replies) > 0, "约束分析 Agent 应返回消息"
    assert replies[0].msg_type == MessageType.CONSTRAINT_REPORT

test("约束分析 Agent - TASK_START 处理", t9)


# ── 测试 10：共享内存 ──────────────────────────────────────────────────
def t10():
    from utils.shared_memory import SharedMemory
    mem = SharedMemory()
    mem.set('test/key', 42)
    assert mem.get('test/key') == 42
    mem.append('test/list', 'a')
    mem.append('test/list', 'b')
    assert mem.get_list('test/list') == ['a', 'b']
    cnt = mem.increment('test/cnt')
    assert cnt == 1

test("共享内存 - 基本读写", t10)


# ── 汇总 ────────────────────────────────────────────────────────────────
print()
print(f"Result: {passed} passed, {failed} failed")
if failed == 0:
    print("All tests passed!")
else:
    print("Some tests FAILED, please check errors above.")
    sys.exit(1)
