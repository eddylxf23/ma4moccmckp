"""
shared_memory.py — Agent 间共享状态管理

提供线程安全的键值存储，作为所有 Agent 之间共享上下文的统一接口。

设计原则：
  - 轻量级：仅封装 dict + 读写锁
  - 类型无关：存储任意 Python 对象
  - 命名空间：支持按 Agent 分区隔离数据
  - 快照：支持导出当前完整状态用于持久化或调试
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional


class SharedMemory:
    """
    线程安全的 Agent 共享内存。

    所有 Agent 通过 ``get``/``set`` 方法访问共享状态。
    支持命名空间前缀（如 ``"pareto/front"``、``"scheduler/iter"``）来隔离不同 Agent 的数据。

    Examples
    --------
    >>> mem = SharedMemory()
    >>> mem.set("scheduler/iter", 0)
    >>> mem.set("pareto/front", [])
    >>> current_iter = mem.get("scheduler/iter", default=0)
    """

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._history: List[Dict[str, Any]] = []  # 变更历史（用于调试）

    # ── 基本读写 ───────────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """读取键值，键不存在时返回 default。"""
        with self._lock:
            return self._store.get(key, default)

    def set(self, key: str, value: Any, record_history: bool = False) -> None:
        """写入键值。``record_history=True`` 时记录变更历史。"""
        with self._lock:
            if record_history:
                old = self._store.get(key)
                self._history.append({
                    "key": key,
                    "old": old,
                    "new": value,
                    "ts": time.time(),
                })
            self._store[key] = value

    def delete(self, key: str) -> bool:
        """删除一个键，返回是否成功。"""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def exists(self, key: str) -> bool:
        """检查键是否存在。"""
        with self._lock:
            return key in self._store

    # ── 批量操作 ───────────────────────────────────────────────────────────────

    def update(self, data: Dict[str, Any]) -> None:
        """批量写入键值对。"""
        with self._lock:
            self._store.update(data)

    def keys(self, prefix: Optional[str] = None) -> List[str]:
        """列出所有键，可按前缀过滤。"""
        with self._lock:
            if prefix is None:
                return list(self._store.keys())
            return [k for k in self._store if k.startswith(prefix)]

    def snapshot(self) -> Dict[str, Any]:
        """导出当前完整状态快照（浅拷贝）。"""
        with self._lock:
            return dict(self._store)

    def clear(self, prefix: Optional[str] = None) -> None:
        """清空存储，可按前缀清空指定命名空间。"""
        with self._lock:
            if prefix is None:
                self._store.clear()
            else:
                keys_to_delete = [k for k in self._store if k.startswith(prefix)]
                for k in keys_to_delete:
                    del self._store[k]

    # ── 原子增减（计数器） ──────────────────────────────────────────────────────

    def increment(self, key: str, delta: int = 1, default: int = 0) -> int:
        """原子自增，返回更新后的值。"""
        with self._lock:
            current = self._store.get(key, default)
            new_val = current + delta
            self._store[key] = new_val
            return new_val

    # ── 列表追加（用于迭代历史记录） ─────────────────────────────────────────────

    def append(self, key: str, item: Any, max_len: Optional[int] = None) -> int:
        """
        向列表类型的键追加元素。若键不存在则创建新列表。
        ``max_len`` 限制列表最大长度（超出时移除最早元素）。
        返回追加后的列表长度。
        """
        with self._lock:
            lst = self._store.get(key, [])
            if not isinstance(lst, list):
                lst = [lst]
            lst.append(item)
            if max_len is not None and len(lst) > max_len:
                lst = lst[-max_len:]
            self._store[key] = lst
            return len(lst)

    def get_list(self, key: str) -> List[Any]:
        """安全获取列表（若不存在返回空列表）。"""
        with self._lock:
            val = self._store.get(key, [])
            return list(val) if isinstance(val, list) else [val]

    # ── 常用键名常量 ──────────────────────────────────────────────────────────

    class Keys:
        """所有 Agent 约定的共享内存键名。"""
        # 约束分析结果
        CONSTRAINT_REPORT = "constraint/report"
        PROBLEM_DIFFICULTY = "constraint/difficulty"

        # 种群和 Pareto 前沿
        CURRENT_POPULATION = "population/current"
        PARETO_FRONT = "pareto/front"
        PARETO_HISTORY = "pareto/history"

        # 调度器状态
        ITER_COUNT = "scheduler/iter"
        STAGNATION_COUNT = "scheduler/stagnation"
        STRATEGY = "scheduler/strategy"
        IS_CONVERGED = "scheduler/converged"

        # 统计
        EVAL_COUNT = "stats/eval_count"
        REPAIR_COUNT = "stats/repair_count"

    def __repr__(self) -> str:
        with self._lock:
            return f"SharedMemory(keys={list(self._store.keys())})"
