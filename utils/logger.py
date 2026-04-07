"""
logger.py — 结构化日志系统

提供统一的日志配置，支持：
  - 控制台彩色输出
  - 文件日志（可选）
  - Agent 消息追踪日志
  - 进度条集成
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional


# ── ANSI 颜色代码 ──────────────────────────────────────────────────────────────
_COLORS = {
    "DEBUG": "\033[36m",    # 青色
    "INFO": "\033[32m",     # 绿色
    "WARNING": "\033[33m",  # 黄色
    "ERROR": "\033[31m",    # 红色
    "CRITICAL": "\033[35m", # 紫色
    "RESET": "\033[0m",
}


class ColoredFormatter(logging.Formatter):
    """带颜色的控制台日志格式化器。"""

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        reset = _COLORS["RESET"]
        record.levelname = f"{color}{record.levelname:8s}{reset}"
        return super().format(record)


def setup_logger(
    name: str = "moccmckp",
    level: str = "INFO",
    log_file: Optional[str] = None,
    colored: bool = True,
) -> logging.Logger:
    """
    配置并返回项目根日志器。

    Parameters
    ----------
    name : str
        日志器名称（通常为项目名）。
    level : str
        日志级别：DEBUG / INFO / WARNING / ERROR。
    log_file : str, optional
        输出日志文件路径，为 None 时仅输出到控制台。
    colored : bool
        是否启用彩色控制台输出。

    Returns
    -------
    logging.Logger
        配置完成的日志器实例。
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_fmt = "%H:%M:%S"

    # ── 控制台 Handler ─────────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    if colored and sys.stdout.isatty():
        formatter = ColoredFormatter(fmt, datefmt=date_fmt)
    else:
        formatter = logging.Formatter(fmt, datefmt=date_fmt)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── 文件 Handler（可选） ───────────────────────────────────────────────────
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
        logger.addHandler(file_handler)

    return logger


class ProgressLogger:
    """简易进度日志器，用于显示优化迭代进度。"""

    def __init__(
        self,
        total: int,
        prefix: str = "优化进度",
        log_every: int = 10,
        logger: Optional[logging.Logger] = None,
    ):
        self.total = total
        self.prefix = prefix
        self.log_every = log_every
        self._logger = logger or logging.getLogger("progress")
        self._start = time.time()
        self._current = 0

    def update(self, step: int, extra: str = "") -> None:
        """更新进度并在必要时打印日志。"""
        self._current = step
        if step % self.log_every == 0 or step == self.total:
            elapsed = time.time() - self._start
            pct = 100.0 * step / self.total if self.total > 0 else 0
            eta = (elapsed / step * (self.total - step)) if step > 0 else 0
            msg = (
                f"{self.prefix}: {step}/{self.total} ({pct:.1f}%) "
                f"| 已用 {elapsed:.1f}s | 预计剩余 {eta:.1f}s"
            )
            if extra:
                msg += f" | {extra}"
            self._logger.info(msg)
