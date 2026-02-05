"""
AI Test Framework - 核心模块

提供测试框架的核心功能：
- 配置加载
- 测试发现
- 执行调度
- 生命周期管理
- Fixture 机制
- 异常处理
- 日志管理
"""

from .config import Config, load_config
from .discovery import discover_tests, test, testclass, skip, skipif, tag
from .scheduler import Scheduler, TestResult, TestStatus
from .lifecycle import setup, teardown, setup_module, teardown_module
from .fixture import fixture, Scope
from .exception import (
    AITestError,
    ConfigError,
    DiscoveryError,
    ExecutionError,
    TimeoutError,
    FixtureError,
)
from .logging import get_logger, configure_logging
from .engine import Engine

__all__ = [
    # Config
    "Config",
    "load_config",
    # Discovery
    "discover_tests",
    "test",
    "testclass",
    "skip",
    "skipif",
    "tag",
    # Scheduler
    "Scheduler",
    "TestResult",
    "TestStatus",
    # Lifecycle
    "setup",
    "teardown",
    "setup_module",
    "teardown_module",
    # Fixture
    "fixture",
    "Scope",
    # Exception
    "AITestError",
    "ConfigError",
    "DiscoveryError",
    "ExecutionError",
    "TimeoutError",
    "FixtureError",
    # Logging
    "get_logger",
    "configure_logging",
    # Engine
    "Engine",
]
