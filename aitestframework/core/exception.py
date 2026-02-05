"""
异常处理模块

定义框架的所有异常类型
"""

from typing import Optional, Any


class AITestError(Exception):
    """框架基础异常"""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigError(AITestError):
    """配置错误"""
    pass


class DiscoveryError(AITestError):
    """测试发现错误"""
    pass


class ExecutionError(AITestError):
    """执行错误"""
    pass


class TimeoutError(AITestError):
    """超时错误"""

    def __init__(self, message: str, timeout: float, test_name: Optional[str] = None):
        super().__init__(message, {"timeout": timeout, "test_name": test_name})
        self.timeout = timeout
        self.test_name = test_name


class FixtureError(AITestError):
    """Fixture 错误"""

    def __init__(self, message: str, fixture_name: Optional[str] = None):
        super().__init__(message, {"fixture_name": fixture_name})
        self.fixture_name = fixture_name


class SkipTest(AITestError):
    """跳过测试（非错误）"""

    def __init__(self, reason: str = ""):
        super().__init__(reason)
        self.reason = reason


class TestFailure(AITestError):
    """测试失败（断言失败）"""

    def __init__(
        self,
        message: str,
        expected: Any = None,
        actual: Any = None,
        diff: Optional[str] = None
    ):
        details = {}
        if expected is not None:
            details["expected"] = expected
        if actual is not None:
            details["actual"] = actual
        if diff is not None:
            details["diff"] = diff
        super().__init__(message, details)
        self.expected = expected
        self.actual = actual
        self.diff = diff
