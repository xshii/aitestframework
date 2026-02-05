"""
日志管理模块

提供：
- 日志级别控制
- 多输出目标（控制台、文件）
- 结构化日志（JSON）
- 测试用例日志隔离
"""

import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
from contextlib import contextmanager

# 框架日志器名称
LOGGER_NAME = "aitest"

# 全局日志配置状态
_configured = False


class JsonFormatter(logging.Formatter):
    """JSON 格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 添加额外字段
        if hasattr(record, "test_name"):
            log_data["test_name"] = record.test_name
        if hasattr(record, "test_id"):
            log_data["test_id"] = record.test_id

        # 异常信息
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


class TestContextFilter(logging.Filter):
    """测试上下文过滤器，添加测试相关信息"""

    def __init__(self):
        super().__init__()
        self.test_name: Optional[str] = None
        self.test_id: Optional[str] = None

    def filter(self, record: logging.LogRecord) -> bool:
        record.test_name = self.test_name
        record.test_id = self.test_id
        return True

    def set_context(self, test_name: str, test_id: str):
        self.test_name = test_name
        self.test_id = test_id

    def clear_context(self):
        self.test_name = None
        self.test_id = None


# 全局上下文过滤器
_context_filter = TestContextFilter()


def configure_logging(
    level: Union[str, int] = "INFO",
    format_str: Optional[str] = None,
    log_file: Optional[str] = None,
    json_format: bool = False,
) -> logging.Logger:
    """
    配置日志系统

    Args:
        level: 日志级别
        format_str: 日志格式字符串
        log_file: 日志文件路径
        json_format: 是否使用 JSON 格式

    Returns:
        配置好的日志器
    """
    global _configured

    logger = logging.getLogger(LOGGER_NAME)

    # 清除现有处理器
    logger.handlers.clear()

    # 设置级别
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)

    # 创建格式化器
    if json_format:
        formatter = JsonFormatter()
    else:
        if format_str is None:
            format_str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        formatter = logging.Formatter(format_str)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_context_filter)
    logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_context_filter)
        logger.addHandler(file_handler)

    # 不传播到根日志器
    logger.propagate = False

    _configured = True
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取日志器

    Args:
        name: 日志器名称，为 None 时返回框架主日志器

    Returns:
        日志器实例
    """
    global _configured

    if not _configured:
        configure_logging()

    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


@contextmanager
def test_log_context(test_name: str, test_id: str):
    """
    测试日志上下文管理器

    在此上下文中的所有日志都会包含测试信息

    Args:
        test_name: 测试名称
        test_id: 测试 ID
    """
    _context_filter.set_context(test_name, test_id)
    try:
        yield
    finally:
        _context_filter.clear_context()


class TestLogger:
    """
    测试专用日志器

    为单个测试用例提供独立的日志记录
    """

    def __init__(self, test_name: str, log_dir: Optional[str] = None):
        self.test_name = test_name
        self.log_dir = Path(log_dir) if log_dir else None
        self.logs: list = []
        self._logger = get_logger(f"test.{test_name}")

    def _log(self, level: int, message: str, **kwargs):
        """内部日志方法"""
        self.logs.append({
            "timestamp": datetime.utcnow().isoformat(),
            "level": logging.getLevelName(level),
            "message": message,
            **kwargs
        })
        self._logger.log(level, message)

    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)

    def get_logs(self) -> list:
        """获取所有日志记录"""
        return self.logs.copy()

    def save(self, filename: Optional[str] = None):
        """保存日志到文件"""
        if self.log_dir is None:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"{self.test_name}.json"

        log_file = self.log_dir / filename
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)
