"""
配置加载模块

支持：
- YAML/JSON 配置文件加载
- 环境变量覆盖
- 运行环境检测
"""

import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    file: Optional[str] = None
    json_format: bool = False


@dataclass
class ExecutionConfig:
    """执行配置"""
    timeout: int = 300  # 默认5分钟超时
    parallel: bool = False
    workers: int = 1
    fail_fast: bool = False
    retry_count: int = 0


@dataclass
class DiscoveryConfig:
    """发现配置"""
    patterns: list = field(default_factory=lambda: ["test_*.py", "*_test.py"])
    ignore_patterns: list = field(default_factory=lambda: ["__pycache__", ".git", ".venv"])
    recursive: bool = True


@dataclass
class Config:
    """框架主配置"""
    # 基础配置
    project_name: str = "aitestframework"
    version: str = "0.1.0"
    test_dirs: list = field(default_factory=lambda: ["tests"])

    # 子配置
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)

    # 环境信息（运行时填充）
    python_version: str = ""
    platform: str = ""
    gpu_available: bool = False

    # 自定义配置
    custom: dict = field(default_factory=dict)

    def __post_init__(self):
        """初始化后检测环境"""
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.platform = sys.platform
        self._detect_gpu()

    def _detect_gpu(self):
        """检测 GPU 可用性"""
        try:
            import torch
            self.gpu_available = torch.cuda.is_available()
        except ImportError:
            self.gpu_available = False

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项，支持点号分隔的路径"""
        parts = key.split(".")
        value = self
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            elif isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "project_name": self.project_name,
            "version": self.version,
            "test_dirs": self.test_dirs,
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file": self.logging.file,
                "json_format": self.logging.json_format,
            },
            "execution": {
                "timeout": self.execution.timeout,
                "parallel": self.execution.parallel,
                "workers": self.execution.workers,
                "fail_fast": self.execution.fail_fast,
                "retry_count": self.execution.retry_count,
            },
            "discovery": {
                "patterns": self.discovery.patterns,
                "ignore_patterns": self.discovery.ignore_patterns,
                "recursive": self.discovery.recursive,
            },
            "python_version": self.python_version,
            "platform": self.platform,
            "gpu_available": self.gpu_available,
            "custom": self.custom,
        }


def load_config(
    config_file: Optional[str] = None,
    env_prefix: str = "AITEST_"
) -> Config:
    """
    加载配置

    优先级（从低到高）：
    1. 默认值
    2. 配置文件
    3. 环境变量

    Args:
        config_file: 配置文件路径，支持 .yaml/.yml/.json
        env_prefix: 环境变量前缀

    Returns:
        Config 对象
    """
    config = Config()

    # 自动查找配置文件
    if config_file is None:
        for name in ["aitest.yaml", "aitest.yml", "aitest.json", ".aitest.yaml"]:
            if Path(name).exists():
                config_file = name
                break

    # 加载配置文件
    if config_file and Path(config_file).exists():
        config = _load_from_file(config_file, config)

    # 环境变量覆盖
    config = _apply_env_overrides(config, env_prefix)

    return config


def _load_from_file(file_path: str, config: Config) -> Config:
    """从文件加载配置"""
    path = Path(file_path)

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(f) or {}
        elif path.suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

    # 更新配置
    if "project_name" in data:
        config.project_name = data["project_name"]
    if "version" in data:
        config.version = data["version"]
    if "test_dirs" in data:
        config.test_dirs = data["test_dirs"]

    # 日志配置
    if "logging" in data:
        log_data = data["logging"]
        if "level" in log_data:
            config.logging.level = log_data["level"]
        if "format" in log_data:
            config.logging.format = log_data["format"]
        if "file" in log_data:
            config.logging.file = log_data["file"]
        if "json_format" in log_data:
            config.logging.json_format = log_data["json_format"]

    # 执行配置
    if "execution" in data:
        exec_data = data["execution"]
        if "timeout" in exec_data:
            config.execution.timeout = exec_data["timeout"]
        if "parallel" in exec_data:
            config.execution.parallel = exec_data["parallel"]
        if "workers" in exec_data:
            config.execution.workers = exec_data["workers"]
        if "fail_fast" in exec_data:
            config.execution.fail_fast = exec_data["fail_fast"]
        if "retry_count" in exec_data:
            config.execution.retry_count = exec_data["retry_count"]

    # 发现配置
    if "discovery" in data:
        disc_data = data["discovery"]
        if "patterns" in disc_data:
            config.discovery.patterns = disc_data["patterns"]
        if "ignore_patterns" in disc_data:
            config.discovery.ignore_patterns = disc_data["ignore_patterns"]
        if "recursive" in disc_data:
            config.discovery.recursive = disc_data["recursive"]

    # 自定义配置
    if "custom" in data:
        config.custom = data["custom"]

    return config


def _apply_env_overrides(config: Config, prefix: str) -> Config:
    """应用环境变量覆盖"""
    env_mappings = {
        f"{prefix}LOG_LEVEL": ("logging", "level"),
        f"{prefix}LOG_FILE": ("logging", "file"),
        f"{prefix}TIMEOUT": ("execution", "timeout", int),
        f"{prefix}PARALLEL": ("execution", "parallel", lambda x: x.lower() == "true"),
        f"{prefix}WORKERS": ("execution", "workers", int),
        f"{prefix}FAIL_FAST": ("execution", "fail_fast", lambda x: x.lower() == "true"),
        f"{prefix}RETRY_COUNT": ("execution", "retry_count", int),
    }

    for env_var, mapping in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            section = mapping[0]
            attr = mapping[1]
            converter = mapping[2] if len(mapping) > 2 else str

            section_obj = getattr(config, section)
            setattr(section_obj, attr, converter(value))

    return config
