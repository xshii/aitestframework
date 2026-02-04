"""
AI Test Framework - AI模型测试框架

集成 aidevtools 算子验证工具，提供完整的模型测试解决方案。

主要功能:
- 测试用例发现与管理
- 测试执行调度（顺序/并行）
- 四状态比对验证（PASS/GOLDEN_SUSPECT/DUT_ISSUE/BOTH_SUSPECT）
- 多格式报告生成
- CI/CD 集成

使用示例:
    from aitestframework import TestRunner
    from aidevtools.compare import CompareEngine

    runner = TestRunner()
    results = runner.run("tests/")
"""

__version__ = "0.1.0"

# 导出 aidevtools 核心组件，方便使用
from aidevtools.compare import (
    CompareEngine,
    CompareConfig,
    CompareResult,
    CompareStatus,
)
from aidevtools.ops import _functional as F
from aidevtools.frontend import DataGenerator

__all__ = [
    # 版本
    "__version__",
    # aidevtools 核心组件
    "CompareEngine",
    "CompareConfig",
    "CompareResult",
    "CompareStatus",
    "F",
    "DataGenerator",
]
