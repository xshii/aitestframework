"""
ADT: aidevtools集成模块 (08-aidevtools-integration.yaml)

适配层，从libs/aidevtools导入核心功能：
- ADT-001: CompareEngine集成 (四状态比对)
- ADT-002: ops算子Golden生成
- ADT-003: 量化格式支持 (BFP/GFloat)
- ADT-004: 数据生成集成
- ADT-005: xlsx工作流集成
- ADT-006: 比对报告集成
- ADT-007: 四状态断言

用法:
    from aitestframework.adt import CompareEngine, CompareStatus
    # 或直接使用
    from libs.aidevtools.compare import CompareEngine
"""

import sys
from pathlib import Path

# 添加libs到Python路径
_libs_path = Path(__file__).parent.parent.parent / 'libs'
if str(_libs_path) not in sys.path:
    sys.path.insert(0, str(_libs_path))

# 从libs/aidevtools导入
try:
    from aidevtools.compare import (
        CompareEngine,
        CompareConfig,
    )
    from aidevtools.compare.types import CompareStatus
    from aidevtools.ops import _functional as F
    from aidevtools.ops.base import get_records, clear as ops_clear

    __all__ = [
        'CompareEngine',
        'CompareConfig',
        'CompareStatus',
        'F',
        'get_records',
        'ops_clear',
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"aidevtools导入失败: {e}")
    __all__ = []
