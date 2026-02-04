"""
AI Test Framework - AI模型测试框架

集成 aidevtools 算子验证工具，提供完整的模型测试解决方案。

模块结构 (按需求文档):
- core/       CORE - 核心框架
- model/      MODEL - AI模型测试
- data/       DATA - 数据管理
- assertion/  ASSERT - 断言与验证
- report/     REPORT - 报告生成
- integration/INTEG - 集成与部署
- extension/  EXT - 扩展性
- adt/        ADT - aidevtools集成适配层

外部库 (libs/):
- aidevtools: 算子验证工具集
- prettycli: CLI美化工具

使用示例:
    from aitestframework.adt import CompareEngine, CompareStatus
    from libs.aidevtools.compare import CompareConfig
"""

__version__ = "0.1.0"

import sys
from pathlib import Path

# 添加libs到Python路径
_libs_path = Path(__file__).parent.parent / 'libs'
if str(_libs_path) not in sys.path:
    sys.path.insert(0, str(_libs_path))

__all__ = [
    "__version__",
]

# 延迟导入aidevtools组件（可选依赖）
def __getattr__(name):
    """延迟导入aidevtools组件"""
    _aidevtools_exports = {
        'CompareEngine': ('aidevtools.compare', 'CompareEngine'),
        'CompareConfig': ('aidevtools.compare', 'CompareConfig'),
        'CompareResult': ('aidevtools.compare', 'CompareResult'),
        'CompareStatus': ('aidevtools.compare.types', 'CompareStatus'),
    }
    if name in _aidevtools_exports:
        module_name, attr_name = _aidevtools_exports[name]
        import importlib
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
