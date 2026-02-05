"""
AI Test Framework - 单元测试配置
"""

import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 确保 libs 在 Python 路径中
_libs_path = _project_root / 'libs'
if str(_libs_path) not in sys.path:
    sys.path.insert(0, str(_libs_path))
