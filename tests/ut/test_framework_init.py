"""
AI Test Framework - 框架初始化测试

测试 aitestframework/__init__.py 的功能
"""

import pytest
import sys


class TestFrameworkInit:
    """测试框架初始化"""

    def test_version_exists(self):
        """测试版本号存在"""
        import aitestframework
        assert hasattr(aitestframework, '__version__')
        assert aitestframework.__version__ == "0.1.0"

    def test_libs_path_in_sys_path(self):
        """测试 libs 路径已添加到 sys.path"""
        import aitestframework
        from pathlib import Path

        libs_path = Path(aitestframework.__file__).parent.parent / 'libs'
        assert str(libs_path) in sys.path

    def test_lazy_import_compare_engine(self):
        """测试延迟导入 CompareEngine"""
        # 这个测试依赖 aidevtools，如果不可用则跳过
        try:
            from aitestframework import CompareEngine
            assert CompareEngine is not None
        except (ImportError, ModuleNotFoundError):
            pytest.skip("aidevtools not available")

    def test_lazy_import_compare_status(self):
        """测试延迟导入 CompareStatus"""
        try:
            from aitestframework import CompareStatus
            assert CompareStatus is not None
        except (ImportError, ModuleNotFoundError):
            pytest.skip("aidevtools not available")

    def test_invalid_attribute_raises_error(self):
        """测试访问无效属性抛出 AttributeError"""
        import aitestframework
        with pytest.raises(AttributeError):
            _ = aitestframework.nonexistent_attribute


class TestFrameworkExports:
    """测试框架导出"""

    def test_all_exports(self):
        """测试 __all__ 包含必要导出"""
        import aitestframework
        assert '__version__' in aitestframework.__all__
