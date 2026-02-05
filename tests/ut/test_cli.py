"""
AI Test Framework - CLI 测试

测试 aitestframework/cli.py 的功能
"""

import pytest
from io import StringIO
import sys


class TestCLI:
    """测试 CLI 入口"""

    def test_main_returns_zero(self):
        """测试 main 函数返回 0"""
        from aitestframework.cli import main
        assert main() == 0

    def test_main_prints_version(self, capsys):
        """测试 main 函数打印版本信息"""
        from aitestframework.cli import main
        main()
        captured = capsys.readouterr()
        assert "AI Test Framework v0.1.0" in captured.out

    def test_main_prints_commands(self, capsys):
        """测试 main 函数打印可用命令"""
        from aitestframework.cli import main
        main()
        captured = capsys.readouterr()
        assert "aitest run" in captured.out
        assert "aitest list" in captured.out
        assert "aitest report" in captured.out

    def test_main_mentions_aidevtools(self, capsys):
        """测试 main 函数提及 aidevtools"""
        from aitestframework.cli import main
        main()
        captured = capsys.readouterr()
        assert "aidevtools" in captured.out
        assert "aidev" in captured.out
