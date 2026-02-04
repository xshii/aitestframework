"""
AI Test Framework 命令行入口

使用方式:
    aitest run tests/          # 执行测试
    aitest list tests/         # 列出测试用例
    aitest report results/     # 生成报告
"""

import sys


def main():
    """CLI 主入口"""
    print("AI Test Framework v0.1.0")
    print("状态: 开发中")
    print()
    print("可用命令:")
    print("  aitest run <path>     执行测试")
    print("  aitest list <path>    列出测试用例")
    print("  aitest report <path>  生成报告")
    print()
    print("aidevtools 工具已集成，可通过 aidev 命令使用:")
    print("  aidev compare         执行比对")
    print("  aidev golden          生成 Golden")
    return 0


if __name__ == "__main__":
    sys.exit(main())
