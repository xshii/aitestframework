#!/bin/bash
# CI 主入口脚本
# 用法: ./scripts/ci/ci.sh [command]
#
# 命令:
#   lint      - 运行代码检查
#   test      - 运行所有测试
#   coverage  - 运行测试并生成覆盖率
#   gate      - 运行质量门禁检查
#   all       - 运行完整CI流程
#   help      - 显示帮助

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

show_help() {
    echo "AI Test Framework CI 工具"
    echo ""
    echo "用法: $0 [command]"
    echo ""
    echo "命令:"
    echo "  lint      运行代码检查 (ruff, pylint)"
    echo "  test      运行所有测试"
    echo "  coverage  运行测试并生成覆盖率报告"
    echo "  gate      运行质量门禁检查"
    echo "  all       运行完整CI流程"
    echo "  setup     安装CI依赖"
    echo "  help      显示此帮助"
    echo ""
    echo "示例:"
    echo "  $0 lint           # 检查代码风格"
    echo "  $0 test           # 运行测试"
    echo "  $0 gate           # 检查质量门禁"
    echo "  $0 all            # 完整CI流程"
}

setup() {
    echo -e "${CYAN}========== 安装CI依赖 ==========${NC}"

    cd "${PROJECT_ROOT}"

    # 创建虚拟环境
    if [[ ! -d ".venv" ]]; then
        echo "创建虚拟环境..."
        python3 -m venv .venv
    fi

    # 激活虚拟环境
    source .venv/bin/activate

    # 安装依赖
    echo "安装依赖..."
    pip install --upgrade pip
    pip install -r requirements/dev.txt

    echo -e "${GREEN}✓ CI依赖安装完成${NC}"
}

run_lint() {
    echo -e "${CYAN}========== 代码检查 ==========${NC}"
    bash "${SCRIPT_DIR}/run_lint.sh" "$@"
}

run_tests() {
    echo -e "${CYAN}========== 运行测试 ==========${NC}"
    bash "${SCRIPT_DIR}/run_tests.sh" all
}

run_coverage() {
    echo -e "${CYAN}========== 测试覆盖率 ==========${NC}"
    bash "${SCRIPT_DIR}/run_tests.sh" all --coverage
}

run_gate() {
    echo -e "${CYAN}========== 质量门禁 ==========${NC}"
    bash "${SCRIPT_DIR}/quality_gate.sh"
}

run_all() {
    echo ""
    echo "========================================"
    echo "         完整 CI 流程"
    echo "========================================"
    echo ""

    local start_time=$(date +%s)
    local exit_code=0

    # 1. 代码检查
    echo -e "${CYAN}[1/3] 代码检查${NC}"
    if ! run_lint; then
        echo -e "${RED}代码检查失败${NC}"
        exit_code=1
    fi

    # 2. 测试 + 覆盖率
    echo ""
    echo -e "${CYAN}[2/3] 测试 + 覆盖率${NC}"
    if ! run_coverage; then
        echo -e "${RED}测试失败${NC}"
        exit_code=1
    fi

    # 3. 质量门禁
    echo ""
    echo -e "${CYAN}[3/3] 质量门禁${NC}"
    if ! run_gate; then
        echo -e "${RED}质量门禁未通过${NC}"
        exit_code=1
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo ""
    echo "========================================"
    echo "         CI 流程完成"
    echo "========================================"
    echo "耗时: ${duration} 秒"

    if [[ ${exit_code} -eq 0 ]]; then
        echo -e "${GREEN}✓ CI 全部通过！${NC}"
    else
        echo -e "${RED}✗ CI 存在问题，请检查${NC}"
    fi

    return ${exit_code}
}

# 主入口
case "${1:-help}" in
    lint)
        shift
        run_lint "$@"
        ;;
    test)
        run_tests
        ;;
    coverage)
        run_coverage
        ;;
    gate)
        run_gate
        ;;
    all)
        run_all
        ;;
    setup)
        setup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}未知命令: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
