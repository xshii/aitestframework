#!/bin/bash
# aidevtools 模块 CI 脚本
# 用法: ./scripts/ci/modules/aidevtools.sh [test|build|lint|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODULE_ROOT="${PROJECT_ROOT}/libs/aidevtools"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

MODULE_NAME="aidevtools"

echo -e "${CYAN}========== ${MODULE_NAME} CI ==========${NC}"

# 切换到项目根目录
cd "${PROJECT_ROOT}"

# 激活虚拟环境（如果存在）
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    echo "使用虚拟环境: .venv"
fi

# 设置环境
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/libs:${PYTHONPATH:-}"

build_cpu_golden() {
    echo -e "${YELLOW}[build] 编译 cpu_golden${NC}"

    if ! command -v cmake &> /dev/null; then
        echo -e "${YELLOW}cmake 未安装，跳过编译${NC}"
        return 0
    fi

    local golden_dir="${MODULE_ROOT}/golden/cpp"
    if [[ -f "${golden_dir}/build.sh" ]]; then
        cd "${golden_dir}"
        chmod +x build.sh
        if ./build.sh; then
            echo -e "${GREEN}✓ cpu_golden 编译成功${NC}"
        else
            echo -e "${YELLOW}⚠ cpu_golden 编译失败，跳过相关测试${NC}"
        fi
    else
        echo -e "${YELLOW}build.sh 不存在，跳过${NC}"
    fi
}

run_lint() {
    echo -e "${YELLOW}[lint] 代码检查${NC}"

    cd "${PROJECT_ROOT}"

    # Ruff
    echo "运行 ruff..."
    ruff check "${MODULE_ROOT}" --exit-zero || true

    # Pylint
    echo "运行 pylint..."
    pylint "${MODULE_ROOT}" --exit-zero || true

    echo -e "${GREEN}✓ 代码检查完成${NC}"
}

run_test() {
    echo -e "${YELLOW}[test] 运行测试${NC}"

    local test_dir="${MODULE_ROOT}/tests"
    if [[ ! -d "${test_dir}" ]]; then
        echo -e "${RED}测试目录不存在: ${test_dir}${NC}"
        return 1
    fi

    python3 -m pytest "${test_dir}" \
        -v --tb=short \
        --junitxml="test-results/${MODULE_NAME}-tests.xml" \
        || return $?

    echo -e "${GREEN}✓ 测试完成${NC}"
}

run_all() {
    local exit_code=0

    build_cpu_golden || exit_code=$?
    run_lint || exit_code=$?
    run_test || exit_code=$?

    return ${exit_code}
}

# 主入口
case "${1:-all}" in
    build)
        build_cpu_golden
        ;;
    lint)
        run_lint
        ;;
    test)
        run_test
        ;;
    all)
        run_all
        ;;
    *)
        echo "用法: $0 [build|lint|test|all]"
        exit 1
        ;;
esac
