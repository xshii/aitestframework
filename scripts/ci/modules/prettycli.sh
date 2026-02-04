#!/bin/bash
# prettycli 模块 CI 脚本
# 用法: ./scripts/ci/modules/prettycli.sh [test|lint|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODULE_ROOT="${PROJECT_ROOT}/libs/prettycli"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

MODULE_NAME="prettycli"

echo -e "${CYAN}========== ${MODULE_NAME} CI ==========${NC}"

# 检查模块是否存在
if [[ ! -d "${MODULE_ROOT}" ]]; then
    echo -e "${YELLOW}模块目录不存在: ${MODULE_ROOT}${NC}"
    echo -e "${YELLOW}跳过 ${MODULE_NAME} CI${NC}"
    exit 0
fi

# 设置环境
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/libs:${PYTHONPATH}"

run_lint() {
    echo -e "${YELLOW}[lint] 代码检查${NC}"

    cd "${PROJECT_ROOT}"

    # Ruff
    echo "运行 ruff..."
    ruff check "${MODULE_ROOT}" --exit-zero || true

    echo -e "${GREEN}✓ 代码检查完成${NC}"
}

run_test() {
    echo -e "${YELLOW}[test] 运行测试${NC}"

    cd "${PROJECT_ROOT}"

    local test_dir="${MODULE_ROOT}/tests"
    if [[ ! -d "${test_dir}" ]]; then
        echo -e "${YELLOW}测试目录不存在，跳过${NC}"
        return 0
    fi

    # 激活虚拟环境
    if [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
    fi

    pytest "${test_dir}" \
        -v --tb=short \
        --junitxml="test-results/${MODULE_NAME}-tests.xml" \
        || return $?

    echo -e "${GREEN}✓ 测试完成${NC}"
}

run_all() {
    local exit_code=0

    run_lint || exit_code=$?
    run_test || exit_code=$?

    return ${exit_code}
}

# 主入口
case "${1:-all}" in
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
        echo "用法: $0 [lint|test|all]"
        exit 1
        ;;
esac
