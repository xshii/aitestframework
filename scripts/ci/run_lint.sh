#!/bin/bash
# 代码检查脚本
# 用法: ./scripts/ci/run_lint.sh [--fix]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "           代码质量检查"
echo "=========================================="

FIX_MODE=false
if [[ "$1" == "--fix" ]]; then
    FIX_MODE=true
    echo "模式: 自动修复"
else
    echo "模式: 仅检查 (使用 --fix 自动修复)"
fi

# 确保在虚拟环境中
if [[ -z "${VIRTUAL_ENV}" ]]; then
    if [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
    fi
fi

EXIT_CODE=0

# 1. Ruff 检查
echo ""
echo -e "${YELLOW}[1/3] Ruff 快速检查${NC}"
echo "----------------------------------------"

if $FIX_MODE; then
    ruff check aitestframework libs/aidevtools --fix || true
    ruff format aitestframework libs/aidevtools || true
else
    if ruff check aitestframework libs/aidevtools; then
        echo -e "${GREEN}✓ Ruff 检查通过${NC}"
    else
        echo -e "${RED}✗ Ruff 发现问题${NC}"
        EXIT_CODE=1
    fi
fi

# 2. Pylint 检查
echo ""
echo -e "${YELLOW}[2/3] Pylint 代码检查${NC}"
echo "----------------------------------------"

PYLINT_THRESHOLD=8.0
PYLINT_OUTPUT=$(pylint aitestframework --exit-zero 2>&1) || true
PYLINT_SCORE=$(echo "${PYLINT_OUTPUT}" | grep "Your code has been rated" | sed 's/.*rated at \([0-9.]*\).*/\1/' || echo "0")

echo "Pylint 评分: ${PYLINT_SCORE}/10.0 (阈值: ${PYLINT_THRESHOLD})"

if (( $(echo "${PYLINT_SCORE} >= ${PYLINT_THRESHOLD}" | bc -l) )); then
    echo -e "${GREEN}✓ Pylint 评分达标${NC}"
else
    echo -e "${RED}✗ Pylint 评分不达标${NC}"
    EXIT_CODE=1
fi

# 3. 类型检查（可选，如果安装了mypy）
echo ""
echo -e "${YELLOW}[3/3] 类型检查 (可选)${NC}"
echo "----------------------------------------"

if command -v mypy &> /dev/null; then
    if mypy aitestframework --ignore-missing-imports --no-error-summary 2>/dev/null; then
        echo -e "${GREEN}✓ 类型检查通过${NC}"
    else
        echo -e "${YELLOW}⚠ 类型检查有警告${NC}"
    fi
else
    echo "跳过 (mypy 未安装)"
fi

# 汇总
echo ""
echo "=========================================="
echo "           检查结果汇总"
echo "=========================================="

if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo -e "${GREEN}✓ 所有检查通过${NC}"
else
    echo -e "${RED}✗ 存在问题，请修复后重新提交${NC}"
fi

exit ${EXIT_CODE}
