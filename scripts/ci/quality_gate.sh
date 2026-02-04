#!/bin/bash
# 质量门禁检查脚本
# 用法: ./scripts/ci/quality_gate.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "=========================================="
echo "           质量门禁检查"
echo "=========================================="

# 门禁阈值配置
COVERAGE_THRESHOLD=80
PYLINT_THRESHOLD=8.0
MAX_RUFF_ERRORS=0

# 确保在虚拟环境中
if [[ -z "${VIRTUAL_ENV}" ]]; then
    if [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
    fi
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/libs:${PYTHONPATH}"

GATE_PASSED=true
declare -a GATE_RESULTS

# 1. 单元测试
echo ""
echo -e "${CYAN}[1/4] 单元测试${NC}"
echo "----------------------------------------"

mkdir -p test-results
if pytest aitestframework/ -v --tb=line --junitxml=test-results/unit-tests.xml 2>/dev/null; then
    GATE_RESULTS+=("${GREEN}✓${NC} 单元测试: 全部通过")
else
    GATE_RESULTS+=("${RED}✗${NC} 单元测试: 存在失败")
    GATE_PASSED=false
fi

# 2. 覆盖率
echo ""
echo -e "${CYAN}[2/4] 代码覆盖率${NC}"
echo "----------------------------------------"

mkdir -p coverage
pytest aitestframework/ --cov=aitestframework --cov-report=xml:coverage/coverage.xml --cov-report=term -q 2>/dev/null || true

if [[ -f "coverage/coverage.xml" ]]; then
    COVERAGE_PERCENT=$(python3 -c "
import xml.etree.ElementTree as ET
tree = ET.parse('coverage/coverage.xml')
root = tree.getroot()
line_rate = float(root.get('line-rate', 0))
print(f'{line_rate * 100:.1f}')
" 2>/dev/null || echo "0")

    echo "覆盖率: ${COVERAGE_PERCENT}%"

    if (( $(echo "${COVERAGE_PERCENT} >= ${COVERAGE_THRESHOLD}" | bc -l) )); then
        GATE_RESULTS+=("${GREEN}✓${NC} 覆盖率: ${COVERAGE_PERCENT}% ≥ ${COVERAGE_THRESHOLD}%")
    else
        GATE_RESULTS+=("${RED}✗${NC} 覆盖率: ${COVERAGE_PERCENT}% < ${COVERAGE_THRESHOLD}%")
        GATE_PASSED=false
    fi
else
    GATE_RESULTS+=("${YELLOW}⚠${NC} 覆盖率: 无法获取")
fi

# 3. Pylint评分
echo ""
echo -e "${CYAN}[3/4] Pylint 代码质量${NC}"
echo "----------------------------------------"

PYLINT_OUTPUT=$(pylint aitestframework --exit-zero 2>&1) || true
PYLINT_SCORE=$(echo "${PYLINT_OUTPUT}" | grep "Your code has been rated" | sed 's/.*rated at \([0-9.]*\).*/\1/' || echo "0")

echo "Pylint评分: ${PYLINT_SCORE}/10"

if (( $(echo "${PYLINT_SCORE} >= ${PYLINT_THRESHOLD}" | bc -l) )); then
    GATE_RESULTS+=("${GREEN}✓${NC} Pylint: ${PYLINT_SCORE} ≥ ${PYLINT_THRESHOLD}")
else
    GATE_RESULTS+=("${RED}✗${NC} Pylint: ${PYLINT_SCORE} < ${PYLINT_THRESHOLD}")
    GATE_PASSED=false
fi

# 4. Ruff检查
echo ""
echo -e "${CYAN}[4/4] Ruff 代码检查${NC}"
echo "----------------------------------------"

RUFF_ERRORS=$(ruff check aitestframework libs/aidevtools --output-format=json 2>/dev/null | python3 -c "import json,sys; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")

echo "Ruff问题数: ${RUFF_ERRORS}"

if [[ "${RUFF_ERRORS}" -le "${MAX_RUFF_ERRORS}" ]]; then
    GATE_RESULTS+=("${GREEN}✓${NC} Ruff: ${RUFF_ERRORS} 个问题 ≤ ${MAX_RUFF_ERRORS}")
else
    GATE_RESULTS+=("${YELLOW}⚠${NC} Ruff: ${RUFF_ERRORS} 个问题 > ${MAX_RUFF_ERRORS} (警告)")
    # Ruff暂时不阻断
fi

# 汇总
echo ""
echo "=========================================="
echo "           质量门禁结果"
echo "=========================================="

for result in "${GATE_RESULTS[@]}"; do
    echo -e "  ${result}"
done

echo ""
echo "----------------------------------------"

if $GATE_PASSED; then
    echo -e "${GREEN}✓ 质量门禁通过！可以合入代码${NC}"
    exit 0
else
    echo -e "${RED}✗ 质量门禁未通过！请修复问题后重新提交${NC}"
    exit 1
fi
