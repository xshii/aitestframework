#!/bin/bash
# 测试运行脚本
# 用法: ./scripts/ci/run_tests.sh [unit|integration|all] [--coverage]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 参数解析
TEST_TYPE="${1:-all}"
COVERAGE=false
for arg in "$@"; do
    if [[ "$arg" == "--coverage" ]]; then
        COVERAGE=true
    fi
done

echo "=========================================="
echo "           测试运行"
echo "=========================================="
echo "类型: ${TEST_TYPE}"
echo "覆盖率: ${COVERAGE}"

# 确保在虚拟环境中
if [[ -z "${VIRTUAL_ENV}" ]]; then
    if [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
    fi
fi

# 设置PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/libs:${PYTHONPATH}"

# 创建输出目录
mkdir -p test-results coverage

EXIT_CODE=0

# 构建pytest参数
PYTEST_ARGS="-v --tb=short"
if $COVERAGE; then
    PYTEST_ARGS="${PYTEST_ARGS} --cov=aitestframework --cov-report=term --cov-report=html:coverage/html --cov-report=xml:coverage/coverage.xml"
fi

# 运行测试
case "${TEST_TYPE}" in
    unit)
        echo ""
        echo -e "${YELLOW}运行单元测试${NC}"
        echo "----------------------------------------"
        pytest aitestframework/ ${PYTEST_ARGS} \
            --junitxml=test-results/unit-tests.xml || EXIT_CODE=$?
        ;;

    integration)
        echo ""
        echo -e "${YELLOW}运行集成测试${NC}"
        echo "----------------------------------------"
        pytest tests/it tests/st ${PYTEST_ARGS} \
            --junitxml=test-results/integration-tests.xml || EXIT_CODE=$?
        ;;

    all)
        echo ""
        echo -e "${YELLOW}运行所有测试${NC}"
        echo "----------------------------------------"

        # 单元测试
        echo ""
        echo "[1/2] 单元测试"
        pytest aitestframework/ ${PYTEST_ARGS} \
            --junitxml=test-results/unit-tests.xml || EXIT_CODE=$?

        # 集成测试
        echo ""
        echo "[2/2] 集成测试"
        pytest tests/it tests/st -v --tb=short \
            --junitxml=test-results/integration-tests.xml || EXIT_CODE=$?
        ;;

    *)
        echo -e "${RED}未知测试类型: ${TEST_TYPE}${NC}"
        echo "用法: $0 [unit|integration|all] [--coverage]"
        exit 1
        ;;
esac

# 覆盖率检查
if $COVERAGE && [[ -f "coverage/coverage.xml" ]]; then
    echo ""
    echo "=========================================="
    echo "           覆盖率报告"
    echo "=========================================="

    COVERAGE_THRESHOLD=80
    COVERAGE_PERCENT=$(python3 -c "
import xml.etree.ElementTree as ET
tree = ET.parse('coverage/coverage.xml')
root = tree.getroot()
line_rate = float(root.get('line-rate', 0))
print(f'{line_rate * 100:.1f}')
")

    echo "行覆盖率: ${COVERAGE_PERCENT}% (阈值: ${COVERAGE_THRESHOLD}%)"

    if (( $(echo "${COVERAGE_PERCENT} >= ${COVERAGE_THRESHOLD}" | bc -l) )); then
        echo -e "${GREEN}✓ 覆盖率达标${NC}"
    else
        echo -e "${RED}✗ 覆盖率不达标${NC}"
        EXIT_CODE=1
    fi

    echo ""
    echo "HTML报告: coverage/html/index.html"
fi

# 汇总
echo ""
echo "=========================================="
echo "           测试结果汇总"
echo "=========================================="

if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo -e "${GREEN}✓ 所有测试通过${NC}"
else
    echo -e "${RED}✗ 测试失败，请检查上述错误${NC}"
fi

exit ${EXIT_CODE}
