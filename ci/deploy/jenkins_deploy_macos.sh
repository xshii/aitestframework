#!/bin/bash
# ci/deploy/jenkins_deploy_macos.sh — macOS 一键部署 + 构建 + 冒烟测试
#
# Usage: ./jenkins_deploy_macos.sh [--repo-url <git-url>]
#
# 无 --repo-url 时跳过 Jenkins 安装，只做构建和冒烟测试。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."
REPO_URL=""
SKIP_JENKINS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-url) REPO_URL="$2"; shift 2 ;;
        --skip-jenkins) SKIP_JENKINS=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================
# 1. Homebrew
# ============================================================
if ! command -v brew &>/dev/null; then
    echo "=== Installing Homebrew ==="
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# ============================================================
# 2. Build dependencies
# ============================================================
echo "=== Installing build dependencies ==="
for pkg in cmake python3 git; do
    if ! command -v "${pkg}" &>/dev/null; then
        brew install "${pkg}"
    else
        echo "${pkg} already installed."
    fi
done

# ============================================================
# 3. Jenkins (optional)
# ============================================================
if [ "${SKIP_JENKINS}" = false ] && [ -n "${REPO_URL}" ]; then
    # Java 17
    if ! /usr/libexec/java_home -v 17 &>/dev/null; then
        echo "=== Installing Java 17 ==="
        brew install openjdk@17
        sudo ln -sfn "$(brew --prefix openjdk@17)/libexec/openjdk.jdk" \
            /Library/Java/JavaVirtualMachines/openjdk-17.jdk
    else
        echo "Java 17 already installed."
    fi
    export JAVA_HOME=$(/usr/libexec/java_home -v 17)

    # Jenkins LTS
    if ! brew list jenkins-lts &>/dev/null; then
        echo "=== Installing Jenkins LTS ==="
        brew install jenkins-lts
    else
        echo "Jenkins LTS already installed."
    fi

    # Start Jenkins
    brew services start jenkins-lts 2>/dev/null || true
    echo "Jenkins starting at http://localhost:8080"

    INIT_PW="${HOME}/.jenkins/secrets/initialAdminPassword"
    if [ -f "${INIT_PW}" ]; then
        echo "Initial admin password: $(cat "${INIT_PW}")"
    fi

    echo ""
    echo "--- Jenkins Job 创建步骤 ---"
    echo "1. 浏览器打开 http://localhost:8080"
    echo "2. New Item → Pipeline → 名称: stubs-build"
    echo "   Pipeline script from SCM → Git → URL: ${REPO_URL}"
    echo "   Script Path: ci/Jenkinsfile → 保存"
    echo "3. 再建一个 smoke-tests Job，Script Path: ci/Jenkinsfile.smoke"
    echo "----------------------------"
fi

# ============================================================
# 4. Build
# ============================================================
echo ""
echo "=== Building project ==="
chmod +x "${PROJECT_ROOT}/stubs/build.sh"
"${PROJECT_ROOT}/stubs/build.sh" -m all

# ============================================================
# 5. Smoke tests (use venv to avoid PEP 668 restrictions)
# ============================================================
echo ""
echo "=== Running smoke tests ==="
VENV_DIR="${PROJECT_ROOT}/build/.venv"
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
pip install --quiet -r "${PROJECT_ROOT}/tests/requirements.txt"
python3 -m pytest "${PROJECT_ROOT}/tests/" -v --platform=func_sim
deactivate

echo ""
echo "=== Done ==="
