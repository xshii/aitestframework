#!/bin/bash
# ci/deploy/jenkins_deploy_linux.sh — One-click Jenkins deployment on Linux (REQ-8.11)
#
# Usage: sudo ./jenkins_deploy_linux.sh [--repo-url <git-url>]
#
# Installs Java 17, Jenkins LTS, build dependencies, and creates pipeline jobs.
# Idempotent: safe to run multiple times.
set -euo pipefail

REPO_URL=""
JENKINS_URL="http://localhost:8080"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------- Parse arguments ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-url) REPO_URL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "${REPO_URL}" ]; then
    echo "Usage: $0 --repo-url <git-url>"
    exit 1
fi

# ---------- Detect package manager ----------
if command -v apt-get &>/dev/null; then
    PKG_MGR="apt"
elif command -v yum &>/dev/null; then
    PKG_MGR="yum"
else
    echo "ERROR: Unsupported package manager. Requires apt or yum."
    exit 1
fi

echo "=== Jenkins Deploy: pkg_mgr=${PKG_MGR} repo=${REPO_URL} ==="

# ---------- Install Java 17 ----------
install_java() {
    if java -version 2>&1 | grep -q '17\.'; then
        echo "Java 17 already installed."
        return
    fi
    echo "Installing Java 17..."
    if [ "${PKG_MGR}" = "apt" ]; then
        apt-get update -qq
        apt-get install -y -qq openjdk-17-jdk
    else
        yum install -y -q java-17-openjdk-devel
    fi
}

# ---------- Install Jenkins LTS ----------
install_jenkins() {
    if systemctl is-active --quiet jenkins 2>/dev/null; then
        echo "Jenkins already running."
        return
    fi
    echo "Installing Jenkins LTS..."
    if [ "${PKG_MGR}" = "apt" ]; then
        curl -fsSL https://pkg.jenkins.io/debian-stable/jenkins.io-2023.key \
            | tee /usr/share/keyrings/jenkins-keyring.asc > /dev/null
        echo "deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] https://pkg.jenkins.io/debian-stable binary/" \
            | tee /etc/apt/sources.list.d/jenkins.list > /dev/null
        apt-get update -qq
        apt-get install -y -qq jenkins
    else
        curl -fsSL https://pkg.jenkins.io/redhat-stable/jenkins.repo \
            -o /etc/yum.repos.d/jenkins.repo
        rpm --import https://pkg.jenkins.io/redhat-stable/jenkins.io-2023.key || true
        yum install -y -q jenkins
    fi
    systemctl enable jenkins
    systemctl start jenkins
}

# ---------- Install build dependencies ----------
install_build_deps() {
    echo "Installing build dependencies..."
    if [ "${PKG_MGR}" = "apt" ]; then
        apt-get install -y -qq cmake gcc python3 python3-pip git
    else
        yum install -y -q cmake gcc python3 python3-pip git
    fi
}

# ---------- Wait for Jenkins to be ready ----------
wait_for_jenkins() {
    echo "Waiting for Jenkins to be ready..."
    local retries=60
    while [ ${retries} -gt 0 ]; do
        if curl -sfo /dev/null "${JENKINS_URL}/login"; then
            echo "Jenkins is ready."
            return
        fi
        retries=$((retries - 1))
        sleep 5
    done
    echo "WARNING: Jenkins did not become ready. Job creation may fail."
}

# ---------- Create Jenkins jobs via CLI ----------
create_jobs() {
    local cli_jar="/tmp/jenkins-cli.jar"

    if [ ! -f "${cli_jar}" ]; then
        curl -sfo "${cli_jar}" "${JENKINS_URL}/jnlpJars/jenkins-cli.jar" || {
            echo "WARNING: Could not download Jenkins CLI. Create jobs manually."
            return
        }
    fi

    echo "Creating/updating stubs-build job..."
    java -jar "${cli_jar}" -s "${JENKINS_URL}" create-job stubs-build \
        < "${SCRIPT_DIR}/job_config_stubs.xml" 2>/dev/null \
        || java -jar "${cli_jar}" -s "${JENKINS_URL}" update-job stubs-build \
            < "${SCRIPT_DIR}/job_config_stubs.xml"

    echo "Creating/updating smoke-tests job..."
    java -jar "${cli_jar}" -s "${JENKINS_URL}" create-job smoke-tests \
        < "${SCRIPT_DIR}/job_config_smoke.xml" 2>/dev/null \
        || java -jar "${cli_jar}" -s "${JENKINS_URL}" update-job smoke-tests \
            < "${SCRIPT_DIR}/job_config_smoke.xml"

    echo "Jobs created/updated successfully."
}

# ---------- Main ----------
install_java
install_jenkins
install_build_deps
wait_for_jenkins
create_jobs

echo "=== Jenkins deployment complete ==="
echo "Access Jenkins at: ${JENKINS_URL}"
echo "Initial admin password: $(cat /var/lib/jenkins/secrets/initialAdminPassword 2>/dev/null || echo 'N/A')"
