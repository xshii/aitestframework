#!/usr/bin/env bash
# 一键部署：创建 venv → 安装依赖 → 启动 Web 服务
# 用法:
#   ./scripts/start.sh              # 正常启动
#   ./scripts/start.sh --debug      # 调试模式（代码修改自动重载）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
DEBUG=false

if [ "${1:-}" = "--debug" ]; then
    DEBUG=true
fi

echo "==> 项目目录: $ROOT"

# --- venv ---
if [ ! -d "$VENV" ]; then
    echo "==> 创建 virtualenv ..."
    python3 -m venv "$VENV"
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"

# --- pip 更新 ---
echo "==> 更新 pip ..."
pip install --upgrade pip -q

# --- 依赖 ---
echo "==> 安装依赖 ..."
pip install -q -e "$ROOT[dev]"

# --- 数据目录 ---
mkdir -p "$ROOT/datastore/registry" "$ROOT/datastore/store" "$ROOT/data"

# --- 端口检测 ---
# Host/port: env vars AITF_HOST / AITF_PORT override config.yaml
PORT="${AITF_PORT:-5000}"

check_port() {
    local port="$1"
    if command -v ss &>/dev/null; then
        ss -tlnp 2>/dev/null | grep -q ":${port} " && return 1
    elif command -v lsof &>/dev/null; then
        lsof -iTCP:"$port" -sTCP:LISTEN &>/dev/null && return 1
    elif command -v netstat &>/dev/null; then
        netstat -tlnp 2>/dev/null | grep -q ":${port} " && return 1
    fi
    return 0
}

if ! check_port "$PORT"; then
    echo "错误: 端口 $PORT 已被占用。" >&2
    echo "请通过环境变量 AITF_PORT 指定其他端口，例如:" >&2
    echo "  AITF_PORT=8080 $0" >&2
    exit 1
fi

# --- 启动 ---
EXTRA_ARGS=""
if [ -n "${AITF_HOST:-}" ]; then EXTRA_ARGS="$EXTRA_ARGS --host $AITF_HOST"; fi
EXTRA_ARGS="$EXTRA_ARGS --port $PORT"
if [ "$DEBUG" = true ]; then EXTRA_ARGS="$EXTRA_ARGS --debug"; fi

echo "==> 启动 Web 服务 (端口: $PORT) ..."
exec aitf web $EXTRA_ARGS
