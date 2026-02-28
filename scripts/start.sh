#!/usr/bin/env bash
# 一键部署：创建 venv → 安装依赖 → 启动 Web 服务
# 用法:
#   ./scripts/start.sh              # 正常启动
#   ./scripts/start.sh --debug      # 调试模式（代码修改自动重载）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
HOST="${AITF_HOST:-0.0.0.0}"
PORT="${AITF_PORT:-5000}"
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

# --- 依赖 ---
echo "==> 安装依赖 ..."
pip install -q -e "$ROOT[dev]"

# --- 数据目录 ---
mkdir -p "$ROOT/datastore/registry" "$ROOT/datastore/store" "$ROOT/data"

# --- 启动 ---
if [ "$DEBUG" = true ]; then
    echo "==> 启动 (debug): http://${HOST}:${PORT}"
    exec python -c "
from aitf.web.app import create_app
app = create_app()
app.run(host='${HOST}', port=${PORT}, debug=True)
"
else
    echo "==> 启动: http://${HOST}:${PORT}"
    exec python -c "
from aitf.web.app import create_app
app = create_app()
app.run(host='${HOST}', port=${PORT}, debug=False)
"
fi
