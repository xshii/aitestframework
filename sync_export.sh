#!/usr/bin/env bash
# sync_export.sh — 在外网机器运行，导出更新包
# 用法: ./sync_export.sh [output_dir]

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${1:-$REPO_DIR}"
DATE=$(date +%Y%m%d)
MARKER_FILE="$REPO_DIR/.last_sync_commit"

cd "$REPO_DIR"

# 确保在 git 仓库中
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "错误: 当前目录不是 git 仓库"
    exit 1
fi

CURRENT_COMMIT=$(git rev-parse HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo "=== aitf 同步导出 ==="
echo "分支: $BRANCH"
echo "当前提交: ${CURRENT_COMMIT:0:12}"

# ---------- git bundle ----------
if [ -f "$MARKER_FILE" ]; then
    LAST_COMMIT=$(cat "$MARKER_FILE")
    echo "上次同步: ${LAST_COMMIT:0:12}"

    # 检查是否有新提交
    NEW_COUNT=$(git rev-list "$LAST_COMMIT..$BRANCH" --count 2>/dev/null || echo "0")
    if [ "$NEW_COUNT" = "0" ]; then
        echo "无新提交，跳过"
        exit 0
    fi

    echo "新增 $NEW_COUNT 个提交，生成增量 bundle..."
    BUNDLE="$OUTPUT_DIR/aitf-${DATE}-incr.bundle"
    git bundle create "$BUNDLE" "$LAST_COMMIT..$BRANCH"
else
    echo "首次导出，生成完整 bundle..."
    BUNDLE="$OUTPUT_DIR/aitf-${DATE}-full.bundle"
    git bundle create "$BUNDLE" --all
fi

echo "Bundle: $BUNDLE ($(du -h "$BUNDLE" | cut -f1))"

# ---------- tar 备份 ----------
TAR="$OUTPUT_DIR/aitf-${DATE}-src.tar.gz"
git archive --format=tar.gz --prefix=aitf-new/ -o "$TAR" HEAD
echo "源码包: $TAR ($(du -h "$TAR" | cut -f1))"

# 记录本次同步点
echo "$CURRENT_COMMIT" > "$MARKER_FILE"

echo ""
echo "=== 导出完成 ==="
echo "请将以下文件传输到内网:"
echo "  $BUNDLE"
echo "  $TAR"
