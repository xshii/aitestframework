#!/usr/bin/env bash
# sync_import.sh — 在内网机器运行，导入更新
# 用法: ./sync_import.sh <bundle_or_tar_file> [target_dir]

set -euo pipefail

INPUT_FILE="${1:?用法: $0 <bundle_or_tar_file> [target_dir]}"
TARGET_DIR="${2:-.}"

if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 文件不存在: $INPUT_FILE"
    exit 1
fi

INPUT_FILE="$(cd "$(dirname "$INPUT_FILE")" && pwd)/$(basename "$INPUT_FILE")"

echo "=== aitf 同步导入 ==="
echo "输入: $INPUT_FILE"
echo "目标: $TARGET_DIR"

# ---------- 判断文件类型 ----------
case "$INPUT_FILE" in
    *.bundle)
        echo ""
        echo "--- git bundle 模式 ---"

        # 验证 bundle
        if ! git bundle verify "$INPUT_FILE" 2>/dev/null; then
            echo "错误: bundle 文件无效或不完整"
            exit 1
        fi
        echo "Bundle 验证通过"

        if [ ! -d "$TARGET_DIR/.git" ]; then
            # 首次：从 bundle 克隆
            echo "首次导入，执行 git clone..."
            git clone "$INPUT_FILE" "$TARGET_DIR"
            echo "克隆完成"
        else
            # 增量更新
            cd "$TARGET_DIR"
            BRANCH=$(git rev-parse --abbrev-ref HEAD)
            BEFORE=$(git rev-parse HEAD)

            echo "当前: ${BEFORE:0:12} ($BRANCH)"

            # 检查是否有未提交的修改
            if ! git diff --quiet HEAD 2>/dev/null; then
                echo "检测到本地修改，先暂存..."
                git stash push -m "sync-import-$(date +%Y%m%d%H%M%S)"
                STASHED=1
            else
                STASHED=0
            fi

            # 从 bundle fetch 并合并
            git fetch "$INPUT_FILE" "$BRANCH:refs/remotes/bundle/$BRANCH" 2>/dev/null \
                || git fetch "$INPUT_FILE" "main:refs/remotes/bundle/main"

            REMOTE_REF=$(git rev-parse refs/remotes/bundle/main 2>/dev/null \
                || git rev-parse "refs/remotes/bundle/$BRANCH")

            git merge --ff-only "$REMOTE_REF" 2>/dev/null \
                || git merge "$REMOTE_REF" -m "Merge from bundle import"

            AFTER=$(git rev-parse HEAD)
            DIFF_COUNT=$(git rev-list "$BEFORE..$AFTER" --count 2>/dev/null || echo "?")
            echo "更新完成: $DIFF_COUNT 个新提交"

            # 恢复暂存
            if [ "$STASHED" = "1" ]; then
                echo "恢复本地修改..."
                git stash pop || echo "警告: stash pop 冲突，请手动解决 (git stash show / git stash pop)"
            fi

            # 显示变更摘要
            echo ""
            echo "--- 变更文件 ---"
            git diff --stat "$BEFORE..$AFTER" 2>/dev/null || true
        fi
        ;;

    *.tar.gz|*.tgz)
        echo ""
        echo "--- tar 源码包模式 ---"

        TMPDIR=$(mktemp -d)
        trap "rm -rf $TMPDIR" EXIT

        tar xzf "$INPUT_FILE" -C "$TMPDIR"

        # 找到解压后的目录
        EXTRACTED=$(find "$TMPDIR" -mindepth 1 -maxdepth 1 -type d | head -1)
        if [ -z "$EXTRACTED" ]; then
            echo "错误: 解压后未找到目录"
            exit 1
        fi

        if [ ! -d "$TARGET_DIR" ] || [ ! "$(ls -A "$TARGET_DIR" 2>/dev/null)" ]; then
            # 首次：直接复制
            mkdir -p "$TARGET_DIR"
            cp -r "$EXTRACTED"/* "$TARGET_DIR/"
            echo "首次导入完成"
        else
            # 对比差异
            echo ""
            echo "--- 文件差异 ---"
            diff -rq "$TARGET_DIR" "$EXTRACTED" \
                --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
                --exclude='build' --exclude='*.egg-info' --exclude='.last_sync_commit' \
                | head -50 || true

            echo ""
            read -p "是否覆盖更新? [y/N] " CONFIRM
            if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
                rsync -av --exclude='.git' --exclude='build' --exclude='__pycache__' \
                    --exclude='*.pyc' --exclude='*.egg-info' \
                    "$EXTRACTED/" "$TARGET_DIR/"
                echo "更新完成"
            else
                echo "已取消"
            fi
        fi
        ;;

    *)
        echo "错误: 不支持的文件格式 (需要 .bundle 或 .tar.gz)"
        exit 1
        ;;
esac

echo ""
echo "=== 导入完成 ==="
