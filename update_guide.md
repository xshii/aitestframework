# 内网环境框架更新指南

本文档介绍在无法直接访问 GitHub 的内网环境中，如何对比差异并更新 aitf 框架。

---

## 方式一：git bundle（推荐）

`git bundle` 将 Git 提交历史打包成单个文件，可以通过 U 盘、共享目录等方式传输。
内网机器收到后可以像操作远程仓库一样 fetch/pull。

### 工作原理

```
外网机器                           内网机器
git bundle create ──> bundle文件 ──> git bundle unbundle / git pull
```

- bundle 文件包含完整的 Git 对象（commits、trees、blobs）
- 支持增量打包：只打包某个基准之后的新提交
- 内网机器可以用 `git fetch` 从 bundle 文件拉取，保留完整提交历史

### 外网机器：导出

```bash
# 首次 —— 打包全部历史
git bundle create aitf-full.bundle --all

# 增量 —— 只打包上次同步之后的新提交
# 假设上次同步到 v1.0 标签或某个 commit hash
git bundle create aitf-incremental.bundle v1.0..main

# 增量 —— 打包最近 N 天的提交
git bundle create aitf-recent.bundle --since=7.days main
```

### 内网机器：导入

```bash
# 验证 bundle 文件完整性
git bundle verify aitf-full.bundle

# --- 首次导入 ---
# 方法 A：直接从 bundle 克隆
git clone aitf-full.bundle aitestframework
cd aitestframework

# 方法 B：已有仓库，添加为远程源
cd aitestframework
git remote add bundle /path/to/aitf-full.bundle
git fetch bundle
git merge bundle/main

# --- 增量更新 ---
# 将 bundle 文件当作远程源 fetch
git fetch /path/to/aitf-incremental.bundle main:refs/remotes/bundle/main
git merge bundle/main
# 或直接 pull
git pull /path/to/aitf-incremental.bundle main
```

---

## 方式二：tar/zip 源码包 + diff 对比

直接下载源码压缩包，解压后与本地目录做对比。

### 外网机器：打包

```bash
cd aitestframework
# 打包干净的源码（排除 git 历史和构建产物）
git archive --format=tar.gz --prefix=aitf-new/ -o aitf-new.tar.gz HEAD
```

### 内网机器：对比并更新

```bash
# 解压新版本
tar xzf aitf-new.tar.gz

# 对比差异
diff -rq aitestframework/ aitf-new/ --exclude='__pycache__' --exclude='.git' --exclude='build' --exclude='*.pyc'

# 查看具体文件差异
diff -u aitestframework/aitf/tc/runner.py aitf-new/aitf/tc/runner.py

# 确认无误后覆盖
rsync -av --exclude='.git' --exclude='build' --exclude='__pycache__' aitf-new/ aitestframework/
```

---

## 自动化脚本

### 脚本 A：外网导出脚本 `sync_export.sh`

在外网机器上运行，自动生成 bundle 或 tar 包。

```bash
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
echo "当前提交: $CURRENT_COMMIT"

# ---------- git bundle ----------
if [ -f "$MARKER_FILE" ]; then
    LAST_COMMIT=$(cat "$MARKER_FILE")
    echo "上次同步: $LAST_COMMIT"

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
```

### 脚本 B：内网导入脚本 `sync_import.sh`

在内网机器上运行，自动从 bundle 或 tar 包更新。

```bash
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

            echo "当前: $BEFORE ($BRANCH)"

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
                echo "已取消。解压目录保留在: $EXTRACTED"
                trap - EXIT  # 不清理临时目录
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
```

---

## 推荐工作流

```
1. 外网开发机修改代码，正常 git commit

2. 准备同步时，在外网运行:
   ./sync_export.sh /tmp/sync/

3. 将 /tmp/sync/ 下的文件拷贝到 U 盘或共享目录

4. 在内网机器运行:
   ./sync_import.sh /mnt/usb/aitf-20260312-incr.bundle ./aitestframework

5. 内网机器拿到完整 git 历史，可正常使用 git log/diff/blame
```

### 注意事项

- 首次同步用 `*-full.bundle`，后续用 `*-incr.bundle` 即可（体积更小）
- bundle 方式保留完整 git 历史，推荐优先使用
- tar 方式适合不需要 git 历史的场景，或作为 bundle 的备份方案
- 导出脚本会自动记录同步点（`.last_sync_commit` 文件），下次运行自动生成增量包
- 内网如有本地修改，导入脚本会自动 `git stash` 保护
