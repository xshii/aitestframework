#!/usr/bin/env python3
"""从 GitHub ZIP 包更新代码，保护内网使用数据。

使用方法:
  1. GitHub 网页 → Code → Download ZIP，下载到内网机器
  2. 执行:
       python scripts/update_from_zip.py ~/Downloads/aitestframework-main.zip

     或指定项目目录:
       python scripts/update_from_zip.py update.zip --project /path/to/aitf

功能:
  - 只覆盖代码文件（.py / .html / .js / .css / .md 等）
  - 跳过所有用户数据（数据库、config.yaml、deps.yaml、datastore 等）
  - 新增的代码文件也会被复制
  - 删除的代码文件会提示但不自动删除
  - 更新前自动备份已修改的文件
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

# ── 保护列表：这些路径永远不会被覆盖 ─────────────────────────────────
PROTECTED_PATHS = {
    "config.yaml",
    "deps.yaml",
    "deps.lock.yaml",
    "testplan.yaml",        # 内网可能有自己的测试计划
}

PROTECTED_DIRS = {
    "build",                # aitf.db, cache, repos
    "datastore",            # golden 数据
    "data",                 # 其他数据目录
    "deps/uploads",         # 上传的依赖包
    ".venv",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".claude",
}

# ── 代码文件扩展名 ──────────────────────────────────────────────────
CODE_EXTENSIONS = {
    ".py", ".html", ".js", ".css", ".json", ".yaml", ".yml",
    ".md", ".txt", ".cfg", ".toml", ".ini", ".sh",
    ".example",             # config.yaml.example 等
}

# 无扩展名但需要更新的文件
CODE_FILES_NO_EXT = {
    "Makefile", "Dockerfile", "Procfile",
    ".gitignore", ".flake8", ".editorconfig",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_protected(rel: str) -> bool:
    """检查相对路径是否属于受保护的数据。"""
    if rel in PROTECTED_PATHS:
        return True
    parts = Path(rel).parts
    for pd in PROTECTED_DIRS:
        pd_parts = Path(pd).parts
        if parts[:len(pd_parts)] == pd_parts:
            return True
    return False


def _is_code_file(rel: str) -> bool:
    """判断文件是否是代码文件。"""
    p = Path(rel)
    if p.suffix in CODE_EXTENSIONS:
        return True
    if p.name in CODE_FILES_NO_EXT:
        return True
    return False


def update(zip_path: Path, project_root: Path, *, dry_run: bool = False):
    """从 ZIP 包提取代码文件，更新到项目目录。"""

    if not zip_path.is_file():
        print(f"错误: ZIP 文件不存在: {zip_path}")
        sys.exit(1)

    if not project_root.is_dir():
        print(f"错误: 项目目录不存在: {project_root}")
        sys.exit(1)

    # 解压到临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        print(f"解压 {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)

        # GitHub ZIP 有一层顶级目录 (e.g. aitestframework-main/)
        top_dirs = [d for d in tmp.iterdir() if d.is_dir()]
        if len(top_dirs) == 1:
            src_root = top_dirs[0]
        else:
            src_root = tmp

        updated = []
        added = []
        skipped_protected = []
        unchanged = []

        # 备份目录
        backup_dir = project_root / "build" / "_update_backup"

        # 遍历 ZIP 中的所有文件
        for src_file in sorted(src_root.rglob("*")):
            if src_file.is_dir():
                continue
            rel = str(src_file.relative_to(src_root))

            # 跳过受保护的数据文件
            if _is_protected(rel):
                skipped_protected.append(rel)
                continue

            # 只处理代码文件
            if not _is_code_file(rel):
                continue

            dst_file = project_root / rel

            if dst_file.exists():
                # 比较是否有变化
                if _sha256(src_file) == _sha256(dst_file):
                    unchanged.append(rel)
                    continue

                if dry_run:
                    updated.append(rel)
                    continue

                # 备份旧文件
                bak = backup_dir / rel
                bak.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dst_file, bak)

                # 覆盖
                shutil.copy2(src_file, dst_file)
                updated.append(rel)
            else:
                if dry_run:
                    added.append(rel)
                    continue

                # 新文件
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                added.append(rel)

        # 检查是否有被删除的代码文件（只在完整 ZIP 时才检查）
        zip_code_count = len(updated) + len(added) + len(unchanged)
        deleted_candidates = []
        if zip_code_count >= 20:  # 完整 ZIP 通常有上百个文件
            for dst_file in sorted(project_root.rglob("*")):
                if dst_file.is_dir():
                    continue
                rel = str(dst_file.relative_to(project_root))
                if _is_protected(rel) or not _is_code_file(rel):
                    continue
                src_file = src_root / rel
                if not src_file.exists():
                    deleted_candidates.append(rel)

        # ── 输出报告 ─────────────────────────────────────────────────
        prefix = "[预览] " if dry_run else ""

        print(f"\n{'═' * 60}")
        print(f"{prefix}更新报告")
        print(f"{'═' * 60}")

        if updated:
            print(f"\n✏️  已更新 ({len(updated)} 个文件):")
            for f in updated:
                print(f"    {f}")

        if added:
            print(f"\n➕ 新增 ({len(added)} 个文件):")
            for f in added:
                print(f"    {f}")

        if deleted_candidates:
            print(f"\n⚠️  以下文件在新版本中已删除，但本地仍保留 ({len(deleted_candidates)}):")
            for f in deleted_candidates:
                print(f"    {f}")
            print("    (如确认不需要，请手动删除)")

        if skipped_protected:
            print(f"\n🛡️  跳过受保护数据 ({len(skipped_protected)} 个):")
            for f in skipped_protected[:5]:
                print(f"    {f}")
            if len(skipped_protected) > 5:
                print(f"    ... 还有 {len(skipped_protected) - 5} 个")

        print(f"\n未变化: {len(unchanged)} 个文件")

        if not dry_run and updated:
            print(f"\n💾 旧版本已备份到: {backup_dir}")

        if not updated and not added:
            print("\n✅ 代码已是最新，无需更新。")
        elif not dry_run:
            print(f"\n✅ 更新完成！共更新 {len(updated)} 个、新增 {len(added)} 个文件。")
            print("   请重启服务使更新生效: python -m aitf.cli web")
        else:
            print(f"\n以上为预览，实际执行请去掉 --dry-run 参数。")


def main():
    parser = argparse.ArgumentParser(
        description="从 GitHub ZIP 包安全更新 AITF 代码",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  # 预览更新内容（不实际修改）
  python scripts/update_from_zip.py update.zip --dry-run

  # 执行更新
  python scripts/update_from_zip.py update.zip

  # 指定项目目录
  python scripts/update_from_zip.py update.zip --project /opt/aitf
""",
    )
    parser.add_argument("zip_file", type=Path, help="GitHub 下载的 ZIP 文件路径")
    parser.add_argument(
        "--project", type=Path, default=None,
        help="项目根目录（默认: 脚本上级目录）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只预览变更，不实际修改文件",
    )
    args = parser.parse_args()

    if args.project:
        project_root = args.project.resolve()
    else:
        project_root = Path(__file__).resolve().parent.parent

    update(args.zip_file.resolve(), project_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
