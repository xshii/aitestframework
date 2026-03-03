"""CLI sub-commands for golden data management.

Convention: every command module exposes ``register(subparsers)`` and
``dispatch(args) -> bool``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from aitf.ds.store import ALLOWED_EXT, GoldenStore

# CLI module identifier — used by the auto-discovery in cli.py
COMMAND = "data"
HELP = "Golden 数据管理"


def _store() -> GoldenStore:
    return GoldenStore()


def _fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1048576:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1048576:.1f} MB"


# -- handlers ----------------------------------------------------------------

def _cmd_list(args: argparse.Namespace) -> None:
    entries = _store().list(model=args.model)
    if not entries:
        print("暂无数据")
        return
    for e in entries:
        print(f"  {e.model}/{e.version}/{e.file}  ({_fmt_size(e.size)})")
    print(f"\n{len(entries)} file(s)")


def _cmd_upload(args: argparse.Namespace) -> None:
    src = Path(args.file)
    if not src.is_file():
        print(f"文件不存在: {src}", file=sys.stderr)
        sys.exit(1)
    if not GoldenStore.allowed(src.name):
        print(f"不支持的格式，仅支持: {', '.join(ALLOWED_EXT)}", file=sys.stderr)
        sys.exit(1)
    dest = _store().save(args.model, args.version, src.name)
    shutil.copy2(src, dest)
    print(f"已上传: {args.model}/{args.version}/{src.name}")


def _cmd_download(args: argparse.Namespace) -> None:
    fp = _store().get_file(args.model, args.version)
    if not fp:
        print(f"不存在: {args.model}/{args.version}", file=sys.stderr)
        sys.exit(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / fp.name
    shutil.copy2(fp, dest)
    print(f"已下载: {dest}")


def _cmd_delete(args: argparse.Namespace) -> None:
    if not _store().delete(args.model, args.version):
        print(f"不存在: {args.model}/{args.version}", file=sys.stderr)
        sys.exit(1)
    print(f"已删除: {args.model}/{args.version}")


def _cmd_download_all(args: argparse.Namespace) -> None:
    store = _store()
    entries = store.list()
    if not entries:
        print("暂无数据", file=sys.stderr)
        sys.exit(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "golden-all.zip"
    buf = store.export_all()
    dest.write_bytes(buf.read())
    print(f"已下载: {dest} ({len(entries)} file(s))")


# -- public interface --------------------------------------------------------

_DISPATCH = {
    "list": _cmd_list,
    "upload": _cmd_upload,
    "download": _cmd_download,
    "download-all": _cmd_download_all,
    "delete": _cmd_delete,
}


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``data`` sub-command and its children."""
    data = subparsers.add_parser(COMMAND, help=HELP)
    data_sub = data.add_subparsers(dest="sub_cmd")

    p = data_sub.add_parser("list", help="列出所有 golden 数据")
    p.add_argument("model", nargs="?", default=None, help="按模型过滤")

    p = data_sub.add_parser("upload", help="上传文件到 model/version")
    p.add_argument("model", help="模型名称")
    p.add_argument("version", help="版本号")
    p.add_argument("file", help="文件路径")

    p = data_sub.add_parser("download", help="下载 model/version 的文件")
    p.add_argument("model", help="模型名称")
    p.add_argument("version", help="版本号")
    p.add_argument("-o", "--output", default=".", help="输出目录（默认当前目录）")

    p = data_sub.add_parser("download-all", help="下载所有 golden 数据为 zip")
    p.add_argument("-o", "--output", default=".", help="输出目录（默认当前目录）")

    p = data_sub.add_parser("delete", help="删除模型的某个版本")
    p.add_argument("model", help="模型名称")
    p.add_argument("version", help="版本号")


def dispatch(args: argparse.Namespace) -> bool:
    """Dispatch to the matching handler. Returns True if handled."""
    handler = _DISPATCH.get(getattr(args, "sub_cmd", None) or "")
    if handler:
        handler(args)
        return True
    return False
