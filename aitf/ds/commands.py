"""CLI sub-commands for golden data management.

Model → Version → Operator → Data Items (category/shape/layout/precision).
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path

from aitf.ds.store import ALLOWED_EXT, DataItem, GoldenStore, OperatorMeta

COMMAND = "data"
HELP = "Golden 数据管理 (模型/版本/算子/数据条目)"


def _store() -> GoldenStore:
    return GoldenStore()


def _fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1048576:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1048576:.1f} MB"


def _cmd_list(args: argparse.Namespace) -> None:
    entries = _store().list(model=args.model)
    if not entries:
        print("暂无数据")
        return
    for e in entries:
        items_desc = ""
        if e.meta.data:
            cats = [d.category for d in e.meta.data]
            items_desc = f"  [{', '.join(cats)}]"
        files = ", ".join(e.files) if e.files else "(无文件)"
        print(f"  {e.model}/{e.version}/{e.operator}{items_desc}  [{files}]  ({_fmt_size(e.total_size)})")
        for d in e.meta.data:
            print(f"    - {d.category}: shape({d.loop},{d.m},{d.n},{d.k}) {d.layout} {d.precision}")
    print(f"\n{len(entries)} operator(s)")


def _cmd_create(args: argparse.Namespace) -> None:
    store = _store()
    meta = OperatorMeta(name=args.operator, data=[])
    store.save_meta(args.model, args.version, args.operator, meta)
    print(f"已创建: {args.model}/{args.version}/{args.operator}")


def _cmd_add_data(args: argparse.Namespace) -> None:
    store = _store()
    meta = store.load_meta(args.model, args.version, args.operator)
    next_seq = max((d.seq for d in meta.data), default=-1) + 1
    item = DataItem(
        seq=next_seq, category=args.category,
        loop=args.loop, m=args.m, n=args.n, k=args.k,
        layout=args.layout, precision=args.precision,
    )
    meta.data.append(item)
    store.save_meta(args.model, args.version, args.operator, meta)
    print(f"已添加: {args.category} shape({args.loop},{args.m},{args.n},{args.k}) {args.layout} {args.precision}")


def _cmd_upload(args: argparse.Namespace) -> None:
    src = Path(args.file)
    if not src.is_file():
        print(f"文件不存在: {src}", file=sys.stderr)
        sys.exit(1)
    if not GoldenStore.allowed(src.name):
        print(f"不支持的格式，仅支持: {', '.join(ALLOWED_EXT)}", file=sys.stderr)
        sys.exit(1)
    dest = _store().save_file(args.model, args.version, args.operator, src.name)
    shutil.copy2(src, dest)
    print(f"已上传: {args.model}/{args.version}/{args.operator}/{src.name}")


def _cmd_download(args: argparse.Namespace) -> None:
    files = _store().get_operator_files(args.model, args.version, args.operator)
    if not files:
        print(f"不存在: {args.model}/{args.version}/{args.operator}", file=sys.stderr)
        sys.exit(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for fp in files:
        dest = out / fp.name
        shutil.copy2(fp, dest)
        print(f"已下载: {dest}")


def _cmd_delete(args: argparse.Namespace) -> None:
    store = _store()
    if args.operator:
        if not store.delete_operator(args.model, args.version, args.operator):
            print(f"不存在: {args.model}/{args.version}/{args.operator}", file=sys.stderr)
            sys.exit(1)
        print(f"已删除: {args.model}/{args.version}/{args.operator}")
    elif args.version:
        if not store.delete_version(args.model, args.version):
            print(f"不存在: {args.model}/{args.version}", file=sys.stderr)
            sys.exit(1)
        print(f"已删除: {args.model}/{args.version}")
    else:
        if not store.delete_model(args.model):
            print(f"不存在: {args.model}", file=sys.stderr)
            sys.exit(1)
        print(f"已删除: {args.model}")


def _cmd_export(args: argparse.Namespace) -> None:
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
    print(f"已导出: {dest} ({len(entries)} operator(s))")


def _cmd_import(args: argparse.Namespace) -> None:
    src = Path(args.file)
    if not src.is_file() or not src.name.endswith(".zip"):
        print(f"需要 .zip 文件: {src}", file=sys.stderr)
        sys.exit(1)
    buf = io.BytesIO(src.read_bytes())
    imported = _store().import_archive(buf)
    print(f"已导入: {len(imported)} 个文件")
    for name in imported:
        print(f"  {name}")


_DISPATCH = {
    "list": _cmd_list,
    "create": _cmd_create,
    "add-data": _cmd_add_data,
    "upload": _cmd_upload,
    "download": _cmd_download,
    "delete": _cmd_delete,
    "export": _cmd_export,
    "import": _cmd_import,
}


def register(subparsers: argparse._SubParsersAction) -> None:
    data = subparsers.add_parser(COMMAND, help=HELP)
    data_sub = data.add_subparsers(dest="sub_cmd")

    p = data_sub.add_parser("list", help="列出所有 golden 数据")
    p.add_argument("model", nargs="?", default=None, help="按模型过滤")

    p = data_sub.add_parser("create", help="创建算子")
    p.add_argument("model", help="模型名称")
    p.add_argument("version", help="版本号")
    p.add_argument("operator", help="算子名称")

    p = data_sub.add_parser("add-data", help="给算子添加数据条目")
    p.add_argument("model", help="模型名称")
    p.add_argument("version", help="版本号")
    p.add_argument("operator", help="算子名称")
    p.add_argument("--category", default="input", help="数据种类 (默认 input)")
    p.add_argument("--loop", type=int, default=1, help="loop (默认 1)")
    p.add_argument("--m", type=int, default=64, help="m (默认 64)")
    p.add_argument("--n", type=int, default=64, help="n (默认 64)")
    p.add_argument("--k", type=int, default=64, help="k (默认 64)")
    p.add_argument("--layout", default="NCHW", help="分型格式 (默认 NCHW)")
    p.add_argument("--precision", default="fp16", help="数据格式 (默认 fp16)")

    p = data_sub.add_parser("upload", help="上传文件到算子")
    p.add_argument("model"); p.add_argument("version"); p.add_argument("operator")
    p.add_argument("file", help="文件路径")

    p = data_sub.add_parser("download", help="下载算子的所有文件")
    p.add_argument("model"); p.add_argument("version"); p.add_argument("operator")
    p.add_argument("-o", "--output", default=".", help="输出目录")

    p = data_sub.add_parser("delete", help="删除模型/版本/算子")
    p.add_argument("model")
    p.add_argument("version", nargs="?", default=None)
    p.add_argument("operator", nargs="?", default=None)

    p = data_sub.add_parser("export", help="导出所有 golden 数据为 zip")
    p.add_argument("-o", "--output", default=".")

    p = data_sub.add_parser("import", help="从 zip 包导入")
    p.add_argument("file", help="zip 文件路径")


def dispatch(args: argparse.Namespace) -> bool:
    handler = _DISPATCH.get(getattr(args, "sub_cmd", None) or "")
    if handler:
        handler(args)
        return True
    return False
