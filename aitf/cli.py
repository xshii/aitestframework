"""Unified CLI entry-point for the AI Test Framework.

Usage::

    aitf data register <case_id> <local_path>
    aitf data list [--platform X] [--model Y]
    aitf data get <case_id>
    aitf data delete <case_id>
    aitf data verify [--case <id>] [--platform X] [--model Y]
    aitf data pull --remote <name> [--case <id>] [--platform X] [--model Y]
    aitf data push --remote <name> --case <id>
    aitf data push-artifacts --remote <name> --case <id> --dir <path>
    aitf data rebuild-cache
    aitf web [--host 0.0.0.0] [--port 5000] [--debug]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from aitf.ds.manager import DataStoreManager


def _json_dumps(obj: object) -> str:
    return json.dumps(obj, indent=2, default=str)


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def _cmd_register(mgr: DataStoreManager, args: argparse.Namespace) -> None:
    case = mgr.register(args.case_id, args.local_path)
    print(_json_dumps(asdict(case)))


def _cmd_list(mgr: DataStoreManager, args: argparse.Namespace) -> None:
    cases = mgr.list(platform=args.platform, model=args.model)
    for c in cases:
        print(f"  {c.case_id:40s}  {c.platform:10s}  {c.model:10s}  {c.variant}")
    print(f"\n{len(cases)} case(s)")


def _cmd_get(mgr: DataStoreManager, args: argparse.Namespace) -> None:
    case = mgr.get(args.case_id)
    print(_json_dumps(asdict(case)))


def _cmd_delete(mgr: DataStoreManager, args: argparse.Namespace) -> None:
    mgr.delete(args.case_id)
    print(f"Deleted {args.case_id}")


def _cmd_verify(mgr: DataStoreManager, args: argparse.Namespace) -> None:
    results = mgr.verify(case_id=args.case)
    ok_count = sum(1 for r in results if r.ok)
    fail_count = len(results) - ok_count
    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(f"  [{status}] {r.case_id} / {r.file_path}")
        if r.error:
            print(f"         error: {r.error}")
    print(f"\n{ok_count} passed, {fail_count} failed")
    if fail_count:
        sys.exit(1)


def _cmd_pull(mgr: DataStoreManager, args: argparse.Namespace) -> None:
    results = mgr.pull(
        args.remote,
        case_id=args.case,
        platform=args.platform,
        model=args.model,
    )
    for r in results:
        print(
            f"  {r.case_id}: {r.files_transferred} transferred, "
            f"{r.files_skipped} skipped, {r.files_failed} failed"
        )


def _cmd_push(mgr: DataStoreManager, args: argparse.Namespace) -> None:
    result = mgr.push(args.remote, args.case)
    print(
        f"  {result.case_id}: {result.files_transferred} transferred, "
        f"{result.files_failed} failed"
    )


def _cmd_push_artifacts(mgr: DataStoreManager, args: argparse.Namespace) -> None:
    result = mgr.push_artifacts(args.remote, args.case, args.dir)
    print(
        f"  {result.case_id}: {result.files_transferred} transferred, "
        f"{result.files_failed} failed"
    )


def _cmd_rebuild_cache(mgr: DataStoreManager, _args: argparse.Namespace) -> None:
    count = mgr.rebuild_cache()
    print(f"Rebuilt cache: {count} case(s)")


def _cmd_web(args: argparse.Namespace) -> None:
    from aitf.web.app import create_app

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aitf", description="AI Test Framework CLI")
    sub = parser.add_subparsers(dest="command")

    # -- data ----------------------------------------------------------------
    data = sub.add_parser("data", help="Test data management")
    data_sub = data.add_subparsers(dest="data_cmd")

    p = data_sub.add_parser("register", help="Register a local case")
    p.add_argument("case_id")
    p.add_argument("local_path")

    p = data_sub.add_parser("list", help="List registered cases")
    p.add_argument("--platform", default=None)
    p.add_argument("--model", default=None)

    p = data_sub.add_parser("get", help="Show case details")
    p.add_argument("case_id")

    p = data_sub.add_parser("delete", help="Delete a case")
    p.add_argument("case_id")

    p = data_sub.add_parser("verify", help="Verify file checksums")
    p.add_argument("--case", default=None)
    p.add_argument("--platform", default=None)
    p.add_argument("--model", default=None)

    p = data_sub.add_parser("pull", help="Pull case data from remote")
    p.add_argument("--remote", required=True)
    p.add_argument("--case", default=None)
    p.add_argument("--platform", default=None)
    p.add_argument("--model", default=None)

    p = data_sub.add_parser("push", help="Push case data to remote")
    p.add_argument("--remote", required=True)
    p.add_argument("--case", required=True)

    p = data_sub.add_parser("push-artifacts", help="Push artifacts to remote")
    p.add_argument("--remote", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--dir", default=None)

    data_sub.add_parser("rebuild-cache", help="Rebuild SQLite cache from YAML")

    # -- web -----------------------------------------------------------------
    p = sub.add_parser("web", help="Start the web server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")

    return parser


def main() -> None:
    """CLI entry-point."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "web":
        _cmd_web(args)
        return

    if args.command != "data" or not args.data_cmd:
        parser.print_help()
        sys.exit(1)

    mgr = DataStoreManager()

    dispatch = {
        "register": _cmd_register,
        "list": _cmd_list,
        "get": _cmd_get,
        "delete": _cmd_delete,
        "verify": _cmd_verify,
        "pull": _cmd_pull,
        "push": _cmd_push,
        "push-artifacts": _cmd_push_artifacts,
        "rebuild-cache": _cmd_rebuild_cache,
    }

    handler = dispatch.get(args.data_cmd)
    if handler:
        handler(mgr, args)
    else:
        parser.print_help()
        sys.exit(1)
