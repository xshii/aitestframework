"""CLI sub-commands for stub code management.

Convention: exposes ``COMMAND``, ``register(subparsers)``, and
``dispatch(args) -> bool``.
"""

from __future__ import annotations

import argparse
import sys

COMMAND = "stub"


# -- handlers ----------------------------------------------------------------

def _cmd_list(args: argparse.Namespace) -> None:
    from aitf.stubs.manager import StubManager

    mgr = StubManager()
    stubs = mgr.list_stubs()
    if not stubs:
        print("No stubs found.")
        return
    for s in stubs:
        srcs = ", ".join(s.sources) if s.sources else "(no sources)"
        print(f"  {s.name:20s}  {srcs}")
    print(f"\n{len(stubs)} stub(s)")


def _cmd_build(args: argparse.Namespace) -> None:
    from aitf.stubs.manager import StubManager

    mgr = StubManager()
    print(f"Building stubs (platform={args.platform}, target={args.target}) ...")
    result = mgr.build(platform=args.platform, target=args.target)
    if result.success:
        print(f"Build succeeded in {result.duration_s:.1f}s")
        print(f"  output: {result.output_dir}")
    else:
        print(f"Build FAILED after {result.duration_s:.1f}s", file=sys.stderr)
        print(result.error_msg, file=sys.stderr)
        sys.exit(1)


def _cmd_new(args: argparse.Namespace) -> None:
    from aitf.stubs.manager import StubManager

    mgr = StubManager()
    try:
        path = mgr.create_from_template(args.name)
    except FileExistsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Created stub: {path}")


def _cmd_changes(args: argparse.Namespace) -> None:
    from aitf.stubs.manager import StubManager

    mgr = StubManager()
    changed = mgr.detect_changes(since=args.since)
    if not changed:
        print("No stub changes detected.")
        return
    for name in changed:
        print(f"  {name}")
    print(f"\n{len(changed)} model(s) changed")


# -- dispatch table ----------------------------------------------------------

_DISPATCH = {
    "list": _cmd_list,
    "build": _cmd_build,
    "new": _cmd_new,
    "changes": _cmd_changes,
}


# -- public interface --------------------------------------------------------

def register(subparsers: argparse._SubParsersAction) -> None:
    """Register ``stub`` sub-commands."""
    stub = subparsers.add_parser(COMMAND, help="Stub code management")
    stub_sub = stub.add_subparsers(dest="sub_cmd")

    stub_sub.add_parser("list", help="List all model stubs")

    p = stub_sub.add_parser("build", help="Build stubs")
    p.add_argument("--platform", default="func_sim", help="Target platform (default: func_sim)")
    p.add_argument("--target", default="all", choices=["app", "ut", "all"],
                    help="Build target (default: all)")

    p = stub_sub.add_parser("new", help="Create a new model stub from template")
    p.add_argument("--name", required=True, help="Model stub name")

    p = stub_sub.add_parser("changes", help="Detect changed stubs")
    p.add_argument("--since", default="HEAD~1", help="Git ref to compare against (default: HEAD~1)")


def dispatch(args: argparse.Namespace) -> bool:
    """Dispatch to the matching handler. Returns True if handled."""
    handler = _DISPATCH.get(getattr(args, "sub_cmd", None) or "")
    if handler:
        handler(args)
        return True
    return False
