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
    aitf deps install [name]
    aitf deps list
    aitf deps lock
    aitf deps clean
    aitf deps doctor
    aitf bundle list
    aitf bundle show <name>
    aitf bundle use <name> [--force]
    aitf bundle install <name>
    aitf bundle export <name> -o <file>
    aitf bundle import <file>
    aitf web [--host 127.0.0.1] [--port 5000] [--debug]
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
# Data sub-command handlers
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
# Deps sub-command handlers
# ---------------------------------------------------------------------------

def _cmd_deps_install(args: argparse.Namespace) -> None:
    from aitf.deps.manager import DepsManager

    mgr = DepsManager()
    mgr.install(name=args.name)
    print("Dependencies installed.")


def _cmd_deps_list(args: argparse.Namespace) -> None:
    from aitf.deps.manager import DepsManager
    from aitf.deps.acquire import is_installed
    from aitf.deps.repo import is_cloned

    mgr = DepsManager()
    cfg = mgr.config

    print("Toolchains:")
    for name, tc in cfg.toolchains.items():
        status = "installed" if is_installed(name, tc.version, mgr.cache_dir) else "not installed"
        print(f"  {name:25s}  {tc.version:12s}  [{status}]")

    print("\nLibraries:")
    for name, lib in cfg.libraries.items():
        status = "installed" if is_installed(name, lib.version, mgr.cache_dir) else "not installed"
        print(f"  {name:25s}  {lib.version:12s}  [{status}]")

    print("\nRepositories:")
    for name, repo in cfg.repos.items():
        status = "cloned" if is_cloned(name, mgr.repos_dir) else "not cloned"
        print(f"  {name:25s}  {repo.ref:12s}  [{status}]")


def _cmd_deps_lock(args: argparse.Namespace) -> None:
    from aitf.deps.manager import DepsManager

    mgr = DepsManager()
    mgr.lock()
    print("Lock file updated: deps.lock.yaml")


def _cmd_deps_clean(args: argparse.Namespace) -> None:
    from aitf.deps.manager import DepsManager

    mgr = DepsManager()
    count = mgr.clean()
    print(f"Cleaned {count} cached directories.")


def _cmd_deps_doctor(args: argparse.Namespace) -> None:
    from aitf.deps.manager import DepsManager

    mgr = DepsManager()
    results = mgr.doctor()
    has_failure = False
    for r in results:
        icon = "\u2713" if r.ok else "\u2717"
        print(f"  {icon} {r.message}")
        if not r.ok:
            has_failure = True
    if has_failure:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Bundle sub-command handlers
# ---------------------------------------------------------------------------

def _cmd_bundle_list(args: argparse.Namespace) -> None:
    from aitf.deps.bundle import BundleManager
    from aitf.deps.manager import DepsManager

    mgr = DepsManager()
    bm = BundleManager(mgr)
    bundles = bm.list_bundles()
    active = bm.active()
    for b in bundles:
        marker = " *" if active and b.name == active.name else ""
        print(f"  {b.name:25s}  [{b.status:10s}]  {b.description}{marker}")
    if not bundles:
        print("  (no bundles defined)")


def _cmd_bundle_show(args: argparse.Namespace) -> None:
    from aitf.deps.bundle import BundleManager
    from aitf.deps.manager import DepsManager

    mgr = DepsManager()
    bm = BundleManager(mgr)
    b = bm.show(args.name)
    print(f"Bundle: {b.name}")
    print(f"Description: {b.description}")
    print(f"Status: {b.status}")
    if b.toolchains:
        print("Toolchains:")
        for name, ver in b.toolchains.items():
            print(f"  {name}: {ver}")
    if b.libraries:
        print("Libraries:")
        for name, ver in b.libraries.items():
            print(f"  {name}: {ver}")
    if b.repos:
        print("Repos:")
        for name, ref in b.repos.items():
            print(f"  {name}: {ref}")
    if b.env:
        print("Env:")
        for key, val in b.env.items():
            print(f"  {key}={val}")


def _cmd_bundle_use(args: argparse.Namespace) -> None:
    from aitf.deps.bundle import BundleManager
    from aitf.deps.manager import DepsManager

    mgr = DepsManager()
    bm = BundleManager(mgr)
    bm.use(args.name, force=args.force)
    print(f"Switched to bundle: {args.name}")


def _cmd_bundle_install(args: argparse.Namespace) -> None:
    from aitf.deps.bundle import BundleManager
    from aitf.deps.manager import DepsManager

    mgr = DepsManager()
    bm = BundleManager(mgr)
    bm.install(args.name)
    print(f"Bundle '{args.name}' dependencies installed.")


def _cmd_bundle_export(args: argparse.Namespace) -> None:
    from aitf.deps.bundle import BundleManager
    from aitf.deps.manager import DepsManager

    mgr = DepsManager()
    bm = BundleManager(mgr)
    path = bm.export_bundle(args.name, args.output)
    print(f"Exported bundle '{args.name}' -> {path}")


def _cmd_bundle_import(args: argparse.Namespace) -> None:
    from aitf.deps.bundle import BundleManager
    from aitf.deps.manager import DepsManager

    mgr = DepsManager()
    bm = BundleManager(mgr)
    name = bm.import_bundle(args.file)
    print(f"Imported bundle: {name}")


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

    # -- deps ----------------------------------------------------------------
    deps = sub.add_parser("deps", help="Dependency management")
    deps_sub = deps.add_subparsers(dest="deps_cmd")

    p = deps_sub.add_parser("install", help="Install dependencies")
    p.add_argument("name", nargs="?", default=None, help="Install only this dependency")

    deps_sub.add_parser("list", help="List all dependencies and status")
    deps_sub.add_parser("lock", help="Generate / update deps.lock.yaml")
    deps_sub.add_parser("clean", help="Remove all cached dependencies")
    deps_sub.add_parser("doctor", help="Run dependency diagnostics")

    # -- bundle --------------------------------------------------------------
    bundle = sub.add_parser("bundle", help="Configuration bundle management")
    bundle_sub = bundle.add_subparsers(dest="bundle_cmd")

    bundle_sub.add_parser("list", help="List all bundles")

    p = bundle_sub.add_parser("show", help="Show bundle details")
    p.add_argument("name")

    p = bundle_sub.add_parser("use", help="Switch to a bundle")
    p.add_argument("name")
    p.add_argument("--force", action="store_true", help="Allow deprecated bundles")

    p = bundle_sub.add_parser("install", help="Install bundle dependencies")
    p.add_argument("name")

    p = bundle_sub.add_parser("export", help="Export bundle as offline archive")
    p.add_argument("name")
    p.add_argument("-o", "--output", required=True, help="Output file path")

    p = bundle_sub.add_parser("import", help="Import bundle from archive")
    p.add_argument("file")

    # -- web -----------------------------------------------------------------
    p = sub.add_parser("web", help="Start the web server")
    p.add_argument("--host", default="127.0.0.1")
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

    if args.command == "deps":
        _dispatch_deps(args, parser)
        return

    if args.command == "bundle":
        _dispatch_bundle(args, parser)
        return

    if args.command == "data" and args.data_cmd:
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
            return

    parser.print_help()
    sys.exit(1)


def _dispatch_deps(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    dispatch = {
        "install": _cmd_deps_install,
        "list": _cmd_deps_list,
        "lock": _cmd_deps_lock,
        "clean": _cmd_deps_clean,
        "doctor": _cmd_deps_doctor,
    }
    handler = dispatch.get(args.deps_cmd or "")
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


def _dispatch_bundle(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    dispatch = {
        "list": _cmd_bundle_list,
        "show": _cmd_bundle_show,
        "use": _cmd_bundle_use,
        "install": _cmd_bundle_install,
        "export": _cmd_bundle_export,
        "import": _cmd_bundle_import,
    }
    handler = dispatch.get(args.bundle_cmd or "")
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)
