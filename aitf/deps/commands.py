"""CLI sub-commands for dependency and bundle management.

Convention: every command module exposes ``register(subparsers)`` and
``dispatch(args) -> bool``.  This module registers two top-level
commands (``deps`` and ``bundle``), so it exposes ``COMMANDS`` instead
of a single ``COMMAND``.
"""

from __future__ import annotations

import argparse
import sys

# Two top-level sub-commands managed by this module.
COMMANDS = ["deps", "bundle"]


# -- deps handlers -----------------------------------------------------------

def _cmd_deps_install(args: argparse.Namespace) -> None:
    from pathlib import Path

    from aitf.deps.manager import DepsManager

    name = args.name
    # If a .yaml file is given, sync from server then install.
    if name and name.endswith(".yaml"):
        yaml_path = Path(name)
        if not yaml_path.is_file():
            print(f"Error: file not found: {name}", file=sys.stderr)
            sys.exit(1)
        _install_from_yaml(yaml_path)
        return

    mgr = DepsManager()
    mgr.install(name=name)
    print("Dependencies installed.")


def _download_archives(server: str, cfg: "DepsConfig") -> None:
    """Download needed archives from *server* to ``deps/uploads/``."""
    import json
    from pathlib import Path
    from urllib.request import Request, urlopen

    from aitf.deps.acquire import archive_candidates
    from aitf.deps.config import detect_platform

    plat = detect_platform()
    needed: set[str] = set()
    for name, tc in cfg.toolchains.items():
        for c in archive_candidates(name, tc.version, plat):
            needed.add(c)
    for name, lib in cfg.libraries.items():
        for c in archive_candidates(name, lib.version, plat):
            needed.add(c)
    if not needed:
        return

    try:
        resp = urlopen(Request(f"{server}/api/deps/uploads"), timeout=30)
        all_archives = json.loads(resp.read())
    except Exception:
        all_archives = []

    to_download = [ar for ar in all_archives if ar["name"] in needed]
    if not to_download:
        print("  No matching archives on server.")
        return

    upload_dir = Path("deps") / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    for i, ar in enumerate(to_download, 1):
        fname = ar["name"]
        dest = upload_dir / fname
        if dest.exists():
            print(f"  [{i}/{len(to_download)}] {fname} (exists, skipped)")
            continue
        print(f"  [{i}/{len(to_download)}] Downloading {fname} ...")
        try:
            dl_url = f"{server}/api/deps/uploads/{fname}/download"
            dl_resp = urlopen(Request(dl_url), timeout=300)
            with open(dest, "wb") as fh:
                while chunk := dl_resp.read(1 << 20):
                    fh.write(chunk)
        except Exception as exc:
            print(f"    Warning: {fname}: {exc}", file=sys.stderr)


def _install_from_yaml(yaml_path: "Path") -> None:
    """Install from a downloaded YAML (single dep or bundle)."""
    from aitf.deps.config import load_deps_config
    from aitf.deps.manager import DepsManager

    cfg = load_deps_config(yaml_path)
    if cfg.server:
        _download_archives(cfg.server.rstrip("/"), cfg)

    mgr = DepsManager(deps_file=str(yaml_path))
    mgr.install(on_progress=lambda cur, total, label: print(f"  [{cur}/{total}] {label}"))
    print("Dependencies installed.")


def _cmd_deps_list(args: argparse.Namespace) -> None:
    from aitf.deps.acquire import is_installed
    from aitf.deps.manager import DepsManager
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


def _cmd_deps_sync(args: argparse.Namespace) -> None:
    """Sync deps config + archives from a remote server, then install."""
    from pathlib import Path
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    from aitf.deps.config import load_deps_config
    from aitf.deps.manager import DepsManager

    bundle_name = args.bundle

    # Resolve server URL: --server flag > config.yaml > deps.yaml (legacy)
    server = args.server
    if not server:
        from aitf.config import load_config
        global_cfg = load_config()
        if global_cfg.server:
            server = f"http://{global_cfg.server}:{global_cfg.port}"
    if not server:
        deps_path = Path("deps.yaml")
        if deps_path.exists():
            cfg = load_deps_config(deps_path)
            server = cfg.server
    if not server:
        print("Error: no server specified. Use --server, set 'server' in config.yaml, "
              "or set 'server' in deps.yaml.", file=sys.stderr)
        sys.exit(1)
    server = server.rstrip("/")

    # 1. Fetch deps.yaml (full config or bundle export)
    if bundle_name:
        url = f"{server}/api/bundles/{bundle_name}/export"
        print(f"Fetching bundle '{bundle_name}' config from {server} ...")
    else:
        url = f"{server}/api/deps/export"
        print(f"Fetching deps.yaml from {server} ...")

    try:
        resp = urlopen(Request(url), timeout=30)
        deps_yaml = resp.read()
    except HTTPError as exc:
        print(f"Error: server returned {exc.code} for {url}", file=sys.stderr)
        sys.exit(1)
    except URLError as exc:
        print(f"Error: cannot reach {server}: {exc.reason}", file=sys.stderr)
        sys.exit(1)

    # 2. Save deps.yaml locally
    deps_path = Path("deps.yaml")
    deps_path.write_bytes(deps_yaml)
    print(f"  Saved {deps_path} ({len(deps_yaml)} bytes)")

    # 3. Download matching archives
    cfg = load_deps_config(deps_path)
    _download_archives(server, cfg)

    # 4. Install from local files
    print("Installing dependencies ...")
    mgr = DepsManager()
    mgr.install(on_progress=lambda cur, total, label: print(f"  [{cur}/{total}] {label}"))
    print("Sync complete.")


# -- bundle handlers ---------------------------------------------------------

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


# -- dispatch tables ---------------------------------------------------------

_DEPS_DISPATCH = {
    "install": _cmd_deps_install,
    "list": _cmd_deps_list,
    "lock": _cmd_deps_lock,
    "clean": _cmd_deps_clean,
    "doctor": _cmd_deps_doctor,
    "sync": _cmd_deps_sync,
}

_BUNDLE_DISPATCH = {
    "list": _cmd_bundle_list,
    "show": _cmd_bundle_show,
    "use": _cmd_bundle_use,
    "install": _cmd_bundle_install,
    "export": _cmd_bundle_export,
    "import": _cmd_bundle_import,
}


# -- public interface --------------------------------------------------------

def register(subparsers: argparse._SubParsersAction) -> None:
    """Register ``deps`` and ``bundle`` sub-commands."""
    # -- deps ----------------------------------------------------------------
    deps = subparsers.add_parser("deps", help="Dependency management")
    deps_sub = deps.add_subparsers(dest="deps_cmd")

    p = deps_sub.add_parser("install", help="Install dependencies")
    p.add_argument("name", nargs="?", default=None, help="Install only this dependency")

    deps_sub.add_parser("list", help="List all dependencies and status")
    deps_sub.add_parser("lock", help="Generate / update deps.lock.yaml")
    deps_sub.add_parser("clean", help="Remove all cached dependencies")
    deps_sub.add_parser("doctor", help="Run dependency diagnostics")

    p = deps_sub.add_parser("sync", help="Sync config + archives from a remote server")
    p.add_argument("--server", default=None, help="Server URL (default: 'server' field in deps.yaml)")
    p.add_argument("--bundle", default=None, help="Only sync a specific bundle")

    # -- bundle --------------------------------------------------------------
    bundle = subparsers.add_parser("bundle", help="Configuration bundle management")
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


def dispatch(args: argparse.Namespace) -> bool:
    """Dispatch to the matching handler. Returns True if handled."""
    if args.command == "deps":
        handler = _DEPS_DISPATCH.get(getattr(args, "deps_cmd", None) or "")
    elif args.command == "bundle":
        handler = _BUNDLE_DISPATCH.get(getattr(args, "bundle_cmd", None) or "")
    else:
        return False

    if handler:
        handler(args)
        return True
    return False
