"""Deps & Bundle web pages."""

from __future__ import annotations

from dataclasses import asdict

from flask import Blueprint, current_app, redirect, render_template, request, url_for

from aitf.deps.acquire import is_installed
from aitf.deps.bundle import BundleManager
from aitf.deps.manager import DepsManager
from aitf.deps.repo import is_cloned
from aitf.deps.types import DepsConfigError

deps_pages_bp = Blueprint("deps_pages", __name__)


def _mgr() -> DepsManager:
    if "deps_manager" not in current_app.config:
        current_app.config["deps_manager"] = DepsManager()
    return current_app.config["deps_manager"]


def _bm() -> BundleManager:
    return BundleManager(_mgr())


def _size_display(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


@deps_pages_bp.route("/deps")
def deps_index():
    mgr = _mgr()
    try:
        cfg = mgr.config
    except DepsConfigError:
        cfg = None

    deps_list = []
    if cfg:
        for name, tc in cfg.toolchains.items():
            deps_list.append({
                "name": name, "type": "toolchain", "version": tc.version,
                "installed": is_installed(name, tc.version, mgr.cache_dir),
                "local_dir": tc.acquire.local_dir or "",
            })
        for name, lib in cfg.libraries.items():
            deps_list.append({
                "name": name, "type": "library", "version": lib.version,
                "installed": is_installed(name, lib.version, mgr.cache_dir),
                "local_dir": lib.acquire.local_dir or "",
            })
        for name, repo in cfg.repos.items():
            deps_list.append({
                "name": name, "type": "repo", "ref": repo.ref,
                "installed": is_cloned(name, mgr.repos_dir),
                "local_dir": "",
            })

    # Uploaded archives
    upload_dir = mgr._root / "deps" / "uploads"
    uploads = []
    if upload_dir.is_dir():
        for p in sorted(upload_dir.glob("*.tar.gz")):
            uploads.append({"name": p.name, "size": _size_display(p.stat().st_size)})

    return render_template("deps.html", deps=deps_list, uploads=uploads, has_config=cfg is not None)


@deps_pages_bp.route("/bundles")
def bundles_index():
    mgr = _mgr()
    try:
        cfg = mgr.config
        bm = BundleManager(mgr)
        bundles = [asdict(b) for b in bm.list_bundles()]
        active = bm.active()
        active_name = active.name if active else None
        # Available deps for the create form
        available = {
            "toolchains": {n: tc.version for n, tc in cfg.toolchains.items()},
            "libraries": {n: lib.version for n, lib in cfg.libraries.items()},
            "repos": {n: rc.ref for n, rc in cfg.repos.items()},
        }
    except DepsConfigError:
        bundles = []
        active_name = None
        available = {"toolchains": {}, "libraries": {}, "repos": {}}

    return render_template(
        "bundles.html", bundles=bundles, active_name=active_name, available=available,
    )
