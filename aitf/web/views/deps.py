"""Deps & Bundle web pages."""

from __future__ import annotations

from dataclasses import asdict

from flask import Blueprint, render_template

from aitf.deps.bundle import BundleManager
from aitf.deps.types import DepsConfigError
from aitf.web.views import get_deps_manager, size_display

deps_pages_bp = Blueprint("deps_pages", __name__)


@deps_pages_bp.route("/deps")
def deps_index():
    mgr = get_deps_manager()
    try:
        cfg = mgr.config
    except DepsConfigError:
        cfg = None

    deps_list = []
    if cfg:
        for name, tc in cfg.toolchains.items():
            deps_list.append({"name": name, "type": "toolchain", "version": tc.version})
        for name, lib in cfg.libraries.items():
            deps_list.append({"name": name, "type": "library", "version": lib.version})
        for name, repo in cfg.repos.items():
            deps_list.append({"name": name, "type": "repo", "version": repo.ref})

    # Uploaded archives
    upload_dir = mgr._root / "deps" / "uploads"
    uploads = []
    if upload_dir.is_dir():
        for p in sorted(upload_dir.glob("*.tar.gz")):
            uploads.append({"name": p.name, "size": size_display(p.stat().st_size)})

    return render_template("deps.html", deps=deps_list, uploads=uploads, has_config=cfg is not None)


@deps_pages_bp.route("/bundles")
def bundles_index():
    mgr = get_deps_manager()
    try:
        cfg = mgr.config
        bm = BundleManager(mgr)
        bundles = [asdict(b) for b in bm.list_bundles()]
        active = bm.active()
        active_name = active.name if active else None
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
