"""Single-page index combining Deps, Bundles and Logs tabs."""

from __future__ import annotations

from dataclasses import asdict

from flask import Blueprint, render_template

from aitf.deps.bundle import BundleManager
from aitf.deps.types import DepsConfigError
from aitf.web.views import build_log_listing, get_deps_manager, log_root, size_display

index_bp = Blueprint("index", __name__)


@index_bp.route("/")
def index():
    mgr = get_deps_manager()

    # -- deps --
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

    upload_dir = mgr.project_root / "deps" / "uploads"
    uploads = []
    if upload_dir.is_dir():
        for p in sorted(upload_dir.glob("*.tar.gz")):
            uploads.append({"name": p.name, "size": size_display(p.stat().st_size)})

    # -- bundles --
    try:
        bm = BundleManager(mgr)
        bundles = [asdict(b) for b in bm.list_bundles()]
        active = bm.active()
        active_name = active.name if active else None
        available = {
            "toolchains": {n: tc.version for n, tc in cfg.toolchains.items()},
            "libraries": {n: lib.version for n, lib in cfg.libraries.items()},
            "repos": {n: rc.ref for n, rc in cfg.repos.items()},
        } if cfg else {"toolchains": {}, "libraries": {}, "repos": {}}
    except DepsConfigError:
        bundles = []
        active_name = None
        available = {"toolchains": {}, "libraries": {}, "repos": {}}

    # -- logs (root listing) --
    log_entries = build_log_listing(log_root())

    return render_template(
        "index.html",
        # deps
        deps=deps_list, uploads=uploads, has_config=cfg is not None,
        # bundles
        bundles=bundles, active_name=active_name, available=available,
        # logs
        log_entries=log_entries,
    )
