"""Single-page index combining Deps, Bundles and Logs tabs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from flask import Blueprint, current_app, render_template

from aitf.deps.bundle import BundleManager
from aitf.deps.types import DepsConfigError
from aitf.web.views import get_deps_manager, size_display

index_bp = Blueprint("index", __name__)


def _log_root() -> Path:
    return Path(current_app.config.get("LOG_ROOT", "build/reports")).resolve()


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

    upload_dir = mgr._root / "deps" / "uploads"
    uploads = []
    if upload_dir.is_dir():
        for p in sorted(upload_dir.glob("*.tar.gz")):
            uploads.append({"name": p.name, "size": size_display(p.stat().st_size)})

    # -- bundles --
    try:
        cfg2 = mgr.config
        bm = BundleManager(mgr)
        bundles = [asdict(b) for b in bm.list_bundles()]
        active = bm.active()
        active_name = active.name if active else None
        available = {
            "toolchains": {n: tc.version for n, tc in cfg2.toolchains.items()},
            "libraries": {n: lib.version for n, lib in cfg2.libraries.items()},
            "repos": {n: rc.ref for n, rc in cfg2.repos.items()},
        }
    except DepsConfigError:
        bundles = []
        active_name = None
        available = {"toolchains": {}, "libraries": {}, "repos": {}}

    # -- logs (root listing) --
    log_root = _log_root()
    log_entries = []
    if log_root.is_dir():
        for child in sorted(log_root.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
            rel = child.relative_to(log_root)
            log_entries.append({
                "name": child.name,
                "path": str(rel),
                "is_dir": child.is_dir(),
                "size_display": size_display(child.stat().st_size) if child.is_file() else "",
            })

    return render_template(
        "index.html",
        # deps
        deps=deps_list, uploads=uploads, has_config=cfg is not None,
        # bundles
        bundles=bundles, active_name=active_name, available=available,
        # logs
        log_entries=log_entries,
    )
