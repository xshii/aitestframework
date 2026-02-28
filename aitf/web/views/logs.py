"""Log browser — directory listing, file viewer, download."""

from __future__ import annotations

from pathlib import Path

from flask import (
    Blueprint,
    abort,
    current_app,
    render_template,
    request,
    send_file,
)

from aitf.web.views import size_display

logs_bp = Blueprint("logs", __name__)

_DEFAULT_LIMIT = 1000  # lines per page


def _log_root() -> Path:
    return Path(current_app.config.get("LOG_ROOT", "build/reports")).resolve()


def _safe_path(subpath: str) -> Path:
    """Resolve *subpath* under log root; abort 404 if escapes."""
    root = _log_root()
    target = (root / subpath).resolve()
    if not target.is_relative_to(root):
        abort(404)
    return target


# -- routes ------------------------------------------------------------------

@logs_bp.route("/logs")
@logs_bp.route("/logs/<path:subpath>")
def log_index(subpath: str = ""):
    target = _safe_path(subpath)
    if not target.is_dir():
        abort(404)

    entries = []
    for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
        rel = child.relative_to(_log_root())
        entries.append({
            "name": child.name,
            "path": str(rel),
            "is_dir": child.is_dir(),
            "size_display": size_display(child.stat().st_size) if child.is_file() else "",
        })

    parent = str(Path(subpath).parent) if subpath else ""
    if parent == ".":
        parent = ""

    return render_template(
        "logs.html",
        entries=entries,
        base_display=subpath or "/",
        parent=parent,
    )


@logs_bp.route("/logs/<path:subpath>/view")
def log_view(subpath: str):
    target = _safe_path(subpath)
    if not target.is_file():
        abort(404)

    offset = request.args.get("offset", 0, type=int)
    limit = request.args.get("limit", _DEFAULT_LIMIT, type=int)

    # Count total lines without loading entire file into memory
    total = 0
    with open(target, encoding="utf-8", errors="replace") as fh:
        for _ in fh:
            total += 1

    # Read only the requested page
    lines = []
    with open(target, encoding="utf-8", errors="replace") as fh:
        for i, raw in enumerate(fh):
            if i < offset:
                continue
            if i >= offset + limit:
                break
            lines.append(raw.rstrip("\n"))
    pages = (total + limit - 1) // limit

    parent = str(Path(subpath).parent)
    if parent == ".":
        parent = ""

    return render_template(
        "log_view.html",
        subpath=subpath,
        filename=Path(subpath).name,
        lines=lines,
        offset=offset,
        limit=limit,
        total_lines=total,
        pages=pages,
        parent=parent,
    )


@logs_bp.route("/logs/<path:subpath>/download")
def log_download(subpath: str):
    target = _safe_path(subpath)
    if not target.is_file():
        abort(404)
    return send_file(target, as_attachment=True, download_name=target.name)
