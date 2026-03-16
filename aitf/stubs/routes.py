"""REST API routes for stub code management.

Self-contained plugin blueprint -- auto-discovered by ``aitf.web.app``.
"""

from __future__ import annotations

from dataclasses import asdict

from flask import Blueprint, current_app, jsonify, request

bp = Blueprint("stubs", __name__)


def _mgr():
    """Return a StubManager using project config paths."""
    from aitf.stubs.manager import StubManager

    cfg = current_app.config.get("AITF_CONFIG")
    if cfg is not None:
        return StubManager(
            stubs_dir=cfg.project_root / "stubs",
            build_dir=cfg.build_root,
        )
    return StubManager()


@bp.route("/api/stubs", methods=["GET"])
def list_stubs():
    """List all model stubs."""
    mgr = _mgr()
    stubs = mgr.list_stubs()
    items = []
    for s in stubs:
        d = asdict(s)
        d["path"] = str(s.path)
        items.append(d)
    return jsonify(items)


@bp.route("/api/stubs/platforms", methods=["GET"])
def list_platforms():
    """List available platform hooks."""
    mgr = _mgr()
    return jsonify(mgr.list_platforms())


@bp.route("/api/stubs/build", methods=["POST"])
def trigger_build():
    """Trigger a stub build."""
    body = request.get_json(silent=True) or {}
    platform = body.get("platform", "func_sim")
    target = body.get("target", "all")

    mgr = _mgr()
    result = mgr.build(platform=platform, target=target)
    return jsonify({
        "success": result.success,
        "output_dir": str(result.output_dir),
        "duration_s": result.duration_s,
        "error_msg": result.error_msg,
    }), 200 if result.success else 500


@bp.route("/api/stubs/changes", methods=["GET"])
def detect_changes():
    """Detect changed stubs since a git ref."""
    since = request.args.get("since", "HEAD~1")
    mgr = _mgr()
    changed = mgr.detect_changes(since=since)
    return jsonify(changed)
