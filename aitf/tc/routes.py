"""Web routes for test-case management (REQ-4 / REQ-7)."""

from __future__ import annotations

import threading

from flask import Blueprint, current_app, jsonify, request

from aitf.tc import store

bp = Blueprint("tc", __name__, template_folder="templates")

_scan_lock = threading.Lock()


def _cases_dir():
    cfg = current_app.config.get("AITF_CONFIG")
    return cfg.project_root / "cases" if cfg else "cases"


# ---------------------------------------------------------------------------
# Suite discovery
# ---------------------------------------------------------------------------

@bp.route("/api/suites", methods=["GET"])
def api_list_suites():
    return jsonify(store.list_suites())


@bp.route("/api/suites/refresh", methods=["POST"])
def api_refresh_suites():
    if not _scan_lock.acquire(blocking=False):
        return jsonify({"ok": True, "count": 0, "msg": "scan already in progress"})
    try:
        count = store.refresh_suites(_cases_dir())
    finally:
        _scan_lock.release()
    return jsonify({"ok": True, "count": count})


# ---------------------------------------------------------------------------
# Executions
# ---------------------------------------------------------------------------

@bp.route("/api/executions", methods=["GET"])
def api_list_executions():
    limit = request.args.get("limit", 50, type=int)
    return jsonify(store.list_executions(limit=limit))


@bp.route("/api/executions/<execution_id>", methods=["GET"])
def api_get_execution(execution_id: str):
    detail = store.get_execution_detail(execution_id)
    if detail is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(detail)


@bp.route("/api/run", methods=["POST"])
def api_run():
    """Trigger test execution in a background thread."""
    from aitf.tc.runner import run_tests_async

    body = request.get_json(silent=True) or {}
    cfg = current_app.config.get("AITF_CONFIG")
    cases_dir = cfg.project_root / "cases" if cfg else "cases"
    db_path = cfg.build_root / "aitf.db" if cfg else "build/aitf.db"

    execution_id = run_tests_async(
        cases_dir, db_path,
        paths=body.get("paths"),
        filter_k=body.get("filter_k"),
        bundle=body.get("bundle"),
        target=body.get("target"),
    )
    if not execution_id:
        return jsonify({"error": "no tests found"}), 400
    return jsonify({"ok": True, "execution_id": execution_id})
