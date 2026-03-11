"""Web routes for test-case management (REQ-4 / REQ-7)."""

from __future__ import annotations

import re
import threading
from pathlib import Path

import yaml
from flask import Blueprint, current_app, jsonify, request

from aitf.tc import store

bp = Blueprint("tc", __name__, template_folder="templates")

_scan_lock = threading.Lock()


def _cases_dir():
    cfg = current_app.config.get("AITF_CONFIG")
    return cfg.project_root / "cases" if cfg else "cases"


def _db_path():
    cfg = current_app.config.get("AITF_CONFIG")
    return cfg.build_root / "aitf.db" if cfg else "build/aitf.db"


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


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

@bp.route("/api/run", methods=["POST"])
def api_run():
    """Trigger test execution in a background thread."""
    from aitf.tc.runner import run_tests_async

    body = request.get_json(silent=True) or {}

    execution_id = run_tests_async(
        _cases_dir(), _db_path(),
        paths=body.get("paths"),
        filter_k=body.get("filter_k"),
        bundle=body.get("bundle"),
        target=body.get("target"),
        golden_model=body.get("golden_model"),
        golden_version=body.get("golden_version"),
    )
    if not execution_id:
        return jsonify({"error": "no tests found"}), 400
    return jsonify({"ok": True, "execution_id": execution_id})


@bp.route("/api/run/plan", methods=["POST"])
def api_run_plan():
    """Trigger testplan execution. Body: {"plan": "testplan.yaml"}"""
    from aitf.tc.runner import run_testplan_async

    body = request.get_json(silent=True) or {}
    plan_file = body.get("plan")
    if not plan_file:
        return jsonify({"error": "missing 'plan' field"}), 400

    cfg = current_app.config.get("AITF_CONFIG")
    plan_path = cfg.project_root / plan_file if cfg else plan_file

    if not plan_path.is_file():
        return jsonify({"error": f"plan not found: {plan_file}"}), 404

    execution_ids = run_testplan_async(_cases_dir(), _db_path(), plan_path)
    if not execution_ids:
        return jsonify({"error": "no tests found in plan"}), 400
    return jsonify({"ok": True, "execution_ids": execution_ids})


# ---------------------------------------------------------------------------
# Helper: list available bundles and golden models (for UI dropdowns)
# ---------------------------------------------------------------------------

@bp.route("/api/tc/options", methods=["GET"])
def api_tc_options():
    """Return available bundles and golden models for the run form."""
    cfg = current_app.config.get("AITF_CONFIG")
    bundles = []
    golden_models = []
    testplans = []

    # Bundles from deps config
    try:
        from aitf.deps.config import load_deps_config
        deps_file = cfg.project_root / "deps.yaml" if cfg else "deps.yaml"
        if deps_file.is_file():
            dcfg = load_deps_config(deps_file)
            bundles = list(dcfg.bundles.keys())
    except Exception:
        pass

    # Golden models/versions from datastore
    try:
        from aitf.ds.store import GoldenStore
        base = current_app.config.get("DATASTORE_BASE_DIR", "datastore")
        gs = GoldenStore(base)
        for model in gs.list_models():
            versions = gs.list_versions(model)
            golden_models.append({"model": model, "versions": versions})
    except Exception:
        pass

    # Testplan files
    try:
        from pathlib import Path
        root = cfg.project_root if cfg else "."
        for f in sorted(Path(root).glob("testplan*.yaml")):
            testplans.append(f.name)
        for f in sorted(Path(root).glob("testplan*.yml")):
            testplans.append(f.name)
    except Exception:
        pass

    return jsonify({
        "bundles": bundles,
        "golden_models": golden_models,
        "testplans": testplans,
    })


# ---------------------------------------------------------------------------
# Testplan generation
# ---------------------------------------------------------------------------

_SAFE_FILENAME_RE = re.compile(r'^[a-zA-Z0-9_\-\u4e00-\u9fff]+$')


@bp.route("/api/tc/plan", methods=["POST"])
def api_save_plan():
    """Save a generated testplan YAML file.

    Body: {"name": "...", "filename": "testplan_xxx.yaml", "plans": [...]}
    Each plan item: {"name": "...", "tests": [...], "bundle": "...",
                     "golden": {"model": "...", "version": "..."}}
    """
    body = request.get_json(silent=True) or {}
    plans = body.get("plans", [])
    if not plans:
        return jsonify({"error": "plans 不能为空"}), 400

    filename = body.get("filename", "").strip()
    if not filename:
        filename = "testplan_custom.yaml"
    # Sanitise — only allow safe names
    stem = Path(filename).stem
    if not _SAFE_FILENAME_RE.match(stem):
        return jsonify({"error": "文件名只能包含字母、数字、下划线、中文"}), 400
    filename = stem + ".yaml"

    plan_data = {"name": body.get("name", stem), "plans": []}
    for item in plans:
        entry = {"name": item.get("name", ""), "tests": item.get("tests", [])}
        if item.get("bundle"):
            entry["bundle"] = item["bundle"]
        golden = item.get("golden")
        if golden and golden.get("model"):
            entry["golden"] = golden
        plan_data["plans"].append(entry)

    cfg = current_app.config.get("AITF_CONFIG")
    root = cfg.project_root if cfg else Path(".")
    out_path = root / filename

    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.dump(plan_data, fh, allow_unicode=True, default_flow_style=False,
                  sort_keys=False)

    return jsonify({"ok": True, "filename": filename})
