"""Web routes for test-case management (REQ-4 / REQ-7)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

import yaml
from flask import Blueprint, current_app, jsonify, request, send_file

from aitf.tc import store
from aitf.tc.models import CaseStatus

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
        params=body.get("params"),
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
        logger.debug("Failed to load bundles for tc/options", exc_info=True)

    # Golden models/versions from datastore
    try:
        from aitf.web.extensions import get_golden_store
        gs = get_golden_store()
        for model in gs.list_models():
            versions = gs.list_versions(model)
            golden_models.append({"model": model, "versions": versions})
    except Exception:
        logger.debug("Failed to load golden models for tc/options", exc_info=True)

    # Testplan files — return {filename, name} pairs
    try:
        root = cfg.project_root if cfg else "."
        for pattern in ("testplan*.yaml", "testplan*.yml"):
            for f in sorted(Path(root).glob(pattern)):
                try:
                    with open(f, encoding="utf-8") as fh:
                        plan_data = yaml.safe_load(fh) or {}
                    plan_name = plan_data.get("name", f.stem)
                except (yaml.YAMLError, OSError):
                    plan_name = f.stem
                testplans.append({"filename": f.name, "name": plan_name})
    except Exception:
        logger.debug("Failed to scan testplans for tc/options", exc_info=True)

    return jsonify({
        "bundles": bundles,
        "golden_models": golden_models,
        "testplans": testplans,
    })


# ---------------------------------------------------------------------------
# Testplan read
# ---------------------------------------------------------------------------

def _safe_plan_path(filename: str) -> Path | None:
    """Resolve plan path safely, preventing path traversal."""
    cfg = current_app.config.get("AITF_CONFIG")
    root = (cfg.project_root if cfg else Path(".")).resolve()
    plan_path = (root / filename).resolve()
    if not plan_path.is_relative_to(root):
        return None
    if not filename.endswith((".yaml", ".yml")):
        return None
    return plan_path


@bp.route("/api/tc/plan/<filename>", methods=["GET"])
def api_get_plan(filename):
    """Return parsed testplan YAML content."""
    plan_path = _safe_plan_path(filename)
    if not plan_path or not plan_path.is_file():
        return jsonify({"error": "not found"}), 404
    with open(plan_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return jsonify(data or {})


@bp.route("/api/tc/plan/<filename>", methods=["DELETE"])
def api_delete_plan(filename):
    """Delete a testplan YAML file."""
    plan_path = _safe_plan_path(filename)
    if not plan_path or not plan_path.is_file():
        return jsonify({"error": "not found"}), 404
    plan_path.unlink()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Testplan generation
# ---------------------------------------------------------------------------

from aitf.tc.models import SAFE_FILENAME_RE as _SAFE_FILENAME_RE


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
        if item.get("target"):
            entry["target"] = item["target"]
        golden = item.get("golden")
        if golden and golden.get("model"):
            entry["golden"] = golden
        params = item.get("params")
        if params and isinstance(params, dict):
            entry["params"] = params
        plan_data["plans"].append(entry)

    cfg = current_app.config.get("AITF_CONFIG")
    root = cfg.project_root if cfg else Path(".")
    out_path = root / filename

    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.dump(plan_data, fh, allow_unicode=True, default_flow_style=False,
                  sort_keys=False)

    return jsonify({"ok": True, "filename": filename})


# ---------------------------------------------------------------------------
# Golden sync to remote server (proxy)
# ---------------------------------------------------------------------------

@bp.route("/api/tc/sync_golden", methods=["POST"])
def api_sync_golden_to_remote():
    """Proxy: export local golden and upload to remote server via SyncClient.

    Body: {"items": [{"model": "m", "version": "v", "operators": ["op1"]}]}
    If operators is empty/missing, uploads the entire version.
    """
    sc = current_app.config.get("SYNC_CLIENT")
    if not sc:
        return jsonify({"error": "未配置远端服务器（仅 client/debug 模式可用）"}), 400

    from aitf.web.extensions import get_golden_store
    gs = get_golden_store()

    body = request.get_json(silent=True) or {}
    items = body.get("items", [])
    if not items:
        return jsonify({"error": "items 不能为空"}), 400

    results = []
    for item in items:
        model = item.get("model", "")
        version = item.get("version", "")
        operators = item.get("operators", [])
        if not model or not version:
            continue
        try:
            if not operators:
                # Upload entire version
                buf = gs.export_version(model, version)
                sc.golden_upload_zip(model, version, None,
                                     buf.read(), f"{model}_{version}.zip")
                results.append({"model": model, "version": version, "ok": True})
            else:
                # Upload specific operators
                for op in operators:
                    buf = gs.export_operator(model, version, op)
                    sc.golden_upload_zip(model, version, op,
                                         buf.read(), f"{model}_{version}_{op}.zip")
                results.append({"model": model, "version": version,
                                "operators": operators, "ok": True})
        except Exception as exc:
            results.append({"model": model, "version": version,
                            "ok": False, "error": str(exc)})

    success = sum(1 for r in results if r.get("ok"))
    return jsonify({"ok": True, "results": results,
                    "success": success, "total": len(results)})


# ---------------------------------------------------------------------------
# Overview dashboard stats (REQ-7.1)
# ---------------------------------------------------------------------------

@bp.route("/api/tc/overview", methods=["GET"])
def api_tc_overview():
    """Return overview dashboard data: latest execution, suite stats, fail top 10."""
    from sqlalchemy import func, select

    from aitf.tc.db import get_session
    from aitf.tc.models import CaseResult, Execution, SuiteInfo

    latest_execution = None
    suite_stats = {"total_suites": 0, "total_cases": 0}
    fail_top10 = []

    try:
        with get_session() as session:
            # Latest execution
            exe = session.execute(
                select(Execution).order_by(Execution.started_at.desc()).limit(1)
            ).scalar_one_or_none()
            if exe:
                latest_execution = exe.to_dict()

            # Suite stats — aggregate in SQL instead of loading all rows
            stats = session.execute(
                select(
                    func.count(SuiteInfo.id),
                    func.coalesce(func.sum(SuiteInfo.case_count), 0),
                )
            ).one()
            suite_stats["total_suites"] = stats[0]
            suite_stats["total_cases"] = stats[1]

            # Fail top 10 — aggregate in SQL with GROUP BY
            fail_rows = session.execute(
                select(
                    CaseResult.suite_class,
                    CaseResult.case_method,
                    func.count().label("cnt"),
                )
                .where(CaseResult.status.in_(CaseStatus.FAILURE))
                .group_by(CaseResult.suite_class, CaseResult.case_method)
                .order_by(func.count().desc())
                .limit(10)
            ).all()
            fail_top10 = [
                {"case_name": f"{row.suite_class}::{row.case_method}",
                 "fail_count": row.cnt}
                for row in fail_rows
            ]
    except Exception:
        logger.warning("Failed to load overview data", exc_info=True)

    return jsonify({
        "latest_execution": latest_execution,
        "suite_stats": suite_stats,
        "fail_top10": fail_top10,
    })


# ---------------------------------------------------------------------------
# Webhook (REQ-7.8)
# ---------------------------------------------------------------------------

@bp.route("/api/webhook", methods=["POST"])
def api_webhook():
    """Receive webhook notifications (e.g. from Jenkins)."""
    # Token validation
    configured_token = current_app.config.get("WEBHOOK_TOKEN")
    if configured_token:
        provided_token = request.headers.get("X-Webhook-Token", "")
        if provided_token != configured_token:
            return jsonify({"error": "invalid token"}), 403

    body = request.get_json(silent=True) or {}
    event = body.get("event")
    if event != "execution_complete":
        return jsonify({"error": f"unsupported event: {event}"}), 400

    execution_id = body.get("execution_id")
    if not execution_id:
        return jsonify({"error": "missing execution_id"}), 400

    summary = body.get("summary", {})
    jenkins_build = body.get("jenkins_build", {})

    from aitf.tc.db import get_session
    from aitf.tc.models import Execution

    try:
        with get_session() as session:
            exe = session.get(Execution, execution_id)
            if exe is None:
                # Create new execution record
                from datetime import UTC, datetime

                now = datetime.now(UTC)
                exe = Execution(
                    id=execution_id,
                    started_at=now,
                    finished_at=now,
                    platform=body.get("platform"),
                    git_commit=body.get("git_commit"),
                    trigger="webhook",
                    total=summary.get("total", 0),
                    passed=summary.get("pass", 0),
                    failed=summary.get("fail", 0),
                    jenkins_job=jenkins_build.get("job_name"),
                    jenkins_build=jenkins_build.get("build_number"),
                    jenkins_url=jenkins_build.get("build_url"),
                    report_json_path=body.get("report_path"),
                )
                total = exe.total or 0
                exe.pass_rate = exe.passed / total if total else 0.0
                session.add(exe)
            else:
                # Update existing execution
                from datetime import UTC, datetime

                exe.finished_at = datetime.now(UTC)
                exe.platform = body.get("platform") or exe.platform
                exe.git_commit = body.get("git_commit") or exe.git_commit
                exe.total = summary.get("total", exe.total)
                exe.passed = summary.get("pass", exe.passed)
                exe.failed = summary.get("fail", exe.failed)
                exe.jenkins_job = jenkins_build.get("job_name") or exe.jenkins_job
                exe.jenkins_build = jenkins_build.get("build_number") or exe.jenkins_build
                exe.jenkins_url = jenkins_build.get("build_url") or exe.jenkins_url
                exe.report_json_path = body.get("report_path") or exe.report_json_path
                total = exe.total or 0
                exe.pass_rate = exe.passed / total if total else 0.0

            session.commit()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "ok", "execution_id": execution_id})


# ---------------------------------------------------------------------------
# Report generation (REQ-4.4)
# ---------------------------------------------------------------------------

@bp.route("/api/executions/<execution_id>/report", methods=["GET"])
def api_execution_report(execution_id: str):
    """Generate and return a report for the given execution."""
    from aitf.tc.report import generate_html_report, generate_json_report

    fmt = request.args.get("format", "json").lower()

    detail = store.get_execution_detail(execution_id)
    if detail is None:
        return jsonify({"error": "not found"}), 404

    cfg = current_app.config.get("AITF_CONFIG")
    output_dir = cfg.build_root / "reports" if cfg else Path("build/reports")

    if fmt == "html":
        path = generate_html_report(execution_id, output_dir)
        return send_file(str(path), mimetype="text/html")
    else:
        path = generate_json_report(execution_id, output_dir)
        return send_file(str(path), mimetype="application/json")
