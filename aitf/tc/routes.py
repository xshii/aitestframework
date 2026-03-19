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
from aitf.web.activity import log_activity

bp = Blueprint("tc", __name__, template_folder="templates")

_scan_lock = threading.Lock()


def _cases_dir():
    cfg = current_app.config.get("AITF_CONFIG")
    return cfg.project_root / "cases" if cfg else "cases"


def _db_path():
    cfg = current_app.config.get("AITF_CONFIG")
    return cfg.build_root / "aitf.db" if cfg else "build/aitf.db"


def _project_root() -> Path:
    cfg = current_app.config.get("AITF_CONFIG")
    return cfg.project_root if cfg else Path(".")


def _require_sync_client():
    """Return SyncClient or None. If None, caller should return 400."""
    return current_app.config.get("SYNC_CLIENT")


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
    log_activity("scan", f"扫描到 {count} 个测试套")
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
    paths = body.get("paths") or []
    log_activity("run", f"执行 {len(paths)} 个模块", target=body.get("target", "本地"))
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
    log_activity("run.plan", f"执行计划 {plan_file} ({len(execution_ids)} 个)")
    return jsonify({"ok": True, "execution_ids": execution_ids})


@bp.route("/api/run/pipeline", methods=["POST"])
def api_run_pipeline():
    """Trigger pipeline execution. Body: {"plan": "testplan.yaml"}

    The testplan must contain a ``pipeline`` field listing platform stages.
    Stages execute sequentially; if any stage fails the pipeline stops.
    Returns pipeline_id immediately.
    """
    from aitf.tc.runner import run_pipeline_async

    body = request.get_json(silent=True) or {}
    plan_file = body.get("plan")
    if not plan_file:
        return jsonify({"error": "missing 'plan' field"}), 400

    cfg = current_app.config.get("AITF_CONFIG")
    plan_path = cfg.project_root / plan_file if cfg else plan_file

    if not plan_path.is_file():
        return jsonify({"error": f"plan not found: {plan_file}"}), 404

    pipeline_id = run_pipeline_async(_cases_dir(), _db_path(), plan_path)
    if not pipeline_id:
        return jsonify({"error": "no pipeline defined in plan"}), 400
    log_activity("run.pipeline", f"流水线 {plan_file}")
    return jsonify({"ok": True, "pipeline_id": pipeline_id})


@bp.route("/api/run/pipeline/targets", methods=["POST"])
def api_run_pipeline_targets():
    """Run tests sequentially on multiple targets (pipeline by target).

    Supports two modes:
    - Ad-hoc: {"paths": [...], "targets": ["env1", "env2"]}
    - Plan:   {"plan": "testplan.yaml", "targets": ["env1", "env2"]}
    Each target becomes a pipeline stage. Stops on first failure.
    """
    from aitf.tc.runner import run_targets_pipeline_async

    body = request.get_json(silent=True) or {}
    targets = body.get("targets", [])
    if not targets or len(targets) < 2:
        return jsonify({"error": "需要至少选择2个执行环境"}), 400

    # If a plan is specified, resolve paths from it
    plan_file = body.get("plan")
    if plan_file:
        from aitf.tc.testplan import load_testplan
        cfg = current_app.config.get("AITF_CONFIG")
        plan_path = cfg.project_root / plan_file if cfg else plan_file
        if not Path(plan_path).is_file():
            return jsonify({"error": f"plan not found: {plan_file}"}), 404
        plan = load_testplan(plan_path)
        # Collect all test paths from the plan
        all_paths = []
        for c in plan.configs:
            all_paths.extend(c.tests)
        body["paths"] = all_paths
        if not body.get("bundle") and plan.configs:
            body["bundle"] = plan.configs[0].bundle

    pipeline_id = run_targets_pipeline_async(
        _cases_dir(), _db_path(),
        targets=targets,
        paths=body.get("paths"),
        filter_k=body.get("filter_k"),
        bundle=body.get("bundle"),
        params=body.get("params"),
    )
    if not pipeline_id:
        return jsonify({"error": "no tests found"}), 400
    log_activity("run.pipeline.targets", f"多环境流水线 ({len(targets)} 个环境)", targets=targets)
    return jsonify({"ok": True, "pipeline_id": pipeline_id})


@bp.route("/api/pipeline/<pipeline_id>", methods=["GET"])
def api_get_pipeline(pipeline_id: str):
    """Return pipeline status by aggregating executions with this pipeline_id."""
    from sqlalchemy import select

    from aitf.tc.db import get_session
    from aitf.tc.models import Execution

    with get_session() as session:
        execs = session.execute(
            select(Execution)
            .where(Execution.pipeline_id == pipeline_id)
            .order_by(Execution.pipeline_stage, Execution.started_at)
        ).scalars().all()

        if not execs:
            return jsonify({"error": "not found"}), 404

        stages: dict[int, list[dict]] = {}
        for exe in execs:
            stage_idx = exe.pipeline_stage or 0
            stages.setdefault(stage_idx, []).append(exe.to_dict())

        # Determine overall status
        all_finished = all(e.finished_at is not None for e in execs)
        any_failed = any(e.failed_total > 0 for e in execs if e.finished_at)
        if all_finished and not any_failed:
            status = "passed"
        elif all_finished and any_failed:
            status = "failed"
        else:
            status = "running"

    return jsonify({
        "pipeline_id": pipeline_id,
        "status": status,
        "stages": stages,
    })


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
        from aitf.tc.testplan import list_plan_files
        testplans = list_plan_files(_project_root())
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
    from aitf.tc.testplan import safe_plan_path
    return safe_plan_path(_project_root(), filename)


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

@bp.route("/api/tc/plan", methods=["POST"])
def api_save_plan():
    """Save a generated testplan YAML file.

    Body: {"name": "...", "filename": "testplan_xxx.yaml", "plans": [...]}
    Each plan item: {"name": "...", "tests": [...], "bundle": "...",
                     "golden": {"model": "...", "version": "..."}}
    """
    from aitf.tc.testplan import save_plan_yaml

    body = request.get_json(silent=True) or {}
    plans = body.get("plans", [])
    if not plans:
        return jsonify({"error": "plans 不能为空"}), 400

    filename = body.get("filename", "").strip() or "testplan_custom"

    plan_data = {"name": body.get("name", Path(filename).stem), "plans": []}
    pipeline = body.get("pipeline")
    if pipeline and isinstance(pipeline, list) and len(pipeline) > 1:
        plan_data["pipeline"] = pipeline
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

    saved, err = save_plan_yaml(_project_root(), filename, plan_data)
    if err:
        return jsonify({"error": err}), 400
    log_activity("plan.save", f"保存计划 {saved}")
    return jsonify({"ok": True, "filename": saved})


# ---------------------------------------------------------------------------
# Golden sync to remote server (proxy)
# ---------------------------------------------------------------------------

@bp.route("/api/tc/sync_golden", methods=["POST"])
def api_sync_golden_to_remote():
    """Proxy: export local golden and upload to remote server via SyncClient.

    Body: {"items": [{"model": "m", "version": "v", "operators": ["op1"]}]}
    If operators is empty/missing, uploads the entire version.
    """
    sc = _require_sync_client()
    if not sc:
        return jsonify({"error": "未配置远端服务器"}), 400

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


@bp.route("/api/tc/sync_plans", methods=["POST"])
def api_sync_plans_from_server():
    """Pull testplans from remote server and save locally."""
    from aitf.tc.testplan import save_plan_yaml

    sc = _require_sync_client()
    if not sc:
        return jsonify({"error": "未配置远端服务器"}), 400

    root = _project_root()
    try:
        plans = sc.list_plans()
        saved = 0
        for p in plans:
            filename = p.get("filename", "")
            if not filename:
                continue
            try:
                plan_data = sc.download_plan(filename)
                save_plan_yaml(root, filename, plan_data)
                saved += 1
            except Exception:
                logger.debug("Failed to download plan %s", filename, exc_info=True)
        return jsonify({"ok": True, "total": len(plans), "saved": saved})
    except Exception as exc:
        return jsonify({"error": f"同步失败: {exc}"}), 502


@bp.route("/api/tc/sync_results", methods=["POST"])
def api_sync_results_from_server():
    """Pull execution results from remote server and save to local DB."""
    sc = _require_sync_client()
    if not sc:
        return jsonify({"error": "未配置远端服务器"}), 400

    try:
        remote_execs = sc.list_executions(limit=50)
        imported = 0
        for exe_summary in remote_execs:
            eid = exe_summary.get("id", "")
            if not eid:
                continue
            local = store.get_execution_detail(eid)
            if local:
                continue
            try:
                detail = sc.get_execution_detail(eid)
                if not detail:
                    continue
                store.import_execution_from_dict(detail, trigger="sync")
                imported += 1
            except Exception:
                logger.debug("Failed to import execution %s", eid, exc_info=True)
        return jsonify({"ok": True, "total": len(remote_execs), "imported": imported})
    except Exception as exc:
        return jsonify({"error": f"同步失败: {exc}"}), 502


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
