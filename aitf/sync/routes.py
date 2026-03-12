"""Server-side sync API — receives data from remote clients."""

from __future__ import annotations

import json
import logging
import socket
import tempfile
from pathlib import Path

import yaml
from flask import Blueprint, current_app, jsonify, request, send_file

from aitf.web.extensions import (
    get_bundle_manager as _bundle_manager,
    get_deps_manager as _deps_manager,
    get_golden_store as _golden_store,
    get_project_root as _project_root,
)

logger = logging.getLogger(__name__)

bp = Blueprint("sync", __name__)


# ---------------------------------------------------------------------------
# ping / info
# ---------------------------------------------------------------------------

@bp.route("/api/sync/ping", methods=["GET"])
def sync_ping():
    """Health check and server info."""
    cfg = current_app.config.get("AITF_CONFIG")
    return jsonify({
        "ok": True,
        "hostname": socket.gethostname(),
        "mode": cfg.mode.value if cfg else "unknown",
        "port": cfg.port if cfg else 5000,
    })


# ---------------------------------------------------------------------------
# Execution results — receive from clients
# ---------------------------------------------------------------------------

@bp.route("/api/sync/results", methods=["POST"])
def receive_results():
    """Receive execution results from a client and store in server DB."""
    from aitf.tc.db import get_session
    from aitf.tc.models import CaseResult, Execution

    body = request.get_json(silent=True)
    if not body or "id" not in body:
        return jsonify({"error": "invalid payload"}), 400

    eid = body["id"]

    with get_session() as session:
        # Deduplicate — skip if already exists
        existing = session.get(Execution, eid)
        if existing:
            return jsonify({"ok": True, "msg": "already exists"})

        exe = Execution(
            id=eid,
            bundle=body.get("bundle"),
            target=body.get("target"),
            golden_model=body.get("golden_model"),
            golden_version=body.get("golden_version"),
            plan_name=body.get("plan_name"),
            platform=body.get("platform"),
            git_commit=body.get("git_commit"),
            trigger=body.get("trigger", "remote"),
            total=body.get("total", 0),
            passed=body.get("passed", 0),
            failed=body.get("failed", 0),
            timeout=body.get("timeout", 0),
            crashed=body.get("crashed", 0),
            skipped=body.get("skipped", 0),
            errored=body.get("errored", 0),
            pass_rate=body.get("pass_rate", 0.0),
        )
        # Parse timestamps
        from datetime import datetime
        for ts_field in ("started_at", "finished_at"):
            val = body.get(ts_field)
            if val:
                try:
                    setattr(exe, ts_field, datetime.fromisoformat(val))
                except (ValueError, TypeError):
                    pass

        session.add(exe)

        for c in body.get("cases", []):
            cr = CaseResult(
                execution_id=eid,
                suite_class=c.get("suite_class", ""),
                case_method=c.get("case_method", ""),
                status=c.get("status", "PENDING"),
                failure_reason=c.get("failure_reason"),
            )
            if c.get("duration_s") is not None:
                cr.duration_s = c["duration_s"]
            if c.get("compare_detail"):
                cr.compare_detail = (
                    json.dumps(c["compare_detail"])
                    if not isinstance(c["compare_detail"], str)
                    else c["compare_detail"]
                )
            session.add(cr)

        session.commit()

    logger.info("Received execution %s from client (%d cases)",
                eid, len(body.get("cases", [])))
    return jsonify({"ok": True, "execution_id": eid})


@bp.route("/api/sync/results", methods=["GET"])
def serve_results():
    """Return recent executions (for clients to view server history)."""
    from aitf.tc import store
    limit = request.args.get("limit", 50, type=int)
    return jsonify(store.list_executions(limit=limit))


@bp.route("/api/sync/results/<execution_id>", methods=["GET"])
def serve_result_detail(execution_id: str):
    """Return single execution detail."""
    from aitf.tc import store
    detail = store.get_execution_detail(execution_id)
    if not detail:
        return jsonify({"error": "not found"}), 404
    return jsonify(detail)


# ---------------------------------------------------------------------------
# Deps — serve archives to clients
# ---------------------------------------------------------------------------

@bp.route("/api/sync/deps/<name>/<version>", methods=["GET"])
def serve_dep_archive(name: str, version: str):
    """Serve a dependency archive for client download."""
    from aitf.deps.acquire import archive_candidates
    from aitf.deps.config import detect_platform

    dm = _deps_manager()
    cache = dm.cache_dir
    plat = detect_platform()

    for candidate in archive_candidates(name, version, plat):
        p = cache / ".downloads" / candidate
        if p.is_file():
            return send_file(str(p), as_attachment=True,
                             download_name=candidate)

    # Try local_dir from config
    cfg = dm.config
    for section in (cfg.toolchains, cfg.libraries):
        if name in section:
            dep = section[name]
            if dep.acquire.local_dir:
                local = dm.project_root / dep.acquire.local_dir
                for candidate in archive_candidates(name, version, plat):
                    p = local / candidate
                    if p.is_file():
                        return send_file(str(p), as_attachment=True,
                                         download_name=candidate)

    return jsonify({"error": f"archive not found: {name}-{version}"}), 404


@bp.route("/api/sync/deps/upload", methods=["POST"])
def receive_dep_archive():
    """Receive a dependency archive from a client."""
    name = request.form.get("name", "").strip()
    version = request.form.get("version", "").strip()
    f = request.files.get("file")
    if not name or not version or not f:
        return jsonify({"error": "name, version, and file are required"}), 400

    from werkzeug.utils import secure_filename as _secure
    dm = _deps_manager()
    dest = dm.cache_dir / ".downloads"
    dest.mkdir(parents=True, exist_ok=True)
    safe_name = _secure(f.filename)
    if not safe_name:
        return jsonify({"error": "invalid filename"}), 400
    target = dest / safe_name
    f.save(str(target))

    logger.info("Received dep archive: %s (%s)", safe_name, name)
    return jsonify({"ok": True, "name": name, "version": version,
                    "filename": safe_name})


# ---------------------------------------------------------------------------
# Bundle — serve / receive
# ---------------------------------------------------------------------------

@bp.route("/api/sync/deps/bundle/<name>", methods=["GET"])
def serve_bundle(name: str):
    """Export and serve a bundle archive."""
    bm = _bundle_manager()
    try:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            out = bm.export_bundle(name, tmp.name)
        return send_file(str(out), as_attachment=True,
                         download_name=f"{name}.tar.gz")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 404


@bp.route("/api/sync/deps/bundle/upload", methods=["POST"])
def receive_bundle():
    """Receive a bundle archive from a client."""
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "file is required"}), 400

    bm = _bundle_manager()
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        f.save(tmp.name)
        bundle_name = bm.import_bundle(tmp.name)

    logger.info("Received bundle archive: %s", bundle_name)
    return jsonify({"ok": True, "name": bundle_name})


# ---------------------------------------------------------------------------
# Test plans — serve / receive
# ---------------------------------------------------------------------------

@bp.route("/api/sync/plans", methods=["GET"])
def serve_plans():
    """List available test plans on the server."""
    root = _project_root()
    plans = []
    for pattern in ("testplan*.yaml", "testplan*.yml"):
        for f in sorted(root.glob(pattern)):
            try:
                with open(f, encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
                plan_name = data.get("name", f.stem)
            except Exception:
                plan_name = f.stem
            plans.append({"filename": f.name, "name": plan_name})
    return jsonify(plans)


@bp.route("/api/sync/plans/<filename>", methods=["GET"])
def serve_plan(filename: str):
    """Serve a test plan YAML as JSON."""
    root = _project_root().resolve()
    plan_path = (root / filename).resolve()
    if not plan_path.is_relative_to(root):
        return jsonify({"error": "invalid path"}), 400
    if not plan_path.is_file():
        return jsonify({"error": "not found"}), 404
    with open(plan_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return jsonify(data or {})


@bp.route("/api/sync/plans/upload", methods=["POST"])
def receive_plan():
    """Receive a test plan from a client and save as YAML."""
    import re
    body = request.get_json(silent=True) or {}
    filename = body.get("filename", "").strip()
    plan_data = body.get("plan")
    if not filename or not plan_data:
        return jsonify({"error": "filename and plan are required"}), 400

    if not filename.endswith((".yaml", ".yml")):
        filename += ".yaml"

    # Sanitise filename — prevent path traversal
    stem = Path(filename).stem
    if not re.match(r'^[a-zA-Z0-9_\-\u4e00-\u9fff]+$', stem):
        return jsonify({"error": "invalid filename"}), 400
    filename = stem + ".yaml"

    root = _project_root()
    out = root / filename
    with open(out, "w", encoding="utf-8") as fh:
        yaml.dump(plan_data, fh, allow_unicode=True, default_flow_style=False,
                  sort_keys=False)

    logger.info("Received test plan: %s", filename)
    return jsonify({"ok": True, "filename": filename})
