"""Server-side sync API — receives data from remote clients."""

from __future__ import annotations

import logging
import socket
import tempfile

import yaml
from flask import Blueprint, current_app, jsonify, request, send_file

from aitf.tc.testplan import list_plan_files, safe_plan_path, save_plan_yaml
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
    from aitf.tc import store

    body = request.get_json(silent=True)
    if not body or "id" not in body:
        return jsonify({"error": "invalid payload"}), 400

    eid = body["id"]
    # Deduplicate
    if store.get_execution_detail(eid):
        return jsonify({"ok": True, "msg": "already exists"})

    store.import_execution_from_dict(body, trigger="remote")
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
    return jsonify(list_plan_files(_project_root()))


@bp.route("/api/sync/plans/<filename>", methods=["GET"])
def serve_plan(filename: str):
    """Serve a test plan YAML as JSON."""
    plan_path = safe_plan_path(_project_root(), filename)
    if plan_path is None:
        return jsonify({"error": "invalid path"}), 400
    if not plan_path.is_file():
        return jsonify({"error": "not found"}), 404
    with open(plan_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return jsonify(data or {})


@bp.route("/api/sync/plans/upload", methods=["POST"])
def receive_plan():
    """Receive a test plan from a client and save as YAML."""
    body = request.get_json(silent=True) or {}
    saved, err = save_plan_yaml(
        _project_root(), body.get("filename", "").strip(), body.get("plan"),
    )
    if err:
        return jsonify({"error": err}), 400
    logger.info("Received test plan: %s", saved)
    return jsonify({"ok": True, "filename": saved})


@bp.route("/api/sync/plans/upload_with_check", methods=["POST"])
def receive_plan_with_golden_check():
    """Receive testplan and compare golden fingerprints.

    Body: {
        "filename": "testplan_xxx.yaml",
        "plan": {...},
        "golden_fingerprints": {
            "model/version": {"operators": {"op1": "sha256...", ...}},
            ...
        }
    }
    """
    body = request.get_json(silent=True) or {}
    client_fps = body.get("golden_fingerprints", {})

    filename, err = save_plan_yaml(
        _project_root(), body.get("filename", "").strip(), body.get("plan"),
    )
    if err:
        return jsonify({"error": err}), 400

    # Compare golden fingerprints
    gs = _golden_store()
    golden_status = []

    for key, client_info in client_fps.items():
        parts = key.split("/", 1)
        if len(parts) != 2:
            continue
        model, version = parts
        client_ops = client_info.get("operators", {})

        # Get server fingerprints
        server_ops = gs.version_fingerprint(model, version)

        if not server_ops:
            # Server has no data for this version at all
            golden_status.append({
                "model": model,
                "version": version,
                "status": "missing",
                "missing_ops": list(client_ops.keys()),
                "diff_ops": [],
            })
            continue

        missing = [op for op in client_ops if op not in server_ops]
        diff = [op for op in client_ops
                if op in server_ops and client_ops[op] != server_ops[op]]

        if not missing and not diff:
            status = "match"
        elif missing and not diff:
            status = "partial"
        else:
            status = "diff"

        golden_status.append({
            "model": model,
            "version": version,
            "status": status,
            "missing_ops": missing,
            "diff_ops": diff,
        })

    logger.info("Received test plan with golden check: %s (golden sets: %d)",
                filename, len(golden_status))
    return jsonify({
        "ok": True,
        "filename": filename,
        "golden_status": golden_status,
    })


# ---------------------------------------------------------------------------
# Golden data — receive from clients
# ---------------------------------------------------------------------------

@bp.route("/api/sync/golden/upload_version", methods=["POST"])
def receive_golden_version():
    """Receive a golden version zip from a client and import it."""
    import io
    import zipfile

    model = request.form.get("model", "").strip()
    version = request.form.get("version", "").strip()
    f = request.files.get("file")
    if not model or not version or not f:
        return jsonify({"error": "model, version, and file are required"}), 400

    gs = _golden_store()
    try:
        buf = io.BytesIO(f.read())
        imported = gs.import_archive(buf)
        gs.log("sync_receive", model=model, version=version,
               files_count=len(imported), source="client")
        logger.info("Received golden version %s/%s (%d files)",
                    model, version, len(imported))
        return jsonify({"ok": True, "model": model, "version": version,
                        "imported": len(imported)})
    except (zipfile.BadZipFile, OSError) as exc:
        return jsonify({"error": f"invalid zip: {exc}"}), 400


@bp.route("/api/sync/golden/upload_operator", methods=["POST"])
def receive_golden_operator():
    """Receive a single golden operator zip from a client and import it."""
    import io
    import zipfile

    model = request.form.get("model", "").strip()
    version = request.form.get("version", "").strip()
    operator = request.form.get("operator", "").strip()
    f = request.files.get("file")
    if not model or not version or not operator or not f:
        return jsonify({"error": "model, version, operator, and file are required"}), 400

    gs = _golden_store()
    try:
        buf = io.BytesIO(f.read())
        imported = gs.import_archive(buf)
        gs.log("sync_receive", model=model, version=version,
               operator=operator, files_count=len(imported), source="client")
        logger.info("Received golden operator %s/%s/%s (%d files)",
                    model, version, operator, len(imported))
        return jsonify({"ok": True, "model": model, "version": version,
                        "operator": operator, "imported": len(imported)})
    except (zipfile.BadZipFile, OSError) as exc:
        return jsonify({"error": f"invalid zip: {exc}"}), 400
