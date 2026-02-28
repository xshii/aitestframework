"""REST API routes for datastore operations."""

from __future__ import annotations

from dataclasses import asdict

from flask import Blueprint, current_app, jsonify, request

from aitf.ds.types import CaseNotFoundError, DataStoreError, SyncError

ds_bp = Blueprint("datastore", __name__)


def _mgr():
    return current_app.config["datastore_manager"]


# -- error handlers ----------------------------------------------------------

@ds_bp.errorhandler(CaseNotFoundError)
def _handle_not_found(exc):
    return jsonify({"error": str(exc)}), 404


@ds_bp.errorhandler(ValueError)
def _handle_value_error(exc):
    return jsonify({"error": str(exc)}), 400


@ds_bp.errorhandler(SyncError)
def _handle_sync_error(exc):
    return jsonify({"error": str(exc)}), 502


@ds_bp.errorhandler(DataStoreError)
def _handle_ds_error(exc):
    return jsonify({"error": str(exc)}), 500


# -- routes ------------------------------------------------------------------

@ds_bp.route("/api/cases", methods=["GET"])
def list_cases():
    cases = _mgr().list(platform=request.args.get("platform"), model=request.args.get("model"))
    return jsonify([asdict(c) for c in cases])


@ds_bp.route("/api/cases/<path:case_id>", methods=["GET"])
def get_case(case_id):
    return jsonify(asdict(_mgr().get(case_id)))


@ds_bp.route("/api/cases", methods=["POST"])
def register_case():
    body = request.get_json(force=True)
    case_id, local_path = body.get("case_id"), body.get("local_path")
    if not case_id or not local_path:
        return jsonify({"error": "case_id and local_path are required"}), 400
    return jsonify(asdict(_mgr().register(case_id, local_path))), 201


@ds_bp.route("/api/cases/<path:case_id>", methods=["DELETE"])
def delete_case(case_id):
    _mgr().delete(case_id)
    return jsonify({"deleted": case_id})


@ds_bp.route("/api/cases/<path:case_id>/pull", methods=["POST"])
def pull_case(case_id):
    body = request.get_json(force=True)
    remote = body.get("remote")
    if not remote:
        return jsonify({"error": "remote is required"}), 400
    results = _mgr().pull(remote, case_id=case_id)
    return jsonify([asdict(r) for r in results])


@ds_bp.route("/api/cases/<path:case_id>/push", methods=["POST"])
def push_case(case_id):
    body = request.get_json(force=True)
    remote = body.get("remote")
    if not remote:
        return jsonify({"error": "remote is required"}), 400
    return jsonify(asdict(_mgr().push(remote, case_id)))


@ds_bp.route("/api/cases/<path:case_id>/push-artifacts", methods=["POST"])
def push_artifacts(case_id):
    body = request.get_json(force=True)
    remote = body.get("remote")
    if not remote:
        return jsonify({"error": "remote is required"}), 400
    return jsonify(asdict(_mgr().push_artifacts(remote, case_id, body.get("artifacts_dir"))))


@ds_bp.route("/api/cases/<path:case_id>/verify", methods=["POST"])
def verify_case(case_id):
    return jsonify([asdict(r) for r in _mgr().verify(case_id=case_id)])


@ds_bp.route("/api/cases/<path:case_id>/versions", methods=["GET"])
def list_versions(case_id):
    case = _mgr().get(case_id)
    return jsonify({"case_id": case_id, "versions": [case.version]})


@ds_bp.route("/api/verify", methods=["POST"])
def verify_all():
    return jsonify([asdict(r) for r in _mgr().verify()])


@ds_bp.route("/api/rebuild-cache", methods=["POST"])
def rebuild_cache():
    return jsonify({"rebuilt": _mgr().rebuild_cache()})
