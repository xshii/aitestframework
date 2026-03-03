"""REST API routes for golden data file management."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request, send_file, send_from_directory

from aitf.ds.store import ALLOWED_EXT, GoldenStore

bp = Blueprint("datastore", __name__, template_folder="templates")


def _store() -> GoldenStore:
    base = current_app.config.get("DATASTORE_BASE_DIR", "datastore")
    return GoldenStore(base)


@bp.route("/api/golden", methods=["GET"])
def list_golden():
    """List all golden entries (one file per model/version)."""
    entries = _store().list()
    return jsonify([
        {"model": e.model, "version": e.version, "file": e.file, "size": e.size}
        for e in entries
    ])


@bp.route("/api/golden/upload", methods=["POST"])
def upload_golden():
    """Upload a file to model/version.  Replaces existing file if any."""
    model = request.form.get("model", "").strip()
    version = request.form.get("version", "").strip()
    f = request.files.get("file")
    if not model or not version or not f or not f.filename:
        return jsonify({"error": "model, version, file are required"}), 400
    if not GoldenStore.allowed(f.filename):
        exts = ", ".join(ALLOWED_EXT)
        return jsonify({"error": f"only {exts} files accepted"}), 400

    dest = _store().save(model, version, f.filename)
    f.save(str(dest))
    return jsonify({"model": model, "version": version, "file": f.filename}), 201


@bp.route("/api/golden/<model>/<version>/download", methods=["GET"])
def download_golden(model, version):
    """Download the golden file for a model/version."""
    fp = _store().get_file(model, version)
    if not fp:
        return jsonify({"error": "not found"}), 404
    return send_from_directory(str(fp.parent), fp.name, as_attachment=True)


@bp.route("/api/golden/download-all", methods=["GET"])
def download_all_golden():
    """Download all golden files as a single zip archive."""
    store = _store()
    if not store.list():
        return jsonify({"error": "no data"}), 404
    buf = store.export_all()
    return send_file(buf, as_attachment=True, download_name="golden-all.zip",
                     mimetype="application/zip")


@bp.route("/api/golden/<model>/<version>", methods=["DELETE"])
def delete_golden(model, version):
    """Delete a model/version entry."""
    if not _store().delete(model, version):
        return jsonify({"error": "not found"}), 404
    return jsonify({"deleted": f"{model}/{version}"})
