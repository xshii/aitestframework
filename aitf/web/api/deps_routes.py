"""REST API routes for dependency & bundle management."""

from __future__ import annotations

from dataclasses import asdict

from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from aitf.deps.bundle import BundleManager
from aitf.deps.manager import DepsManager
from aitf.deps.types import BundleNotFoundError, DepsConfigError, DepsError
from aitf.web import tasks

deps_bp = Blueprint("deps", __name__)


def _mgr() -> DepsManager:
    from flask import current_app
    if "deps_manager" not in current_app.config:
        current_app.config["deps_manager"] = DepsManager()
    return current_app.config["deps_manager"]


def _bm() -> BundleManager:
    return BundleManager(_mgr())


# -- error handlers ----------------------------------------------------------

@deps_bp.errorhandler(BundleNotFoundError)
def _handle_not_found(exc):
    return jsonify({"error": str(exc)}), 404


@deps_bp.errorhandler(DepsConfigError)
def _handle_config_error(exc):
    return jsonify({"error": str(exc)}), 400


@deps_bp.errorhandler(DepsError)
def _handle_deps_error(exc):
    return jsonify({"error": str(exc)}), 500


# -- task polling ------------------------------------------------------------

@deps_bp.route("/api/tasks/<task_id>", methods=["GET"])
def get_task(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "task not found"}), 404
    return jsonify({
        "id": task.id, "status": task.status,
        "step": task.step, "done": task.done, "total": task.total,
        "error": task.error, "logs": task.logs,
    })


# -- deps routes -------------------------------------------------------------

@deps_bp.route("/api/deps", methods=["GET"])
def list_deps():
    from aitf.deps.acquire import is_installed
    from aitf.deps.repo import is_cloned

    mgr = _mgr()
    cfg = mgr.config
    result = {"toolchains": {}, "libraries": {}, "repos": {}}

    for name, tc in cfg.toolchains.items():
        result["toolchains"][name] = {
            "version": tc.version, "installed": is_installed(name, tc.version, mgr.cache_dir),
        }
    for name, lib in cfg.libraries.items():
        result["libraries"][name] = {
            "version": lib.version, "installed": is_installed(name, lib.version, mgr.cache_dir),
        }
    for name, repo in cfg.repos.items():
        result["repos"][name] = {
            "ref": repo.ref, "cloned": is_cloned(name, mgr.repos_dir),
        }
    return jsonify(result)


@deps_bp.route("/api/deps/install", methods=["POST"])
def install_dep():
    body = request.get_json(silent=True) or {}
    name = body.get("name")
    mgr = _mgr()

    def _run(task):
        mgr.install(name=name, on_progress=task.progress)

    task = tasks.submit(_run)
    return jsonify({"task_id": task.id}), 202


@deps_bp.route("/api/deps/clean", methods=["POST"])
def clean_deps():
    return jsonify({"removed": _mgr().clean()})


@deps_bp.route("/api/deps/doctor", methods=["GET"])
def doctor():
    return jsonify([asdict(r) for r in _mgr().doctor()])


# -- bundle routes -----------------------------------------------------------

@deps_bp.route("/api/bundles", methods=["GET"])
def list_bundles():
    bm = _bm()
    active = bm.active()
    return jsonify([
        {**asdict(b), "active": active is not None and b.name == active.name}
        for b in bm.list_bundles()
    ])


@deps_bp.route("/api/bundles/<name>", methods=["GET"])
def show_bundle(name):
    return jsonify(asdict(_bm().show(name)))


@deps_bp.route("/api/bundles/<name>/install", methods=["POST"])
def install_bundle(name):
    bm = _bm()

    def _run(task):
        bm.install(name, on_progress=task.progress)

    task = tasks.submit(_run)
    return jsonify({"task_id": task.id}), 202


@deps_bp.route("/api/bundles/<name>/use", methods=["POST"])
def use_bundle(name):
    force = (request.get_json(silent=True) or {}).get("force", False)
    bm = _bm()

    def _run(task):
        bm.use(name, force=force, on_progress=task.progress)

    task = tasks.submit(_run)
    return jsonify({"task_id": task.id}), 202


# -- upload / download routes -----------------------------------------------

@deps_bp.route("/api/deps/upload", methods=["POST"])
def upload_dep():
    """Upload a .tar.gz archive to deps/uploads/."""
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "file is required"}), 400
    if not f.filename.endswith(".tar.gz"):
        return jsonify({"error": "only .tar.gz files accepted"}), 400

    mgr = _mgr()
    upload_dir = mgr._root / "deps" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f.filename
    f.save(dest)
    return jsonify({"saved": f.filename, "path": str(dest), "size": dest.stat().st_size}), 201


@deps_bp.route("/api/deps/uploads", methods=["GET"])
def list_uploads():
    """List uploaded archives in deps/uploads/."""
    mgr = _mgr()
    upload_dir = mgr._root / "deps" / "uploads"
    if not upload_dir.is_dir():
        return jsonify([])
    files = []
    for p in sorted(upload_dir.glob("*.tar.gz")):
        files.append({"name": p.name, "size": p.stat().st_size})
    return jsonify(files)


@deps_bp.route("/api/deps/uploads/<filename>", methods=["DELETE"])
def delete_upload(filename):
    """Delete an uploaded archive."""
    mgr = _mgr()
    target = (mgr._root / "deps" / "uploads" / filename).resolve()
    if not target.is_file():
        return jsonify({"error": "file not found"}), 404
    target.unlink()
    return jsonify({"deleted": filename})


@deps_bp.route("/api/deps/uploads/<filename>/download", methods=["GET"])
def download_upload(filename):
    """Download an uploaded archive."""
    mgr = _mgr()
    target = (mgr._root / "deps" / "uploads" / filename).resolve()
    if not target.is_file():
        return jsonify({"error": "file not found"}), 404
    return send_file(target, as_attachment=True, download_name=filename)


@deps_bp.route("/api/bundles", methods=["POST"])
def create_bundle():
    """Create or update a bundle in deps.yaml."""
    body = request.get_json(force=True)
    name = body.get("name")
    if not name:
        return jsonify({"error": "name is required"}), 400

    from aitf.deps.config import save_deps_config
    from aitf.deps.types import BundleConfig

    mgr = _mgr()
    cfg = mgr.config
    cfg.bundles[name] = BundleConfig(
        name=name,
        description=body.get("description", ""),
        status=body.get("status", "testing"),
        toolchains=body.get("toolchains", {}),
        libraries=body.get("libraries", {}),
        repos=body.get("repos", {}),
        env=body.get("env", {}),
    )
    save_deps_config(cfg, mgr._deps_file)
    mgr.reload()
    return jsonify({"created": name}), 201


@deps_bp.route("/api/bundles/<name>", methods=["DELETE"])
def delete_bundle(name):
    """Delete a bundle from deps.yaml."""
    from aitf.deps.config import save_deps_config

    mgr = _mgr()
    cfg = mgr.config
    if name not in cfg.bundles:
        return jsonify({"error": "bundle not found"}), 404
    del cfg.bundles[name]
    if cfg.active_bundle == name:
        cfg.active_bundle = None
    save_deps_config(cfg, mgr._deps_file)
    mgr.reload()
    return jsonify({"deleted": name})


@deps_bp.route("/api/bundles/<name>/export", methods=["GET"])
def export_bundle(name):
    """Export bundle config as a deps.yaml file for local use."""
    import io
    import yaml as _yaml

    mgr = _mgr()
    cfg = mgr.config
    bundle = _bm().show(name)

    # Build a minimal deps.yaml containing only deps referenced by this bundle
    data: dict = {}
    tc_section = {}
    for tc_name, tc_ver in bundle.toolchains.items():
        tc = cfg.toolchains.get(tc_name)
        if tc:
            tc_section[tc_name] = {"version": tc_ver, "sha256": tc.sha256,
                                   "bin_dir": tc.bin_dir, "env": tc.env,
                                   "acquire": {"local_dir": tc.acquire.local_dir,
                                               "script": tc.acquire.script}}
    if tc_section:
        data["toolchains"] = tc_section

    lib_section = {}
    for lib_name, lib_ver in bundle.libraries.items():
        lib = cfg.libraries.get(lib_name)
        if lib:
            lib_section[lib_name] = {"version": lib_ver, "sha256": lib.sha256,
                                     "build_system": lib.build_system,
                                     "acquire": {"local_dir": lib.acquire.local_dir,
                                                 "script": lib.acquire.script}}
    if lib_section:
        data["libraries"] = lib_section

    repo_section = {}
    for repo_name, repo_ref in bundle.repos.items():
        repo = cfg.repos.get(repo_name)
        if repo:
            repo_section[repo_name] = {"url": repo.url, "ref": repo_ref,
                                       "depth": repo.depth,
                                       "build_script": repo.build_script}
    if repo_section:
        data["repos"] = repo_section

    data["bundles"] = {name: {
        "description": bundle.description, "status": bundle.status,
        "toolchains": bundle.toolchains, "libraries": bundle.libraries,
        "repos": bundle.repos, "env": bundle.env,
    }}
    data["active"] = name

    buf = io.BytesIO()
    buf.write(_yaml.dump(data, default_flow_style=False,
                         allow_unicode=True, sort_keys=False).encode("utf-8"))
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f"{name}-deps.yaml", mimetype="text/yaml")


@deps_bp.route("/api/bundles/import", methods=["POST"])
def import_bundle():
    """Import a bundle .tar.gz archive."""
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "file is required"}), 400
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    archive = tmp / f.filename
    f.save(archive)
    bm = _bm()
    bundle_name = bm.import_bundle(archive)
    return jsonify({"imported": bundle_name}), 201
