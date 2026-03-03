"""REST API routes for dependency & bundle management.

Self-contained plugin blueprint — auto-discovered by ``aitf.web.app``.

The web UI is a **configuration depot**.  Users upload dependencies
(archive + metadata, repo URL, or script), browse/download them as YAML,
and compose bundles.  Actual installation happens via CLI.
"""

from __future__ import annotations

import io
import re
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path

import yaml as _yaml
from flask import Blueprint, current_app, jsonify, request, send_file
from werkzeug.utils import secure_filename

from aitf.deps.config import strip_none
from aitf.deps.types import BundleNotFoundError, BundleStatus, DepsConfigError, DepsError

bp = Blueprint("deps", __name__, template_folder="templates")

# Guards concurrent writes to deps.yaml.
_config_lock = threading.Lock()


# -- helpers ----------------------------------------------------------------

def _mgr():
    """Return the shared DepsManager, creating on first call."""
    from aitf.deps.manager import DepsManager

    if "deps_manager" not in current_app.config:
        aitf_cfg = current_app.config.get("AITF_CONFIG")
        if aitf_cfg is not None:
            current_app.config["deps_manager"] = DepsManager(
                project_root=str(aitf_cfg.project_root),
                build_dir=str(aitf_cfg.build_root),
            )
        else:
            current_app.config["deps_manager"] = DepsManager()
    return current_app.config["deps_manager"]


def _bm():
    """Return a BundleManager wrapping the shared DepsManager."""
    from aitf.deps.bundle import BundleManager

    return BundleManager(_mgr())


def _upload_dir() -> Path:
    return _mgr().project_root / "deps" / "uploads"


def _upload_files() -> set[str]:
    d = _upload_dir()
    return {p.name for p in d.glob("*.tar.gz")} if d.is_dir() else set()


def _extract_versions(dep_name: str, filenames: set[str]) -> list[str]:
    """Extract available versions for *dep_name* from upload filenames."""
    prefix = dep_name + "-"
    versions: set[str] = set()
    for f in filenames:
        if not f.startswith(prefix) or not f.endswith(".tar.gz"):
            continue
        middle = f[len(prefix):-len(".tar.gz")]
        m = re.match(r'^(.+?)(?:-(?:linux|darwin|windows)-\w+)?$', middle)
        if m:
            versions.add(m.group(1))
    return sorted(versions, reverse=True)


def _safe_upload_path(filename: str) -> Path | None:
    safe_name = secure_filename(filename)
    if not safe_name:
        return None
    upload = _upload_dir().resolve()
    target = (upload / safe_name).resolve()
    if not target.is_relative_to(upload):
        return None
    return target


def _send_yaml(data: dict, download_name: str):
    """Serialize *data* to YAML and return as a downloadable response."""
    buf = io.BytesIO()
    buf.write(_yaml.dump(data, default_flow_style=False,
                         allow_unicode=True, sort_keys=False).encode("utf-8"))
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=download_name, mimetype="text/yaml")


def _validate_bundle_status(status: str):
    """Return an error response tuple if *status* is invalid, else None."""
    valid = {s.value for s in BundleStatus}
    if status not in valid:
        return jsonify({"error": f"invalid status, must be one of: {', '.join(sorted(valid))}"}), 400
    return None


def _save_cfg(mgr):
    """Persist config and reload (call inside _config_lock)."""
    from aitf.deps.config import save_deps_config
    save_deps_config(mgr.config, mgr.deps_file)
    mgr.reload()


# -- error handlers ----------------------------------------------------------

@bp.errorhandler(BundleNotFoundError)
def _handle_not_found(exc):
    return jsonify({"error": str(exc)}), 404


@bp.errorhandler(DepsConfigError)
def _handle_config_error(exc):
    return jsonify({"error": str(exc)}), 400


@bp.errorhandler(DepsError)
def _handle_deps_error(exc):
    return jsonify({"error": str(exc)}), 500


# -- deps routes -------------------------------------------------------------

@bp.route("/api/deps", methods=["GET"])
def list_deps():
    """Flat list of all dependencies with available versions."""
    mgr = _mgr()
    cfg = mgr.config
    available = _upload_files()
    items = []

    for dep_type, section in [("toolchain", cfg.toolchains), ("library", cfg.libraries)]:
        for name, dep in section.items():
            versions = _extract_versions(name, available)
            if dep.version not in versions:
                versions.append(dep.version)
            versions.sort(reverse=True)
            items.append({"name": name, "type": dep_type,
                          "version": dep.version, "versions": versions,
                          "order": dep.order, "install_dir": dep.install_dir or ""})
    for name, rc in cfg.repos.items():
        items.append({"name": name, "type": "repo",
                      "version": rc.ref, "versions": [rc.ref],
                      "order": rc.order, "install_dir": rc.install_dir or ""})
    return jsonify(items)


@bp.route("/api/deps", methods=["POST"])
def add_dep():
    """Add a dependency (or a new version of an existing one).

    Multipart form: name, type (toolchain/library/repo), version,
    and optional file (.tar.gz or script).
    For repos: url field.
    """
    from aitf.deps.types import AcquireConfig, LibraryConfig, RepoConfig, ToolchainConfig

    name = (request.form.get("name") or "").strip()
    dep_type = (request.form.get("type") or "").strip()
    version = (request.form.get("version") or "").strip()
    order = int(request.form.get("order") or 0)
    install_dir = (request.form.get("install_dir") or "").strip() or None
    if not name:
        return jsonify({"error": "name is required"}), 400
    if not version:
        return jsonify({"error": "version is required"}), 400

    mgr = _mgr()
    f = request.files.get("file")

    with _config_lock:
        cfg = mgr.config

        # Determine type: explicit > existing config
        if not dep_type:
            if name in cfg.toolchains:
                dep_type = "toolchain"
            elif name in cfg.libraries:
                dep_type = "library"
            elif name in cfg.repos:
                dep_type = "repo"
            else:
                return jsonify({"error": "type is required for new dependencies"}), 400

        # Save uploaded file if present
        if f and f.filename:
            upload = _upload_dir()
            upload.mkdir(parents=True, exist_ok=True)
            if f.filename.endswith(".tar.gz"):
                dest = upload / f"{name}-{version}.tar.gz"
                f.save(dest)
            else:
                f.save(upload / secure_filename(f.filename))

        # Create/update config entry
        acq = AcquireConfig(local_dir="deps/uploads/")
        if dep_type == "toolchain":
            cfg.toolchains[name] = ToolchainConfig(
                name=name, version=version, acquire=acq,
                order=order, install_dir=install_dir,
            )
        elif dep_type == "library":
            cfg.libraries[name] = LibraryConfig(
                name=name, version=version, acquire=acq,
                order=order, install_dir=install_dir,
            )
        elif dep_type == "repo":
            url = (request.form.get("url") or "").strip()
            if not url and name in cfg.repos:
                url = cfg.repos[name].url
            cfg.repos[name] = RepoConfig(name=name, url=url, ref=version,
                                         order=order, install_dir=install_dir)
        else:
            return jsonify({"error": f"unknown type: {dep_type}"}), 400

        _save_cfg(mgr)

    return jsonify({"added": name, "version": version}), 201


@bp.route("/api/deps/<name>", methods=["DELETE"])
def delete_dep(name):
    """Delete a dependency from deps.yaml and its archives."""
    mgr = _mgr()
    with _config_lock:
        cfg = mgr.config
        found = False
        for section in (cfg.toolchains, cfg.libraries, cfg.repos):
            if name in section:
                del section[name]
                found = True
                break
        if not found:
            return jsonify({"error": f"dependency '{name}' not found"}), 404

        # Remove from any bundles that reference it
        for b in cfg.bundles.values():
            for cat in (b.toolchains, b.libraries, b.repos):
                cat.pop(name, None)

        _save_cfg(mgr)

    # Delete associated archives
    upload = _upload_dir()
    if upload.is_dir():
        for p in upload.glob(f"{name}-*.tar.gz"):
            p.unlink(missing_ok=True)

    return jsonify({"deleted": name})


@bp.route("/api/deps/export", methods=["GET"])
def export_deps():
    """Return the full deps.yaml as a downloadable YAML file."""
    mgr = _mgr()
    deps_path = mgr.deps_file
    if not deps_path.is_file():
        return jsonify({"error": "deps.yaml not found"}), 404
    return send_file(deps_path, as_attachment=True,
                     download_name="deps.yaml", mimetype="text/yaml")


@bp.route("/api/deps/<name>/export", methods=["GET"])
def export_single_dep(name):
    """Return a minimal deps.yaml for a single dependency."""
    mgr = _mgr()
    cfg = mgr.config
    version = request.args.get("version")

    data: dict = {}

    if name in cfg.toolchains:
        tc = cfg.toolchains[name]
        data["toolchains"] = {name: {"version": version or tc.version}}
    elif name in cfg.libraries:
        lib = cfg.libraries[name]
        data["libraries"] = {name: {"version": version or lib.version}}
    elif name in cfg.repos:
        rc = cfg.repos[name]
        data["repos"] = {name: {"url": rc.url, "ref": version or rc.ref}}
    else:
        return jsonify({"error": f"dependency '{name}' not found"}), 404

    return _send_yaml(data, f"{name}.yaml")


# -- instance info & sync ---------------------------------------------------

@bp.route("/api/info")
def instance_info():
    """Return current mode / server / port for the frontend."""
    aitf_cfg = current_app.config.get("AITF_CONFIG")
    if aitf_cfg is None:
        return jsonify({"mode": "standalone", "server": "", "port": 5000})
    return jsonify({"mode": aitf_cfg.mode.value, "server": aitf_cfg.server, "port": aitf_cfg.port})


@bp.route("/api/deps/sync", methods=["POST"])
def sync_from_server():
    """Client-mode: pull deps.yaml from the remote server and reload."""
    from urllib.request import Request, urlopen

    aitf_cfg = current_app.config.get("AITF_CONFIG")
    if not aitf_cfg or not aitf_cfg.server_url:
        return jsonify({"error": "sync is only available in client mode"}), 400

    export_url = f"{aitf_cfg.server_url}/api/deps/export"
    try:
        resp = urlopen(Request(export_url), timeout=30)
        deps_yaml = resp.read()
    except Exception as exc:
        return jsonify({"error": f"failed to fetch from server: {exc}"}), 502

    mgr = _mgr()
    with _config_lock:
        mgr.deps_file.write_bytes(deps_yaml)
        mgr.reload()

    return jsonify({"ok": True, "bytes": len(deps_yaml)})


# -- upload / download routes (used by CLI sync) ----------------------------

@bp.route("/api/deps/uploads", methods=["GET"])
def list_uploads():
    upload = _upload_dir()
    if not upload.is_dir():
        return jsonify([])
    return jsonify([
        {"name": p.name, "size": p.stat().st_size}
        for p in sorted(upload.glob("*.tar.gz"))
    ])


@bp.route("/api/deps/uploads/<filename>", methods=["DELETE"])
def delete_upload(filename):
    target = _safe_upload_path(filename)
    if not target:
        return jsonify({"error": "file not found"}), 404
    try:
        target.unlink(missing_ok=False)
    except FileNotFoundError:
        return jsonify({"error": "file not found"}), 404
    return jsonify({"deleted": filename})


@bp.route("/api/deps/uploads/<filename>/download", methods=["GET"])
def download_upload(filename):
    target = _safe_upload_path(filename)
    if not target or not target.is_file():
        return jsonify({"error": "file not found"}), 404
    return send_file(target, as_attachment=True, download_name=target.name)


# -- bundle routes -----------------------------------------------------------

@bp.route("/api/bundles", methods=["GET"])
def list_bundles():
    mgr = _mgr()
    active_name = mgr.config.active_bundle
    return jsonify([
        {**asdict(b), "active": b.name == active_name}
        for b in _bm().list_bundles()
    ])


@bp.route("/api/bundles/<name>", methods=["GET"])
def show_bundle(name):
    return jsonify(asdict(_bm().show(name)))


@bp.route("/api/bundles", methods=["POST"])
def create_bundle():
    from aitf.deps.types import BundleConfig

    body = request.get_json(silent=True) or {}
    name = body.get("name")
    if not name:
        return jsonify({"error": "name is required"}), 400

    status = body.get("status", "testing")
    err = _validate_bundle_status(status)
    if err:
        return err

    mgr = _mgr()
    with _config_lock:
        cfg = mgr.config
        cfg.bundles[name] = BundleConfig(
            name=name,
            description=body.get("description", ""),
            status=status,
            toolchains=body.get("toolchains", {}),
            libraries=body.get("libraries", {}),
            repos=body.get("repos", {}),
            env=body.get("env", {}),
        )
        _save_cfg(mgr)
    return jsonify({"created": name}), 201


@bp.route("/api/bundles/<name>", methods=["PUT"])
def update_bundle(name):
    from aitf.deps.types import BundleConfig

    body = request.get_json(silent=True) or {}
    mgr = _mgr()
    with _config_lock:
        cfg = mgr.config
        if name not in cfg.bundles:
            return jsonify({"error": "bundle not found"}), 404

        status = body.get("status", cfg.bundles[name].status)
        err = _validate_bundle_status(status)
        if err:
            return err

        old = cfg.bundles[name]
        cfg.bundles[name] = BundleConfig(
            name=name,
            description=body.get("description", old.description),
            status=status,
            toolchains=body.get("toolchains", old.toolchains),
            libraries=body.get("libraries", old.libraries),
            repos=body.get("repos", old.repos),
            env=body.get("env", old.env),
        )
        _save_cfg(mgr)
    return jsonify({"updated": name})


@bp.route("/api/bundles/<name>", methods=["DELETE"])
def delete_bundle(name):
    mgr = _mgr()
    with _config_lock:
        cfg = mgr.config
        if name not in cfg.bundles:
            return jsonify({"error": "bundle not found"}), 404
        del cfg.bundles[name]
        if cfg.active_bundle == name:
            cfg.active_bundle = None
        _save_cfg(mgr)
    return jsonify({"deleted": name})


@bp.route("/api/bundles/<name>/export", methods=["GET"])
def export_bundle(name):
    """Export bundle config as a deps.yaml file for local use."""
    mgr = _mgr()
    cfg = mgr.config
    bundle = _bm().show(name)

    data: dict = {}

    tc_section = {}
    for tc_name, tc_ver in bundle.toolchains.items():
        if tc_name in cfg.toolchains:
            tc_section[tc_name] = {"version": tc_ver}
    if tc_section:
        data["toolchains"] = tc_section

    lib_section = {}
    for lib_name, lib_ver in bundle.libraries.items():
        if lib_name in cfg.libraries:
            lib_section[lib_name] = {"version": lib_ver}
    if lib_section:
        data["libraries"] = lib_section

    repo_section = {}
    for repo_name, repo_ref in bundle.repos.items():
        repo_cfg = cfg.repos.get(repo_name)
        if repo_cfg:
            repo_section[repo_name] = {"url": repo_cfg.url, "ref": repo_ref}
    if repo_section:
        data["repos"] = repo_section

    data["bundles"] = {name: strip_none({
        "description": bundle.description or None, "status": bundle.status,
        "toolchains": bundle.toolchains, "libraries": bundle.libraries,
        "repos": bundle.repos,
    })}
    data["active"] = name

    return _send_yaml(data, f"{name}-deps.yaml")


@bp.route("/api/bundles/import", methods=["POST"])
def import_bundle():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "file is required"}), 400

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / secure_filename(f.filename)
        f.save(archive)
        bundle_name = _bm().import_bundle(archive)
    return jsonify({"imported": bundle_name}), 201
