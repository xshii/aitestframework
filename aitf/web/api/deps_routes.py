"""REST API routes for dependency & bundle management."""

from __future__ import annotations

from dataclasses import asdict

from flask import Blueprint, jsonify, request

from aitf.deps.manager import DepsManager
from aitf.deps.bundle import BundleManager
from aitf.deps.types import BundleNotFoundError, DepsConfigError, DepsError

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
    _mgr().install(name=body.get("name"))
    return jsonify({"status": "ok"})


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


@deps_bp.route("/api/bundles/<name>/use", methods=["POST"])
def use_bundle(name):
    force = (request.get_json(silent=True) or {}).get("force", False)
    _bm().use(name, force=force)
    return jsonify({"status": "ok", "active": name})
