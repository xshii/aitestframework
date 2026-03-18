"""REST API routes for execution environment (target) management.

Auto-discovered by ``aitf.web.app`` via the ``bp`` attribute.
"""

from __future__ import annotations

import logging
import socket
import threading
from pathlib import Path

import yaml
from flask import Blueprint, jsonify, request

bp = Blueprint("runner", __name__, template_folder="templates")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _targets_path() -> Path:
    """Return the resolved path to targets.yaml."""
    from aitf.web.extensions import get_project_root
    root = get_project_root()
    for candidate in [root / "runner" / "targets.yaml",
                      root / "targets.yaml",
                      root / "config" / "targets.yaml"]:
        if candidate.is_file():
            return candidate
    # default location (may not exist yet)
    return root / "targets.yaml"


def _default_targets() -> dict:
    """Generate default targets from config.yaml (local + server if configured)."""
    from flask import current_app
    targets: dict = {
        "local": {"type": "local", "build_dir": "build/"},
    }
    cfg = current_app.config.get("AITF_CONFIG")
    if cfg and cfg.server:
        targets["server"] = {
            "type": "remote",
            "host": cfg.server,
            "port": cfg.port,
            "user": "",
            "auth": {"method": "key"},
            "remote_dir": "/tmp/aitf",
        }
    return targets


def _load_raw() -> dict:
    """Load raw YAML dict from targets.yaml.  Auto-creates with defaults if missing."""
    p = _targets_path()
    if not p.is_file():
        defaults = _default_targets()
        _save_raw(defaults)
        return defaults
    with open(p, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("targets", {}) if isinstance(data, dict) else {}


def _save_raw(targets: dict) -> None:
    """Write targets dict back to targets.yaml."""
    p = _targets_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        yaml.dump({"targets": targets}, fh, default_flow_style=False,
                   allow_unicode=True, sort_keys=False)


def _target_to_json(name: str, raw: dict) -> dict:
    """Convert a single raw target entry to a JSON-friendly dict."""
    return {
        "name": name,
        "type": raw.get("type", "local"),
        "host": raw.get("host", ""),
        "port": raw.get("port", 22),
        "user": raw.get("user", ""),
        "auth_method": (raw.get("auth") or {}).get("method", "key"),
        "remote_dir": raw.get("remote_dir", ""),
        "build_dir": raw.get("build_dir", ""),
        "env": raw.get("env", {}),
        "pre_commands": raw.get("pre_commands", []),
        "post_commands": raw.get("post_commands", []),
    }


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@bp.route("/api/targets", methods=["GET"])
def list_targets():
    """List all configured execution targets."""
    raw = _load_raw()
    return jsonify([_target_to_json(n, v) for n, v in raw.items()])


@bp.route("/api/targets/<name>", methods=["GET"])
def get_target(name):
    """Get a single target by name."""
    raw = _load_raw()
    if name not in raw:
        return jsonify({"error": f"目标 '{name}' 不存在"}), 404
    return jsonify(_target_to_json(name, raw[name]))


@bp.route("/api/targets", methods=["POST"])
def add_target():
    """Add a new target."""
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        return jsonify({"error": "名称不能为空"}), 400

    raw = _load_raw()
    if name in raw:
        return jsonify({"error": f"目标 '{name}' 已存在"}), 409

    raw[name] = _body_to_raw(body)
    _save_raw(raw)
    return jsonify({"ok": True, "name": name})


@bp.route("/api/targets/<name>", methods=["PUT"])
def update_target(name):
    """Update an existing target."""
    body = request.get_json(silent=True) or {}
    raw = _load_raw()
    if name not in raw:
        return jsonify({"error": f"目标 '{name}' 不存在"}), 404

    raw[name] = _body_to_raw(body)
    _save_raw(raw)
    return jsonify({"ok": True})


@bp.route("/api/targets/<name>", methods=["DELETE"])
def delete_target(name):
    """Delete a target."""
    raw = _load_raw()
    if name not in raw:
        return jsonify({"error": f"目标 '{name}' 不存在"}), 404
    del raw[name]
    _save_raw(raw)
    return jsonify({"ok": True})


@bp.route("/api/targets/<name>/test", methods=["POST"])
def test_target(name):
    """Test connectivity to a target. Returns {ok, message}."""
    raw = _load_raw()
    if name not in raw:
        return jsonify({"error": f"目标 '{name}' 不存在"}), 404

    t = raw[name]
    target_type = t.get("type", "local")

    if target_type == "local":
        return jsonify({"ok": True, "message": "本地环境可用"})

    host = t.get("host", "")
    port = int(t.get("port", 22))
    if not host:
        return jsonify({"ok": False, "message": "未配置主机地址"})

    # TCP connectivity check (non-blocking, 5s timeout)
    try:
        sock = socket.create_connection((host, port), timeout=5)
        sock.close()
        return jsonify({"ok": True, "message": f"{host}:{port} 可达"})
    except OSError as exc:
        return jsonify({"ok": False, "message": f"{host}:{port} 不可达: {exc}"})


# ---------------------------------------------------------------------------
# helpers (body parsing)
# ---------------------------------------------------------------------------

def _body_to_raw(body: dict) -> dict:
    """Convert JSON request body to targets.yaml format."""
    target_type = body.get("type", "local")
    entry: dict = {"type": target_type}

    if target_type == "remote":
        if body.get("host"):
            entry["host"] = body["host"]
        entry["port"] = int(body.get("port", 22))
        if body.get("user"):
            entry["user"] = body["user"]

        auth_method = body.get("auth_method", "key")
        auth: dict = {"method": auth_method}
        if auth_method == "key" and body.get("key_file"):
            auth["key_file"] = body["key_file"]
        if auth:
            entry["auth"] = auth

        if body.get("remote_dir"):
            entry["remote_dir"] = body["remote_dir"]

    if body.get("build_dir"):
        entry["build_dir"] = body["build_dir"]
    if body.get("env"):
        entry["env"] = body["env"]
    if body.get("pre_commands"):
        entry["pre_commands"] = body["pre_commands"]
    if body.get("post_commands"):
        entry["post_commands"] = body["post_commands"]

    return entry
