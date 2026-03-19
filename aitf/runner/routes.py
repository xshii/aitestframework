"""REST API routes for execution environment (target) management.

Auto-discovered by ``aitf.web.app`` via the ``bp`` attribute.
Supports the multi-port target model with platform lifecycle.
"""

from __future__ import annotations

import logging
import socket
import threading
from pathlib import Path

import yaml
from flask import Blueprint, Response, current_app, jsonify, request
from aitf.web.activity import log_activity

bp = Blueprint("runner", __name__, template_folder="templates")

logger = logging.getLogger(__name__)

PLATFORMS = ["pc_func", "pc_perf", "fpga", "emu", "eda"]

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
    return root / "targets.yaml"


def _default_targets_data() -> dict:
    """Generate default targets.yaml content."""
    from flask import current_app
    data: dict = {
        "platforms": {
            "pc_func": {"label": "PC功能测试"},
            "pc_perf": {"label": "PC性能测试"},
            "fpga": {"label": "FPGA验证"},
            "emu": {"label": "仿真验证"},
            "eda": {"label": "EDA综合仿真"},
        },
        "targets": {
            "local": {"platform": "pc_func", "build_dir": "build/"},
        },
    }
    cfg = current_app.config.get("AITF_CONFIG")
    if cfg and cfg.server:
        data["targets"]["server"] = {
            "platform": "pc_func",
            "ports": {
                "ctrl": {
                    "type": "ssh",
                    "host": cfg.server,
                    "port": cfg.port,
                    "user": "",
                },
            },
            "remote_dir": "/tmp/aitf",
        }
    return data


def _load_raw() -> dict:
    """Load raw YAML from targets.yaml. Auto-creates with defaults if missing."""
    p = _targets_path()
    if not p.is_file():
        defaults = _default_targets_data()
        _save_raw(defaults)
        return defaults
    with open(p, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def _save_raw(data: dict) -> None:
    """Write full data dict back to targets.yaml."""
    p = _targets_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False,
                   allow_unicode=True, sort_keys=False)


def _target_to_json(name: str, raw: dict) -> dict:
    """Convert a single raw target entry to a JSON-friendly dict."""
    ports = raw.get("ports", {})
    # Backward compat: single host → synthesize a ctrl port
    if not ports and raw.get("host"):
        ports = {"ctrl": {
            "type": raw.get("type", "ssh"),
            "host": raw["host"],
            "port": raw.get("port", 22),
            "user": raw.get("user", ""),
        }}

    # Derive display fields from ports
    primary_host = ""
    primary_user = ""
    for p in ports.values():
        if isinstance(p, dict) and p.get("host"):
            primary_host = f"{p['host']}:{p.get('port', '')}"
            primary_user = p.get("user", "")
            break

    return {
        "name": name,
        "platform": raw.get("platform", ""),
        "pool": raw.get("pool", ""),
        "ports": ports,
        "host": primary_host,
        "user": primary_user,
        "remote_dir": raw.get("remote_dir", ""),
        "build_dir": raw.get("build_dir", ""),
        "env": raw.get("env", {}),
        "is_local": not ports,
    }


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@bp.route("/api/targets/platforms", methods=["GET"])
def list_platforms():
    """Return available platform categories with labels."""
    data = _load_raw()
    raw_platforms = data.get("platforms", {})
    result = []
    for p in PLATFORMS:
        pcfg = raw_platforms.get(p, {})
        label = pcfg.get("label", p) if isinstance(pcfg, dict) else p
        has_steps = bool(pcfg.get("steps")) if isinstance(pcfg, dict) else False
        result.append({"name": p, "label": label, "has_steps": has_steps})
    return jsonify(result)


@bp.route("/api/targets", methods=["GET"])
def list_targets():
    """List all configured execution targets."""
    data = _load_raw()
    raw = data.get("targets", {})
    return jsonify([_target_to_json(n, v) for n, v in raw.items()
                    if isinstance(v, dict)])


@bp.route("/api/targets/<name>", methods=["GET"])
def get_target(name):
    """Get a single target by name."""
    data = _load_raw()
    raw = data.get("targets", {})
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

    data = _load_raw()
    targets = data.setdefault("targets", {})
    if name in targets:
        return jsonify({"error": f"目标 '{name}' 已存在"}), 409

    targets[name] = _body_to_raw(body)
    _save_raw(data)
    log_activity("target.add", f"添加环境 {name}", platform=body.get("platform", ""))
    return jsonify({"ok": True, "name": name})


@bp.route("/api/targets/<name>", methods=["PUT"])
def update_target(name):
    """Update an existing target."""
    body = request.get_json(silent=True) or {}
    data = _load_raw()
    targets = data.get("targets", {})
    if name not in targets:
        return jsonify({"error": f"目标 '{name}' 不存在"}), 404

    targets[name] = _body_to_raw(body)
    _save_raw(data)
    log_activity("target.edit", f"编辑环境 {name}")
    return jsonify({"ok": True})


@bp.route("/api/targets/<name>", methods=["DELETE"])
def delete_target(name):
    """Delete a target."""
    data = _load_raw()
    targets = data.get("targets", {})
    if name not in targets:
        return jsonify({"error": f"目标 '{name}' 不存在"}), 404
    del targets[name]
    _save_raw(data)
    log_activity("target.delete", f"删除环境 {name}")
    return jsonify({"ok": True})


@bp.route("/api/targets/<name>/test", methods=["POST"])
def test_target(name):
    """Test connectivity to a target's ports. Returns per-port status."""
    data = _load_raw()
    targets = data.get("targets", {})
    if name not in targets:
        return jsonify({"error": f"目标 '{name}' 不存在"}), 404

    t = targets[name]
    ports = t.get("ports", {})

    # No ports → local target
    if not ports:
        return jsonify({"ok": True, "message": "本地环境可用", "ports": {}})

    port_results = {}
    all_ok = True
    for pname, pcfg in ports.items():
        if not isinstance(pcfg, dict):
            continue
        host = pcfg.get("host", "")
        port = int(pcfg.get("port", 0))
        if not host or not port:
            port_results[pname] = {"ok": True, "message": "无需连接"}
            continue
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.close()
            port_results[pname] = {"ok": True, "message": f"{host}:{port} 可达"}
        except OSError as exc:
            port_results[pname] = {"ok": False, "message": f"{host}:{port} 不可达: {exc}"}
            all_ok = False

    summary = "全部端口可达" if all_ok else "部分端口不可达"
    return jsonify({"ok": all_ok, "message": summary, "ports": port_results})


# ---------------------------------------------------------------------------
# Platform config API
# ---------------------------------------------------------------------------

@bp.route("/api/targets/pools", methods=["GET"])
def list_pools():
    """Return pool summary: which pools exist, how many targets each has."""
    data = _load_raw()
    raw_targets = data.get("targets", {})
    pools: dict[str, dict] = {}
    for name, t in raw_targets.items():
        if not isinstance(t, dict):
            continue
        pool_name = t.get("pool", "")
        platform = t.get("platform", "")
        if pool_name:
            if pool_name not in pools:
                pools[pool_name] = {"pool": pool_name, "platform": platform,
                                    "total": 0, "targets": []}
            pools[pool_name]["total"] += 1
            pools[pool_name]["targets"].append(name)
    return jsonify(list(pools.values()))


@bp.route("/api/platforms", methods=["GET"])
def list_platform_configs():
    """Return platform lifecycle configurations."""
    data = _load_raw()
    raw = data.get("platforms", {})
    result = {}
    for name in PLATFORMS:
        pcfg = raw.get(name, {})
        if not isinstance(pcfg, dict):
            pcfg = {}
        result[name] = {
            "name": name,
            "label": pcfg.get("label", name),
            "steps": pcfg.get("steps", []),
            "env": pcfg.get("env", {}),
            "timeout": pcfg.get("timeout", 300),
        }
    return jsonify(result)


@bp.route("/api/platforms/<name>", methods=["PUT"])
def update_platform(name):
    """Update a platform lifecycle configuration."""
    if name not in PLATFORMS:
        return jsonify({"error": f"未知平台: {name}"}), 400

    body = request.get_json(silent=True) or {}
    data = _load_raw()
    platforms = data.setdefault("platforms", {})
    platforms[name] = {
        "label": body.get("label", name),
        "steps": body.get("steps", []),
        "env": body.get("env", {}),
        "timeout": int(body.get("timeout", 300)),
    }
    _save_raw(data)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# helpers (body parsing)
# ---------------------------------------------------------------------------

def _body_to_raw(body: dict) -> dict:
    """Convert JSON request body to targets.yaml format."""
    entry: dict = {}

    if body.get("platform"):
        entry["platform"] = body["platform"]
    if body.get("pool"):
        entry["pool"] = body["pool"]

    # Multi-port model
    ports = body.get("ports")
    if ports and isinstance(ports, dict):
        clean_ports = {}
        for pname, pcfg in ports.items():
            if not isinstance(pcfg, dict):
                continue
            port_entry: dict = {"type": pcfg.get("type", "ssh")}
            if pcfg.get("host"):
                port_entry["host"] = pcfg["host"]
            if pcfg.get("port"):
                port_entry["port"] = int(pcfg["port"])
            if pcfg.get("user"):
                port_entry["user"] = pcfg["user"]
            if pcfg.get("auth"):
                port_entry["auth"] = pcfg["auth"]
            if pcfg.get("baudrate"):
                port_entry["baudrate"] = int(pcfg["baudrate"])
            if pcfg.get("device"):
                port_entry["device"] = pcfg["device"]
            clean_ports[pname] = port_entry
        if clean_ports:
            entry["ports"] = clean_ports

    if body.get("remote_dir"):
        entry["remote_dir"] = body["remote_dir"]
    if body.get("build_dir"):
        entry["build_dir"] = body["build_dir"]
    if body.get("env"):
        entry["env"] = body["env"]

    return entry


# ---------------------------------------------------------------------------
# Export / Sync
# ---------------------------------------------------------------------------

@bp.route("/api/targets/export", methods=["GET"])
def export_targets():
    """Export raw targets.yaml content (used by client sync)."""
    p = _targets_path()
    if not p.is_file():
        return Response("", status=404)
    return Response(p.read_bytes(), mimetype="text/yaml")


@bp.route("/api/targets/sync", methods=["POST"])
def sync_targets_from_server():
    """Client-mode: pull targets.yaml from the remote server and reload."""
    import urllib.error
    from urllib.request import Request, urlopen

    aitf_cfg = current_app.config.get("AITF_CONFIG")
    if not aitf_cfg or not aitf_cfg.server_url:
        return jsonify({"error": "仅在客户端/调试模式下可用"}), 400

    export_url = f"{aitf_cfg.server_url}/api/targets/export"
    try:
        resp = urlopen(Request(export_url), timeout=30)
        raw_yaml = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return jsonify({"error": "服务器上尚无 targets.yaml"})
        return jsonify({"error": f"从服务器获取失败: HTTP {exc.code}"})
    except Exception as exc:
        return jsonify({"error": f"从服务器获取失败: {exc}"})

    p = _targets_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(raw_yaml)

    data = yaml.safe_load(raw_yaml) or {}
    total = len(data.get("targets", {}))
    log_activity("target.sync", f"从服务器同步 {total} 个环境")
    return jsonify({"ok": True, "total": total})
