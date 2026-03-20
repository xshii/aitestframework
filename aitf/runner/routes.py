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

_SCRIPT_SUFFIXES = {".sh", ".bash", ".py"}
from aitf.web.activity import log_activity

bp = Blueprint("runner", __name__, template_folder="templates")

logger = logging.getLogger(__name__)

PLATFORMS = ["pc_func", "pc_perf", "fpga", "emu", "eda"]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _targets_path() -> Path:
    """Return the resolved path to targets.yaml.

    Preferred location: data/config/targets.yaml (via AitfConfig).
    Falls back to legacy locations for backward compatibility.
    """
    cfg = current_app.config.get("AITF_CONFIG")
    if cfg:
        primary = cfg.targets_file
        if primary.is_file():
            return primary
    # Legacy fallback
    from aitf.web.extensions import get_project_root
    root = get_project_root()
    for candidate in [root / "runner" / "targets.yaml",
                      root / "targets.yaml"]:
        if candidate.is_file():
            return candidate
    # Default: new location
    if cfg:
        cfg.targets_file.parent.mkdir(parents=True, exist_ok=True)
        return cfg.targets_file
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


@bp.route("/api/targets/health", methods=["GET"])
def all_targets_health():
    """Quick health check for all targets (non-local only)."""
    p = _targets_path()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.is_file() else {}
    results = {}
    for name, t in data.items():
        ports = t.get("ports", {})
        if not ports:
            results[name] = "local"
            continue
        all_ok = True
        for pname, pcfg in ports.items():
            if not isinstance(pcfg, dict):
                continue
            host = pcfg.get("host", "")
            port = int(pcfg.get("port", 0))
            if not host or not port:
                continue
            try:
                sock = socket.create_connection((host, port), timeout=3)
                sock.close()
            except OSError:
                all_ok = False
                break
        results[name] = "ok" if all_ok else "unreachable"
    return jsonify(results)


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
# Schedule policies (idle auto-run)
# ---------------------------------------------------------------------------

def _proxy_to_server(path: str):
    """If client mode, proxy GET/POST to server. Returns response or None."""
    from aitf.web.proxy import proxy_if_client
    return proxy_if_client(path)


@bp.route("/api/schedules", methods=["GET"])
def list_schedules():
    """Return all schedule rules. Client mode → proxy to server."""
    proxied = _proxy_to_server("/api/schedules")
    if proxied:
        return proxied

    from aitf.runner.schedule import load_schedules
    cfg = current_app.config.get("AITF_CONFIG")
    root = cfg.project_root if cfg else "."
    rules = load_schedules(root)
    return jsonify([r.to_dict() for r in rules])


@bp.route("/api/schedules", methods=["POST"])
def save_schedules_api():
    """Save all schedule rules. Client mode → proxy to server."""
    proxied = _proxy_to_server("/api/schedules")
    if proxied:
        return proxied

    from aitf.runner.schedule import ScheduleRule, save_schedules

    body = request.get_json(silent=True)
    if not isinstance(body, list):
        return jsonify({"error": "expected a JSON array"}), 400

    cfg = current_app.config.get("AITF_CONFIG")
    root = cfg.project_root if cfg else "."

    _fields = ScheduleRule.__dataclass_fields__
    rules = [
        ScheduleRule(**{k: item[k] for k in _fields if k in item})
        for item in body
    ]
    save_schedules(root, rules)
    log_activity("schedule.save", f"保存 {len(rules)} 条调度策略")
    return jsonify({"ok": True, "count": len(rules)})


@bp.route("/api/schedules/options", methods=["GET"])
def schedule_options():
    """Return available targets, pools, and test plans. Client → proxy."""
    proxied = _proxy_to_server("/api/schedules/options")
    if proxied:
        return proxied

    cfg = current_app.config.get("AITF_CONFIG")
    root = cfg.project_root if cfg else Path(".")

    # Targets + pools
    p = _targets_path()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.is_file() else {}
    targets = list(data.keys())
    pools: dict[str, int] = {}
    for name, tcfg in data.items():
        pool = tcfg.get("pool", "")
        if pool:
            pools[pool] = pools.get(pool, 0) + 1

    # Test plans
    plans = []
    for d in [root / "testplans", root]:
        if not isinstance(d, Path):
            d = Path(d)
        for f in sorted(d.glob("*.yaml")) if d.is_dir() else []:
            if f.stem.startswith("test") or "plan" in f.stem:
                plans.append(f.name)

    return jsonify({
        "targets": targets,
        "pools": [{"name": k, "count": v} for k, v in pools.items()],
        "plans": plans,
    })


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


# ---------------------------------------------------------------------------
# Restart scripts — per-port upload & one-click restart
# ---------------------------------------------------------------------------

def _runner_subdir(name: str) -> Path:
    """Return (and create) a subdirectory under runner/."""
    cfg = current_app.config.get("AITF_CONFIG")
    root = cfg.project_root if cfg else Path(".")
    d = root / "runner" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _scripts_dir() -> Path:
    return _runner_subdir("scripts")


def _files_dir() -> Path:
    return _runner_subdir("files")


@bp.route("/api/targets/<name>/files", methods=["GET"])
def list_target_files(name):
    """List uploaded files for a target (scripts, firmware, etc.)."""
    d = _files_dir() / name
    if not d.is_dir():
        return jsonify([])
    files = []
    for p in sorted(d.iterdir()):
        if p.is_file():
            files.append({"name": p.name, "size": p.stat().st_size,
                          "is_script": p.suffix in _SCRIPT_SUFFIXES})
    return jsonify(files)


@bp.route("/api/targets/<name>/files", methods=["POST"])
def upload_target_file(name):
    """Upload a file (restart script, firmware, etc.) for a target.

    Form fields: file (required), category (optional: restart/firmware/other)
    """
    from werkzeug.utils import secure_filename as sf

    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "file is required"}), 400

    d = _files_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    safe = sf(f.filename)
    if not safe:
        return jsonify({"error": "invalid filename"}), 400
    dest = d / safe
    f.save(dest)
    # Make scripts executable
    if dest.suffix in _SCRIPT_SUFFIXES:
        dest.chmod(0o755)
    log_activity("file.upload", f"{name}/{safe}")
    return jsonify({"ok": True, "name": safe, "size": dest.stat().st_size})


@bp.route("/api/targets/<name>/files/<filename>", methods=["DELETE"])
def delete_target_file(name, filename):
    """Delete an uploaded file."""
    from werkzeug.utils import secure_filename as sf
    safe = sf(filename)
    p = (_files_dir() / name / safe).resolve()
    if not p.is_relative_to(_files_dir()) or not p.is_file():
        return jsonify({"error": "not found"}), 404
    p.unlink()
    log_activity("file.delete", f"{name}/{safe}")
    return jsonify({"ok": True})


@bp.route("/api/targets/<name>/scripts/<port_name>", methods=["POST"])
def upload_restart_script(name, port_name):
    """Upload a restart script for a specific port.

    Saves as ``scripts/<name>_<port>_restart.sh`` with executable permission.
    """
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "file is required"}), 400

    dest = _scripts_dir() / f"{name}_{port_name}_restart.sh"
    f.save(dest)
    dest.chmod(0o755)
    log_activity("script.upload", f"{name}/{port_name}")
    return jsonify({"ok": True, "name": dest.name})


@bp.route("/api/targets/<name>/restart", methods=["POST"])
def restart_target(name):
    """Execute startup/restart pipeline for a target.

    Uses the declarative ``startup.steps`` from targets.yaml if configured,
    otherwise falls back to legacy per-port restart scripts.
    """
    from aitf.runner.startup import StartupEngine, parse_startup_config

    p = _targets_path()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.is_file() else {}
    target_cfg = data.get(name)
    if target_cfg is None:
        return jsonify({"error": f"target '{name}' not found"}), 404

    cfg = current_app.config.get("AITF_CONFIG")
    root = Path(cfg.project_root) if cfg else Path(".")

    # Try declarative startup pipeline first
    startup_cfg = parse_startup_config(target_cfg)
    if startup_cfg and startup_cfg.steps:

        # Get DepsManager if deps are configured
        deps_mgr = None
        if startup_cfg.deps:
            try:
                from aitf.web.extensions import get_deps_manager
                deps_mgr = get_deps_manager()
            except Exception:
                pass

        engine = StartupEngine(name, target_cfg, root, deps_mgr)
        results = engine.run(startup_cfg)
        all_ok = all(r.success for r in results)
        log_activity("restart", f"{name} {'✓' if all_ok else '✗'} ({len(results)} steps)")
        return jsonify({
            "ok": all_ok,
            "results": [r.to_dict() for r in results],
        })

    # Fallback: convert legacy per-port restart scripts into steps
    from aitf.runner.startup import StartupConfig, StartupStep

    body = request.get_json(silent=True) or {}
    port_filter = body.get("port", "")
    prefix = f"{name}_"
    scripts = sorted(_scripts_dir().glob(f"{prefix}*_restart.sh"))
    if port_filter:
        scripts = [s for s in scripts if f"{name}_{port_filter}_restart" in s.name]
    if not scripts:
        return jsonify({"error": "没有配置 startup 步骤，也没有找到重启脚本"}), 404

    # Build steps from legacy scripts
    steps = [
        StartupStep(
            name=s.name[len(prefix):].rsplit("_restart", 1)[0],
            type="script",
            script=str(s.relative_to(root)),
        )
        for s in scripts
    ]
    engine = StartupEngine(name, target_cfg, root)
    results = engine.run(StartupConfig(steps=steps))
    all_ok = all(r.success for r in results)
    log_activity("restart", f"{name} {'✓' if all_ok else '✗'}")
    return jsonify({"ok": all_ok, "results": [r.to_dict() for r in results]})
