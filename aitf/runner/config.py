"""Platform and target configuration loader.

Configuration format (targets.yaml)::

    targets:
      local:
        platform: pc_func

      fpga-board-1:
        platform: fpga
        pool: fpga-A
        remote_dir: /opt/aitf_run
        ports:
          ctrl:
            host: 192.168.1.10
            port: 22
            type: ssh
            user: root
          jtag:
            host: 192.168.1.10
            port: 3333
            type: tcp
          console:
            host: 192.168.1.10
            port: 4001
            type: tcp
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PLATFORMS = ["pc_func", "pc_perf", "fpga", "emu", "eda"]


@dataclass
class ExecuteResult:
    """Result of running a command on a target."""
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    output_path: str | None = None

PLATFORM_LABELS = {
    "pc_func": "PC功能测试",
    "pc_perf": "PC性能测试",
    "fpga": "FPGA验证",
    "emu": "仿真验证",
    "eda": "EDA综合仿真",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PortConfig:
    """A single port/interface on a target."""
    name: str               # e.g. "ctrl", "jtag", "console"
    type: str = "ssh"       # ssh | tcp
    host: str = ""
    port: int = 0
    user: str = ""
    auth: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "type": self.type,
            "host": self.host, "port": self.port,
            "user": self.user, "auth": self.auth,
        }


@dataclass
class TargetConfig:
    """A concrete execution target (machine/board)."""
    name: str
    platform: str = ""
    pool: str = ""              # pool name for resource pooling
    ports: dict[str, PortConfig] = field(default_factory=dict)
    remote_dir: str = ""
    build_dir: str = ""
    env: dict[str, str] = field(default_factory=dict)

    @property
    def is_local(self) -> bool:
        return not self.ports

    @property
    def primary_host(self) -> str:
        for p in self.ports.values():
            if p.host:
                return p.host
        return ""

    def to_api_dict(self) -> dict:
        ports_dict = {n: p.to_dict() for n, p in self.ports.items()}
        host_str = ""
        user_str = ""
        for p in self.ports.values():
            if p.host:
                host_str = f"{p.host}:{p.port}"
                user_str = p.user
                break
        return {
            "name": self.name, "platform": self.platform,
            "pool": self.pool, "ports": ports_dict,
            "host": host_str, "user": user_str,
            "remote_dir": self.remote_dir, "build_dir": self.build_dir,
            "env": self.env, "is_local": self.is_local,
        }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_runner_config(path: str | Path) -> dict[str, TargetConfig]:
    """Load targets from a YAML file. Returns targets dict keyed by name."""
    path = Path(path)
    if not path.is_file():
        logger.warning("Runner config not found: %s", path)
        return {}

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    return _parse_targets(raw.get("targets", {}))


def _parse_targets(raw: dict) -> dict[str, TargetConfig]:
    targets: dict[str, TargetConfig] = {}
    for name, attrs in raw.items():
        if not isinstance(attrs, dict):
            continue

        ports: dict[str, PortConfig] = {}
        raw_ports = attrs.get("ports", {})
        if isinstance(raw_ports, dict):
            for pname, pcfg in raw_ports.items():
                if not isinstance(pcfg, dict):
                    continue
                ports[pname] = PortConfig(
                    name=pname,
                    type=pcfg.get("type", "ssh"),
                    host=pcfg.get("host", ""),
                    port=int(pcfg.get("port", 0)),
                    user=pcfg.get("user", ""),
                    auth=pcfg.get("auth", {}),
                )

        # Backward compat: flat host → synthesize ctrl port
        if not ports and attrs.get("host"):
            ports["ctrl"] = PortConfig(
                name="ctrl", type="ssh",
                host=attrs["host"],
                port=int(attrs.get("port", 22)),
                user=attrs.get("user", ""),
                auth=attrs.get("auth", {}),
            )

        targets[name] = TargetConfig(
            name=name,
            platform=attrs.get("platform", ""),
            pool=attrs.get("pool", ""),
            ports=ports,
            remote_dir=attrs.get("remote_dir", ""),
            build_dir=attrs.get("build_dir", ""),
            env=dict(attrs.get("env", {})),
        )

    logger.info("Loaded %d target(s)", len(targets))
    return targets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def expand_vars(template: str, **kwargs) -> str:
    """Replace ``{var_name}`` placeholders with values from kwargs."""
    def _replace(m):
        return str(kwargs.get(m.group(1), m.group(0)))
    return re.sub(r"\{(\w+)\}", _replace, template)


def group_targets_by_platform(
    targets: dict[str, TargetConfig],
) -> dict[str, list[TargetConfig]]:
    groups: dict[str, list[TargetConfig]] = {}
    for t in targets.values():
        if t.platform:
            groups.setdefault(t.platform, []).append(t)
    return groups


def pick_target(
    targets: dict[str, TargetConfig],
    *, platform: str = "", pool: str = "",
) -> list[TargetConfig]:
    """Return matching targets, ordered for failover.

    If *pool* is given, return targets in that pool.
    If *platform* is given, return targets of that platform.
    Pool takes precedence over platform.
    """
    candidates = list(targets.values())
    if pool:
        candidates = [t for t in candidates if t.pool == pool]
    elif platform:
        candidates = [t for t in candidates if t.platform == platform]
    return candidates
