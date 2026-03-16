"""Shared data types and helpers for the execution framework."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass
class TargetConfig:
    """Describes a local or remote execution target."""

    name: str
    type: str  # "local" or "remote"
    host: str | None = None
    port: int = 22
    user: str | None = None
    auth: dict | None = None  # {"method": "key", "key_file": "..."} or {"method": "password"}
    remote_dir: str | None = None
    build_dir: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    pre_commands: list[str] = field(default_factory=list)
    post_commands: list[str] = field(default_factory=list)
    upload_extra: list[str] = field(default_factory=list)


@dataclass
class ExecuteResult:
    """Result of running a command on a target."""

    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    output_path: str | None = None


# ------------------------------------------------------------------
# Loader
# ------------------------------------------------------------------

def load_targets(path: str | Path) -> dict[str, TargetConfig]:
    """Load target definitions from a YAML file.

    Expected format::

        targets:
          local:
            type: local
            build_dir: build/
            env:
              LD_LIBRARY_PATH: "{build_dir}/lib"
          sim-server:
            type: remote
            host: 192.168.1.100
            port: 22
            user: test
            auth:
              method: key
              key_file: ~/.ssh/id_rsa
            remote_dir: /tmp/aitf
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Targets file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict) or "targets" not in raw:
        raise ValueError(f"Invalid targets file: expected top-level 'targets' key in {path}")

    configs: dict[str, TargetConfig] = {}
    for name, attrs in raw["targets"].items():
        if not isinstance(attrs, dict):
            logger.warning("Skipping target %r: expected a mapping", name)
            continue

        # Resolve simple {build_dir} placeholders inside env values.
        build_dir = attrs.get("build_dir", "")
        env = {}
        for k, v in attrs.get("env", {}).items():
            env[k] = str(v).replace("{build_dir}", build_dir)

        configs[name] = TargetConfig(
            name=name,
            type=attrs.get("type", "local"),
            host=attrs.get("host"),
            port=int(attrs.get("port", 22)),
            user=attrs.get("user"),
            auth=attrs.get("auth"),
            remote_dir=attrs.get("remote_dir"),
            build_dir=build_dir or None,
            env=env,
            pre_commands=list(attrs.get("pre_commands", [])),
            post_commands=list(attrs.get("post_commands", [])),
            upload_extra=list(attrs.get("upload_extra", [])),
        )

    logger.info("Loaded %d target(s) from %s", len(configs), path)
    return configs
