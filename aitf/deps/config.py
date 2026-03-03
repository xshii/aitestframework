"""Parse and validate deps.yaml configuration."""

from __future__ import annotations

import logging
import platform as _platform
from pathlib import Path

import yaml

from aitf.deps.types import (
    AcquireConfig,
    BundleConfig,
    DepsConfigError,
    LibraryConfig,
    RepoConfig,
    ToolchainConfig,
)

logger = logging.getLogger(__name__)

DEFAULT_DEPS_FILE = "deps.yaml"


def detect_platform() -> str:
    return f"{_platform.system().lower()}-{_platform.machine()}"


def strip_none(d: dict) -> dict:
    """Return a copy of *d* with all ``None``-valued keys removed."""
    return {k: v for k, v in d.items() if v is not None}


def _parse_acquire(raw: dict) -> AcquireConfig:
    acq = raw.get("acquire") or {}
    return AcquireConfig(
        local_dir=acq.get("local_dir"),
        script=acq.get("script"),
    )


def _serialize_acquire(acq: AcquireConfig) -> dict:
    return strip_none({
        "local_dir": acq.local_dir,
        "script": acq.script,
    })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DepsConfig:
    """Parsed representation of ``deps.yaml``."""

    def __init__(self) -> None:
        self.server: str | None = None
        self.toolchains: dict[str, ToolchainConfig] = {}
        self.libraries: dict[str, LibraryConfig] = {}
        self.repos: dict[str, RepoConfig] = {}
        self.bundles: dict[str, BundleConfig] = {}
        self.active_bundle: str | None = None


def load_deps_config(path: str | Path = DEFAULT_DEPS_FILE) -> DepsConfig:
    p = Path(path)
    if not p.exists():
        raise DepsConfigError(f"Configuration file not found: {p}")

    try:
        with open(p, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise DepsConfigError(f"Failed to parse {p}: {exc}") from exc

    if not isinstance(data, dict):
        raise DepsConfigError(f"Expected a YAML mapping in {p}, got {type(data).__name__}")

    cfg = DepsConfig()

    for name, raw in data.get("toolchains", {}).items():
        cfg.toolchains[name] = ToolchainConfig(
            name=name, version=str(raw.get("version", "")),
            sha256=raw.get("sha256", {}), bin_dir=raw.get("bin_dir"),
            env=raw.get("env", {}), acquire=_parse_acquire(raw),
            install_dir=raw.get("install_dir"),
            order=int(raw.get("order", 0)),
        )

    for name, raw in data.get("libraries", {}).items():
        cfg.libraries[name] = LibraryConfig(
            name=name, version=str(raw.get("version", "")),
            sha256=str(raw.get("sha256", "")),
            build_system=raw.get("build_system", "cmake"),
            cmake_args=raw.get("cmake_args", []),
            build_script=raw.get("build_script"), acquire=_parse_acquire(raw),
            install_dir=raw.get("install_dir"),
            order=int(raw.get("order", 0)),
        )

    for name, raw in data.get("repos", {}).items():
        cfg.repos[name] = RepoConfig(
            name=name, url=raw.get("url", ""), ref=str(raw.get("ref", "main")),
            depth=raw.get("depth"), sparse_checkout=raw.get("sparse_checkout", []),
            build_script=raw.get("build_script"), env=raw.get("env", {}),
            install_dir=raw.get("install_dir"),
            order=int(raw.get("order", 0)),
        )

    for name, raw in data.get("bundles", {}).items():
        cfg.bundles[name] = BundleConfig(
            name=name, description=raw.get("description", ""),
            status=raw.get("status", "testing"),
            toolchains=raw.get("toolchains", {}), libraries=raw.get("libraries", {}),
            repos=raw.get("repos", {}), env=raw.get("env", {}),
        )

    cfg.active_bundle = data.get("active")
    cfg.server = data.get("server")
    if cfg.server:
        logger.warning(
            "'server' field in deps.yaml is deprecated — "
            "please move it to config.yaml instead."
        )
    return cfg


def save_deps_config(cfg: DepsConfig, path: str | Path = DEFAULT_DEPS_FILE) -> None:
    data: dict = {}

    if cfg.toolchains:
        data["toolchains"] = {
            name: strip_none({
                "version": tc.version, "sha256": tc.sha256, "bin_dir": tc.bin_dir,
                "env": tc.env, "acquire": _serialize_acquire(tc.acquire),
                "install_dir": tc.install_dir,
                "order": tc.order or None,
            })
            for name, tc in cfg.toolchains.items()
        }
    if cfg.libraries:
        data["libraries"] = {
            name: strip_none({
                "version": lib.version, "sha256": lib.sha256,
                "build_system": lib.build_system, "cmake_args": lib.cmake_args,
                "build_script": lib.build_script,
                "acquire": _serialize_acquire(lib.acquire),
                "install_dir": lib.install_dir,
                "order": lib.order or None,
            })
            for name, lib in cfg.libraries.items()
        }
    if cfg.repos:
        data["repos"] = {
            name: strip_none({
                "url": r.url, "ref": r.ref, "depth": r.depth,
                "sparse_checkout": r.sparse_checkout, "build_script": r.build_script,
                "env": r.env,
                "install_dir": r.install_dir,
                "order": r.order or None,
            })
            for name, r in cfg.repos.items()
        }
    if cfg.bundles:
        data["bundles"] = {
            name: strip_none({
                "description": b.description, "status": b.status,
                "toolchains": b.toolchains, "libraries": b.libraries,
                "repos": b.repos, "env": b.env,
            })
            for name, b in cfg.bundles.items()
        }
    if cfg.active_bundle:
        data["active"] = cfg.active_bundle
    # server field is deprecated in deps.yaml — no longer written out

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
