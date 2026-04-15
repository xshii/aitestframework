"""Parse and validate deps.yaml configuration."""

from __future__ import annotations

import logging
import platform as _platform
from pathlib import Path

import yaml

from aitf.deps.types import (
    AcquireConfig,
    ArtifactToolAcquire,
    BundleConfig,
    DepsConfigError,
    RepoConfig,
    ToolchainConfig,
)

logger = logging.getLogger(__name__)

DEFAULT_DEPS_FILE = "deps.yaml"


def detect_platform() -> str:
    return f"{_platform.system().lower()}-{_platform.machine()}"


def strip_none(d: dict) -> dict:
    """Return a copy of *d* with ``None`` / empty-container values removed."""
    return {k: v for k, v in d.items() if v not in (None, "", {}, [])}


#: Yaml keys under ``acquire.artifact_tool:`` that the framework consumes
#: directly. Everything else becomes a free-form placeholder.
_ARTIFACT_TOOL_RESERVED = frozenset({"extract"})


def _parse_acquire(raw: dict) -> AcquireConfig:
    acq = raw.get("acquire") or {}
    at_raw = acq.get("artifact_tool")
    artifact_tool = None
    if at_raw is not None:  # presence (even {}) is the trigger
        if not isinstance(at_raw, dict):
            raise DepsConfigError("acquire.artifact_tool must be a mapping")
        artifact_tool = ArtifactToolAcquire(
            extract=bool(at_raw.get("extract", False)),
            placeholders={
                k: str(v) for k, v in at_raw.items()
                if k not in _ARTIFACT_TOOL_RESERVED
            },
        )
    return AcquireConfig(
        local_dir=acq.get("local_dir"),
        srcpkg=acq.get("srcpkg"),
        script=acq.get("script"),
        install_dir=acq.get("install_dir") or None,
        artifact_tool=artifact_tool,
    )


def _serialize_acquire(acq: AcquireConfig) -> dict:
    artifact_tool = None
    if acq.artifact_tool is not None:
        artifact_tool = strip_none({
            "extract": acq.artifact_tool.extract or None,
            **acq.artifact_tool.placeholders,
        })
    return strip_none({
        "local_dir": acq.local_dir,
        "srcpkg": acq.srcpkg,
        "script": acq.script,
        "install_dir": acq.install_dir,
        "artifact_tool": artifact_tool,
    })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DepsConfig:
    """Parsed representation of ``deps.yaml``."""

    def __init__(self) -> None:
        self.server: str | None = None
        self.toolchains: dict[str, ToolchainConfig] = {}
        self.repos: dict[str, RepoConfig] = {}
        self.bundles: dict[str, BundleConfig] = {}
        self.active_bundle: str | None = None


def _parse_order(raw) -> float:
    try:
        return float(raw or 0)
    except (TypeError, ValueError):
        return 0.0


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

    for name, raw in (data.get("toolchains") or {}).items():
        cfg.toolchains[name] = ToolchainConfig(
            name=name,
            version=str(raw.get("version", "")),
            acquire=_parse_acquire(raw),
            order=_parse_order(raw.get("order")),
        )

    for name, raw in (data.get("repos") or {}).items():
        branch = raw.get("branch")
        tag = raw.get("tag")
        commitno = raw.get("commitno")
        if not any((branch, tag, commitno)):
            raise DepsConfigError(
                f"repos.{name}: one of branch / tag / commitno must be set"
            )
        cfg.repos[name] = RepoConfig(
            name=name,
            url=raw.get("url", ""),
            branch=branch,
            tag=tag,
            commitno=commitno,
            depth=raw.get("depth"),
            sparse_checkout=raw.get("sparse_checkout", []),
            build_script=raw.get("build_script"),
            install_dir=raw.get("install_dir") or None,
            order=_parse_order(raw.get("order")),
        )

    for name, raw in (data.get("bundles") or {}).items():
        cfg.bundles[name] = BundleConfig(
            name=name,
            description=raw.get("description", ""),
            status=raw.get("status", "testing"),
            toolchains=raw.get("toolchains", {}),
            repos=raw.get("repos", {}),
        )

    cfg.active_bundle = data.get("active")
    cfg.server = data.get("server")
    if cfg.server:
        logger.warning(
            "'server' field in deps.yaml is deprecated — "
            "please move it to config.yaml instead."
        )
    return cfg


def _serialize_repo(r: RepoConfig) -> dict:
    return strip_none({
        "url": r.url,
        "branch": r.branch,
        "tag": r.tag,
        "commitno": r.commitno,
        "depth": r.depth,
        "sparse_checkout": r.sparse_checkout,
        "build_script": r.build_script,
        "install_dir": r.install_dir,
        "order": r.order or None,
    })


def save_deps_config(cfg: DepsConfig, path: str | Path = DEFAULT_DEPS_FILE) -> None:
    data: dict = {}

    if cfg.toolchains:
        data["toolchains"] = {
            name: strip_none({
                "version": tc.version,
                "acquire": _serialize_acquire(tc.acquire),
                "order": tc.order or None,
            })
            for name, tc in cfg.toolchains.items()
        }
    if cfg.repos:
        data["repos"] = {name: _serialize_repo(r) for name, r in cfg.repos.items()}
    if cfg.bundles:
        data["bundles"] = {
            name: strip_none({
                "description": b.description,
                "status": b.status,
                "toolchains": b.toolchains,
                "repos": b.repos,
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
