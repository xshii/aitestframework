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
    """Return a platform tag such as ``linux-x86_64``."""
    system = _platform.system().lower()
    machine = _platform.machine()
    return f"{system}-{machine}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_acquire(raw: dict | None) -> AcquireConfig:
    if not raw:
        return AcquireConfig()
    return AcquireConfig(
        local_dir=raw.get("local_dir"),
        script=raw.get("script"),
    )


def _parse_toolchain(name: str, raw: dict) -> ToolchainConfig:
    return ToolchainConfig(
        name=name,
        version=str(raw.get("version", "")),
        sha256=raw.get("sha256", {}),
        bin_dir=raw.get("bin_dir"),
        env=raw.get("env", {}),
        acquire=_parse_acquire(raw.get("acquire")),
    )


def _parse_library(name: str, raw: dict) -> LibraryConfig:
    return LibraryConfig(
        name=name,
        version=str(raw.get("version", "")),
        sha256=str(raw.get("sha256", "")),
        build_system=raw.get("build_system", "cmake"),
        cmake_args=raw.get("cmake_args", []),
        build_script=raw.get("build_script"),
        acquire=_parse_acquire(raw.get("acquire")),
    )


def _parse_repo(name: str, raw: dict) -> RepoConfig:
    return RepoConfig(
        name=name,
        url=raw.get("url", ""),
        ref=str(raw.get("ref", "main")),
        depth=raw.get("depth"),
        sparse_checkout=raw.get("sparse_checkout", []),
        build_script=raw.get("build_script"),
        env=raw.get("env", {}),
    )


def _parse_bundle(name: str, raw: dict) -> BundleConfig:
    return BundleConfig(
        name=name,
        description=raw.get("description", ""),
        status=raw.get("status", "testing"),
        toolchains=raw.get("toolchains", {}),
        libraries=raw.get("libraries", {}),
        repos=raw.get("repos", {}),
        env=raw.get("env", {}),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DepsConfig:
    """Parsed representation of ``deps.yaml``.

    Attributes:
        toolchains: Toolchain definitions keyed by name.
        libraries: Library definitions keyed by name.
        repos: Repository definitions keyed by name.
        bundles: Bundle definitions keyed by name.
        active_bundle: Name of the currently active bundle, or ``None``.
    """

    def __init__(
        self,
        toolchains: dict[str, ToolchainConfig] | None = None,
        libraries: dict[str, LibraryConfig] | None = None,
        repos: dict[str, RepoConfig] | None = None,
        bundles: dict[str, BundleConfig] | None = None,
        active_bundle: str | None = None,
    ) -> None:
        self.toolchains: dict[str, ToolchainConfig] = toolchains or {}
        self.libraries: dict[str, LibraryConfig] = libraries or {}
        self.repos: dict[str, RepoConfig] = repos or {}
        self.bundles: dict[str, BundleConfig] = bundles or {}
        self.active_bundle: str | None = active_bundle

    # -- serialisation helpers -----------------------------------------------

    def to_dict(self) -> dict:
        """Serialise back to a plain dict (for writing deps.yaml)."""
        data: dict = {}
        if self.toolchains:
            data["toolchains"] = {
                name: {
                    "version": tc.version,
                    **({"sha256": tc.sha256} if tc.sha256 else {}),
                    **({"bin_dir": tc.bin_dir} if tc.bin_dir else {}),
                    **({"env": tc.env} if tc.env else {}),
                    **({"acquire": _acquire_dict(tc.acquire)} if _has_acquire(tc.acquire) else {}),
                }
                for name, tc in self.toolchains.items()
            }
        if self.libraries:
            data["libraries"] = {
                name: {
                    "version": lib.version,
                    **({"sha256": lib.sha256} if lib.sha256 else {}),
                    "build_system": lib.build_system,
                    **({"cmake_args": lib.cmake_args} if lib.cmake_args else {}),
                    **({"build_script": lib.build_script} if lib.build_script else {}),
                    **({"acquire": _acquire_dict(lib.acquire)} if _has_acquire(lib.acquire) else {}),
                }
                for name, lib in self.libraries.items()
            }
        if self.repos:
            data["repos"] = {
                name: {
                    "url": repo.url,
                    "ref": repo.ref,
                    **({"depth": repo.depth} if repo.depth else {}),
                    **({"sparse_checkout": repo.sparse_checkout} if repo.sparse_checkout else {}),
                    **({"build_script": repo.build_script} if repo.build_script else {}),
                    **({"env": repo.env} if repo.env else {}),
                }
                for name, repo in self.repos.items()
            }
        if self.bundles:
            data["bundles"] = {
                name: {
                    "description": b.description,
                    "status": b.status,
                    **({"toolchains": b.toolchains} if b.toolchains else {}),
                    **({"libraries": b.libraries} if b.libraries else {}),
                    **({"repos": b.repos} if b.repos else {}),
                    **({"env": b.env} if b.env else {}),
                }
                for name, b in self.bundles.items()
            }
        if self.active_bundle:
            data["active"] = self.active_bundle
        return data


def _has_acquire(acq: AcquireConfig) -> bool:
    return bool(acq.local_dir or acq.script)


def _acquire_dict(acq: AcquireConfig) -> dict:
    d: dict = {}
    if acq.local_dir:
        d["local_dir"] = acq.local_dir
    if acq.script:
        d["script"] = acq.script
    return d


def load_deps_config(path: str | Path = DEFAULT_DEPS_FILE) -> DepsConfig:
    """Load and parse ``deps.yaml``.

    Args:
        path: Path to the deps configuration file.

    Returns:
        Parsed :class:`DepsConfig`.

    Raises:
        DepsConfigError: If the file is missing or malformed.
    """
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

    toolchains = {
        name: _parse_toolchain(name, raw)
        for name, raw in data.get("toolchains", {}).items()
    }
    libraries = {
        name: _parse_library(name, raw)
        for name, raw in data.get("libraries", {}).items()
    }
    repos = {
        name: _parse_repo(name, raw)
        for name, raw in data.get("repos", {}).items()
    }
    bundles = {
        name: _parse_bundle(name, raw)
        for name, raw in data.get("bundles", {}).items()
    }
    active = data.get("active")

    return DepsConfig(
        toolchains=toolchains,
        libraries=libraries,
        repos=repos,
        bundles=bundles,
        active_bundle=active,
    )


def save_deps_config(cfg: DepsConfig, path: str | Path = DEFAULT_DEPS_FILE) -> None:
    """Write the config back to ``deps.yaml``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        yaml.dump(cfg.to_dict(), fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
