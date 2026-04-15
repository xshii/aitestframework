"""Data classes and exceptions for the deps module."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class BundleStatus(str, Enum):
    """Lifecycle status of a configuration bundle."""

    VERIFIED = "verified"
    TESTING = "testing"
    DEPRECATED = "deprecated"


# ---------------------------------------------------------------------------
# Configuration data classes
# ---------------------------------------------------------------------------

@dataclass
class ArtifactToolAcquire:
    """Acquire a toolchain via the external artifact tool (see bootstrap.py).

    The presence of an ``acquire.artifact_tool:`` block triggers this code
    path. *extract* is the only framework-consumed key (controls whether
    system tar / unzip / unrar runs after download). Every other yaml key
    under ``artifact_tool:`` is collected into *placeholders* and made
    available as ``{name}`` substitutions inside
    :data:`aitf.deps.bootstrap.ARTIFACT_TOOL_FLAGS` — adding a new
    placeholder requires no Python changes.

    Two placeholders are auto-injected by the framework and cannot be
    overridden from yaml: ``{version}`` (from ``toolchains.<name>.version``)
    and ``{install_dir}`` (computed at runtime).
    """

    extract: bool = False
    placeholders: dict[str, str] = field(default_factory=dict)


@dataclass
class AcquireConfig:
    """How to obtain a toolchain archive.

    *local_dir* / *srcpkg* cover the pre-staged-archive case. *script*
    wraps an arbitrary transfer (bash script). *artifact_tool* bypasses
    the archive pipeline entirely (see bootstrap.py).

    *install_dir* is the unified target directory: archives extract
    here, scripts receive it as a positional arg, and the artifact tool
    downloads/extracts here. When unset, the framework falls back to
    ``<cache_dir>/<name>``.
    """

    local_dir: str | None = None
    srcpkg: str | None = None
    script: str | None = None
    install_dir: str | None = None
    artifact_tool: ArtifactToolAcquire | None = None


def _resolve(override: str | None, name: str,
             default_base: Path, build_dir: Path) -> Path:
    """Pick the concrete install directory for a dep.

    ``override`` (yaml ``install_dir``) wins when set — absolute kept,
    relative joined to *build_dir*. Fallback is ``default_base / name``.
    """
    if override:
        p = Path(override)
        return p if p.is_absolute() else build_dir / p
    return default_base / name


@dataclass
class ToolchainConfig:
    """Toolchain entry in deps.yaml."""

    name: str
    version: str
    acquire: AcquireConfig = field(default_factory=AcquireConfig)
    order: float = 0.0

    def install_path(self, default_base: Path, build_dir: Path) -> Path:
        return _resolve(self.acquire.install_dir, self.name, default_base, build_dir)


@dataclass
class RepoConfig:
    """Git repository dependency entry in deps.yaml.

    Ref is a three-way exclusive choice: *branch* / *tag* / *commitno*.
    Resolution priority is ``commitno > tag > branch``.
    """

    name: str
    url: str
    branch: str | None = None
    tag: str | None = None
    commitno: str | None = None
    depth: int | None = None
    sparse_checkout: list[str] = field(default_factory=list)
    build_script: str | None = None
    install_dir: str | None = None
    order: float = 0.0

    @property
    def resolved_ref(self) -> str:
        """Return the effective ref following commitno > tag > branch."""
        return self.commitno or self.tag or self.branch or "main"

    @property
    def is_commit(self) -> bool:
        return self.commitno is not None

    def install_path(self, default_base: Path, build_dir: Path) -> Path:
        return _resolve(self.install_dir, self.name, default_base, build_dir)


@dataclass
class BundleConfig:
    """A named set of toolchain + repo versions."""

    name: str
    description: str = ""
    status: str = "testing"
    toolchains: dict[str, str] = field(default_factory=dict)
    repos: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class DiagResult:
    """Single diagnostic check result."""

    check: str
    ok: bool
    message: str


@dataclass
class LockEntry:
    """One entry in the lock file."""

    name: str
    version: str = ""
    ref: str = ""
    commit: str = ""
    installed_at: str = ""


@dataclass
class LockFile:
    """Content of deps.lock.yaml."""

    generated_at: str = ""
    platform: str = ""
    toolchains: dict[str, LockEntry] = field(default_factory=dict)
    repos: dict[str, LockEntry] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DepsError(Exception):
    """Base exception for dependency operations."""


class DepsConfigError(DepsError):
    """Raised when deps.yaml is invalid or missing."""


class AcquireError(DepsError):
    """Raised when dependency acquisition fails."""


class BundleError(DepsError):
    """Raised for bundle-related errors."""


class BundleNotFoundError(BundleError, KeyError):
    """Raised when a bundle name does not exist."""


class RepoError(DepsError):
    """Raised when a git operation fails."""
