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
class AcquireConfig:
    """How to obtain a toolchain or library archive."""

    local_dir: str | None = None
    script: str | None = None


@dataclass
class ToolchainConfig:
    """Toolchain entry in deps.yaml."""

    name: str
    version: str
    sha256: dict[str, str] = field(default_factory=dict)
    bin_dir: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    acquire: AcquireConfig = field(default_factory=AcquireConfig)
    install_dir: str | None = None
    order: int = 0


@dataclass
class LibraryConfig:
    """Third-party C/C++ library entry in deps.yaml."""

    name: str
    version: str
    sha256: str = ""
    build_system: str = "cmake"
    cmake_args: list[str] = field(default_factory=list)
    build_script: str | None = None
    acquire: AcquireConfig = field(default_factory=AcquireConfig)
    install_dir: str | None = None
    order: int = 0


@dataclass
class RepoConfig:
    """Git repository dependency entry in deps.yaml."""

    name: str
    url: str
    ref: str = "main"
    depth: int | None = None
    sparse_checkout: list[str] = field(default_factory=list)
    build_script: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    install_dir: str | None = None
    order: int = 0


@dataclass
class BundleConfig:
    """A named set of toolchain + library + repo versions."""

    name: str
    description: str = ""
    status: str = "testing"
    toolchains: dict[str, str] = field(default_factory=dict)
    libraries: dict[str, str] = field(default_factory=dict)
    repos: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


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
    sha256: str = ""
    ref: str = ""
    commit: str = ""
    installed_at: str = ""


@dataclass
class LockFile:
    """Content of deps.lock.yaml."""

    generated_at: str = ""
    platform: str = ""
    toolchains: dict[str, LockEntry] = field(default_factory=dict)
    libraries: dict[str, LockEntry] = field(default_factory=dict)
    repos: dict[str, LockEntry] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

def resolve_dep_dir(dep, default_base: Path, build_dir: Path) -> Path:
    """Resolve a dependency's install directory.

    If *dep.install_dir* is set, resolve it (relative to *build_dir*).
    Otherwise fall back to ``default_base / dep.name``.
    """
    if dep.install_dir:
        p = Path(dep.install_dir)
        return p if p.is_absolute() else build_dir / p
    return default_base / dep.name


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
