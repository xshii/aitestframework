"""Data classes and exceptions for the deps module."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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
    remote: bool = False
    script: str | None = None


@dataclass
class RemoteDepotConfig:
    """SFTP remote server that hosts dependency archives."""

    host: str
    user: str
    path: str
    port: int = 22
    key_file: str | None = None


@dataclass
class ToolchainConfig:
    """Toolchain entry in deps.yaml."""

    name: str
    version: str
    sha256: dict[str, str] = field(default_factory=dict)
    bin_dir: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    acquire: AcquireConfig = field(default_factory=AcquireConfig)


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
