"""Data classes and exceptions for the datastore module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class DataType(str, Enum):
    """The four kinds of binary data in a case package."""

    WEIGHTS = "weights"
    INPUTS = "inputs"
    GOLDEN = "golden"
    ARTIFACTS = "artifacts"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FileEntry:
    """A single file within a case data package."""

    path: str
    size: int = 0
    checksum: str = ""


@dataclass
class CaseData:
    """Metadata and file inventory for one test-data case.

    ``platform``, ``model``, ``variant`` are auto-derived from ``case_id``
    (format: ``<platform>/<model>/<variant>``).  Explicitly passed values
    are silently overwritten by ``__post_init__``.
    """

    case_id: str
    name: str = ""
    platform: str = field(default="", init=False)
    model: str = field(default="", init=False)
    variant: str = field(default="", init=False)
    version: str = "v1"
    files: dict[str, list[FileEntry]] = field(default_factory=dict)
    source: str = "local"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        parts = self.case_id.strip("/").split("/")
        if len(parts) == 3:
            self.platform, self.model, self.variant = parts


@dataclass
class RemoteConfig:
    """Connection details for an SFTP remote."""

    name: str
    host: str
    user: str
    path: str
    port: int = 22
    ssh_key: str | None = None


@dataclass
class VerifyResult:
    """Result of verifying one file's integrity."""

    case_id: str
    file_path: str
    expected_checksum: str
    actual_checksum: str
    ok: bool
    error: str | None = None


@dataclass
class SyncResult:
    """Summary of a pull / push operation."""

    case_id: str
    direction: str  # "pull" or "push"
    files_transferred: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    bytes_transferred: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no files failed."""
        return self.files_failed == 0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DataStoreError(Exception):
    """Base exception for datastore operations."""


class CaseNotFoundError(DataStoreError, KeyError):
    """Raised when a case_id does not exist."""


class IntegrityError(DataStoreError):
    """Raised when a checksum verification fails."""


class SyncError(DataStoreError):
    """Raised when an SFTP sync operation fails."""
