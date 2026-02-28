"""SHA-256 integrity helpers for case data files."""

from __future__ import annotations

import hashlib
from pathlib import Path

from aitf.ds.types import CaseData, FileEntry, VerifyResult

_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB


def compute_sha256(path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        path: Filesystem path to the file.

    Returns:
        Hex-encoded SHA-256 digest prefixed with ``sha256:``.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def scan_directory(case_dir: str | Path, data_type: str) -> list[FileEntry]:
    """Walk *case_dir*/*data_type*/ and return a :class:`FileEntry` per file.

    Args:
        case_dir: Root directory for the case (e.g. ``store/npu/tdd/fp32_basic``).
        data_type: Sub-directory name (``weights``, ``inputs``, ``golden``, or
            ``artifacts``).

    Returns:
        Sorted list of :class:`FileEntry` objects.
    """
    target = Path(case_dir) / data_type
    if not target.is_dir():
        return []

    entries: list[FileEntry] = []
    for fp in sorted(target.rglob("*")):
        if not fp.is_file():
            continue
        rel = fp.relative_to(Path(case_dir))
        entries.append(
            FileEntry(
                path=str(rel),
                size=fp.stat().st_size,
                checksum=compute_sha256(fp),
            )
        )
    return entries


def verify_case(case_dir: str | Path, case: CaseData) -> list[VerifyResult]:
    """Verify every registered file in *case* against its on-disk checksum.

    Args:
        case_dir: Root directory for the case.
        case: The :class:`CaseData` whose files to verify.

    Returns:
        A list of :class:`VerifyResult`, one per file.
    """
    results: list[VerifyResult] = []
    for _dtype, entries in case.files.items():
        for entry in entries:
            fp = Path(case_dir) / entry.path
            if not fp.exists():
                results.append(
                    VerifyResult(
                        case_id=case.case_id,
                        file_path=entry.path,
                        expected_checksum=entry.checksum,
                        actual_checksum="",
                        ok=False,
                        error="file not found",
                    )
                )
                continue
            actual = compute_sha256(fp)
            results.append(
                VerifyResult(
                    case_id=case.case_id,
                    file_path=entry.path,
                    expected_checksum=entry.checksum,
                    actual_checksum=actual,
                    ok=(actual == entry.checksum),
                )
            )
    return results
