"""SQLite cache layer — derived data that can be rebuilt from YAML at any time."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from sqlalchemy.orm import Session

from aitf.comm.models import CaseDataRow, FileEntryRow
from aitf.ds import registry as reg
from aitf.ds.types import CaseData, FileEntry

# Auto-derived from CaseData — stays in sync when fields are added/removed.
# _INIT_FIELDS: only fields passable to CaseData.__init__ (excludes init=False).
# _ALL_FIELDS: every scalar field (for writing to CaseDataRow which accepts all).
_INIT_FIELDS = tuple(
    f.name for f in dataclasses.fields(CaseData) if f.name != "files" and f.init
)
_ALL_FIELDS = tuple(
    f.name for f in dataclasses.fields(CaseData) if f.name != "files"
)


def _row_to_case(row: CaseDataRow) -> CaseData:
    files: dict[str, list[FileEntry]] = {}
    for f in row.files:
        files.setdefault(f.data_type, []).append(
            FileEntry(path=f.path, size=f.size, checksum=f.checksum)
        )
    return CaseData(**{k: getattr(row, k) for k in _INIT_FIELDS}, files=files)


def _case_to_rows(case: CaseData) -> tuple[CaseDataRow, list[FileEntryRow]]:
    row = CaseDataRow(**{k: getattr(case, k) for k in _ALL_FIELDS})
    file_rows = [
        FileEntryRow(
            case_id=case.case_id, data_type=dtype,
            path=e.path, size=e.size, checksum=e.checksum,
        )
        for dtype, entries in case.files.items()
        for e in entries
    ]
    return row, file_rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rebuild_cache(session: Session, registry_dir: str | Path) -> int:
    """Drop all cached rows and repopulate from YAML."""
    session.query(FileEntryRow).delete()
    session.query(CaseDataRow).delete()
    session.flush()

    cases = reg.load_all_cases(registry_dir)
    for case in cases:
        row, file_rows = _case_to_rows(case)
        session.add(row)
        session.add_all(file_rows)
    session.commit()
    return len(cases)


def query_case(session: Session, case_id: str) -> CaseData | None:
    """Look up a single case in the cache."""
    row = session.get(CaseDataRow, case_id)
    return _row_to_case(row) if row else None


def query_cases(
    session: Session,
    platform: str | None = None,
    model: str | None = None,
) -> list[CaseData]:
    """Query cases with optional filters."""
    q = session.query(CaseDataRow)
    if platform:
        q = q.filter(CaseDataRow.platform == platform)
    if model:
        q = q.filter(CaseDataRow.model == model)
    return [_row_to_case(r) for r in q.all()]


def upsert_case(session: Session, case: CaseData) -> None:
    """Insert or replace a case in the cache."""
    existing = session.get(CaseDataRow, case.case_id)
    if existing:
        session.delete(existing)
        session.flush()

    row, file_rows = _case_to_rows(case)
    session.add(row)
    session.add_all(file_rows)
    session.commit()


def delete_case(session: Session, case_id: str) -> None:
    """Delete a case from the cache."""
    row = session.get(CaseDataRow, case_id)
    if row:
        session.delete(row)
        session.commit()
