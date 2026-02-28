"""YAML-based case registry — the source of truth for case metadata.

Files are named ``<platform>_<model>.yaml`` and stored under
``datastore/registry/``.  Each YAML file contains a mapping of
``case_id -> case dict``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from aitf.ds.types import CaseData, FileEntry

# Fields needing special deserialization — everything else is auto-mapped.
_SPECIAL_FIELDS = frozenset({"case_id", "files", "created_at", "updated_at"})

# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _parse_dt(val: Any) -> datetime:
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        return datetime.fromisoformat(val)
    return datetime.now(timezone.utc)


def case_to_dict(case: CaseData) -> dict[str, Any]:
    """Serialise a :class:`CaseData` to a plain dict suitable for YAML."""
    d = asdict(case)
    del d["case_id"]  # case_id is the YAML key
    # Coerce StrEnum keys to plain str for YAML safety
    d["files"] = {str(k): v for k, v in d["files"].items()}
    d["created_at"] = case.created_at.isoformat()
    d["updated_at"] = case.updated_at.isoformat()
    return d


def dict_to_case(case_id: str, d: dict[str, Any]) -> CaseData:
    """Deserialise a plain dict (from YAML) into a :class:`CaseData`.

    Simple init fields are auto-mapped via ``dataclasses.fields()``;
    ``files`` and datetime fields get special handling.
    """
    files: dict[str, list[FileEntry]] = {
        dtype: [FileEntry(**e) for e in entries]
        for dtype, entries in (d.get("files") or {}).items()
    }
    # Auto-map simple init fields from dict, using dataclass defaults
    simple: dict[str, Any] = {}
    for f in dataclasses.fields(CaseData):
        if not f.init or f.name in _SPECIAL_FIELDS:
            continue
        default = f.default if f.default is not dataclasses.MISSING else ""
        simple[f.name] = d.get(f.name, default)

    return CaseData(
        case_id=case_id,
        **simple,
        files=files,
        created_at=_parse_dt(d.get("created_at")),
        updated_at=_parse_dt(d.get("updated_at")),
    )


# ---------------------------------------------------------------------------
# Registry file helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_all_cases(registry_dir: str | Path) -> list[CaseData]:
    """Load every case from all YAML files in *registry_dir*."""
    rd = Path(registry_dir)
    if not rd.is_dir():
        return []
    return [
        dict_to_case(cid, cdict)
        for yf in sorted(rd.glob("*.yaml"))
        for cid, cdict in _load_yaml(yf).items()
    ]


def load_case(registry_dir: str | Path, case_id: str) -> CaseData | None:
    """Load a single case by *case_id*."""
    rd = Path(registry_dir)
    if not rd.is_dir():
        return None
    for yf in sorted(rd.glob("*.yaml")):
        data = _load_yaml(yf)
        if case_id in data:
            return dict_to_case(case_id, data[case_id])
    return None


def save_case(registry_dir: str | Path, case: CaseData) -> None:
    """Create or update a case entry in its YAML file."""
    rd = Path(registry_dir)
    yf = rd / f"{case.platform}_{case.model}.yaml"
    data = _load_yaml(yf)
    data[case.case_id] = case_to_dict(case)
    _save_yaml(yf, data)


def delete_case(registry_dir: str | Path, case_id: str) -> bool:
    """Remove a case entry. Returns ``True`` if found and deleted."""
    rd = Path(registry_dir)
    if not rd.is_dir():
        return False
    for yf in sorted(rd.glob("*.yaml")):
        data = _load_yaml(yf)
        if case_id in data:
            del data[case_id]
            if data:
                _save_yaml(yf, data)
            else:
                yf.unlink()
            return True
    return False
