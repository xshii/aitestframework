"""Shared fixtures for datastore tests."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aitf.comm import db as core_db
from aitf.comm.db import get_session, init_db
from aitf.ds.types import CaseData, FileEntry


@pytest.fixture()
def tmp_db(tmp_path):
    """Provide a fresh in-memory-like SQLite database per test."""
    db_path = str(tmp_path / "test.db")
    core_db.reset()
    init_db(db_path)
    yield db_path
    core_db.reset()


@pytest.fixture()
def session(tmp_db):
    """Provide a SQLAlchemy session bound to the temp database."""
    s = get_session()
    yield s
    s.close()


@pytest.fixture()
def registry_dir(tmp_path):
    """Create an empty registry directory."""
    d = tmp_path / "registry"
    d.mkdir()
    return d


@pytest.fixture()
def store_dir(tmp_path):
    """Create a store directory with a sample case on disk."""
    store = tmp_path / "store"
    case_dir = store / "npu" / "tdd" / "fp32_basic"

    # weights
    w = case_dir / "weights"
    w.mkdir(parents=True)
    (w / "conv_w.bin").write_bytes(b"\x01\x02\x03\x04" * 16)
    (w / "conv_b.bin").write_bytes(b"\x05\x06\x07\x08" * 8)

    # golden
    g = case_dir / "golden"
    g.mkdir()
    (g / "output.bin").write_bytes(b"\xAA\xBB" * 32)

    return store


@pytest.fixture()
def sample_case():
    """Return a CaseData instance with known values."""
    now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    return CaseData(
        case_id="npu/tdd/fp32_basic",
        name="tdd/fp32_basic",
        files={
            "weights": [
                FileEntry(path="weights/conv_w.bin", size=64, checksum="sha256:aaa"),
                FileEntry(path="weights/conv_b.bin", size=32, checksum="sha256:bbb"),
            ],
            "golden": [
                FileEntry(path="golden/output.bin", size=64, checksum="sha256:ccc"),
            ],
        },
        created_at=now,
        updated_at=now,
    )
