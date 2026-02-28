"""Shared fixtures for web tests."""

from __future__ import annotations

import pytest

from aitf.comm import db as core_db
from aitf.web.app import create_app


@pytest.fixture()
def app(tmp_path):
    """Create a Flask test app with temp directories."""
    core_db.reset()
    base = tmp_path / "datastore"
    base.mkdir()
    db_path = str(tmp_path / "test.db")

    # Create a case directory for registration tests
    case_dir = tmp_path / "local_case"
    (case_dir / "weights").mkdir(parents=True)
    (case_dir / "weights" / "w.bin").write_bytes(b"\x01" * 32)

    app = create_app(
        config={
            "TESTING": True,
            "DATASTORE_BASE_DIR": str(base),
            "DATASTORE_DB_PATH": db_path,
            "LOCAL_CASE_DIR": str(case_dir),
        }
    )
    yield app
    core_db.reset()


@pytest.fixture()
def client(app):
    """Flask test client."""
    return app.test_client()
