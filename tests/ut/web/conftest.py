"""Shared fixtures for web tests."""

from __future__ import annotations

import pytest

from aitf.config import AitfConfig
from aitf.web.app import create_app


@pytest.fixture()
def app(tmp_path):
    """Create a Flask test app with temp directories."""
    base = tmp_path / "datastore"
    base.mkdir()

    app = create_app(
        config={
            "TESTING": True,
            "DATASTORE_BASE_DIR": str(base),
        },
        aitf_config=AitfConfig(project_root=tmp_path),
    )
    yield app


@pytest.fixture()
def client(app):
    """Flask test client."""
    return app.test_client()
