"""Shared fixtures for integration tests.

Integration tests exercise cross-module interactions using
Flask test clients.  No real network or build artifacts needed.
urllib is patched so sync endpoints route through the WSGI test app.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from aitf.config import AitfConfig, Mode
from aitf.web.app import create_app


@pytest.fixture()
def project_root(tmp_path):
    """Minimal project tree with deps.yaml, cases dir, and datastore."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "build" / "cache").mkdir(parents=True)
    (root / "build" / "repos").mkdir(parents=True)
    (root / "deps" / "uploads").mkdir(parents=True)
    (root / "datastore").mkdir()

    deps_cfg = {
        "toolchains": {
            "npu-compiler": {
                "version": "2.1.0",
                "sha256": {},
                "bin_dir": "bin",
                "env": {"NPU_CC": "{install_dir}/bin/npu-gcc"},
                "acquire": {"local_dir": "deps/uploads/"},
            },
        },
        "libraries": {
            "json-c": {
                "version": "0.17",
                "sha256": "",
                "build_system": "cmake",
                "acquire": {"local_dir": "deps/uploads/"},
            },
        },
        "repos": {},
        "bundles": {
            "npu-v2.1": {
                "description": "NPU test env v2.1",
                "status": "verified",
                "toolchains": {"npu-compiler": "2.1.0"},
                "libraries": {"json-c": "0.17"},
            },
        },
        "active": "npu-v2.1",
    }
    with open(root / "deps.yaml", "w") as fh:
        yaml.dump(deps_cfg, fh)

    cases = root / "cases" / "npu" / "operators"
    cases.mkdir(parents=True)
    (cases / "test_demo.py").write_text(
        "import unittest\n"
        "class TestDemo(unittest.TestCase):\n"
        "    def test_one(self): pass\n"
        "    def test_two(self): pass\n",
        encoding="utf-8",
    )

    return root


def _make_app(project_root: Path, mode: Mode, server: str = "",
              port: int = 8080):
    """Create a Flask app for the given mode."""
    if mode == Mode.DEBUG:
        patcher = patch("aitf.config.is_local_ip", return_value=True)
        patcher.start()
        cfg = AitfConfig(project_root=project_root, server=server or "127.0.0.1", port=port)
        patcher.stop()
    elif mode == Mode.CLIENT:
        cfg = AitfConfig(project_root=project_root, server=server or "192.0.2.1", port=port)
    else:
        cfg = AitfConfig(project_root=project_root)

    app = create_app(
        config={
            "TESTING": True,
            "DATASTORE_BASE_DIR": str(project_root / "datastore"),
        },
        aitf_config=cfg,
    )

    from aitf.deps.manager import DepsManager
    app.config["deps_manager"] = DepsManager(
        project_root=str(project_root),
        deps_file="deps.yaml",
        build_dir="build",
    )

    return app


def _urlopen_via_client(client):
    """Return a fake urlopen that routes requests through the test client."""
    def fake_urlopen(url_or_req, *args, **kwargs):
        if isinstance(url_or_req, urllib.request.Request):
            url = url_or_req.full_url
        else:
            url = url_or_req
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path
        if parsed.query:
            path = f"{path}?{parsed.query}"
        resp = client.get(path)
        if resp.status_code >= 400:
            raise urllib.error.HTTPError(
                url, resp.status_code, "Error", {}, io.BytesIO(resp.data))
        mock_resp = MagicMock()
        mock_resp.read.return_value = resp.data
        mock_resp.status = resp.status_code
        return mock_resp
    return fake_urlopen


@pytest.fixture()
def standalone_app(project_root):
    """Flask app in standalone mode (no sync)."""
    return _make_app(project_root, Mode.STANDALONE)


@pytest.fixture()
def standalone_client(standalone_app):
    return standalone_app.test_client()


@pytest.fixture()
def debug_app(project_root):
    """Flask app in debug mode (server + client on same host)."""
    return _make_app(project_root, Mode.DEBUG)


@pytest.fixture()
def debug_client(debug_app):
    """Test client with urllib patched to route through WSGI app."""
    client = debug_app.test_client()
    fake = _urlopen_via_client(client)
    with patch("urllib.request.urlopen", fake):
        yield client


@pytest.fixture()
def debug_no_deps_app(tmp_path):
    """Debug mode app WITHOUT deps.yaml — simulates fresh project."""
    root = tmp_path / "bare"
    root.mkdir()
    (root / "build").mkdir()
    (root / "datastore").mkdir()
    (root / "cases").mkdir()

    patcher = patch("aitf.config.is_local_ip", return_value=True)
    patcher.start()
    cfg = AitfConfig(project_root=root, server="127.0.0.1", port=8080)
    patcher.stop()

    app = create_app(
        config={"TESTING": True, "DATASTORE_BASE_DIR": str(root / "datastore")},
        aitf_config=cfg,
    )
    from aitf.deps.manager import DepsManager
    app.config["deps_manager"] = DepsManager(
        project_root=str(root), deps_file="deps.yaml", build_dir="build",
    )
    return app


@pytest.fixture()
def debug_no_deps_client(debug_no_deps_app):
    """Debug client whose project has no deps.yaml."""
    client = debug_no_deps_app.test_client()
    fake = _urlopen_via_client(client)
    with patch("urllib.request.urlopen", fake):
        yield client
