"""Integration tests: real client ↔ server sync over HTTP.

A standalone Flask server runs in a **separate process** on a random port.
A client-mode Flask app has its SyncClient pointed at that server.
All sync operations go through real HTTP — no mocking.

Process isolation is critical: ``aitf.tc.db`` uses module-level globals
(``_engine``, ``_SessionFactory``).  If server and client share the same
process, the second ``init_db()`` call overwrites the first, causing both
to share one SQLite database and breaking execution-sync assertions.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import socket
import time

import pytest
import yaml

from aitf.config import AitfConfig
from aitf.deps.manager import DepsManager
from aitf.sync.client import SyncClient
from aitf.web.app import create_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_project(tmp_path, name, *, with_deps=True):
    """Create a project directory with optional deps.yaml."""
    root = tmp_path / name
    root.mkdir()
    (root / "build" / "cache").mkdir(parents=True)
    (root / "build" / "repos").mkdir(parents=True)
    (root / "deps" / "uploads").mkdir(parents=True)
    (root / "datastore").mkdir()

    if with_deps:
        deps_cfg = {
            "toolchains": {
                "npu-compiler": {
                    "version": "2.1.0",
                    "sha256": {},
                    "bin_dir": "bin",
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
            "bundles": {},
            "active": "",
        }
        with open(root / "deps.yaml", "w") as fh:
            yaml.dump(deps_cfg, fh)

    return root


def _run_server(root_str: str, port: int) -> None:
    """Entry point for the server subprocess.

    All imports are local so the child process gets its own module state.
    """
    from pathlib import Path

    from aitf.config import AitfConfig as _Cfg
    from aitf.deps.manager import DepsManager as _DM
    from aitf.web.app import create_app as _create

    # Suppress Werkzeug request logging in tests
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    root = Path(root_str)
    cfg = _Cfg(project_root=root)
    app = _create(
        config={
            "TESTING": False,
            "DATASTORE_BASE_DIR": str(root / "datastore"),
        },
        aitf_config=cfg,
    )
    app.config["deps_manager"] = _DM(
        project_root=str(root), deps_file="deps.yaml", build_dir="build",
    )
    app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def server(tmp_path):
    """Start a standalone Flask server in a separate process."""
    root = _make_project(tmp_path, "server")
    port = _free_port()

    proc = multiprocessing.Process(
        target=_run_server,
        args=(str(root), port),
        daemon=True,
    )
    proc.start()

    # Wait for server to be ready
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)
    else:
        proc.kill()
        raise RuntimeError("Server did not start in time")

    yield {"port": port, "root": root, "url": f"http://127.0.0.1:{port}"}

    proc.kill()
    proc.join(timeout=2)


@pytest.fixture()
def client_app(tmp_path, server):
    """Client-mode Flask app whose SyncClient points to the real server."""
    root = _make_project(tmp_path, "client", with_deps=False)
    cfg = AitfConfig(project_root=root, server="127.0.0.1", port=server["port"])
    app = create_app(
        config={
            "TESTING": True,
            "DATASTORE_BASE_DIR": str(root / "datastore"),
        },
        aitf_config=cfg,
    )
    app.config["deps_manager"] = DepsManager(
        project_root=str(root), deps_file="deps.yaml", build_dir="build",
    )
    return app


@pytest.fixture()
def client(client_app):
    return client_app.test_client()


@pytest.fixture()
def sync_client(server):
    """A raw SyncClient connected to the real server."""
    return SyncClient(server["url"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestServerHealth:

    def test_ping(self, sync_client):
        assert sync_client.ping() is True


# ---------------------------------------------------------------------------
# Golden sync: server → client
# ---------------------------------------------------------------------------

class TestGoldenSync:

    def _upload_golden(self, sync_client, tmp_path):
        """Upload golden data to the server via real HTTP."""
        f = tmp_path / "input.bin"
        f.write_bytes(b"\x00" * 16)
        sync_client.golden_upload_files(
            model="test_model", version="v1", operator="op_add",
            file_paths=[f],
        )

    def test_client_syncs_golden_from_server(self, client, sync_client, tmp_path):
        """Client pulls golden data from server via real HTTP."""
        self._upload_golden(sync_client, tmp_path)

        resp = client.post(
            "/api/golden/sync",
            data=json.dumps({"source": ""}),
            content_type="application/json",
        )
        data = resp.get_json()
        assert resp.status_code == 200
        assert "error" not in data, f"Unexpected error: {data.get('error')}"
        assert data["count"] >= 1

    def test_client_can_list_synced_models(self, client, sync_client, tmp_path):
        """After sync, client should list the synced model."""
        self._upload_golden(sync_client, tmp_path)
        client.post(
            "/api/golden/sync",
            data=json.dumps({"source": ""}),
            content_type="application/json",
        )

        resp = client.get("/api/golden/models")
        models = resp.get_json()
        assert "test_model" in models


# ---------------------------------------------------------------------------
# Deps sync: server → client
# ---------------------------------------------------------------------------

class TestDepsSync:

    def test_client_syncs_deps_from_server(self, client):
        """Client pulls deps.yaml from server via real HTTP."""
        resp = client.post("/api/deps/sync")
        data = resp.get_json()
        assert resp.status_code == 200
        assert "error" not in data, f"Unexpected error: {data.get('error')}"
        assert data["total"] == 2  # npu-compiler + json-c

    def test_synced_deps_visible_on_client(self, client):
        """After sync, client should list the server's deps."""
        client.post("/api/deps/sync")
        resp = client.get("/api/deps")
        data = resp.get_json()
        names = {d["name"] for d in data}
        assert "npu-compiler" in names
        assert "json-c" in names


# ---------------------------------------------------------------------------
# TC plan sync: server → client
# ---------------------------------------------------------------------------

class TestPlanSync:

    def _upload_plan(self, sync_client):
        """Upload a test plan to the server via real HTTP."""
        plan = {
            "name": "server plan",
            "plans": [{"name": "group1", "tests": ["test_a", "test_b"]}],
        }
        assert sync_client.upload_plan("testplan_server", plan) is True

    def test_client_syncs_plans_from_server(self, client, sync_client):
        """Client pulls test plans from server via real HTTP."""
        self._upload_plan(sync_client)

        resp = client.post(
            "/api/tc/sync_plans", content_type="application/json")
        data = resp.get_json()
        assert resp.status_code == 200
        assert "error" not in data, f"Unexpected error: {data.get('error')}"
        assert data["total"] >= 1
        assert data["saved"] >= 1


# ---------------------------------------------------------------------------
# Execution results sync: server → client
# ---------------------------------------------------------------------------

class TestResultsSync:

    def _upload_execution(self, sync_client):
        """Upload execution via real HTTP (SyncClient), not test client."""
        payload = {
            "id": "srv-exec-001",
            "total": 10, "passed": 8, "failed": 2,
            "started_at": "2026-01-15T10:00:00",
            "cases": [
                {"suite_class": "TestA", "case_method": "test_1", "status": "PASS"},
                {"suite_class": "TestA", "case_method": "test_2", "status": "FAIL",
                 "failure_reason": "assert 1 == 2"},
            ],
        }
        assert sync_client.upload_execution(payload) is True

    def test_client_syncs_results_from_server(self, client, sync_client):
        """Client pulls execution results from server via real HTTP."""
        self._upload_execution(sync_client)

        resp = client.post(
            "/api/tc/sync_results", content_type="application/json")
        data = resp.get_json()
        assert resp.status_code == 200
        assert "error" not in data, f"Unexpected error: {data.get('error')}"
        assert data["total"] >= 1
        assert data["imported"] >= 1

    def test_synced_execution_has_cases(self, client, sync_client):
        """After sync, execution detail should include case results."""
        self._upload_execution(sync_client)
        client.post("/api/tc/sync_results", content_type="application/json")

        resp = client.get("/api/executions/srv-exec-001")
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["id"] == "srv-exec-001"
        assert len(data["cases"]) == 2


# ---------------------------------------------------------------------------
# Round trip: client → server → client
# ---------------------------------------------------------------------------

class TestRoundTrip:

    def test_plan_upload_then_sync_back(self, client, sync_client):
        """Upload a plan to server, then sync it back to client."""
        plan_data = {
            "name": "roundtrip plan",
            "plans": [{"name": "rt", "tests": ["test_rt"]}],
        }
        assert sync_client.upload_plan("testplan_roundtrip", plan_data) is True

        resp = client.post(
            "/api/tc/sync_plans", content_type="application/json")
        data = resp.get_json()
        assert data["saved"] >= 1

    def test_execution_upload_then_sync_back(self, client, sync_client):
        """Upload execution to server, then sync it back to client."""
        payload = {
            "id": "rt-exec-001",
            "total": 3, "passed": 3, "failed": 0,
            "started_at": "2026-02-01T12:00:00",
            "cases": [],
        }
        assert sync_client.upload_execution(payload) is True

        resp = client.post(
            "/api/tc/sync_results", content_type="application/json")
        data = resp.get_json()
        assert data["imported"] >= 1
