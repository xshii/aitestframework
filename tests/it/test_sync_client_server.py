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

import io
import json
import logging
import multiprocessing
import socket
import tarfile
import time
import zipfile

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
# Golden upload to remote (client → server): /api/tc/sync_golden
# ---------------------------------------------------------------------------

class TestGoldenUploadToRemote:

    def _seed_golden_on_client(self, client):
        """Create golden data on the client via its local API."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("test_model/v1/op_add/input.bin", b"\x00" * 16)
        buf.seek(0)
        resp = client.post("/api/golden/import", data={
            "file": (buf, "golden.zip"),
        }, content_type="multipart/form-data")
        assert resp.status_code == 201

    def test_upload_version_to_server(self, client, sync_client):
        """Client pushes golden version to server via /api/tc/sync_golden."""
        self._seed_golden_on_client(client)

        resp = client.post(
            "/api/tc/sync_golden",
            data=json.dumps({
                "items": [{"model": "test_model", "version": "v1"}],
            }),
            content_type="application/json",
        )
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["success"] == 1

        # Verify data arrived on server
        models = sync_client.golden_list_models()
        assert "test_model" in models

    def test_upload_operator_to_server(self, client, sync_client):
        """Client pushes a specific operator to server."""
        self._seed_golden_on_client(client)

        resp = client.post(
            "/api/tc/sync_golden",
            data=json.dumps({
                "items": [{"model": "test_model", "version": "v1",
                           "operators": ["op_add"]}],
            }),
            content_type="application/json",
        )
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["success"] == 1

        # Verify operator exists on server
        operators = sync_client.golden_list_operators("test_model", "v1")
        assert "op_add" in operators


# ---------------------------------------------------------------------------
# Golden fingerprint
# ---------------------------------------------------------------------------

class TestGoldenFingerprint:

    def test_fingerprint_after_upload(self, sync_client, tmp_path):
        """Fingerprint should return operator hashes for uploaded data."""
        f = tmp_path / "input.bin"
        f.write_bytes(b"\xAB" * 32)
        sync_client.golden_upload_files(
            model="fp_model", version="v1", operator="op1",
            file_paths=[f],
        )

        fp = sync_client.golden_version_fingerprint("fp_model", "v1")
        assert fp["model"] == "fp_model"
        assert fp["version"] == "v1"
        assert "op1" in fp["operators"]
        assert isinstance(fp["operators"]["op1"], str)
        assert len(fp["operators"]["op1"]) > 0


# ---------------------------------------------------------------------------
# Deps archive upload / download
# ---------------------------------------------------------------------------

class TestDepsArchiveSync:

    def _make_archive(self, tmp_path, name, version):
        """Create a minimal tar.gz archive for a dependency."""
        archive = tmp_path / f"{name}-{version}.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            data = b"fake dep content"
            info = tarfile.TarInfo(name=f"{name}-{version}/README")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        return archive

    def test_upload_dep_archive(self, sync_client, tmp_path):
        """Upload a dependency archive to the server."""
        archive = self._make_archive(tmp_path, "mylib", "1.0")
        result = sync_client.deps_upload_archive("mylib", "1.0", archive)
        assert result["ok"] is True
        assert result["name"] == "mylib"
        assert result["version"] == "1.0"

    def test_upload_then_download_dep_archive(self, sync_client, tmp_path):
        """Upload a dep archive, then download it back."""
        archive = self._make_archive(tmp_path, "mylib", "1.0")
        sync_client.deps_upload_archive("mylib", "1.0", archive)

        dest = tmp_path / "downloaded"
        dest.mkdir()
        downloaded = sync_client.deps_download_archive("mylib", "1.0", dest)
        assert downloaded.is_file()
        assert downloaded.stat().st_size > 0
        # Verify it's a valid tar.gz
        with tarfile.open(downloaded, "r:gz") as tf:
            names = tf.getnames()
            assert any("README" in n for n in names)


# ---------------------------------------------------------------------------
# Bundle upload / download
# ---------------------------------------------------------------------------

class TestBundleSync:

    def _create_bundle_on_server(self, sync_client, server):
        """Create a bundle on the server via its API so export works."""
        import urllib.request
        url = f"{server['url']}/api/bundles"
        data = json.dumps({
            "name": "test_bundle",
            "description": "A test bundle",
            "status": "verified",
            "toolchains": {"npu-compiler": "2.1.0"},
            "libraries": {"json-c": "0.17"},
        }).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=10)
        assert resp.status in (200, 201)

    def test_bundle_upload(self, sync_client, tmp_path):
        """Upload a bundle archive to the server."""
        # Create a minimal bundle tar.gz
        archive = tmp_path / "mybundle.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            meta = yaml.dump({
                "name": "mybundle",
                "description": "test",
                "status": "verified",
                "toolchains": {},
                "libraries": {},
                "repos": {},
                "env": {},
            }).encode()
            info = tarfile.TarInfo(name="mybundle/bundle.yaml")
            info.size = len(meta)
            tf.addfile(info, io.BytesIO(meta))
        result = sync_client.bundle_upload("mybundle", archive)
        assert result["ok"] is True
        assert result["name"] == "mybundle"

    def test_bundle_download(self, sync_client, server, tmp_path):
        """Create bundle on server, then download it."""
        self._create_bundle_on_server(sync_client, server)

        dest = tmp_path / "downloaded"
        dest.mkdir()
        downloaded = sync_client.bundle_download("test_bundle", dest)
        assert downloaded.is_file()
        assert downloaded.stat().st_size > 0
        # Verify it's a valid tar.gz
        with tarfile.open(downloaded, "r:gz") as tf:
            names = tf.getnames()
            assert any("bundle.yaml" in n for n in names)


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
