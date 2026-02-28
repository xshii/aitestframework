"""Tests for web.api.deps_routes — deps & bundle REST endpoints."""

from __future__ import annotations

import io
import json
import tarfile
import time
from pathlib import Path

import pytest
import yaml

from aitf.web.app import create_app


@pytest.fixture()
def project_root(tmp_path):
    """Minimal project tree with deps.yaml."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "build" / "cache").mkdir(parents=True)
    (root / "build" / "repos").mkdir(parents=True)
    (root / "deps" / "uploads").mkdir(parents=True)

    cfg = {
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
        "repos": {
            "npu-runtime": {
                "url": "git@10.0.0.1:hw/npu-runtime.git",
                "ref": "main",
                "depth": 1,
            },
        },
        "bundles": {
            "npu-v2.1": {
                "description": "NPU test env v2.1",
                "status": "verified",
                "toolchains": {"npu-compiler": "2.1.0"},
                "libraries": {"json-c": "0.17"},
                "repos": {"npu-runtime": "main"},
                "env": {"NPU_SDK_VERSION": "2.1"},
            },
        },
        "active": "npu-v2.1",
    }
    with open(root / "deps.yaml", "w") as fh:
        yaml.dump(cfg, fh)
    return root


@pytest.fixture()
def app(project_root, tmp_path):
    from aitf.comm import db as core_db
    core_db.reset()

    db_path = str(tmp_path / "test.db")
    a = create_app(config={
        "TESTING": True,
        "DATASTORE_BASE_DIR": str(tmp_path / "datastore"),
        "DATASTORE_DB_PATH": db_path,
        "deps_manager": None,  # force lazy-create below
    })
    # Inject a DepsManager pointing at our temp project
    from aitf.deps.manager import DepsManager
    a.config["deps_manager"] = DepsManager(
        project_root=str(project_root),
        deps_file="deps.yaml",
        build_dir="build",
    )
    yield a
    core_db.reset()


@pytest.fixture()
def client(app):
    return app.test_client()


# ---------------------------------------------------------------------------
# Deps routes
# ---------------------------------------------------------------------------

class TestListDeps:
    def test_list_deps(self, client):
        resp = client.get("/api/deps")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "npu-compiler" in data["toolchains"]
        assert "json-c" in data["libraries"]
        assert "npu-runtime" in data["repos"]

    def test_toolchain_not_installed(self, client):
        data = client.get("/api/deps").get_json()
        assert data["toolchains"]["npu-compiler"]["installed"] is False


class TestDoctor:
    def test_doctor(self, client):
        resp = client.get("/api/deps/doctor")
        assert resp.status_code == 200
        data = resp.get_json()
        assert any(r["check"] == "config" for r in data)


class TestClean:
    def test_clean(self, client, project_root):
        # Create a fake cached dep
        (project_root / "build" / "cache" / "dummy-1.0").mkdir()
        resp = client.post("/api/deps/clean")
        assert resp.status_code == 200
        assert resp.get_json()["removed"] == 1


class TestInstallAsync:
    def test_install_returns_task_id(self, client):
        resp = client.post(
            "/api/deps/install",
            data=json.dumps({"name": "npu-compiler"}),
            content_type="application/json",
        )
        assert resp.status_code == 202
        data = resp.get_json()
        assert "task_id" in data

    def test_task_polling(self, client):
        resp = client.post(
            "/api/deps/install",
            data=json.dumps({}),
            content_type="application/json",
        )
        task_id = resp.get_json()["task_id"]
        # Poll until done (max 5s)
        for _ in range(50):
            r = client.get(f"/api/tasks/{task_id}")
            assert r.status_code == 200
            if r.get_json()["status"] != "running":
                break
            time.sleep(0.1)
        final = client.get(f"/api/tasks/{task_id}").get_json()
        assert final["status"] in ("done", "failed")

    def test_task_not_found(self, client):
        resp = client.get("/api/tasks/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Upload / download routes
# ---------------------------------------------------------------------------

def _make_tar_gz(name: str = "test-1.0.tar.gz", content: bytes = b"hello") -> tuple[io.BytesIO, str]:
    """Create a minimal tar.gz in memory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="test-1.0/readme.txt")
        info.size = len(content)
        tf.addfile(info, io.BytesIO(content))
    buf.seek(0)
    return buf, name


class TestUpload:
    def test_upload_success(self, client):
        buf, name = _make_tar_gz()
        resp = client.post(
            "/api/deps/upload",
            data={"file": (buf, name)},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["saved"] == name
        assert data["size"] > 0

    def test_upload_no_file(self, client):
        resp = client.post("/api/deps/upload")
        assert resp.status_code == 400

    def test_upload_wrong_extension(self, client):
        buf = io.BytesIO(b"not tar")
        resp = client.post(
            "/api/deps/upload",
            data={"file": (buf, "bad.zip")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400


class TestListUploads:
    def test_empty(self, client):
        resp = client.get("/api/deps/uploads")
        assert resp.status_code == 200
        assert resp.get_json() == []

    def test_after_upload(self, client):
        buf, name = _make_tar_gz()
        client.post("/api/deps/upload", data={"file": (buf, name)}, content_type="multipart/form-data")
        resp = client.get("/api/deps/uploads")
        data = resp.get_json()
        assert len(data) == 1
        assert data[0]["name"] == name


class TestDownloadUpload:
    def test_download(self, client):
        buf, name = _make_tar_gz()
        client.post("/api/deps/upload", data={"file": (buf, name)}, content_type="multipart/form-data")
        resp = client.get(f"/api/deps/uploads/{name}/download")
        assert resp.status_code == 200
        assert len(resp.data) > 0

    def test_download_not_found(self, client):
        resp = client.get("/api/deps/uploads/no-such.tar.gz/download")
        assert resp.status_code == 404


class TestDeleteUpload:
    def test_delete(self, client):
        buf, name = _make_tar_gz()
        client.post("/api/deps/upload", data={"file": (buf, name)}, content_type="multipart/form-data")
        resp = client.delete(f"/api/deps/uploads/{name}")
        assert resp.status_code == 200
        assert resp.get_json()["deleted"] == name
        # Verify gone
        resp = client.get(f"/api/deps/uploads/{name}/download")
        assert resp.status_code == 404

    def test_delete_not_found(self, client):
        resp = client.delete("/api/deps/uploads/nonexistent.tar.gz")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Bundle routes
# ---------------------------------------------------------------------------

class TestListBundles:
    def test_list(self, client):
        resp = client.get("/api/bundles")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) >= 1
        names = {b["name"] for b in data}
        assert "npu-v2.1" in names


class TestShowBundle:
    def test_show(self, client):
        resp = client.get("/api/bundles/npu-v2.1")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["name"] == "npu-v2.1"
        assert "npu-compiler" in data["toolchains"]

    def test_not_found(self, client):
        resp = client.get("/api/bundles/nonexistent")
        assert resp.status_code == 404


class TestCreateBundle:
    def test_create(self, client):
        resp = client.post(
            "/api/bundles",
            data=json.dumps({
                "name": "new-bundle",
                "description": "test",
                "status": "testing",
                "toolchains": {"npu-compiler": "2.1.0"},
            }),
            content_type="application/json",
        )
        assert resp.status_code == 201
        # Verify it exists
        resp = client.get("/api/bundles/new-bundle")
        assert resp.status_code == 200

    def test_missing_name(self, client):
        resp = client.post(
            "/api/bundles",
            data=json.dumps({"description": "no name"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_invalid_status(self, client):
        resp = client.post(
            "/api/bundles",
            data=json.dumps({"name": "bad", "status": "bogus"}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "invalid status" in resp.get_json()["error"]


class TestDeleteBundle:
    def test_delete(self, client):
        resp = client.delete("/api/bundles/npu-v2.1")
        assert resp.status_code == 200
        # Verify gone
        resp = client.get("/api/bundles/npu-v2.1")
        assert resp.status_code == 404

    def test_delete_not_found(self, client):
        resp = client.delete("/api/bundles/nonexistent")
        assert resp.status_code == 404


class TestExportBundle:
    def test_export(self, client):
        resp = client.get("/api/bundles/npu-v2.1/export")
        assert resp.status_code == 200
        # Should be valid YAML
        data = yaml.safe_load(resp.data)
        assert "bundles" in data
        assert "npu-v2.1" in data["bundles"]


class TestInstallBundle:
    def test_install_returns_task(self, client):
        resp = client.post("/api/bundles/npu-v2.1/install")
        assert resp.status_code == 202
        assert "task_id" in resp.get_json()


class TestUseBundle:
    def test_use_returns_task(self, client):
        resp = client.post(
            "/api/bundles/npu-v2.1/use",
            data=json.dumps({"force": False}),
            content_type="application/json",
        )
        assert resp.status_code == 202
        assert "task_id" in resp.get_json()


class TestConcurrentDownloads:
    """Verify that multiple concurrent download requests don't break."""

    def test_concurrent_downloads(self, client):
        """Upload one file, then download it from multiple 'clients'."""
        buf, name = _make_tar_gz()
        client.post("/api/deps/upload", data={"file": (buf, name)}, content_type="multipart/form-data")

        results = []
        for _ in range(10):
            resp = client.get(f"/api/deps/uploads/{name}/download")
            results.append(resp.status_code)

        assert all(code == 200 for code in results)
        # All should return the same content
        sizes = set()
        for _ in range(10):
            resp = client.get(f"/api/deps/uploads/{name}/download")
            sizes.add(len(resp.data))
        assert len(sizes) == 1
