"""Tests for deps & bundle REST endpoints (configuration depot).

The web API exposes dependency CRUD, archive upload/download/delete,
and bundle CRUD/export/import.
"""

from __future__ import annotations

import io
import json
import tarfile
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
    from aitf.config import AitfConfig

    a = create_app(
        config={
            "TESTING": True,
            "DATASTORE_BASE_DIR": str(tmp_path / "datastore"),
            "deps_manager": None,
        },
        aitf_config=AitfConfig(project_root=project_root),
    )
    from aitf.deps.manager import DepsManager
    a.config["deps_manager"] = DepsManager(
        project_root=str(project_root),
        deps_file="deps.yaml",
        build_dir="build",
    )
    yield a


@pytest.fixture()
def client(app):
    return app.test_client()


def _make_tar_gz(name: str = "test-1.0.tar.gz", content: bytes = b"hello") -> tuple[io.BytesIO, str]:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="test-1.0/readme.txt")
        info.size = len(content)
        tf.addfile(info, io.BytesIO(content))
    buf.seek(0)
    return buf, name


# ---------------------------------------------------------------------------
# Deps listing (flat)
# ---------------------------------------------------------------------------

class TestListDeps:
    def test_list_returns_flat_list(self, client):
        resp = client.get("/api/deps")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        names = {d["name"] for d in data}
        assert {"npu-compiler", "json-c", "npu-runtime"} == names

    def test_each_dep_has_version_and_versions(self, client):
        data = client.get("/api/deps").get_json()
        for dep in data:
            assert "version" in dep
            assert "versions" in dep
            assert isinstance(dep["versions"], list)
            assert dep["version"] in dep["versions"]

    def test_add_version_adds_to_versions_list(self, client):
        buf, _ = _make_tar_gz("npu-compiler-3.0.0.tar.gz")
        client.post("/api/deps", data={
            "name": "npu-compiler", "version": "3.0.0",
            "file": (buf, "npu-compiler-3.0.0.tar.gz"),
        }, content_type="multipart/form-data")
        data = client.get("/api/deps").get_json()
        tc = next(d for d in data if d["name"] == "npu-compiler")
        assert "3.0.0" in tc["versions"]

    def test_type_field_present(self, client):
        data = client.get("/api/deps").get_json()
        types = {d["name"]: d["type"] for d in data}
        assert types["npu-compiler"] == "toolchain"
        assert types["json-c"] == "library"
        assert types["npu-runtime"] == "repo"


# ---------------------------------------------------------------------------
# Add / delete dependencies
# ---------------------------------------------------------------------------

class TestAddDep:
    def test_add_toolchain(self, client):
        buf, _ = _make_tar_gz("my-tool-1.0.tar.gz")
        resp = client.post("/api/deps", data={
            "name": "my-tool", "type": "toolchain", "version": "1.0",
            "file": (buf, "my-tool-1.0.tar.gz"),
        }, content_type="multipart/form-data")
        assert resp.status_code == 201
        data = client.get("/api/deps").get_json()
        names = {d["name"] for d in data}
        assert "my-tool" in names

    def test_add_repo(self, client):
        resp = client.post("/api/deps", data={
            "name": "my-repo", "type": "repo", "version": "dev",
            "url": "git@host:group/my-repo.git",
        }, content_type="multipart/form-data")
        assert resp.status_code == 201
        data = client.get("/api/deps").get_json()
        assert any(d["name"] == "my-repo" and d["version"] == "dev" for d in data)

    def test_add_missing_name(self, client):
        resp = client.post("/api/deps", data={
            "type": "toolchain", "version": "1.0",
        }, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_add_missing_type_for_new(self, client):
        resp = client.post("/api/deps", data={
            "name": "brand-new", "version": "1.0",
        }, content_type="multipart/form-data")
        assert resp.status_code == 400


class TestDeleteDep:
    def test_delete(self, client):
        resp = client.delete("/api/deps/json-c")
        assert resp.status_code == 200
        data = client.get("/api/deps").get_json()
        assert not any(d["name"] == "json-c" for d in data)

    def test_delete_not_found(self, client):
        resp = client.delete("/api/deps/nonexistent")
        assert resp.status_code == 404

    def test_delete_removes_from_bundles(self, client):
        client.delete("/api/deps/json-c")
        b = client.get("/api/bundles/npu-v2.1").get_json()
        assert "json-c" not in b.get("libraries", {})


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExportDeps:
    def test_export_full_yaml(self, client):
        resp = client.get("/api/deps/export")
        assert resp.status_code == 200
        data = yaml.safe_load(resp.data)
        assert "toolchains" in data
        assert "npu-compiler" in data["toolchains"]

    def test_export_is_valid_yaml(self, client):
        resp = client.get("/api/deps/export")
        data = yaml.safe_load(resp.data)
        assert isinstance(data, dict)
        assert data["toolchains"]["npu-compiler"]["version"] == "2.1.0"


class TestExportSingleDep:
    def test_export_toolchain(self, client):
        resp = client.get("/api/deps/npu-compiler/export")
        assert resp.status_code == 200
        data = yaml.safe_load(resp.data)
        assert "toolchains" in data
        assert data["toolchains"]["npu-compiler"]["version"] == "2.1.0"

    def test_export_repo(self, client):
        resp = client.get("/api/deps/npu-runtime/export")
        assert resp.status_code == 200
        data = yaml.safe_load(resp.data)
        assert "repos" in data
        assert data["repos"]["npu-runtime"]["url"] == "git@10.0.0.1:hw/npu-runtime.git"

    def test_export_with_version_override(self, client):
        resp = client.get("/api/deps/npu-compiler/export?version=3.0.0")
        data = yaml.safe_load(resp.data)
        assert data["toolchains"]["npu-compiler"]["version"] == "3.0.0"

    def test_export_not_found(self, client):
        resp = client.get("/api/deps/nonexistent/export")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Upload listing / download / delete (CLI-facing routes)
# ---------------------------------------------------------------------------

def _add_dep_with_file(client, name="test", version="1.0"):
    """Add a dep via POST /api/deps with an archive file."""
    buf, _ = _make_tar_gz(f"{name}-{version}.tar.gz")
    client.post("/api/deps", data={
        "name": name, "type": "toolchain", "version": version,
        "file": (buf, f"{name}-{version}.tar.gz"),
    }, content_type="multipart/form-data")
    return f"{name}-{version}.tar.gz"


class TestListUploads:
    def test_empty(self, client):
        resp = client.get("/api/deps/uploads")
        assert resp.status_code == 200
        assert resp.get_json() == []

    def test_after_add(self, client):
        _add_dep_with_file(client)
        data = client.get("/api/deps/uploads").get_json()
        assert len(data) >= 1


class TestDownloadUpload:
    def test_download(self, client):
        fname = _add_dep_with_file(client)
        resp = client.get(f"/api/deps/uploads/{fname}/download")
        assert resp.status_code == 200

    def test_download_not_found(self, client):
        resp = client.get("/api/deps/uploads/no-such.tar.gz/download")
        assert resp.status_code == 404


class TestDeleteUpload:
    def test_delete(self, client):
        fname = _add_dep_with_file(client)
        resp = client.delete(f"/api/deps/uploads/{fname}")
        assert resp.status_code == 200

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
        names = {b["name"] for b in resp.get_json()}
        assert "npu-v2.1" in names


class TestShowBundle:
    def test_show(self, client):
        resp = client.get("/api/bundles/npu-v2.1")
        assert resp.status_code == 200
        assert resp.get_json()["name"] == "npu-v2.1"

    def test_not_found(self, client):
        resp = client.get("/api/bundles/nonexistent")
        assert resp.status_code == 404


class TestCreateBundle:
    def test_create(self, client):
        resp = client.post("/api/bundles",
                           data=json.dumps({"name": "new-bundle", "status": "testing",
                                            "toolchains": {"npu-compiler": "2.1.0"}}),
                           content_type="application/json")
        assert resp.status_code == 201
        resp = client.get("/api/bundles/new-bundle")
        assert resp.status_code == 200

    def test_missing_name(self, client):
        resp = client.post("/api/bundles",
                           data=json.dumps({"description": "no name"}),
                           content_type="application/json")
        assert resp.status_code == 400

    def test_invalid_status(self, client):
        resp = client.post("/api/bundles",
                           data=json.dumps({"name": "bad", "status": "bogus"}),
                           content_type="application/json")
        assert resp.status_code == 400


class TestUpdateBundle:
    def test_update(self, client):
        resp = client.put("/api/bundles/npu-v2.1",
                          data=json.dumps({"description": "updated"}),
                          content_type="application/json")
        assert resp.status_code == 200
        data = client.get("/api/bundles/npu-v2.1").get_json()
        assert data["description"] == "updated"

    def test_update_not_found(self, client):
        resp = client.put("/api/bundles/nonexistent",
                          data=json.dumps({"description": "x"}),
                          content_type="application/json")
        assert resp.status_code == 404

    def test_update_invalid_status(self, client):
        resp = client.put("/api/bundles/npu-v2.1",
                          data=json.dumps({"status": "bogus"}),
                          content_type="application/json")
        assert resp.status_code == 400


class TestDeleteBundle:
    def test_delete(self, client):
        resp = client.delete("/api/bundles/npu-v2.1")
        assert resp.status_code == 200
        resp = client.get("/api/bundles/npu-v2.1")
        assert resp.status_code == 404

    def test_delete_not_found(self, client):
        resp = client.delete("/api/bundles/nonexistent")
        assert resp.status_code == 404


class TestExportBundle:
    def test_export(self, client):
        resp = client.get("/api/bundles/npu-v2.1/export")
        assert resp.status_code == 200
        data = yaml.safe_load(resp.data)
        assert "bundles" in data
        assert "npu-v2.1" in data["bundles"]


class TestConcurrentDownloads:
    def test_concurrent_downloads(self, client):
        fname = _add_dep_with_file(client)
        results = [client.get(f"/api/deps/uploads/{fname}/download").status_code for _ in range(10)]
        assert all(code == 200 for code in results)
