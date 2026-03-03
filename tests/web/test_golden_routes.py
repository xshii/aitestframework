"""Tests for golden data REST endpoints (single file per model/version)."""

from __future__ import annotations

import io
import zipfile

import pytest

from aitf.config import AitfConfig
from aitf.web.app import create_app


@pytest.fixture()
def app(tmp_path):
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
    return app.test_client()


def _upload(client, model="tdd", version="v1", name="weights.bin", content=b"data"):
    return client.post(
        "/api/golden/upload",
        data={"model": model, "version": version, "file": (io.BytesIO(content), name)},
        content_type="multipart/form-data",
    )


class TestUpload:
    def test_upload_success(self, client):
        resp = _upload(client)
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["model"] == "tdd"
        assert data["version"] == "v1"
        assert data["file"] == "weights.bin"

    def test_upload_replaces_old(self, client):
        _upload(client, name="old.bin", content=b"old")
        _upload(client, name="new.bin", content=b"new-data")
        items = client.get("/api/golden").get_json()
        assert len(items) == 1
        assert items[0]["file"] == "new.bin"

    def test_upload_missing_fields(self, client):
        resp = client.post("/api/golden/upload")
        assert resp.status_code == 400

    def test_upload_bad_extension(self, client):
        resp = _upload(client, name="data.csv")
        assert resp.status_code == 400
        assert "only" in resp.get_json()["error"]

    @pytest.mark.parametrize("ext", [".pth", ".bin", ".zip", ".tar", ".tar.gz"])
    def test_upload_allowed_extensions(self, client, ext):
        resp = _upload(client, name=f"model{ext}")
        assert resp.status_code == 201


class TestList:
    def test_empty(self, client):
        resp = client.get("/api/golden")
        assert resp.status_code == 200
        assert resp.get_json() == []

    def test_after_upload(self, client):
        _upload(client)
        items = client.get("/api/golden").get_json()
        assert len(items) == 1
        it = items[0]
        assert it["model"] == "tdd"
        assert it["version"] == "v1"
        assert it["file"] == "weights.bin"
        assert it["size"] > 0


class TestDownload:
    def test_download(self, client):
        _upload(client, content=b"hello")
        resp = client.get("/api/golden/tdd/v1/download")
        assert resp.status_code == 200
        assert resp.data == b"hello"

    def test_download_not_found(self, client):
        resp = client.get("/api/golden/no/such/download")
        assert resp.status_code == 404


class TestDownloadAll:
    def test_download_all_empty(self, client):
        resp = client.get("/api/golden/download-all")
        assert resp.status_code == 404

    def test_download_all(self, client):
        _upload(client, model="tdd", version="v1", content=b"aaa")
        _upload(client, model="resnet", version="v2", name="model.pth", content=b"bbb")
        resp = client.get("/api/golden/download-all")
        assert resp.status_code == 200
        assert resp.content_type == "application/zip"
        zf = zipfile.ZipFile(io.BytesIO(resp.data))
        names = sorted(zf.namelist())
        assert len(names) == 2
        assert "tdd/v1/weights.bin" in names
        assert "resnet/v2/model.pth" in names
        assert zf.read("tdd/v1/weights.bin") == b"aaa"
        assert zf.read("resnet/v2/model.pth") == b"bbb"


class TestDelete:
    def test_delete(self, client):
        _upload(client)
        resp = client.delete("/api/golden/tdd/v1")
        assert resp.status_code == 200
        assert resp.get_json()["deleted"] == "tdd/v1"
        # Verify gone
        assert client.get("/api/golden").get_json() == []

    def test_delete_not_found(self, client):
        resp = client.delete("/api/golden/no/such")
        assert resp.status_code == 404
