"""Tests for golden data REST endpoints (model/version/operator/data-items)."""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from aitf.config import AitfConfig
from aitf.web.app import create_app


@pytest.fixture()
def app(tmp_path):
    base = tmp_path / "datastore"
    base.mkdir()
    app = create_app(
        config={"TESTING": True, "DATASTORE_BASE_DIR": str(base)},
        aitf_config=AitfConfig(project_root=tmp_path),
    )
    yield app


@pytest.fixture()
def client(app):
    return app.test_client()


def _create(client, model="tdd", version="v1", operator="matmul",
            data_items=None, **extra):
    data = {"model": model, "version": version, "operator": operator}
    if data_items is not None:
        data["data_items"] = json.dumps(data_items)
    data.update(extra)
    return client.post("/api/golden/create", data=data,
                       content_type="multipart/form-data")


def _upload(client, model="tdd", version="v1", operator="matmul",
            name="weights.bin", content=b"data"):
    return client.post(
        "/api/golden/upload",
        data={"model": model, "version": version, "operator": operator,
              "file": (io.BytesIO(content), name)},
        content_type="multipart/form-data",
    )


class TestCreate:
    def test_create_empty(self, client):
        resp = _create(client)
        assert resp.status_code == 201
        d = resp.get_json()
        assert d["operator"] == "matmul"
        assert d["meta"]["data"] == []

    def test_create_with_data_items(self, client):
        items = [
            {"category": "input", "loop": 2, "m": 128, "n": 256, "k": 64,
             "layout": "NHWC", "precision": "fp32"},
            {"category": "weight", "loop": 1, "m": 64, "n": 64, "k": 32,
             "layout": "FRACTAL_NZ", "precision": "bf16"},
        ]
        resp = _create(client, data_items=items)
        assert resp.status_code == 201
        meta = resp.get_json()["meta"]
        assert len(meta["data"]) == 2
        assert meta["data"][0]["category"] == "input"
        assert meta["data"][0]["m"] == 128
        assert meta["data"][1]["category"] == "weight"
        assert meta["data"][1]["precision"] == "bf16"

    def test_create_missing_fields(self, client):
        resp = client.post("/api/golden/create", data={"model": "tdd"})
        assert resp.status_code == 400

    def test_create_with_file(self, client):
        resp = client.post(
            "/api/golden/create",
            data={"model": "tdd", "version": "v1", "operator": "matmul",
                  "file": (io.BytesIO(b"w"), "model.bin")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 201
        assert "model.bin" in resp.get_json()["files"]


class TestDataItems:
    def test_add_data_item(self, client):
        _create(client)
        resp = client.post(
            "/api/golden/tdd/v1/matmul/data-item",
            json={"category": "output", "loop": 1, "m": 32, "n": 32, "k": 16,
                  "layout": "NCHW", "precision": "fp16"},
        )
        assert resp.status_code == 201
        assert len(resp.get_json()["data"]) == 1
        assert resp.get_json()["data"][0]["category"] == "output"

    def test_add_multiple_items(self, client):
        _create(client)
        client.post("/api/golden/tdd/v1/matmul/data-item",
                     json={"category": "input"})
        client.post("/api/golden/tdd/v1/matmul/data-item",
                     json={"category": "weight"})
        resp = client.get("/api/golden/tdd/v1/matmul/meta")
        assert len(resp.get_json()["data"]) == 2

    def test_delete_data_item(self, client):
        _create(client, data_items=[
            {"category": "input"}, {"category": "output"}, {"category": "weight"},
        ])
        resp = client.delete("/api/golden/tdd/v1/matmul/data-item/1")
        assert resp.status_code == 200
        data = resp.get_json()["data"]
        assert len(data) == 2
        assert data[0]["category"] == "input"
        assert data[1]["category"] == "weight"

    def test_delete_data_item_out_of_range(self, client):
        _create(client)
        resp = client.delete("/api/golden/tdd/v1/matmul/data-item/99")
        assert resp.status_code == 400

    def test_duplicate_category_seq_rejected(self, client):
        _create(client, data_items=[{"category": "input", "seq": 0}])
        resp = client.post("/api/golden/tdd/v1/matmul/data-item",
                           json={"category": "input", "seq": 0})
        assert resp.status_code == 409
        assert "序号重复" in resp.get_json()["error"]

    def test_same_seq_different_category_ok(self, client):
        _create(client, data_items=[{"category": "input", "seq": 0}])
        resp = client.post("/api/golden/tdd/v1/matmul/data-item",
                           json={"category": "output", "seq": 0})
        assert resp.status_code == 201

    def test_create_with_duplicate_items_rejected(self, client):
        resp = _create(client, data_items=[
            {"category": "input", "seq": 0},
            {"category": "input", "seq": 0},
        ])
        assert resp.status_code == 409


class TestUpload:
    def test_upload_success(self, client):
        resp = _upload(client)
        assert resp.status_code == 201
        assert "weights.bin" in resp.get_json()["files"]

    def test_upload_missing_fields(self, client):
        resp = client.post("/api/golden/upload")
        assert resp.status_code == 400

    def test_upload_bad_extension(self, client):
        resp = _upload(client, name="data.csv")
        assert resp.status_code == 400

    @pytest.mark.parametrize("ext", [".pth", ".bin", ".zip", ".tar", ".tar.gz"])
    def test_upload_allowed_extensions(self, client, ext):
        resp = _upload(client, name=f"model{ext}")
        assert resp.status_code == 201

    def test_upload_auto_parse_name(self, client):
        resp = client.post(
            "/api/golden/upload",
            data={"operator": "conv2d",
                  "file": (io.BytesIO(b"pkg"), "resnet_v2.bin")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 201
        assert resp.get_json()["model"] == "resnet"
        assert resp.get_json()["version"] == "v2"


class TestList:
    def test_empty(self, client):
        assert client.get("/api/golden").get_json() == []

    def test_after_create_with_items(self, client):
        _create(client, data_items=[{"category": "input", "m": 32}])
        items = client.get("/api/golden").get_json()
        assert len(items) == 1
        assert items[0]["meta"]["data"][0]["category"] == "input"
        assert items[0]["meta"]["data"][0]["m"] == 32

    def test_filter_by_model(self, client):
        _create(client, model="tdd")
        _create(client, model="resnet", operator="conv")
        assert len(client.get("/api/golden?model=tdd").get_json()) == 1


class TestMeta:
    def test_get_meta(self, client):
        _create(client, data_items=[{"category": "bias", "precision": "bf16"}])
        resp = client.get("/api/golden/tdd/v1/matmul/meta")
        assert resp.status_code == 200
        assert resp.get_json()["data"][0]["precision"] == "bf16"

    def test_update_meta(self, client):
        _create(client)
        resp = client.put("/api/golden/tdd/v1/matmul/meta",
                          json={"name": "matmul", "data": [
                              {"category": "input", "m": 512},
                              {"category": "output", "m": 256},
                          ]})
        assert resp.status_code == 200
        assert len(resp.get_json()["data"]) == 2


class TestDownload:
    def test_download_file(self, client):
        _upload(client, content=b"hello")
        resp = client.get("/api/golden/tdd/v1/matmul/weights.bin/download")
        assert resp.status_code == 200
        assert resp.data == b"hello"

    def test_download_not_found(self, client):
        assert client.get("/api/golden/no/such/op/f.bin/download").status_code == 404


class TestDownloadAll:
    def test_download_all_empty(self, client):
        assert client.get("/api/golden/download-all").status_code == 404

    def test_download_all(self, client):
        _create(client, data_items=[{"category": "input"}])
        _upload(client, content=b"aaa")
        _create(client, model="resnet", version="v2", operator="conv")
        _upload(client, model="resnet", version="v2", operator="conv",
                name="model.pth", content=b"bbb")
        resp = client.get("/api/golden/download-all")
        assert resp.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(resp.data))
        data_files = [n for n in zf.namelist() if not n.endswith("golden_meta.json")]
        assert len(data_files) == 2


class TestImport:
    def test_import_archive(self, client):
        _create(client, data_items=[{"category": "input", "m": 128}])
        _upload(client)
        export_resp = client.get("/api/golden/download-all")
        client.delete("/api/golden/tdd/v1/matmul")
        resp = client.post("/api/golden/import",
                           data={"file": (io.BytesIO(export_resp.data), "backup.zip")},
                           content_type="multipart/form-data")
        assert resp.status_code == 201
        items = client.get("/api/golden").get_json()
        assert len(items) == 1
        assert items[0]["meta"]["data"][0]["m"] == 128


class TestDelete:
    def test_delete_operator(self, client):
        _create(client); _upload(client)
        assert client.delete("/api/golden/tdd/v1/matmul").status_code == 200
        assert client.get("/api/golden").get_json() == []

    def test_delete_version(self, client):
        _create(client, operator="op1"); _create(client, operator="op2")
        assert client.delete("/api/golden/tdd/v1").status_code == 200
        assert client.get("/api/golden").get_json() == []

    def test_delete_model(self, client):
        _create(client, version="v1"); _create(client, version="v2", operator="conv")
        assert client.delete("/api/golden/tdd").status_code == 200
        assert client.get("/api/golden").get_json() == []

    def test_delete_single_file(self, client):
        _upload(client, name="a.bin", content=b"a")
        _upload(client, name="b.bin", content=b"b")
        assert client.delete("/api/golden/tdd/v1/matmul/a.bin").status_code == 200
        items = client.get("/api/golden").get_json()
        assert "b.bin" in items[0]["files"]
        assert "a.bin" not in items[0]["files"]

    def test_delete_not_found(self, client):
        assert client.delete("/api/golden/no/such/op").status_code == 404


class TestDownloadVersion:
    def test_download_version(self, client):
        _create(client, data_items=[{"category": "input"}])
        _upload(client, content=b"aaa")
        _create(client, operator="conv", data_items=[{"category": "weight"}])
        _upload(client, operator="conv", name="w.bin", content=b"bbb")
        resp = client.get("/api/golden/tdd/v1/download")
        assert resp.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(resp.data))
        data_files = [n for n in zf.namelist() if not n.endswith("golden_meta.json")]
        assert len(data_files) == 2

    def test_download_version_not_found(self, client):
        assert client.get("/api/golden/nomodel/noversion/download").status_code == 404

    def test_download_version_excludes_other_versions(self, client):
        _create(client, version="v1")
        _upload(client, version="v1", content=b"v1data")
        _create(client, version="v2", operator="conv")
        _upload(client, version="v2", operator="conv", name="v2.bin", content=b"v2data")
        resp = client.get("/api/golden/tdd/v1/download")
        zf = zipfile.ZipFile(io.BytesIO(resp.data))
        assert all("v1" in n for n in zf.namelist())


class TestDuplicateOperator:
    def test_create_duplicate_operator_rejected(self, client):
        resp = _create(client)
        assert resp.status_code == 201
        resp2 = _create(client)
        assert resp2.status_code == 409
        assert "已存在" in resp2.get_json()["error"]

    def test_different_operators_ok(self, client):
        assert _create(client, operator="matmul").status_code == 201
        assert _create(client, operator="softmax").status_code == 201


class TestChoices:
    def test_choices(self, client):
        d = client.get("/api/golden/choices").get_json()
        assert "input" in d["category"]
        assert "FP16" in d["precision"]
        assert "NCHW" in d["layout"]
