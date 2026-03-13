"""Tests for golden fingerprint API and GoldenStore fingerprint methods."""

from __future__ import annotations

import io
import json

import pytest

from aitf.config import AitfConfig
from aitf.ds.store import FP_FILE, GoldenStore
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


@pytest.fixture()
def store(tmp_path):
    base = tmp_path / "ds"
    base.mkdir()
    return GoldenStore(base)


def _create_operator(client, model="m1", version="v1", operator="conv2d"):
    return client.post(
        "/api/golden/create",
        data={"model": model, "version": version, "operator": operator},
        content_type="multipart/form-data",
    )


def _upload_file(client, model="m1", version="v1", operator="conv2d",
                 name="data.bin", content=b"hello"):
    return client.post(
        "/api/golden/upload",
        data={
            "model": model, "version": version, "operator": operator,
            "file": (io.BytesIO(content), name),
        },
        content_type="multipart/form-data",
    )


class TestStoreFingerprint:
    def test_fingerprint_empty_operator(self, store):
        """Operator with no data files should still produce a fingerprint."""
        op_dir = store.root / "m" / "v" / "op"
        op_dir.mkdir(parents=True)
        fp = store.operator_fingerprint("m", "v", "op")
        assert fp is not None
        assert len(fp) == 64  # SHA-256 hex length

    def test_fingerprint_changes_on_mutation(self, store):
        op_dir = store.root / "m" / "v" / "op"
        op_dir.mkdir(parents=True)
        (op_dir / "a.bin").write_bytes(b"aaa")
        fp1 = store.update_fingerprint("m", "v", "op")

        (op_dir / "a.bin").write_bytes(b"bbb")
        store._invalidate_fingerprint("m", "v", "op")
        fp2 = store.operator_fingerprint("m", "v", "op")
        assert fp1 != fp2

    def test_fingerprint_cached(self, store):
        op_dir = store.root / "m" / "v" / "op"
        op_dir.mkdir(parents=True)
        (op_dir / "a.bin").write_bytes(b"aaa")
        fp1 = store.operator_fingerprint("m", "v", "op")

        # Cache file should exist
        assert (op_dir / FP_FILE).is_file()

        # Modify data but don't invalidate — should return cached value
        (op_dir / "a.bin").write_bytes(b"bbb")
        fp2 = store.operator_fingerprint("m", "v", "op")
        assert fp1 == fp2  # stale cache

    def test_save_file_invalidates(self, store):
        op_dir = store.root / "m" / "v" / "op"
        op_dir.mkdir(parents=True)
        (op_dir / "a.bin").write_bytes(b"aaa")
        store.update_fingerprint("m", "v", "op")

        # save_file should delete the cache
        store.save_file("m", "v", "op", "b.bin")
        assert not (op_dir / FP_FILE).is_file()

    def test_delete_file_invalidates(self, store):
        op_dir = store.root / "m" / "v" / "op"
        op_dir.mkdir(parents=True)
        (op_dir / "a.bin").write_bytes(b"aaa")
        store.update_fingerprint("m", "v", "op")

        store.delete_file("m", "v", "op", "a.bin")
        assert not (op_dir / FP_FILE).is_file()

    def test_version_fingerprint(self, store):
        for op in ("conv2d", "matmul"):
            op_dir = store.root / "m" / "v" / op
            op_dir.mkdir(parents=True)
            (op_dir / "data.bin").write_bytes(f"data-{op}".encode())

        fps = store.version_fingerprint("m", "v")
        assert set(fps.keys()) == {"conv2d", "matmul"}
        assert fps["conv2d"] != fps["matmul"]

    def test_nonexistent_operator(self, store):
        assert store.operator_fingerprint("m", "v", "nope") is None


class TestFingerprintRoute:
    def test_version_fingerprint_api(self, client):
        _create_operator(client, operator="conv2d")
        _upload_file(client, operator="conv2d", content=b"data1")

        resp = client.get("/api/golden/m1/v1/fingerprint")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["model"] == "m1"
        assert data["version"] == "v1"
        assert "conv2d" in data["operators"]
        assert len(data["operators"]["conv2d"]) == 64

    def test_empty_version_fingerprint(self, client):
        resp = client.get("/api/golden/m1/v1/fingerprint")
        assert resp.status_code == 200
        assert resp.get_json()["operators"] == {}
