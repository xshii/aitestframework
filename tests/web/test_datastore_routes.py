"""Tests for web.api.ds_routes — Flask REST endpoints."""

from __future__ import annotations

import json

import pytest


def _register(client, app, case_id="npu/tdd/fp32"):
    """Helper: register a case and return the response."""
    local_path = app.config["LOCAL_CASE_DIR"]
    return client.post(
        "/api/cases",
        data=json.dumps({"case_id": case_id, "local_path": local_path}),
        content_type="application/json",
    )


class TestListCases:
    def test_empty(self, client):
        resp = client.get("/api/cases")
        assert resp.status_code == 200
        assert resp.get_json() == []

    def test_with_cases(self, client, app):
        _register(client, app, "npu/tdd/a")
        _register(client, app, "gpu/fdd/b")
        resp = client.get("/api/cases")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == 2

    def test_filter_platform(self, client, app):
        _register(client, app, "npu/tdd/a")
        _register(client, app, "gpu/fdd/b")
        resp = client.get("/api/cases?platform=npu")
        data = resp.get_json()
        assert len(data) == 1
        assert data[0]["platform"] == "npu"

    def test_filter_model(self, client, app):
        _register(client, app, "npu/tdd/a")
        _register(client, app, "npu/fdd/b")
        resp = client.get("/api/cases?model=fdd")
        data = resp.get_json()
        assert len(data) == 1
        assert data[0]["model"] == "fdd"


class TestGetCase:
    def test_found(self, client, app):
        _register(client, app)
        resp = client.get("/api/cases/npu/tdd/fp32")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["case_id"] == "npu/tdd/fp32"

    def test_not_found(self, client):
        resp = client.get("/api/cases/no/such/case")
        assert resp.status_code == 404


class TestRegisterCase:
    def test_success(self, client, app):
        resp = _register(client, app)
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["case_id"] == "npu/tdd/fp32"
        assert "weights" in data["files"]

    def test_missing_fields(self, client):
        resp = client.post(
            "/api/cases",
            data=json.dumps({"case_id": "npu/tdd/fp32"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_bad_case_id(self, client, app):
        resp = client.post(
            "/api/cases",
            data=json.dumps({"case_id": "bad", "local_path": app.config["LOCAL_CASE_DIR"]}),
            content_type="application/json",
        )
        assert resp.status_code == 400


class TestDeleteCase:
    def test_success(self, client, app):
        _register(client, app)
        resp = client.delete("/api/cases/npu/tdd/fp32")
        assert resp.status_code == 200
        # Verify gone
        resp = client.get("/api/cases/npu/tdd/fp32")
        assert resp.status_code == 404

    def test_not_found(self, client):
        resp = client.delete("/api/cases/no/such/case")
        assert resp.status_code == 404


class TestVerifyCase:
    def test_verify_empty_case(self, client, app):
        _register(client, app)
        resp = client.post("/api/cases/npu/tdd/fp32/verify")
        assert resp.status_code == 200
        # Files exist in registry but not in store, so results may show failures
        data = resp.get_json()
        assert isinstance(data, list)


class TestVerifyAll:
    def test_verify_all(self, client, app):
        _register(client, app)
        resp = client.post("/api/verify")
        assert resp.status_code == 200


class TestRebuildCache:
    def test_rebuild(self, client, app):
        _register(client, app)
        resp = client.post("/api/rebuild-cache")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["rebuilt"] >= 1


class TestVersions:
    def test_list_versions(self, client, app):
        _register(client, app)
        resp = client.get("/api/cases/npu/tdd/fp32/versions")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "v1" in data["versions"]
