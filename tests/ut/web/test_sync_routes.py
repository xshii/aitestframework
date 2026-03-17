"""Tests for sync API routes (server-side)."""

from __future__ import annotations

import json

import pytest

from aitf.config import AitfConfig
from aitf.web.app import create_app


@pytest.fixture()
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "build").mkdir()
    return root


@pytest.fixture()
def app(project):
    base = project / "datastore"
    base.mkdir()
    app = create_app(
        config={"TESTING": True, "DATASTORE_BASE_DIR": str(base)},
        aitf_config=AitfConfig(project_root=project),
    )
    yield app


@pytest.fixture()
def client(app):
    return app.test_client()


class TestPing:
    def test_ping(self, client):
        resp = client.get("/api/sync/ping")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert "hostname" in data


class TestResults:
    def test_receive_results(self, client):
        payload = {
            "id": "sync-test-001",
            "total": 5,
            "passed": 3,
            "failed": 2,
            "started_at": "2026-01-01T00:00:00",
            "cases": [
                {"suite_class": "TestA", "case_method": "test_1",
                 "status": "PASS"},
                {"suite_class": "TestA", "case_method": "test_2",
                 "status": "FAIL", "failure_reason": "assert failed"},
            ],
        }
        resp = client.post(
            "/api/sync/results",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

    def test_receive_results_dedup(self, client):
        payload = {"id": "sync-dedup-001", "total": 1, "passed": 1, "cases": []}
        client.post("/api/sync/results", data=json.dumps(payload),
                    content_type="application/json")
        resp = client.post("/api/sync/results", data=json.dumps(payload),
                           content_type="application/json")
        assert resp.status_code == 200
        assert "already exists" in resp.get_json().get("msg", "")

    def test_receive_invalid(self, client):
        resp = client.post("/api/sync/results", data=json.dumps({}),
                           content_type="application/json")
        assert resp.status_code == 400

    def test_serve_results(self, client):
        resp = client.get("/api/sync/results")
        assert resp.status_code == 200


class TestPlans:
    def test_serve_plans_empty(self, client):
        resp = client.get("/api/sync/plans")
        assert resp.status_code == 200
        assert resp.get_json() == []

    def test_upload_and_serve_plan(self, client):
        plan = {"name": "test plan", "plans": [{"name": "g1", "tests": ["t1"]}]}
        resp = client.post(
            "/api/sync/plans/upload",
            data=json.dumps({"filename": "testplan_sync_test", "plan": plan}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

        resp = client.get("/api/sync/plans/testplan_sync_test.yaml")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["name"] == "test plan"

    def test_upload_invalid_filename(self, client):
        resp = client.post(
            "/api/sync/plans/upload",
            data=json.dumps({"filename": "bad file!@#", "plan": {"a": 1}}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_path_traversal_rejected(self, client):
        resp = client.get("/api/sync/plans/../../etc/passwd")
        assert resp.status_code in (400, 404)
