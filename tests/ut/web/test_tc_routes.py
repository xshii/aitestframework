"""Tests for tc (test-case management) routes."""

from __future__ import annotations

import json

import pytest

from aitf.config import AitfConfig
from aitf.web.app import create_app


@pytest.fixture()
def project(tmp_path):
    """Project root with a minimal cases dir."""
    root = tmp_path / "project"
    root.mkdir()
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


class TestSuites:
    def test_list_suites_empty(self, client):
        resp = client.get("/api/suites")
        assert resp.status_code == 200
        assert resp.get_json() == []

    def test_refresh_and_list(self, client):
        resp = client.post("/api/suites/refresh",
                           content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["count"] >= 1

        resp = client.get("/api/suites")
        suites = resp.get_json()
        assert len(suites) >= 1
        assert suites[0]["class_name"] == "TestDemo"


class TestExecutions:
    def test_list_empty(self, client):
        resp = client.get("/api/executions")
        assert resp.status_code == 200
        assert resp.get_json() == []

    def test_get_missing(self, client):
        resp = client.get("/api/executions/nonexistent")
        assert resp.status_code == 404


class TestOverview:
    def test_overview_empty_db(self, client):
        resp = client.get("/api/tc/overview")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["latest_execution"] is None
        assert data["suite_stats"]["total_suites"] == 0
        assert data["fail_top10"] == []


class TestOptions:
    def test_options(self, client):
        resp = client.get("/api/tc/options")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "bundles" in data
        assert "golden_models" in data
        assert "testplans" in data


class TestPlan:
    def test_save_and_get_plan(self, client):
        resp = client.post(
            "/api/tc/plan",
            data=json.dumps({
                "name": "test plan",
                "filename": "testplan_unit",
                "plans": [{"name": "g1", "tests": ["test_one"]}],
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True
        assert resp.get_json()["filename"] == "testplan_unit.yaml"

        resp = client.get("/api/tc/plan/testplan_unit.yaml")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["name"] == "test plan"
        assert len(data["plans"]) == 1

    def test_save_empty_plans(self, client):
        resp = client.post(
            "/api/tc/plan",
            data=json.dumps({"plans": []}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_save_bad_filename(self, client):
        resp = client.post(
            "/api/tc/plan",
            data=json.dumps({
                "filename": "bad name!@#",
                "plans": [{"name": "g1", "tests": ["t"]}],
            }),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_delete_plan(self, client):
        # Create then delete
        client.post(
            "/api/tc/plan",
            data=json.dumps({
                "filename": "testplan_del",
                "plans": [{"name": "g1", "tests": ["t"]}],
            }),
            content_type="application/json",
        )
        resp = client.delete("/api/tc/plan/testplan_del.yaml")
        assert resp.status_code == 200

        resp = client.get("/api/tc/plan/testplan_del.yaml")
        assert resp.status_code == 404

    def test_get_missing_plan(self, client):
        resp = client.get("/api/tc/plan/nonexistent.yaml")
        assert resp.status_code == 404


class TestWebhook:
    def test_webhook_missing_event(self, client):
        resp = client.post(
            "/api/webhook",
            data=json.dumps({"event": "unknown"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_webhook_missing_execution_id(self, client):
        resp = client.post(
            "/api/webhook",
            data=json.dumps({"event": "execution_complete"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_webhook_creates_execution(self, client):
        resp = client.post(
            "/api/webhook",
            data=json.dumps({
                "event": "execution_complete",
                "execution_id": "wh-test-001",
                "summary": {"total": 10, "pass": 8, "fail": 2},
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"

        # Verify execution exists
        resp = client.get("/api/executions/wh-test-001")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total"] == 10
        assert data["passed"] == 8

    def test_webhook_token_required(self, app, client):
        app.config["WEBHOOK_TOKEN"] = "secret123"
        resp = client.post(
            "/api/webhook",
            data=json.dumps({
                "event": "execution_complete",
                "execution_id": "wh-test-002",
            }),
            content_type="application/json",
        )
        assert resp.status_code == 403

        resp = client.post(
            "/api/webhook",
            data=json.dumps({
                "event": "execution_complete",
                "execution_id": "wh-test-002",
                "summary": {"total": 1},
            }),
            content_type="application/json",
            headers={"X-Webhook-Token": "secret123"},
        )
        assert resp.status_code == 200


class TestQueue:
    def test_queue_status_empty(self, client):
        resp = client.get("/api/queue")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["targets"] == {}
        assert data["recent"] == []

    def test_submit_and_queue_status(self, client):
        # Refresh suites first so tests can be found
        client.post("/api/suites/refresh", content_type="application/json")

        resp = client.post(
            "/api/run",
            data=json.dumps({"paths": ["npu/operators/test_demo.py"], "target": "local"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert "queue_item_id" in data

    def test_cancel_nonexistent(self, client):
        resp = client.post("/api/queue/nonexistent/cancel",
                           content_type="application/json")
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is False


class TestTimeline:
    def test_timeline_empty(self, client):
        resp = client.get("/api/executions/timeline?hours=24")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "lanes" in data
        assert "since" in data
        assert "now" in data


class TestCaseHistory:
    def test_history_empty(self, client):
        resp = client.get("/api/cases/TestDemo/test_one/history")
        assert resp.status_code == 200
        assert resp.get_json() == []


class TestRerunFailed:
    def test_rerun_nonexistent(self, client):
        resp = client.post("/api/executions/nonexistent/rerun-failed",
                           content_type="application/json",
                           data="{}")
        assert resp.status_code == 404


class TestCleanup:
    def test_cleanup_no_old_data(self, client):
        resp = client.post(
            "/api/executions/cleanup",
            data=json.dumps({"keep_days": 30}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["deleted"] == 0


class TestCSRF:
    def test_post_with_text_plain_rejected(self, client):
        """POST with text/plain Content-Type and a body should be rejected."""
        resp = client.post(
            "/api/suites/refresh",
            data="hello",
            content_type="text/plain",
        )
        assert resp.status_code == 415

    def test_post_json_accepted(self, client):
        resp = client.post(
            "/api/suites/refresh",
            content_type="application/json",
        )
        assert resp.status_code == 200
