"""Integration tests: debug mode sync across all three modules.

In debug mode, the server syncs from itself.  urllib is patched in
conftest so all sync HTTP calls route through the WSGI test app.
"""

from __future__ import annotations

import io
import json
import zipfile


# ---------------------------------------------------------------------------
# Golden sync (data tab)
# ---------------------------------------------------------------------------

class TestGoldenSyncDebugMode:

    def _seed_golden(self, client):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("input.bin", b"\x00" * 16)
        buf.seek(0)
        resp = client.post("/api/golden/upload", data={
            "model": "testmodel", "version": "v1", "operator": "op_add",
            "file": (buf, "input.bin"),
        }, content_type="multipart/form-data")
        assert resp.status_code in (200, 201)

    def test_sync_empty_source_auto_fills(self, debug_client):
        """Empty source should auto-fill from config, not error."""
        self._seed_golden(debug_client)
        resp = debug_client.post(
            "/api/golden/sync",
            data=json.dumps({"source": ""}),
            content_type="application/json",
        )
        data = resp.get_json()
        assert resp.status_code == 200
        assert "error" not in data
        assert isinstance(data["count"], int)
        assert data["count"] >= 0

    def test_standalone_sync_returns_error(self, standalone_client):
        resp = standalone_client.post(
            "/api/golden/sync",
            data=json.dumps({"source": ""}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "error" in resp.get_json()


# ---------------------------------------------------------------------------
# Deps sync (deps tab)
# ---------------------------------------------------------------------------

class TestDepsSyncDebugMode:

    def test_sync_returns_total(self, debug_client):
        resp = debug_client.post("/api/deps/sync")
        data = resp.get_json()
        assert resp.status_code == 200
        assert "error" not in data, f"Unexpected error: {data.get('error')}"
        assert data["total"] == 2  # npu-compiler + json-c

    def test_standalone_returns_error(self, standalone_client):
        resp = standalone_client.post("/api/deps/sync")
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_sync_preserves_deps(self, debug_client):
        debug_client.post("/api/deps/sync")
        resp = debug_client.get("/api/deps")
        names = {d["name"] for d in resp.get_json()}
        assert "npu-compiler" in names
        assert "json-c" in names

    def test_sync_no_deps_yaml_returns_404_chinese(self, debug_no_deps_client):
        """When server has no deps.yaml, should return Chinese 404 message."""
        resp = debug_no_deps_client.post("/api/deps/sync")
        assert resp.status_code == 404
        msg = resp.get_json()["error"]
        assert "deps.yaml" in msg


# ---------------------------------------------------------------------------
# TC sync (cases tab)
# ---------------------------------------------------------------------------

class TestTcSyncDebugMode:

    def _seed_plan_on_server(self, client):
        plan = {"name": "from server", "plans": [{"name": "g1", "tests": ["t1"]}]}
        client.post(
            "/api/sync/plans/upload",
            data=json.dumps({"filename": "testplan_from_server", "plan": plan}),
            content_type="application/json",
        )

    def test_sync_plans(self, debug_client):
        self._seed_plan_on_server(debug_client)
        resp = debug_client.post(
            "/api/tc/sync_plans", content_type="application/json")
        data = resp.get_json()
        assert resp.status_code == 200
        assert "error" not in data
        assert "total" in data
        assert "saved" in data

    def test_sync_results(self, debug_client):
        resp = debug_client.post(
            "/api/tc/sync_results", content_type="application/json")
        data = resp.get_json()
        assert resp.status_code == 200
        assert "error" not in data
        assert "total" in data
        assert "imported" in data

    def test_standalone_returns_error(self, standalone_client):
        resp = standalone_client.post(
            "/api/tc/sync_plans", content_type="application/json")
        assert resp.status_code == 400
        assert "error" in resp.get_json()


# ---------------------------------------------------------------------------
# Error message consistency — all Chinese
# ---------------------------------------------------------------------------

class TestSyncErrorConsistency:

    def test_golden_error_is_chinese(self, standalone_client):
        resp = standalone_client.post(
            "/api/golden/sync",
            data=json.dumps({"source": ""}),
            content_type="application/json",
        )
        msg = resp.get_json()["error"]
        assert "请提供" in msg or "配置" in msg

    def test_deps_error_is_chinese(self, standalone_client):
        msg = standalone_client.post("/api/deps/sync").get_json()["error"]
        assert "客户端" in msg or "模式" in msg

    def test_tc_error_is_chinese(self, standalone_client):
        resp = standalone_client.post(
            "/api/tc/sync_plans", content_type="application/json")
        msg = resp.get_json()["error"]
        assert "服务器" in msg or "配置" in msg


# ---------------------------------------------------------------------------
# Cross-module: upload → sync round-trip
# ---------------------------------------------------------------------------

class TestCrossModuleSync:

    def test_golden_upload_then_sync(self, debug_client):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("data.bin", b"\x01" * 32)
        buf.seek(0)
        debug_client.post("/api/golden/upload", data={
            "model": "m1", "version": "v1", "operator": "op1",
            "file": (buf, "data.bin"),
        }, content_type="multipart/form-data")

        resp = debug_client.post(
            "/api/golden/sync",
            data=json.dumps({"source": ""}),
            content_type="application/json",
        )
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["count"] >= 1

    def test_execution_upload_then_sync(self, debug_client):
        """Upload execution via server API, then sync_results picks it up."""
        payload = {
            "id": "cross-test-001",
            "total": 5, "passed": 3, "failed": 2,
            "started_at": "2026-01-01T00:00:00",
            "cases": [
                {"suite_class": "TestA", "case_method": "test_1", "status": "PASS"},
            ],
        }
        debug_client.post(
            "/api/sync/results",
            data=json.dumps(payload),
            content_type="application/json",
        )

        resp = debug_client.post(
            "/api/tc/sync_results", content_type="application/json")
        data = resp.get_json()
        assert resp.status_code == 200
        # Total should reflect the execution we uploaded
        assert data["total"] >= 1
