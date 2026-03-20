"""Tests for ExecutionQueue — state management, persistence, pool resolution."""

from __future__ import annotations

import json
import time

import pytest

from aitf.tc.queue import ExecutionQueue, QueueItem, QueueStatus, Trigger


@pytest.fixture()
def queue(tmp_path):
    """Create a queue with tmp dirs."""
    cases = tmp_path / "cases"
    cases.mkdir()
    db = tmp_path / "test.db"
    return ExecutionQueue(str(cases), str(db), str(tmp_path))


class TestQueueBasics:
    def test_submit_creates_item(self, queue):
        item = queue.submit("local", ["test_demo.py"])
        assert item.status == QueueStatus.QUEUED or item.status == QueueStatus.RUNNING
        assert item.target == "local"
        assert item.trigger == Trigger.MANUAL

    def test_cancel_queued(self, queue):
        # Submit two items to same target — second will be queued
        queue.submit("tgt1", ["a.py"], label="first")
        item2 = queue.submit("tgt1", ["b.py"], label="second")
        # First is running, second should be queued
        status = queue.get_status()
        queued = status["targets"].get("tgt1", {}).get("queued", [])
        if queued:
            assert queue.cancel(item2.id) is True
            assert item2.status == QueueStatus.CANCELLED

    def test_cancel_nonexistent(self, queue):
        assert queue.cancel("nonexistent") is False

    def test_get_status_structure(self, queue):
        status = queue.get_status()
        assert "targets" in status
        assert "recent" in status
        assert isinstance(status["targets"], dict)
        assert isinstance(status["recent"], list)


class TestQueuePersistence:
    def test_save_and_load(self, tmp_path):
        cases = tmp_path / "cases"
        cases.mkdir()
        db = tmp_path / "test.db"

        q1 = ExecutionQueue(str(cases), str(db), str(tmp_path))
        q1.submit("local", ["test.py"], label="persisted")
        # Wait a moment for the item to start (or stay queued)
        time.sleep(0.1)

        # Create a new queue instance — should restore state
        q2 = ExecutionQueue(str(cases), str(db), str(tmp_path))
        status = q2.get_status()
        all_items = (
            [status["targets"][t]["running"] for t in status["targets"]
             if status["targets"][t]["running"]]
            + [q for t in status["targets"]
               for q in status["targets"][t]["queued"]]
            + status["recent"]
        )
        assert len(all_items) >= 1

    def test_running_items_marked_failed_on_restore(self, tmp_path):
        state_file = tmp_path / "data" / "build" / "queue_state.json"
        state_file.parent.mkdir(parents=True)
        state_file.write_text(json.dumps([{
            "id": "test123",
            "target": "local",
            "status": "running",
            "submitted_at": time.time(),
            "started_at": time.time(),
            "finished_at": 0,
            "paths": ["test.py"],
            "label": "was running",
        }]))

        cases = tmp_path / "cases"
        cases.mkdir()
        q = ExecutionQueue(str(cases), str(tmp_path / "test.db"), str(tmp_path))
        status = q.get_status()
        # Should be in recent as failed
        recent = status["recent"]
        assert any(r["id"] == "test123" and r["status"] == QueueStatus.FAILED
                   for r in recent)


class TestQueueItemSerialization:
    def test_to_dict_excludes_run_kwargs(self):
        item = QueueItem(id="abc", target="local", paths=["t.py"],
                         run_kwargs={"bundle": "v1"})
        d = item.to_dict()
        assert "run_kwargs" not in d
        assert d["target"] == "local"
        assert d["trigger"] == Trigger.MANUAL


class TestPoolResolution:
    def test_pool_resolves_to_member(self, tmp_path):
        # Create targets.yaml with a pool
        config_dir = tmp_path / "data" / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "targets.yaml").write_text(
            "fpga1:\n  pool: fpga_pool\n"
            "fpga2:\n  pool: fpga_pool\n"
        )

        cases = tmp_path / "cases"
        cases.mkdir()
        q = ExecutionQueue(str(cases), str(tmp_path / "t.db"), str(tmp_path))
        item = q.submit("fpga_pool", ["test.py"])
        assert item.target in ("fpga1", "fpga2")
        assert item.pool == "fpga_pool"

    def test_non_pool_target_unchanged(self, queue):
        item = queue.submit("specific-target", ["test.py"])
        assert item.target == "specific-target"
        assert item.pool == ""
