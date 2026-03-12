"""SyncWorker — background thread for uploading execution results."""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class SyncWorker:
    """Background worker that uploads local execution results to the server.

    Results are queued in a local JSON file for persistence across restarts.
    The worker thread periodically flushes the queue.
    """

    def __init__(self, sync_client, queue_dir: Path,
                 interval: int = 30, max_retries: int = 3) -> None:
        self._client = sync_client
        self._queue_path = queue_dir / "sync_queue.json"
        self._interval = interval
        self._max_retries = max_retries
        self._lock = threading.Lock()
        self._queue: list[dict] = self._load_queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- queue persistence -------------------------------------------------

    def _load_queue(self) -> list[dict]:
        if self._queue_path.is_file():
            try:
                return json.loads(self._queue_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def _save_queue(self) -> None:
        self._queue_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._queue_path.write_text(
                json.dumps(self._queue, indent=2), encoding="utf-8",
            )

    # -- public API --------------------------------------------------------

    def enqueue(self, execution_data: dict) -> None:
        """Add an execution result to the upload queue."""
        entry = {
            "data": execution_data,
            "retries": 0,
            "queued_at": time.time(),
            "hostname": socket.gethostname(),
        }
        with self._lock:
            # Deduplicate by execution_id
            eid = execution_data.get("id", "")
            self._queue = [e for e in self._queue
                           if e["data"].get("id") != eid]
            self._queue.append(entry)
        self._save_queue()
        logger.info("Enqueued execution %s for sync", eid)

    def start(self) -> None:
        """Start the background sync thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="aitf-sync-worker", daemon=True,
        )
        self._thread.start()
        logger.info("SyncWorker started (interval=%ds)", self._interval)

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._stop.set()

    def flush(self) -> int:
        """Manually flush the queue. Returns number of successfully uploaded."""
        return self._do_flush()

    def queue_size(self) -> int:
        with self._lock:
            return len(self._queue)

    def queue_info(self) -> list[dict]:
        """Return summary of queued items."""
        with self._lock:
            return [
                {
                    "execution_id": e["data"].get("id"),
                    "retries": e["retries"],
                    "queued_at": e["queued_at"],
                }
                for e in self._queue
            ]

    # -- internal ----------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._do_flush()
            except Exception:
                logger.exception("SyncWorker flush error")

    def _do_flush(self) -> int:
        with self._lock:
            pending = list(self._queue)

        if not pending:
            return 0

        succeeded = []
        failed = []

        for entry in pending:
            eid = entry["data"].get("id", "?")
            try:
                ok = self._client.upload_execution(entry["data"])
                if ok:
                    succeeded.append(entry)
                    logger.info("Synced execution %s", eid)
                else:
                    entry["retries"] += 1
                    failed.append(entry)
            except Exception as exc:
                entry["retries"] += 1
                logger.warning("Failed to sync %s (attempt %d): %s",
                               eid, entry["retries"], exc)
                failed.append(entry)

        # Remove succeeded and over-retried from queue
        with self._lock:
            self._queue = [
                e for e in failed
                if e["retries"] < self._max_retries
            ]

        dropped = [e for e in failed if e["retries"] >= self._max_retries]
        for e in dropped:
            logger.warning("Dropped execution %s after %d retries",
                           e["data"].get("id"), e["retries"])

        self._save_queue()
        return len(succeeded)
