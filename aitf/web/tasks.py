"""Lightweight in-memory background task runner with progress tracking."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field


_MAX_FINISHED_TASKS = 200
_FINISHED_TTL = 3600  # seconds


@dataclass
class Task:
    id: str
    status: str = "running"  # running | done | failed
    step: str = ""
    done: int = 0
    total: int = 0
    error: str = ""
    logs: list[str] = field(default_factory=list)
    finished_at: float = 0.0

    def progress(self, done: int, total: int, step: str) -> None:
        self.done, self.total, self.step = done, total, step
        self.logs.append(f"[{done}/{total}] {step}")


_tasks: dict[str, Task] = {}
_lock = threading.Lock()


def _cleanup() -> None:
    """Remove expired finished tasks, keeping at most _MAX_FINISHED_TASKS."""
    now = time.monotonic()
    expired = [
        tid for tid, t in _tasks.items()
        if t.status in ("done", "failed")
        and t.finished_at > 0
        and now - t.finished_at > _FINISHED_TTL
    ]
    for tid in expired:
        del _tasks[tid]

    finished = [
        (tid, t.finished_at)
        for tid, t in _tasks.items()
        if t.status in ("done", "failed") and t.finished_at > 0
    ]
    if len(finished) > _MAX_FINISHED_TASKS:
        finished.sort(key=lambda x: x[1])
        for tid, _ in finished[: len(finished) - _MAX_FINISHED_TASKS]:
            _tasks.pop(tid, None)


def submit(fn) -> Task:
    task = Task(id=uuid.uuid4().hex[:12])
    with _lock:
        _cleanup()
        _tasks[task.id] = task

    def _run():
        try:
            fn(task)
            task.status = "done"
            task.logs.append("done")
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.logs.append(f"error: {e}")
        finally:
            task.finished_at = time.monotonic()

    threading.Thread(target=_run, daemon=True).start()
    return task


def get(task_id: str) -> Task | None:
    return _tasks.get(task_id)
