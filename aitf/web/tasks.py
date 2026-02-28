"""Lightweight in-memory background task runner with progress tracking."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field


@dataclass
class Task:
    id: str
    status: str = "running"  # running | done | failed
    step: str = ""
    done: int = 0
    total: int = 0
    error: str = ""
    logs: list[str] = field(default_factory=list)

    def progress(self, done: int, total: int, step: str) -> None:
        self.done, self.total, self.step = done, total, step
        self.logs.append(f"[{done}/{total}] {step}")


_tasks: dict[str, Task] = {}


def submit(fn) -> Task:
    task = Task(id=uuid.uuid4().hex[:12])
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

    threading.Thread(target=_run, daemon=True).start()
    return task


def get(task_id: str) -> Task | None:
    return _tasks.get(task_id)
