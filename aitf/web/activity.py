"""Activity log — ring buffer of recent user actions."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field

_MAX = 100
_lock = threading.Lock()
_log: deque[dict] = deque(maxlen=_MAX)


@dataclass
class ActivityEntry:
    ts: float
    action: str       # e.g. "run", "sync", "target.add"
    detail: str = ""  # human-readable summary
    extra: dict = field(default_factory=dict)


def log_activity(action: str, detail: str = "", **extra) -> None:
    """Append an activity entry to the ring buffer."""
    entry = ActivityEntry(ts=time.time(), action=action, detail=detail, extra=extra)
    with _lock:
        _log.append(asdict(entry))


def get_activity(limit: int = _MAX) -> list[dict]:
    """Return recent activity entries, newest first."""
    with _lock:
        items = list(_log)
    items.reverse()
    return items[:limit]
