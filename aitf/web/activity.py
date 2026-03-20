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


def _get_client_ip() -> str:
    """Best-effort client IP from Flask request context."""
    try:
        from flask import request
        xff = request.headers.get("X-Forwarded-For", "")
        if xff:
            return xff.split(",")[0].strip()  # first IP = real client
        return request.remote_addr or ""
    except (ImportError, RuntimeError):
        return ""


def log_activity(action: str, detail: str = "", **extra) -> None:
    """Append an activity entry to the ring buffer.

    Automatically records the client IP from the current Flask request.
    """
    ip = extra.pop("ip", None) or _get_client_ip()
    if ip:
        extra["ip"] = ip
    entry = ActivityEntry(ts=time.time(), action=action, detail=detail, extra=extra)
    with _lock:
        _log.append(asdict(entry))


def get_activity(limit: int = _MAX) -> list[dict]:
    """Return recent activity entries, newest first."""
    with _lock:
        items = list(_log)
    items.reverse()
    return items[:limit]
