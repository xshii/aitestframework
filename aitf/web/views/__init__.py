"""Shared helpers for web views."""

from __future__ import annotations

from pathlib import Path

from flask import current_app


def size_display(size: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def log_root() -> Path:
    """Resolved path to the log/report directory."""
    return Path(current_app.config.get("LOG_ROOT", "build/reports")).resolve()


def build_log_listing(directory: Path) -> list[dict]:
    """Build a sorted list of directory entries with metadata."""
    if not directory.is_dir():
        return []
    entries = []
    for child in sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
        is_dir = child.is_dir()
        entries.append({
            "name": child.name,
            "path": str(child.relative_to(log_root())),
            "is_dir": is_dir,
            "size_display": "" if is_dir else size_display(child.stat().st_size),
        })
    return entries
