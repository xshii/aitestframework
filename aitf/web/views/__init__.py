"""Shared helpers for web views."""

from __future__ import annotations

from flask import current_app

from aitf.deps.bundle import BundleManager
from aitf.deps.manager import DepsManager


def get_deps_manager() -> DepsManager:
    """Return the shared DepsManager instance, creating on first call."""
    if "deps_manager" not in current_app.config:
        current_app.config["deps_manager"] = DepsManager()
    return current_app.config["deps_manager"]


def get_bundle_manager() -> BundleManager:
    return BundleManager(get_deps_manager())


def size_display(size: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
