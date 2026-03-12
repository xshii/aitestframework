"""Unified object factories for Flask request context.

All route modules should use these instead of creating their own instances.
Objects are cached per-request via Flask ``g``.
"""

from __future__ import annotations

from flask import current_app, g


def get_golden_store():
    """Return a GoldenStore (or CachedGoldenStore in CLIENT mode), cached per request."""
    if "golden_store" not in g:
        sc = current_app.config.get("SYNC_CLIENT")
        base = current_app.config.get("DATASTORE_BASE_DIR", "datastore")
        if sc:
            from aitf.sync.cache import CachedGoldenStore
            g.golden_store = CachedGoldenStore(base, sc)
        else:
            from aitf.ds.store import GoldenStore
            g.golden_store = GoldenStore(base)
    return g.golden_store


def get_deps_manager():
    """Return a DepsManager, cached per request."""
    if "deps_manager" not in g:
        from aitf.deps.manager import DepsManager
        cfg = current_app.config.get("AITF_CONFIG")
        if cfg:
            g.deps_manager = DepsManager(
                project_root=str(cfg.project_root),
                build_dir=str(cfg.build_root),
            )
        else:
            g.deps_manager = DepsManager()
    return g.deps_manager


def get_bundle_manager():
    """Return a BundleManager wrapping the shared DepsManager, cached per request."""
    if "bundle_manager" not in g:
        from aitf.deps.bundle import BundleManager
        g.bundle_manager = BundleManager(get_deps_manager())
    return g.bundle_manager


def get_project_root():
    """Return the project root Path."""
    from pathlib import Path
    cfg = current_app.config.get("AITF_CONFIG")
    return cfg.project_root if cfg else Path(".")
