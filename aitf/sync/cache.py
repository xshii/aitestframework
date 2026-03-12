"""CachedGoldenStore — transparent remote-backed golden data with local cache."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from aitf.ds.store import GoldenStore

logger = logging.getLogger(__name__)

_MANIFEST = "sync_manifest.json"


class CachedGoldenStore(GoldenStore):
    """GoldenStore that transparently fetches from a remote server.

    Inherits all read/write methods from :class:`GoldenStore`.
    On read, if data is not found locally, fetches from the remote server
    and caches it.  Writes go to the local cache only — use
    :meth:`push_operator` to upload to the server.
    """

    def __init__(self, cache_dir: str | Path, sync_client) -> None:
        super().__init__(cache_dir)
        self._client = sync_client
        self._manifest_path = Path(cache_dir) / _MANIFEST
        self._manifest: dict = self._load_manifest()

    # -- manifest ----------------------------------------------------------

    def _load_manifest(self) -> dict:
        if self._manifest_path.is_file():
            try:
                return json.loads(self._manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_manifest(self) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
            json.dumps(self._manifest, indent=2), encoding="utf-8",
        )

    def _cache_key(self, model: str, version: str, operator: str) -> str:
        return f"{model}/{version}/{operator}"

    def _is_cached(self, model: str, version: str, operator: str) -> bool:
        key = self._cache_key(model, version, operator)
        if key not in self._manifest:
            return False
        # Also check that local dir actually exists
        d = self._root / model / version / operator
        return d.is_dir() and any(d.iterdir())

    def _mark_cached(self, model: str, version: str, operator: str) -> None:
        key = self._cache_key(model, version, operator)
        self._manifest[key] = {"cached_at": time.time()}
        self._save_manifest()

    # -- remote fetch ------------------------------------------------------

    def _ensure_operator(self, model: str, version: str, operator: str) -> None:
        """Fetch operator data from remote if not cached locally."""
        if self._is_cached(model, version, operator):
            return
        try:
            # Download zip and extract into our store root
            # The zip contains paths like model/version/operator/files
            self._client.golden_download_operator(
                model, version, operator,
                self._root.parent,  # extract to store parent so paths align
            )
            self._mark_cached(model, version, operator)
        except Exception as exc:
            logger.warning("Failed to fetch golden %s/%s/%s: %s",
                           model, version, operator, exc)

    def _ensure_version(self, model: str, version: str) -> None:
        """Fetch all operators for a version if none are cached."""
        d = self._root / model / version
        if d.is_dir() and any(d.iterdir()):
            return
        try:
            self._client.golden_download_version(
                model, version,
                self._root.parent,
            )
            # Mark all downloaded operators
            for op_dir in d.iterdir():
                if op_dir.is_dir():
                    self._mark_cached(model, version, op_dir.name)
        except Exception as exc:
            logger.warning("Failed to fetch golden %s/%s: %s",
                           model, version, exc)

    # -- override read methods to add remote fetch -------------------------

    def list_models(self) -> list[str]:
        # Merge local + remote
        local = super().list_models()
        try:
            remote = self._client.golden_list_models()
            return sorted(set(local) | set(remote))
        except Exception:
            return local

    def list_versions(self, model: str) -> list[str]:
        local = super().list_versions(model)
        try:
            remote = self._client.golden_list_versions(model)
            return sorted(set(local) | set(remote))
        except Exception:
            return local

    def list_operators(self, model: str, version: str) -> list[str]:
        # Try remote first to get full list
        try:
            remote = self._client.golden_list_operators(model, version)
            if remote:
                return remote
        except Exception:
            pass
        return super().list_operators(model, version)

    def get_operator_files(self, model: str, version: str,
                           operator: str) -> list[Path]:
        self._ensure_operator(model, version, operator)
        return super().get_operator_files(model, version, operator)

    def get_file(self, model: str, version: str, operator: str,
                 filename: str) -> Path | None:
        self._ensure_operator(model, version, operator)
        return super().get_file(model, version, operator, filename)

    def load_meta(self, model, version, operator):
        self._ensure_operator(model, version, operator)
        return super().load_meta(model, version, operator)

    # -- upload to server --------------------------------------------------

    def push_operator(self, model: str, version: str,
                      operator: str) -> bool:
        """Upload a local operator's files to the remote server."""
        files = super().get_operator_files(model, version, operator)
        if not files:
            logger.warning("No files to push for %s/%s/%s",
                           model, version, operator)
            return False
        try:
            self._client.golden_upload_files(model, version, operator, files)
            logger.info("Pushed golden %s/%s/%s (%d files)",
                        model, version, operator, len(files))
            return True
        except Exception as exc:
            logger.warning("Failed to push golden %s/%s/%s: %s",
                           model, version, operator, exc)
            return False

    # -- cache management --------------------------------------------------

    def invalidate(self, model: str, version: str,
                   operator: str | None = None) -> None:
        """Remove cached data, forcing re-fetch on next access."""
        import shutil

        if operator:
            key = self._cache_key(model, version, operator)
            self._manifest.pop(key, None)
            d = self._root / model / version / operator
            if d.is_dir():
                shutil.rmtree(d)
        else:
            # Invalidate entire version
            prefix = f"{model}/{version}/"
            keys_to_remove = [k for k in self._manifest if k.startswith(prefix)]
            for k in keys_to_remove:
                self._manifest.pop(k)
            d = self._root / model / version
            if d.is_dir():
                shutil.rmtree(d)
        self._save_manifest()

    def cache_stats(self) -> dict:
        """Return cache statistics."""
        total_size = 0
        count = 0
        if self._root.is_dir():
            for f in self._root.rglob("*"):
                if f.is_file() and f.name != _MANIFEST:
                    total_size += f.stat().st_size
                    count += 1
        return {
            "operators": len(self._manifest),
            "files": count,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
        }
