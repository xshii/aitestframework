"""DataStoreManager — orchestrates registry, cache, integrity, and SFTP."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import yaml
from sqlalchemy.orm import Session

from aitf.comm.db import get_session, init_db
from aitf.ds import cache, integrity, registry
from aitf.ds.sftp import SftpClient
from aitf.ds.types import (
    CaseData,
    CaseNotFoundError,
    DataType,
    RemoteConfig,
    SyncResult,
    VerifyResult,
)

logger = logging.getLogger(__name__)


@contextmanager
def _session() -> Iterator[Session]:
    s = get_session()
    try:
        yield s
    finally:
        s.close()


class DataStoreManager:
    """Central facade for all data-store operations.

    Args:
        base_dir: Root directory containing ``registry/`` and ``store/``.
        db_path: Path to the SQLite database file.
    """

    def __init__(
        self,
        base_dir: str | Path = "datastore",
        db_path: str = "data/aitf.db",
    ) -> None:
        self._base = Path(base_dir)
        self._registry_dir = self._base / "registry"
        self._store_dir = self._base / "store"
        self._registry_dir.mkdir(parents=True, exist_ok=True)
        self._store_dir.mkdir(parents=True, exist_ok=True)

        init_db(db_path)

        # Auto-rebuild cache when SQLite is empty but YAML has data
        from aitf.comm.models import CaseDataRow

        with _session() as s:
            if s.query(CaseDataRow).count() == 0:
                yaml_cases = registry.load_all_cases(self._registry_dir)
                if yaml_cases:
                    logger.info("Auto-rebuilding cache from YAML (%d cases)", len(yaml_cases))
                    cache.rebuild_cache(s, self._registry_dir)

    # -- registry / cache ----------------------------------------------------

    def register(self, case_id: str, local_path: str | Path) -> CaseData:
        """Scan a local directory and register it as a case."""
        parts = case_id.strip("/").split("/")
        if len(parts) != 3:
            raise ValueError(f"case_id must be <platform>/<model>/<variant>, got: {case_id}")

        lp = Path(local_path)

        files: dict[str, list] = {}
        for dtype in DataType:
            entries = integrity.scan_directory(lp, dtype)
            if entries:
                files[dtype] = entries

        now = datetime.now(timezone.utc)
        case = CaseData(
            case_id=case_id,
            name=f"{parts[1]}/{parts[2]}",
            files=files,
            created_at=now,
            updated_at=now,
        )

        registry.save_case(self._registry_dir, case)
        with _session() as s:
            cache.upsert_case(s, case)
        return case

    def get(self, case_id: str) -> CaseData:
        """Retrieve a case by its identifier."""
        with _session() as s:
            result = cache.query_case(s, case_id)
        if result is None:
            raise CaseNotFoundError(case_id)
        return result

    def delete(self, case_id: str) -> None:
        """Delete a case from both YAML and SQLite."""
        if not registry.delete_case(self._registry_dir, case_id):
            raise CaseNotFoundError(case_id)
        with _session() as s:
            cache.delete_case(s, case_id)

    def list(
        self,
        platform: str | None = None,
        model: str | None = None,
    ) -> list[CaseData]:
        """List cases with optional filters."""
        with _session() as s:
            return cache.query_cases(s, platform=platform, model=model)

    def rebuild_cache(self) -> int:
        """Rebuild the SQLite cache from YAML."""
        with _session() as s:
            return cache.rebuild_cache(s, self._registry_dir)

    # -- integrity -----------------------------------------------------------

    def verify(self, case_id: str | None = None) -> list[VerifyResult]:
        """Verify file checksums for one or all cases."""
        if case_id:
            case = self.get(case_id)
            return integrity.verify_case(self._store_dir / case_id, case)

        results: list[VerifyResult] = []
        for case in self.list():
            results.extend(integrity.verify_case(self._store_dir / case.case_id, case))
        return results

    # -- SFTP ----------------------------------------------------------------

    def _load_remote(self, remote_name: str) -> RemoteConfig:
        cfg_path = self._base / "remote.yaml"
        if not cfg_path.exists():
            raise ValueError(f"remote.yaml not found in {self._base}")
        with open(cfg_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        remotes = data.get("remotes", {})
        if remote_name not in remotes:
            raise ValueError(f"Remote '{remote_name}' not found in remote.yaml")
        r = remotes[remote_name]
        auth = r.get("auth", {})
        return RemoteConfig(
            name=remote_name,
            host=r["host"],
            user=r["user"],
            path=r.get("path", "/"),
            port=r.get("port", 22),
            ssh_key=auth.get("key_file"),
        )

    def pull(
        self,
        remote_name: str,
        case_id: str | None = None,
        platform: str | None = None,
        model: str | None = None,
    ) -> list[SyncResult]:
        """Pull case data from a remote server."""
        remote = self._load_remote(remote_name)
        cases = [self.get(case_id)] if case_id else self.list(platform=platform, model=model)
        results: list[SyncResult] = []
        with SftpClient(remote) as client:
            for case in cases:
                results.append(client.pull_case(case.case_id, str(self._store_dir), remote.path))
        return results

    def push(self, remote_name: str, case_id: str) -> SyncResult:
        """Push a case to a remote server."""
        remote = self._load_remote(remote_name)
        self.get(case_id)  # ensure exists
        with SftpClient(remote) as client:
            return client.push_case(case_id, str(self._store_dir), remote.path)

    def push_artifacts(
        self,
        remote_name: str,
        case_id: str,
        artifacts_dir: str | Path | None = None,
    ) -> SyncResult:
        """Push only artifacts for a case."""
        remote = self._load_remote(remote_name)
        self.get(case_id)  # ensure exists
        with SftpClient(remote) as client:
            return client.push_case(
                case_id, str(self._store_dir), remote.path,
                data_types=(DataType.ARTIFACTS,),
                src_override=artifacts_dir,
            )

    # -- path helper ---------------------------------------------------------

    def get_dir(self, case_id: str, data_type: str) -> Path:
        """Return the absolute path to a case's data sub-directory."""
        return (self._store_dir / case_id / data_type).resolve()
