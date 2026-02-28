"""SFTP client for pulling/pushing case data to a remote server."""

from __future__ import annotations

import logging
import os
import stat
import time
from pathlib import Path, PurePosixPath
from typing import Callable

import paramiko

from aitf.ds.types import DataType, RemoteConfig, SyncError, SyncResult

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY = 5  # seconds
_PULL_TYPES = (DataType.WEIGHTS.value, DataType.INPUTS.value, DataType.GOLDEN.value)
_PUSH_TYPES = tuple(dt.value for dt in DataType)


class SftpClient:
    """Managed SFTP connection for syncing case data."""

    def __init__(self, config: RemoteConfig) -> None:
        self._config = config
        self._transport: paramiko.Transport | None = None
        self._sftp: paramiko.SFTPClient | None = None

    def connect(self) -> None:
        """Open the SSH transport and SFTP channel."""
        self._transport = paramiko.Transport((self._config.host, self._config.port))
        if self._config.ssh_key:
            pkey = paramiko.RSAKey.from_private_key_file(self._config.ssh_key)
            self._transport.connect(username=self._config.user, pkey=pkey)
        else:
            self._transport.connect(username=self._config.user)
        self._sftp = paramiko.SFTPClient.from_transport(self._transport)

    def close(self) -> None:
        """Close the SFTP channel and transport."""
        if self._sftp:
            self._sftp.close()
            self._sftp = None
        if self._transport:
            self._transport.close()
            self._transport = None

    def __enter__(self) -> SftpClient:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def sftp(self) -> paramiko.SFTPClient:
        if self._sftp is None:
            raise SyncError("SFTP client not connected")
        return self._sftp

    # -- internal helpers ----------------------------------------------------

    def _mkdir_p(self, remote_dir: str) -> None:
        parts = PurePosixPath(remote_dir).parts
        current = ""
        for part in parts:
            current = f"{current}/{part}" if current else part
            if current == "/":
                continue
            try:
                self.sftp.stat(current)
            except FileNotFoundError:
                self.sftp.mkdir(current)

    def _retry(self, fn: Callable[[], None], label: str) -> None:
        """Call *fn* with retries."""
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                fn()
                return
            except Exception as exc:
                if attempt == _MAX_RETRIES:
                    raise SyncError(
                        f"Failed after {_MAX_RETRIES} attempts: {label}: {exc}"
                    ) from exc
                logger.warning(
                    "Attempt %d/%d failed for %s: %s — retrying in %ds",
                    attempt, _MAX_RETRIES, label, exc, _RETRY_DELAY,
                )
                time.sleep(_RETRY_DELAY)

    def _do_transfer(
        self,
        result: SyncResult,
        fn: Callable[[], None],
        local_path: str,
        label: str,
    ) -> None:
        """Run a transfer, accumulating stats into *result*."""
        try:
            self._retry(fn, label)
            result.files_transferred += 1
            result.bytes_transferred += os.path.getsize(local_path)
        except SyncError as exc:
            result.files_failed += 1
            result.errors.append(str(exc))

    # -- public API ----------------------------------------------------------

    def pull_case(
        self,
        case_id: str,
        local_store: str | Path,
        remote_base: str,
        data_types: tuple[str, ...] = _PULL_TYPES,
    ) -> SyncResult:
        """Download case files from the remote server."""
        result = SyncResult(case_id=case_id, direction="pull")

        for dtype in data_types:
            remote_dir = f"{remote_base}/{case_id}/{dtype}"
            local_dir = Path(local_store) / case_id / dtype

            try:
                entries = self.sftp.listdir_attr(remote_dir)
            except FileNotFoundError:
                continue

            for entry in entries:
                if stat.S_ISDIR(entry.st_mode or 0):
                    continue
                local_file = local_dir / entry.filename

                if local_file.exists() and local_file.stat().st_size == (entry.st_size or 0):
                    result.files_skipped += 1
                    continue

                remote_file = f"{remote_dir}/{entry.filename}"
                lf = str(local_file)

                def do_get(r=remote_file, l=lf):  # noqa: E741
                    Path(l).parent.mkdir(parents=True, exist_ok=True)
                    self.sftp.get(r, l)

                self._do_transfer(result, do_get, lf, lf)

        return result

    def push_case(
        self,
        case_id: str,
        local_store: str | Path,
        remote_base: str,
        data_types: tuple[str, ...] = _PUSH_TYPES,
        src_override: str | Path | None = None,
    ) -> SyncResult:
        """Upload case files to the remote server."""
        result = SyncResult(case_id=case_id, direction="push")

        for dtype in data_types:
            local_dir = Path(src_override) if src_override else Path(local_store) / case_id / dtype
            if not local_dir.is_dir():
                continue
            remote_dir = f"{remote_base}/{case_id}/{dtype}"

            for fp in sorted(local_dir.rglob("*")):
                if not fp.is_file():
                    continue
                rel = fp.relative_to(local_dir)
                remote_file = f"{remote_dir}/{rel}"
                lf = str(fp)

                def do_put(r=remote_file, l=lf):  # noqa: E741
                    self._mkdir_p(str(PurePosixPath(r).parent))
                    self.sftp.put(l, r)

                self._do_transfer(result, do_put, lf, lf)

        return result
