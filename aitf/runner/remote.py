"""RemoteExecutor — run commands on a remote machine via SSH (paramiko)."""

from __future__ import annotations

import logging
import os
import stat
import time
from pathlib import Path, PurePosixPath

from .executor import ExecuteResult, TargetConfig

logger = logging.getLogger(__name__)

# Paramiko is an optional dependency — import lazily so the rest of the
# framework can work without it.
try:
    import paramiko
except ImportError:  # pragma: no cover
    paramiko = None  # type: ignore[assignment]

_PARAMIKO_MISSING = (
    "paramiko is required for remote execution but is not installed. "
    "Install it with:  pip install paramiko"
)


class RemoteExecutor:
    """Execute commands on a remote target over SSH."""

    def __init__(self, config: TargetConfig) -> None:
        if paramiko is None:
            raise ImportError(_PARAMIKO_MISSING)
        self.config = config
        self._client: paramiko.SSHClient | None = None
        self._sftp: paramiko.SFTPClient | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open an SSH connection to the remote host."""
        if self._client is not None:
            return

        cfg = self.config
        if not cfg.host:
            raise ValueError("TargetConfig.host must be set for remote targets")

        logger.info("Connecting to %s@%s:%d", cfg.user or "(default)", cfg.host, cfg.port)

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kw: dict = {
            "hostname": cfg.host,
            "port": cfg.port,
        }
        if cfg.user:
            connect_kw["username"] = cfg.user

        auth = cfg.auth or {}
        method = auth.get("method", "key")
        if method == "key":
            key_file = auth.get("key_file")
            if key_file:
                connect_kw["key_filename"] = os.path.expanduser(key_file)
        elif method == "password":
            password = auth.get("password")
            if password:
                connect_kw["password"] = password
        else:
            logger.warning("Unknown auth method %r; falling back to agent/defaults", method)

        try:
            client.connect(**connect_kw)
        except Exception as exc:
            raise ConnectionError(
                f"SSH connection to {cfg.host}:{cfg.port} failed: {exc}"
            ) from exc

        self._client = client
        logger.info("SSH connection established to %s:%d", cfg.host, cfg.port)

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def run(
        self,
        command: str,
        timeout: int = 300,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> ExecuteResult:
        """Execute *command* on the remote host.

        Pre-commands from the target config are prepended automatically.
        """
        if self._client is None:
            raise RuntimeError("Not connected — call connect() first")

        # Build the full command: pre_commands + env exports + cd + command.
        parts: list[str] = list(self.config.pre_commands)

        merged_env: dict[str, str] = {}
        merged_env.update(self.config.env)
        if env:
            merged_env.update(env)
        for k, v in merged_env.items():
            parts.append(f"export {k}={_shell_quote(v)}")

        work_dir = cwd or self.config.remote_dir
        if work_dir:
            parts.append(f"cd {_shell_quote(work_dir)}")

        parts.append(command)
        full_command = " && ".join(parts)

        logger.debug("RemoteExecutor.run: %s (timeout=%ds)", full_command, timeout)

        start = time.monotonic()
        try:
            _, stdout_ch, stderr_ch = self._client.exec_command(
                full_command, timeout=timeout
            )
            stdout_str = stdout_ch.read().decode("utf-8", errors="replace")
            stderr_str = stderr_ch.read().decode("utf-8", errors="replace")
            returncode = stdout_ch.channel.recv_exit_status()
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error("Remote command failed: %s", exc)
            return ExecuteResult(
                returncode=-1,
                stdout="",
                stderr=str(exc),
                duration_s=round(elapsed, 3),
            )

        elapsed = time.monotonic() - start

        result = ExecuteResult(
            returncode=returncode,
            stdout=stdout_str,
            stderr=stderr_str,
            duration_s=round(elapsed, 3),
        )
        logger.debug(
            "RemoteExecutor.run finished: rc=%d duration=%.3fs",
            result.returncode,
            result.duration_s,
        )
        return result

    # ------------------------------------------------------------------
    # File transfer
    # ------------------------------------------------------------------

    def upload(self, local_path: str, remote_path: str) -> None:
        """Upload a file or directory to the remote host via SFTP."""
        sftp = self._get_sftp()
        local = Path(local_path)

        if local.is_dir():
            self._upload_dir(sftp, local, PurePosixPath(remote_path))
        elif local.is_file():
            self._mkdir_p(sftp, str(PurePosixPath(remote_path).parent))
            logger.info("Uploading %s -> %s", local_path, remote_path)
            sftp.put(str(local), remote_path)
        else:
            raise FileNotFoundError(f"Local path does not exist: {local_path}")

    def download(self, remote_path: str, local_path: str) -> None:
        """Download a file from the remote host via SFTP."""
        sftp = self._get_sftp()
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading %s -> %s", remote_path, local_path)
        sftp.get(remote_path, str(local))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the SFTP and SSH sessions."""
        if self._sftp is not None:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            logger.info("SSH connection closed")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_sftp(self) -> paramiko.SFTPClient:
        if self._client is None:
            raise RuntimeError("Not connected — call connect() first")
        if self._sftp is None:
            self._sftp = self._client.open_sftp()
        return self._sftp

    def _upload_dir(
        self,
        sftp: paramiko.SFTPClient,
        local_dir: Path,
        remote_dir: PurePosixPath,
    ) -> None:
        """Recursively upload a directory."""
        self._mkdir_p(sftp, str(remote_dir))
        for item in local_dir.iterdir():
            remote_item = remote_dir / item.name
            if item.is_dir():
                self._upload_dir(sftp, item, remote_item)
            else:
                logger.debug("Uploading %s -> %s", item, remote_item)
                sftp.put(str(item), str(remote_item))

    @staticmethod
    def _mkdir_p(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
        """Create remote directory tree (equivalent of mkdir -p)."""
        parts = PurePosixPath(remote_dir).parts
        current = ""
        for part in parts:
            current = f"{current}/{part}" if current else part
            if not current or current == "/":
                current = "/"
                continue
            try:
                sftp.stat(current)
            except FileNotFoundError:
                logger.debug("Creating remote directory: %s", current)
                sftp.mkdir(current)


def _shell_quote(value: str) -> str:
    """Quote a value for safe use in a shell command."""
    return "'" + value.replace("'", "'\\''") + "'"
