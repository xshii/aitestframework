"""Port connectors — Python-native connections to target ports.

Only SSH and TCP are implemented. Other protocols (serial, JTAG, etc.)
can be added when actually needed.
"""

from __future__ import annotations

import logging
import socket
import time
from pathlib import Path, PurePosixPath

logger = logging.getLogger(__name__)


class ConnectError(Exception):
    """Raised when a port connection fails."""


# ---------------------------------------------------------------------------
# SSH connector (paramiko)
# ---------------------------------------------------------------------------

class SSHConnector:
    """SSH connection — supports exec, upload, download."""

    def __init__(self, host: str, port: int = 22, user: str = "",
                 auth: dict | None = None):
        self.host = host
        self.port = port or 22
        self.user = user
        self.auth = auth or {}
        self._client = None
        self._sftp = None

    def connect(self) -> None:
        if self._client is not None:
            return
        try:
            import paramiko
        except ImportError:
            raise ImportError(
                "paramiko is required for SSH. Install: pip install paramiko")

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        kw: dict = {"hostname": self.host, "port": self.port}
        if self.user:
            kw["username"] = self.user

        method = self.auth.get("method", "key")
        if method == "key" and self.auth.get("key_file"):
            import os
            kw["key_filename"] = os.path.expanduser(self.auth["key_file"])
        elif method == "password" and self.auth.get("password"):
            kw["password"] = self.auth["password"]

        try:
            client.connect(**kw)
        except Exception as exc:
            raise ConnectError(f"SSH {self.host}:{self.port}: {exc}") from exc

        self._client = client
        logger.info("SSH connected: %s@%s:%d", self.user or "-", self.host, self.port)

    def disconnect(self) -> None:
        if self._sftp:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    @property
    def connected(self) -> bool:
        if self._client is None:
            return False
        t = self._client.get_transport()
        return t is not None and t.is_active()

    def exec(self, command: str, *, timeout: int = 300,
             env: dict | None = None, cwd: str | None = None) -> tuple[int, str, str]:
        """Returns (returncode, stdout, stderr)."""
        if self._client is None:
            raise RuntimeError("SSH not connected")
        parts: list[str] = []
        if env:
            for k, v in env.items():
                parts.append(f"export {k}='{v}'")
        if cwd:
            parts.append(f"cd '{cwd}'")
        parts.append(command)
        full = " && ".join(parts)

        try:
            _, out_ch, err_ch = self._client.exec_command(full, timeout=timeout)
            stdout = out_ch.read().decode("utf-8", errors="replace")
            stderr = err_ch.read().decode("utf-8", errors="replace")
            rc = out_ch.channel.recv_exit_status()
        except Exception as exc:
            return -1, "", str(exc)
        return rc, stdout, stderr

    def upload(self, local_path: str, remote_path: str) -> None:
        sftp = self._get_sftp()
        local = Path(local_path)
        if local.is_dir():
            self._upload_dir(sftp, local, PurePosixPath(remote_path))
        elif local.is_file():
            self._mkdir_p(sftp, str(PurePosixPath(remote_path).parent))
            sftp.put(str(local), remote_path)
        else:
            raise FileNotFoundError(f"Not found: {local_path}")

    def download(self, remote_path: str, local_path: str) -> None:
        sftp = self._get_sftp()
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote_path, local_path)

    def _get_sftp(self):
        if self._client is None:
            raise RuntimeError("SSH not connected")
        if self._sftp is None:
            self._sftp = self._client.open_sftp()
        return self._sftp

    def _upload_dir(self, sftp, local_dir: Path, remote_dir: PurePosixPath):
        self._mkdir_p(sftp, str(remote_dir))
        for item in local_dir.iterdir():
            r = remote_dir / item.name
            if item.is_dir():
                self._upload_dir(sftp, item, r)
            else:
                sftp.put(str(item), str(r))

    @staticmethod
    def _mkdir_p(sftp, remote_dir: str):
        parts = PurePosixPath(remote_dir).parts
        cur = ""
        for p in parts:
            cur = f"{cur}/{p}" if cur else p
            if cur in ("", "/"):
                cur = "/"
                continue
            try:
                sftp.stat(cur)
            except FileNotFoundError:
                sftp.mkdir(cur)


# ---------------------------------------------------------------------------
# TCP connector (raw socket)
# ---------------------------------------------------------------------------

class TCPConnector:
    """Raw TCP — for debug ports, serial-over-network, etc."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        if self._sock is not None:
            return
        try:
            self._sock = socket.create_connection((self.host, self.port), timeout=10)
        except OSError as exc:
            raise ConnectError(f"TCP {self.host}:{self.port}: {exc}") from exc

    def disconnect(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def send(self, data: bytes) -> None:
        if not self._sock:
            raise RuntimeError("TCP not connected")
        self._sock.sendall(data)

    def recv(self, size: int = 4096, *, timeout: int = 30) -> bytes:
        if not self._sock:
            raise RuntimeError("TCP not connected")
        self._sock.settimeout(timeout)
        return self._sock.recv(size)

    def read_until(self, pattern: str, *, timeout: int = 300) -> str:
        if not self._sock:
            raise RuntimeError("TCP not connected")
        self._sock.settimeout(timeout)
        buf = b""
        pat = pattern.encode("utf-8")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                chunk = self._sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
                if pat in buf:
                    break
            except socket.timeout:
                break
        return buf.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_connector(port_cfg: dict) -> SSHConnector | TCPConnector:
    """Create a connector from a port config dict."""
    ptype = port_cfg.get("type", "ssh")
    if ptype == "ssh":
        return SSHConnector(
            host=port_cfg.get("host", ""),
            port=int(port_cfg.get("port", 22)),
            user=port_cfg.get("user", ""),
            auth=port_cfg.get("auth"),
        )
    elif ptype == "tcp":
        return TCPConnector(
            host=port_cfg.get("host", ""),
            port=int(port_cfg.get("port", 0)),
        )
    else:
        raise ValueError(f"Unknown port type: {ptype!r} (supported: ssh, tcp)")
