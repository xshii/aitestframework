"""Unified project configuration (config.yaml).

Determines the runtime mode based on ``server`` field:

- **standalone** — ``server`` is empty or config.yaml is missing.
  Local machine acts as server, binds to ``0.0.0.0``.
- **client** — ``server`` is a remote IP.
  Connects to remote server for sync; also starts local web UI.
- **debug** — ``server`` equals a local IP.
  Starts both server and client on the same machine (for testing).

配置示例::

    # 场景 1: 不写 server → 本机即服务器（standalone）
    port: 8080

    # 场景 2: server 是远程 IP → 客户端，连接远程服务器
    server: "192.168.1.100"
    port: 8080

    # 场景 3: server 是本机 IP → 调试模式，同时启动服务端和客户端
    server: "192.168.1.50"   # 本机 IP
    port: 8080
"""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_FILE = "config.yaml"


class Mode(str, Enum):
    STANDALONE = "standalone"
    CLIENT = "client"
    DEBUG = "debug"          # server IP == local IP → both roles


def get_local_ips() -> set[str]:
    """Return all IP addresses on this machine."""
    local_ips: set[str] = {"127.0.0.1", "::1", "localhost"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            local_ips.add(info[4][0])
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ips.add(s.getsockname()[0])
    except OSError:
        pass
    return local_ips


def is_local_ip(ip: str) -> bool:
    """Return True if *ip* resolves to an address on this machine."""
    return ip in get_local_ips()


def _determine_mode(server: str) -> Mode:
    if not server:
        return Mode.STANDALONE
    if is_local_ip(server):
        return Mode.DEBUG
    return Mode.CLIENT


@dataclass
class AitfConfig:
    """Global configuration loaded from ``config.yaml``.

    Attributes:
        server: 服务器 IP。空 = 本机即服务器; 远程 IP = 客户端模式;
                本机 IP = 调试模式（同时启动服务端和客户端）。
        port: Web 服务端口。
    """

    server: str = ""
    port: int = 5000
    mode: Mode = field(default=Mode.STANDALONE, init=False)
    project_root: Path = field(default_factory=lambda: Path(".").resolve())
    dbox_enabled: bool = False
    dbox_server: str = ""
    dbox_upload_path: str = "/api/upload"
    _build_root: Path | None = field(default=None, repr=False)
    _datastore_dir: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.mode = _determine_mode(self.server)

    # -- path properties ---------------------------------------------------

    @property
    def build_root(self) -> Path:
        if self._build_root is not None:
            return self._build_root
        return self.project_root / "build"

    @property
    def datastore_dir(self) -> Path:
        if self._datastore_dir is not None:
            return self._datastore_dir
        return self.project_root / "datastore"

    @property
    def server_url(self) -> str | None:
        """Full base URL for client/debug sync. None for standalone."""
        if self.mode in (Mode.CLIENT, Mode.DEBUG):
            return f"http://{self.server}:{self.port}"
        return None

    @property
    def bind_host(self) -> str:
        """Host address the web server should bind to.

        standalone/debug → 0.0.0.0 (accept remote connections)
        client → 127.0.0.1 (local only, sync to remote server)
        """
        if self.mode == Mode.CLIENT:
            return "127.0.0.1"
        return "0.0.0.0"

    @property
    def is_server(self) -> bool:
        """Whether this instance acts as a server (accepts data from clients)."""
        return self.mode in (Mode.STANDALONE, Mode.DEBUG)

    @property
    def is_client(self) -> bool:
        """Whether this instance syncs data to a remote server."""
        return self.mode in (Mode.CLIENT, Mode.DEBUG)


def load_config(
    path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> AitfConfig:
    """Load ``config.yaml`` and return an :class:`AitfConfig`.

    Missing file is not an error — returns a standalone default config.
    """
    if project_root is None:
        project_root = Path(".").resolve()
    else:
        project_root = Path(project_root).resolve()

    if path is None:
        path = project_root / DEFAULT_CONFIG_FILE
    else:
        path = Path(path)

    server = ""
    port = 5000
    build_root: Path | None = None
    datastore_dir: Path | None = None
    data: dict = {}

    if path.is_file():
        try:
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            logger.warning("Failed to parse %s: %s", path, exc)
            data = {}

        if isinstance(data, dict):
            server = str(data.get("server", "") or "")
            port = int(data.get("port", 5000))
            raw_build = data.get("build_root")
            if raw_build:
                p = Path(raw_build)
                build_root = p if p.is_absolute() else (project_root / p).resolve()
            raw_ds = data.get("datastore_dir")
            if raw_ds:
                p = Path(raw_ds)
                datastore_dir = p if p.is_absolute() else (project_root / p).resolve()
        else:
            logger.warning("Expected a YAML mapping in %s, got %s", path, type(data).__name__)
    else:
        logger.debug("Config file not found: %s — using defaults", path)

    dbox_enabled = bool(data.get("dbox_enabled", False)) if isinstance(data, dict) else False
    dbox_server = str(data.get("dbox_server", "") or "") if isinstance(data, dict) else ""
    dbox_upload_path = str(data.get("dbox_upload_path", "/api/upload") or "/api/upload") if isinstance(data, dict) else "/api/upload"

    return AitfConfig(
        server=server, port=port, project_root=project_root,
        dbox_enabled=dbox_enabled, dbox_server=dbox_server,
        dbox_upload_path=dbox_upload_path,
        _build_root=build_root, _datastore_dir=datastore_dir,
    )
