"""Unified project configuration (config.yaml).

Determines the runtime mode based on the ``server`` field:

- **standalone** — ``server`` is empty or config.yaml is missing
- **server** — ``server`` matches a local IP address
- **client** — ``server`` is a remote IP address
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
    SERVER = "server"
    CLIENT = "client"


def is_local_ip(ip: str) -> bool:
    """Return True if *ip* resolves to an address on this machine."""
    if ip in ("localhost", "127.0.0.1", "::1"):
        return True
    local_ips: set[str] = {"127.0.0.1", "::1"}
    # Addresses from hostname
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            local_ips.add(info[4][0])
    except OSError:
        pass
    # Preferred outbound IP (UDP connect trick)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ips.add(s.getsockname()[0])
    except OSError:
        pass
    return ip in local_ips


def _determine_mode(server: str) -> Mode:
    if not server:
        return Mode.STANDALONE
    if is_local_ip(server):
        return Mode.SERVER
    return Mode.CLIENT


@dataclass
class AitfConfig:
    """Global configuration loaded from ``config.yaml``."""

    server: str = ""
    port: int = 5000
    mode: Mode = field(default=Mode.STANDALONE, init=False)
    project_root: Path = field(default_factory=lambda: Path(".").resolve())
    # None → derive from project_root at access time
    _build_root: Path | None = field(default=None, repr=False)
    _datastore_dir: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.mode = _determine_mode(self.server)

    # -- path properties (default = project_root-relative) -------------------

    @property
    def build_root(self) -> Path:
        """Root for build artefacts (cache, repos). Default: ``<project_root>/build``."""
        if self._build_root is not None:
            return self._build_root
        return self.project_root / "build"

    @property
    def datastore_dir(self) -> Path:
        """Golden-data store location. Default: ``<project_root>/datastore``."""
        if self._datastore_dir is not None:
            return self._datastore_dir
        return self.project_root / "datastore"

    @property
    def server_url(self) -> str | None:
        """Full base URL for client mode, ``None`` otherwise."""
        if self.mode == Mode.CLIENT:
            return f"http://{self.server}:{self.port}"
        return None

    @property
    def bind_host(self) -> str:
        """Host address the web server should bind to."""
        return "0.0.0.0" if self.mode == Mode.SERVER else "127.0.0.1"


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

    return AitfConfig(
        server=server, port=port, project_root=project_root,
        _build_root=build_root, _datastore_dir=datastore_dir,
    )
