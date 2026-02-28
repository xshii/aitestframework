"""Dependency acquisition — local dir, SFTP remote, or fetch script."""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import tarfile
from pathlib import Path

from aitf.deps.config import detect_platform
from aitf.deps.types import (
    AcquireConfig,
    AcquireError,
    LibraryConfig,
    RemoteDepotConfig,
    ToolchainConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def run_script(script: str, args: list[str], *, project_root: Path, timeout: int = 600) -> None:
    """Execute ``bash <script> <args...>``. Raises :class:`AcquireError` on failure."""
    script_path = project_root / script
    if not script_path.is_file():
        raise AcquireError(f"Script not found: {script_path}")

    result = subprocess.run(
        ["bash", str(script_path), *args],
        capture_output=True, text=True, cwd=str(project_root), timeout=timeout,
    )
    if result.returncode != 0:
        raise AcquireError(
            f"Script failed (exit {result.returncode}): {script}\n{result.stderr.strip()}"
        )


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

def install_toolchain(
    tc: ToolchainConfig, *, cache_dir: Path, project_root: Path,
    remote: RemoteDepotConfig | None = None,
) -> Path:
    """Acquire and install a toolchain, returning the install directory."""
    install_dir = cache_dir / f"{tc.name}-{tc.version}"
    if install_dir.is_dir():
        return install_dir

    archive = _locate_archive(
        tc.name, tc.version, tc.acquire, project_root, cache_dir,
        remote=remote, remote_subdir="toolchains",
    )

    expected_sha = tc.sha256.get(detect_platform())
    if expected_sha:
        _verify_sha256(archive, expected_sha, f"{tc.name}-{tc.version}")

    _unpack(archive, install_dir)
    return install_dir


def install_library(
    lib: LibraryConfig, *, cache_dir: Path, project_root: Path,
    remote: RemoteDepotConfig | None = None,
) -> Path:
    """Acquire and install a C/C++ library, returning the install directory."""
    install_dir = cache_dir / f"{lib.name}-{lib.version}"
    if install_dir.is_dir():
        return install_dir

    archive = _locate_archive(
        lib.name, lib.version, lib.acquire, project_root, cache_dir,
        remote=remote, remote_subdir="libraries",
    )

    if lib.sha256:
        _verify_sha256(archive, lib.sha256, f"{lib.name}-{lib.version}")

    _unpack(archive, install_dir)

    if lib.build_script:
        run_script(lib.build_script, [str(install_dir), str(install_dir)],
                   project_root=project_root, timeout=1800)
    return install_dir


def _verify_sha256(archive: Path, expected: str, label: str) -> None:
    actual = sha256_file(archive)
    if actual != expected:
        archive.unlink(missing_ok=True)
        raise AcquireError(f"SHA-256 mismatch for {label}: expected {expected}, got {actual}")


# ---------------------------------------------------------------------------
# Archive location (3-tier: local → remote SFTP → script)
# ---------------------------------------------------------------------------

def _locate_archive(
    name: str, version: str, acquire: AcquireConfig,
    project_root: Path, cache_dir: Path,
    *, remote: RemoteDepotConfig | None = None, remote_subdir: str = "",
) -> Path:
    downloads = cache_dir / ".downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    plat = detect_platform()

    # Priority 1: local_dir
    if acquire.local_dir:
        found = _find_archive(project_root / acquire.local_dir, name, version, plat)
        if found:
            return found

    # Priority 2: SFTP remote depot
    if acquire.remote and remote:
        fetched = _fetch_from_remote(remote, remote_subdir, name, version, plat, downloads)
        if fetched:
            return fetched

    # Priority 3: fetch script
    if acquire.script:
        run_script(acquire.script, [version, str(downloads)], project_root=project_root)
        found = _find_archive(downloads, name, version, plat)
        if found:
            return found

    raise AcquireError(
        f"Could not find archive for {name}-{version}. "
        f"Place it in '{acquire.local_dir or 'deps/uploads/'}' or provide a fetch script."
    )


def _find_archive(directory: Path, name: str, version: str, plat: str) -> Path | None:
    """Search *directory* for a matching ``.tar.gz`` (platform-specific first)."""
    for candidate in (f"{name}-{version}-{plat}.tar.gz", f"{name}-{version}.tar.gz"):
        p = directory / candidate
        if p.is_file():
            return p
    return None


def _fetch_from_remote(
    remote: RemoteDepotConfig, subdir: str,
    name: str, version: str, plat: str,
    downloads: Path,
) -> Path | None:
    """Try to download an archive from the SFTP remote depot."""
    try:
        import paramiko
    except ImportError:
        logger.warning("paramiko not available, skipping remote fetch")
        return None

    remote_base = f"{remote.path.rstrip('/')}/{subdir}" if subdir else remote.path.rstrip("/")
    candidates = [f"{name}-{version}-{plat}.tar.gz", f"{name}-{version}.tar.gz"]

    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        connect_kw: dict = {"hostname": remote.host, "port": remote.port, "username": remote.user}
        if remote.key_file:
            connect_kw["key_filename"] = remote.key_file
        ssh.connect(**connect_kw)
        sftp = ssh.open_sftp()
        try:
            for filename in candidates:
                remote_path = f"{remote_base}/{filename}"
                local_path = downloads / filename
                try:
                    sftp.stat(remote_path)
                    logger.info("Downloading %s:%s", remote.host, remote_path)
                    sftp.get(remote_path, str(local_path))
                    return local_path
                except FileNotFoundError:
                    continue
        finally:
            sftp.close()
            ssh.close()
    except Exception as exc:
        logger.warning("Remote fetch failed: %s", exc)

    return None


def _unpack(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest, filter="data")


def is_installed(name: str, version: str, cache_dir: Path) -> bool:
    return (cache_dir / f"{name}-{version}").is_dir()


def clean_cache(cache_dir: Path) -> int:
    if not cache_dir.is_dir():
        return 0
    count = 0
    for child in cache_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
            count += 1
    return count
