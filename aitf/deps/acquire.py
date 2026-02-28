"""Dependency acquisition — local upload directory or fetch script."""

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
    ToolchainConfig,
)

logger = logging.getLogger(__name__)

DOWNLOADS_DIR = ".downloads"


# ---------------------------------------------------------------------------
# SHA-256 helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Archive naming
# ---------------------------------------------------------------------------

def _archive_name(name: str, version: str, platform_tag: str | None = None) -> str:
    """Return expected archive filename: ``<name>-<version>[-<platform>].tar.gz``."""
    if platform_tag:
        return f"{name}-{version}-{platform_tag}.tar.gz"
    return f"{name}-{version}.tar.gz"


def _find_archive(
    local_dir: Path,
    name: str,
    version: str,
) -> Path | None:
    """Search *local_dir* for a matching archive.

    Tries platform-specific name first, then generic.
    """
    plat = detect_platform()
    for candidate in (
        _archive_name(name, version, plat),
        _archive_name(name, version),
    ):
        p = local_dir / candidate
        if p.is_file():
            return p
    return None


# ---------------------------------------------------------------------------
# Fetch via script
# ---------------------------------------------------------------------------

def _run_fetch_script(
    script: str,
    version: str,
    output_dir: Path,
    *,
    project_root: Path,
) -> None:
    """Execute a user-provided fetch script.

    Interface: ``bash <script> <version> <output_dir>``
    Exit code 0 = success, else failure.
    """
    script_path = project_root / script
    if not script_path.is_file():
        raise AcquireError(f"Fetch script not found: {script_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Running fetch script: %s %s %s", script_path, version, output_dir)

    result = subprocess.run(
        ["bash", str(script_path), version, str(output_dir)],
        capture_output=True,
        text=True,
        cwd=str(project_root),
        timeout=600,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise AcquireError(
            f"Fetch script failed (exit {result.returncode}): {script}\n{stderr}"
        )


# ---------------------------------------------------------------------------
# Install (unpack) logic
# ---------------------------------------------------------------------------

def _unpack(archive: Path, dest: Path) -> None:
    """Extract a ``.tar.gz`` archive into *dest*."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest, filter="data")
    logger.info("Unpacked %s -> %s", archive.name, dest)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def install_toolchain(
    tc: ToolchainConfig,
    *,
    cache_dir: Path,
    project_root: Path,
) -> Path:
    """Acquire and install a toolchain, returning the install directory.

    1. Check cache — already installed?
    2. Check local_dir for a pre-uploaded archive.
    3. Run fetch script.
    4. Verify SHA-256.
    5. Unpack to ``cache_dir/<name>-<version>/``.

    Args:
        tc: Toolchain configuration.
        cache_dir: Directory for unpacked dependencies (``build/cache/``).
        project_root: Project root for resolving relative paths.

    Returns:
        Path to the installed directory.

    Raises:
        AcquireError: If acquisition or verification fails.
    """
    install_dir = cache_dir / f"{tc.name}-{tc.version}"
    if install_dir.is_dir():
        logger.info("Toolchain %s %s already cached", tc.name, tc.version)
        return install_dir

    archive = _locate_archive(tc.name, tc.version, tc.acquire, project_root, cache_dir)

    # Verify SHA-256
    plat = detect_platform()
    expected_sha = tc.sha256.get(plat)
    if expected_sha:
        actual = sha256_file(archive)
        if actual != expected_sha:
            archive.unlink(missing_ok=True)
            raise AcquireError(
                f"SHA-256 mismatch for {tc.name}-{tc.version}: "
                f"expected {expected_sha}, got {actual}"
            )

    _unpack(archive, install_dir)
    return install_dir


def install_library(
    lib: LibraryConfig,
    *,
    cache_dir: Path,
    project_root: Path,
) -> Path:
    """Acquire and install a C/C++ library, returning the install directory.

    Same flow as :func:`install_toolchain` but with a single SHA-256 value.
    """
    install_dir = cache_dir / f"{lib.name}-{lib.version}"
    if install_dir.is_dir():
        logger.info("Library %s %s already cached", lib.name, lib.version)
        return install_dir

    archive = _locate_archive(lib.name, lib.version, lib.acquire, project_root, cache_dir)

    if lib.sha256:
        actual = sha256_file(archive)
        if actual != lib.sha256:
            archive.unlink(missing_ok=True)
            raise AcquireError(
                f"SHA-256 mismatch for {lib.name}-{lib.version}: "
                f"expected {lib.sha256}, got {actual}"
            )

    _unpack(archive, install_dir)

    # Run library build if specified
    if lib.build_script:
        _run_build_script(lib.build_script, install_dir, install_dir, project_root=project_root)

    return install_dir


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _locate_archive(
    name: str,
    version: str,
    acquire: AcquireConfig,
    project_root: Path,
    cache_dir: Path,
) -> Path:
    """Find or fetch the archive for a dependency."""
    downloads = cache_dir / DOWNLOADS_DIR
    downloads.mkdir(parents=True, exist_ok=True)

    # Priority 1: local_dir
    if acquire.local_dir:
        local = project_root / acquire.local_dir
        found = _find_archive(local, name, version)
        if found:
            logger.info("Found local archive: %s", found)
            return found

    # Priority 2: fetch script
    if acquire.script:
        _run_fetch_script(acquire.script, version, downloads, project_root=project_root)
        found = _find_archive(downloads, name, version)
        if found:
            return found
        # Also check local_dir in case the script saved it there
        if acquire.local_dir:
            local = project_root / acquire.local_dir
            found = _find_archive(local, name, version)
            if found:
                return found

    raise AcquireError(
        f"Could not find archive for {name}-{version}. "
        f"Place it in '{acquire.local_dir or 'deps/uploads/'}' or provide a fetch script."
    )


def _run_build_script(
    script: str,
    repo_dir: Path,
    install_dir: Path,
    *,
    project_root: Path,
) -> None:
    """Execute a user-provided build script: ``bash <script> <repo_dir> <install_dir>``."""
    script_path = project_root / script
    if not script_path.is_file():
        raise AcquireError(f"Build script not found: {script_path}")

    logger.info("Running build script: %s", script_path)
    result = subprocess.run(
        ["bash", str(script_path), str(repo_dir), str(install_dir)],
        capture_output=True,
        text=True,
        cwd=str(project_root),
        timeout=1800,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise AcquireError(
            f"Build script failed (exit {result.returncode}): {script}\n{stderr}"
        )


def is_installed(name: str, version: str, cache_dir: Path) -> bool:
    """Check whether a dependency is already unpacked in *cache_dir*."""
    return (cache_dir / f"{name}-{version}").is_dir()


def clean_cache(cache_dir: Path) -> int:
    """Remove all cached dependencies. Returns number of directories removed."""
    if not cache_dir.is_dir():
        return 0
    count = 0
    for child in cache_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
            count += 1
    return count
