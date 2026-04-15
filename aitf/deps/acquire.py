"""Toolchain acquisition — local dir, fetch script, or artifact tool."""

from __future__ import annotations

import itertools
import logging
import shutil
import subprocess
import tarfile
import zipfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aitf.deps.config import detect_platform
from aitf.deps.types import (
    AcquireConfig,
    AcquireError,
    ToolchainConfig,
)

logger = logging.getLogger(__name__)

# Step = (order, label, zero-arg callable)
Step = tuple[float, str, Callable[[], None]]


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def run_script(script: str, args: list[str], *, project_root: Path, timeout: int = 600) -> None:
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


def archive_candidates(name: str, version: str, plat: str) -> tuple[str, ...]:
    """Default filename convention when ``acquire.srcpkg`` is not set."""
    stems = (f"{name}-{version}-{plat}", f"{name}-{version}")
    exts = (".tar.gz", ".tgz", ".tar.xz", ".tar.bz2", ".tar", ".zip", ".rar")
    return tuple(f"{s}{e}" for s in stems for e in exts)


def run_ordered(
    steps: Iterable[Step], *,
    on_progress: Callable[[int, int, str], None] | None = None,
    on_error: Callable[[str, BaseException], None] | None = None,
) -> None:
    """Execute install steps honoring ``order`` groups.

    Steps with identical ``order`` run concurrently via a thread pool;
    groups run sequentially in ascending order. Exceptions are routed to
    *on_error* instead of aborting the run — this matches both the CLI
    and the framework's "best-effort install" behaviour.
    """
    ordered = sorted(steps, key=lambda s: s[0])
    total = len(ordered)
    done = 0

    def _run(label: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:
            if on_error:
                on_error(label, exc)
            else:
                logger.error("Failed to install %s: %s", label, exc)

    for _, group in itertools.groupby(ordered, key=lambda s: s[0]):
        batch = list(group)
        if len(batch) == 1:
            _, label, fn = batch[0]
            done += 1
            if on_progress:
                on_progress(done, total, label)
            _run(label, fn)
            continue

        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futs = {pool.submit(_run, label, fn): label for _, label, fn in batch}
            for fut in futs:
                fut.result()
                done += 1
                if on_progress:
                    on_progress(done, total, futs[fut])


# ---------------------------------------------------------------------------
# Install flow
# ---------------------------------------------------------------------------

def install_toolchain(
    tc: ToolchainConfig, *, cache_dir: Path, project_root: Path,
    install_dir: Path | None = None,
) -> Path:
    """Install a toolchain — dispatches on acquire mode."""
    target = install_dir if install_dir else cache_dir / tc.name
    if target.is_dir() and any(target.iterdir()):
        return target

    # Artifact-tool path bypasses the archive/unpack pipeline: the
    # external tool downloads (and optionally extracts) directly into
    # install_dir.
    if tc.acquire.artifact_tool is not None:
        from aitf.deps.bootstrap import fetch_via_tool
        fetch_via_tool(
            version=tc.version,
            install_dir=target,
            extract=tc.acquire.artifact_tool.extract,
            placeholders=tc.acquire.artifact_tool.placeholders,
        )
        return target

    archive = _locate_archive(tc.name, tc.version, tc.acquire, project_root, target)
    if archive is not None:
        _unpack(archive, target)
    return target


# ---------------------------------------------------------------------------
# Archive location (3-tier: local -> script -> remote server)
# ---------------------------------------------------------------------------

def _scan_dir(directory: Path, srcpkg: str | None,
              name: str, version: str, plat: str) -> Path | None:
    """Look for an archive in *directory*: explicit *srcpkg* wins, then
    the default ``{name}-{version}[-{platform}].{ext}`` convention."""
    if not directory.is_dir():
        return None
    if srcpkg:
        p = directory / srcpkg
        if p.is_file():
            return p
    for candidate in archive_candidates(name, version, plat):
        p = directory / candidate
        if p.is_file():
            return p
    return None


def _locate_archive(
    name: str, version: str, acquire: AcquireConfig,
    project_root: Path, install_dir: Path,
) -> Path | None:
    """Return the archive path, or ``None`` if *install_dir* already holds
    unpacked content written directly by a fetch script."""
    install_dir.mkdir(parents=True, exist_ok=True)
    plat = detect_platform()

    # Tier 1: local directory (+ optional explicit filename)
    if acquire.local_dir:
        found = _scan_dir(project_root / acquire.local_dir,
                          acquire.srcpkg, name, version, plat)
        if found:
            return found

    # Tier 2: fetch script — writes into install_dir
    if acquire.script:
        run_script(
            acquire.script, [version, str(install_dir)],
            project_root=project_root,
        )
        found = _scan_dir(install_dir, acquire.srcpkg, name, version, plat)
        if found:
            return found
        # Script may have written extracted content directly; treat as done.
        if any(install_dir.iterdir()):
            return None

    # Tier 3: remote server (CLIENT mode)
    found = _try_remote_download(name, version, install_dir)
    if found:
        return found

    raise AcquireError(
        f"Could not find archive for {name}-{version}. "
        f"Place it in '{acquire.local_dir or 'deps/uploads/'}' or provide a fetch script."
    )


def _try_remote_download(name: str, version: str, downloads: Path) -> Path | None:
    """Attempt to download a dep archive from the remote server."""
    try:
        from flask import current_app
        sc = current_app.config.get("SYNC_CLIENT")
        if sc is None:
            return None
    except (ImportError, RuntimeError):
        return None

    try:
        dest = sc.deps_download_archive(name, version, downloads)
        if dest.is_file():
            logger.info("Downloaded dep %s-%s from remote server", name, version)
            return dest
    except Exception as exc:
        logger.debug("Remote download failed for %s-%s: %s", name, version, exc)
    return None


# Preserved for compatibility with external callers / tests.
def _find_archive(directory: Path, name: str, version: str, plat: str) -> Path | None:
    return _scan_dir(directory, None, name, version, plat)


# ---------------------------------------------------------------------------
# Unpack — supports tar.gz / tar.xz / tar.bz2 / tar / zip / rar
# ---------------------------------------------------------------------------

_TAR_MODES: dict[tuple[str, ...], str] = {
    (".tar.gz", ".tgz"): "r:gz",
    (".tar.xz",):        "r:xz",
    (".tar.bz2",):       "r:bz2",
    (".tar",):           "r:",
}


def _unpack(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    try:
        for exts, mode in _TAR_MODES.items():
            if name.endswith(exts):
                with tarfile.open(archive, mode) as tf:
                    tf.extractall(dest, filter="data")
                return
        if name.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dest)
            return
        if name.endswith(".rar"):
            r = subprocess.run(
                ["unrar", "x", "-y", "-inul", str(archive), str(dest) + "/"],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                raise AcquireError(f"unrar failed: {r.stderr.strip()}")
            return
        raise AcquireError(f"Unsupported archive type: {archive.name}")
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise


def is_installed(name: str, version: str, cache_dir: Path) -> bool:
    d = cache_dir / name
    return d.is_dir() and any(d.iterdir())


def clean_cache(cache_dir: Path) -> int:
    if not cache_dir.is_dir():
        return 0
    count = 0
    for child in cache_dir.iterdir():
        if child.is_dir() and not child.name.startswith("."):
            shutil.rmtree(child)
            count += 1
    return count
