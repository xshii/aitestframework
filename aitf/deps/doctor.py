"""Dependency diagnostics — ``aitf deps doctor``."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from aitf.deps.acquire import is_installed
from aitf.deps.config import DepsConfig, detect_platform
from aitf.deps.lock import load_lock
from aitf.deps.repo import get_head_commit, is_cloned
from aitf.deps.types import DiagLevel, DiagResult

logger = logging.getLogger(__name__)


def run_diagnostics(
    cfg: DepsConfig,
    *,
    cache_dir: Path,
    repos_dir: Path,
    project_root: Path,
    lock_path: Path | None = None,
) -> list[DiagResult]:
    """Run all diagnostic checks and return results.

    Args:
        cfg: Parsed deps.yaml.
        cache_dir: ``build/cache/`` directory.
        repos_dir: ``build/repos/`` directory.
        project_root: Project root for resolving script paths.
        lock_path: Path to deps.lock.yaml (optional).

    Returns:
        List of :class:`DiagResult` entries.
    """
    results: list[DiagResult] = []

    results.append(_check_config_valid(cfg))
    results.extend(_check_toolchains(cfg, cache_dir))
    results.extend(_check_libraries(cfg, cache_dir))
    results.extend(_check_repos(cfg, repos_dir))
    results.extend(_check_scripts(cfg, project_root))
    results.extend(_check_build_tools())
    if lock_path:
        results.extend(_check_lock_sync(cfg, cache_dir, repos_dir, lock_path))

    return results


def _check_config_valid(cfg: DepsConfig) -> DiagResult:
    """Check that deps.yaml parsed without issues."""
    # If we got here, it was already parsed successfully
    return DiagResult(
        check="deps.yaml configuration",
        level=DiagLevel.PASS,
        message="deps.yaml parsed successfully",
    )


def _check_toolchains(cfg: DepsConfig, cache_dir: Path) -> list[DiagResult]:
    results: list[DiagResult] = []
    for name, tc in cfg.toolchains.items():
        if is_installed(name, tc.version, cache_dir):
            results.append(DiagResult(
                check=f"toolchain:{name}",
                level=DiagLevel.PASS,
                message=f"{name} {tc.version} installed",
            ))
        else:
            results.append(DiagResult(
                check=f"toolchain:{name}",
                level=DiagLevel.FAIL,
                message=f"{name} {tc.version} not installed — run: aitf deps install {name}",
            ))
    return results


def _check_libraries(cfg: DepsConfig, cache_dir: Path) -> list[DiagResult]:
    results: list[DiagResult] = []
    for name, lib in cfg.libraries.items():
        if is_installed(name, lib.version, cache_dir):
            results.append(DiagResult(
                check=f"library:{name}",
                level=DiagLevel.PASS,
                message=f"{name} {lib.version} installed",
            ))
        else:
            results.append(DiagResult(
                check=f"library:{name}",
                level=DiagLevel.FAIL,
                message=f"{name} {lib.version} not installed — run: aitf deps install {name}",
            ))
    return results


def _check_repos(cfg: DepsConfig, repos_dir: Path) -> list[DiagResult]:
    results: list[DiagResult] = []
    for name, repo in cfg.repos.items():
        if is_cloned(name, repos_dir):
            repo_dir = repos_dir / name
            try:
                commit = get_head_commit(repo_dir)
                results.append(DiagResult(
                    check=f"repo:{name}",
                    level=DiagLevel.PASS,
                    message=f"{name} HEAD={commit[:8]} ref={repo.ref}",
                ))
            except Exception as exc:
                results.append(DiagResult(
                    check=f"repo:{name}",
                    level=DiagLevel.FAIL,
                    message=f"{name} git error: {exc}",
                ))
        else:
            results.append(DiagResult(
                check=f"repo:{name}",
                level=DiagLevel.FAIL,
                message=f"{name} not cloned — run: aitf deps install {name}",
            ))
    return results


def _check_scripts(cfg: DepsConfig, project_root: Path) -> list[DiagResult]:
    results: list[DiagResult] = []
    scripts: set[str] = set()

    for tc in cfg.toolchains.values():
        if tc.acquire.script:
            scripts.add(tc.acquire.script)
    for lib in cfg.libraries.values():
        if lib.acquire.script:
            scripts.add(lib.acquire.script)
        if lib.build_script:
            scripts.add(lib.build_script)
    for repo in cfg.repos.values():
        if repo.build_script:
            scripts.add(repo.build_script)

    for script in sorted(scripts):
        path = project_root / script
        if path.is_file():
            results.append(DiagResult(
                check=f"script:{script}",
                level=DiagLevel.PASS,
                message=f"{script} exists",
            ))
        else:
            results.append(DiagResult(
                check=f"script:{script}",
                level=DiagLevel.FAIL,
                message=f"{script} not found",
            ))
    return results


def _check_build_tools() -> list[DiagResult]:
    """Check that cmake and git are available on PATH."""
    results: list[DiagResult] = []
    for tool in ("cmake", "git"):
        if shutil.which(tool):
            results.append(DiagResult(
                check=f"tool:{tool}",
                level=DiagLevel.PASS,
                message=f"{tool} available",
            ))
        else:
            results.append(DiagResult(
                check=f"tool:{tool}",
                level=DiagLevel.FAIL,
                message=f"{tool} not found on PATH — please install it",
            ))
    return results


def _check_lock_sync(
    cfg: DepsConfig,
    cache_dir: Path,
    repos_dir: Path,
    lock_path: Path,
) -> list[DiagResult]:
    """Check that deps.lock.yaml is in sync with installed state."""
    lock = load_lock(lock_path)
    if lock is None:
        return [DiagResult(
            check="lock_file",
            level=DiagLevel.FAIL,
            message="deps.lock.yaml not found — run: aitf deps lock",
        )]

    results: list[DiagResult] = []

    # Check toolchains in lock vs config
    for name, tc in cfg.toolchains.items():
        entry = lock.toolchains.get(name)
        if not entry:
            results.append(DiagResult(
                check=f"lock:{name}",
                level=DiagLevel.FAIL,
                message=f"{name} not in lock file — run: aitf deps lock",
            ))
        elif entry.version != tc.version:
            results.append(DiagResult(
                check=f"lock:{name}",
                level=DiagLevel.FAIL,
                message=(
                    f"{name} version mismatch: config={tc.version} lock={entry.version}"
                    " — run: aitf deps lock"
                ),
            ))

    # Check libraries in lock vs config
    for name, lib in cfg.libraries.items():
        entry = lock.libraries.get(name)
        if not entry:
            results.append(DiagResult(
                check=f"lock:{name}",
                level=DiagLevel.FAIL,
                message=f"{name} not in lock file — run: aitf deps lock",
            ))
        elif entry.version != lib.version:
            results.append(DiagResult(
                check=f"lock:{name}",
                level=DiagLevel.FAIL,
                message=(
                    f"{name} version mismatch: config={lib.version} lock={entry.version}"
                    " — run: aitf deps lock"
                ),
            ))

    if not results:
        results.append(DiagResult(
            check="lock_file",
            level=DiagLevel.PASS,
            message="deps.lock.yaml in sync",
        ))

    return results
