"""Dependency diagnostics — ``aitf deps doctor``."""

from __future__ import annotations

import shutil
from pathlib import Path

from aitf.deps.acquire import is_installed
from aitf.deps.config import DepsConfig
from aitf.deps.lock import load_lock
from aitf.deps.repo import get_head_commit, is_cloned
from aitf.deps.types import DiagResult


def run_diagnostics(
    cfg: DepsConfig, *, cache_dir: Path, repos_dir: Path,
    project_root: Path, lock_path: Path | None = None,
) -> list[DiagResult]:
    """Run all diagnostic checks and return results."""
    results: list[DiagResult] = [
        DiagResult(check="config", ok=True, message="deps.yaml parsed successfully"),
    ]

    # Toolchains & libraries
    for name, tc in cfg.toolchains.items():
        installed = is_installed(name, tc.version, cache_dir)
        results.append(DiagResult(
            check=f"toolchain:{name}", ok=installed,
            message=(f"{name} {tc.version} installed" if installed
                     else f"{name} {tc.version} not installed — run: aitf deps install {name}"),
        ))
    for name, lib in cfg.libraries.items():
        installed = is_installed(name, lib.version, cache_dir)
        results.append(DiagResult(
            check=f"library:{name}", ok=installed,
            message=(f"{name} {lib.version} installed" if installed
                     else f"{name} {lib.version} not installed — run: aitf deps install {name}"),
        ))

    # Repos
    for name, repo in cfg.repos.items():
        cloned = is_cloned(name, repos_dir)
        if cloned:
            try:
                commit = get_head_commit(repos_dir / name)
                results.append(DiagResult(
                    check=f"repo:{name}", ok=True,
                    message=f"{name} HEAD={commit[:8]} ref={repo.ref}",
                ))
            except Exception as exc:
                results.append(DiagResult(
                    check=f"repo:{name}", ok=False, message=f"{name} git error: {exc}",
                ))
        else:
            results.append(DiagResult(
                check=f"repo:{name}", ok=False,
                message=f"{name} not cloned — run: aitf deps install {name}",
            ))

    # Scripts
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
    for s in sorted(scripts):
        exists = (project_root / s).is_file()
        results.append(DiagResult(
            check=f"script:{s}", ok=exists,
            message=f"{s} {'exists' if exists else 'not found'}",
        ))

    # Build tools
    for tool in ("cmake", "git"):
        avail = shutil.which(tool) is not None
        results.append(DiagResult(
            check=f"tool:{tool}", ok=avail,
            message=f"{tool} {'available' if avail else 'not found on PATH'}",
        ))

    # Lock file sync
    if lock_path:
        lock = load_lock(lock_path)
        if lock is None:
            results.append(DiagResult(
                check="lock_file", ok=False,
                message="deps.lock.yaml not found — run: aitf deps lock",
            ))
        else:
            mismatches = []
            for name, tc in cfg.toolchains.items():
                entry = lock.toolchains.get(name)
                if not entry or entry.version != tc.version:
                    mismatches.append(name)
            for name, lib in cfg.libraries.items():
                entry = lock.libraries.get(name)
                if not entry or entry.version != lib.version:
                    mismatches.append(name)
            if mismatches:
                results.append(DiagResult(
                    check="lock_file", ok=False,
                    message=f"Lock file out of sync for: {', '.join(mismatches)} — run: aitf deps lock",
                ))
            else:
                results.append(DiagResult(
                    check="lock_file", ok=True, message="deps.lock.yaml in sync",
                ))

    return results
