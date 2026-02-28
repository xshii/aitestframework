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
    results: list[DiagResult] = [
        DiagResult(check="config", ok=True, message="deps.yaml parsed successfully"),
    ]

    # Toolchains & libraries — identical check pattern, table-driven
    for dep_type, deps in [("toolchain", cfg.toolchains), ("library", cfg.libraries)]:
        for name, dep in deps.items():
            ok = is_installed(name, dep.version, cache_dir)
            msg = (f"{name} {dep.version} installed" if ok
                   else f"{name} {dep.version} not installed — run: aitf deps install {name}")
            results.append(DiagResult(check=f"{dep_type}:{name}", ok=ok, message=msg))

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

    # Scripts — collect from all deps in one pass
    scripts: set[str] = set()
    for dep in (*cfg.toolchains.values(), *cfg.libraries.values()):
        if dep.acquire.script:
            scripts.add(dep.acquire.script)
    for dep in (*cfg.libraries.values(), *cfg.repos.values()):
        if dep.build_script:
            scripts.add(dep.build_script)
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

    # Lock file sync — table-driven mismatch check
    if lock_path:
        lock = load_lock(lock_path)
        if lock is None:
            results.append(DiagResult(
                check="lock_file", ok=False,
                message="deps.lock.yaml not found — run: aitf deps lock",
            ))
        else:
            mismatches = []
            for cfg_section, lock_section in [
                (cfg.toolchains, lock.toolchains),
                (cfg.libraries, lock.libraries),
            ]:
                for name, dep in cfg_section.items():
                    entry = lock_section.get(name)
                    if not entry or entry.version != dep.version:
                        mismatches.append(name)
            ok = not mismatches
            msg = ("deps.lock.yaml in sync" if ok
                   else f"Lock file out of sync for: {', '.join(mismatches)} — run: aitf deps lock")
            results.append(DiagResult(check="lock_file", ok=ok, message=msg))

    return results
