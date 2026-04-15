"""Dependency diagnostics — ``aitf deps doctor``."""

from __future__ import annotations

import shutil
from pathlib import Path

from aitf.deps.config import DepsConfig
from aitf.deps.lock import load_lock
from aitf.deps.repo import get_head_commit
from aitf.deps.types import DiagResult


def run_diagnostics(
    cfg: DepsConfig, *, cache_dir: Path, repos_dir: Path,
    project_root: Path, build_dir: Path | None = None,
    lock_path: Path | None = None,
) -> list[DiagResult]:
    bdir = build_dir or cache_dir.parent
    results: list[DiagResult] = [
        DiagResult(check="config", ok=True, message="deps.yaml parsed successfully"),
    ]

    # Toolchains
    for name, tc in cfg.toolchains.items():
        target = tc.install_path(cache_dir, bdir)
        ok = target.is_dir() and any(target.iterdir())
        msg = (f"{name} {tc.version} installed" if ok
               else f"{name} {tc.version} not installed — run: aitf deps install {name}")
        results.append(DiagResult(check=f"toolchain:{name}", ok=ok, message=msg))

    # Repos
    for name, rc in cfg.repos.items():
        target = rc.install_path(repos_dir, bdir)
        if target.is_dir() and (target / ".git").exists():
            try:
                commit = get_head_commit(target)
                results.append(DiagResult(
                    check=f"repo:{name}", ok=True,
                    message=f"{name} HEAD={commit[:8]} ref={rc.resolved_ref}",
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

    # Scripts referenced anywhere in the config
    scripts = {
        *(tc.acquire.script for tc in cfg.toolchains.values() if tc.acquire.script),
        *(rc.build_script for rc in cfg.repos.values() if rc.build_script),
    }
    for s in sorted(scripts):
        exists = (project_root / s).is_file()
        results.append(DiagResult(
            check=f"script:{s}", ok=exists,
            message=f"{s} {'exists' if exists else 'not found'}",
        ))

    # Build tools
    for tool in ("git",):
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
            mismatches = [
                name for name, tc in cfg.toolchains.items()
                if (e := lock.toolchains.get(name)) is None or e.version != tc.version
            ] + [
                name for name, rc in cfg.repos.items()
                if (e := lock.repos.get(name)) is None or e.ref != rc.resolved_ref
            ]
            ok = not mismatches
            msg = ("deps.lock.yaml in sync" if ok
                   else f"Lock file out of sync for: {', '.join(mismatches)} — run: aitf deps lock")
            results.append(DiagResult(check="lock_file", ok=ok, message=msg))

    return results
