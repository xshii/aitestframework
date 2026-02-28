"""Lock file (deps.lock.yaml) management."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

from aitf.deps.config import DepsConfig, detect_platform
from aitf.deps.repo import get_head_commit
from aitf.deps.types import LockEntry, LockFile

logger = logging.getLogger(__name__)

DEFAULT_LOCK_FILE = "deps.lock.yaml"


def generate_lock(cfg: DepsConfig, cache_dir: Path, repos_dir: Path) -> LockFile:
    """Inspect installed dependencies and generate a lock file."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    lock = LockFile(generated_at=now, platform=detect_platform())

    for name, tc in cfg.toolchains.items():
        if (cache_dir / f"{name}-{tc.version}").is_dir():
            lock.toolchains[name] = LockEntry(
                name=name, version=tc.version,
                sha256=tc.sha256.get(detect_platform(), ""), installed_at=now,
            )

    for name, lib in cfg.libraries.items():
        if (cache_dir / f"{name}-{lib.version}").is_dir():
            lock.libraries[name] = LockEntry(
                name=name, version=lib.version, sha256=lib.sha256, installed_at=now,
            )

    for name, repo in cfg.repos.items():
        repo_dir = repos_dir / name
        if repo_dir.is_dir() and (repo_dir / ".git").exists():
            lock.repos[name] = LockEntry(
                name=name, ref=repo.ref,
                commit=get_head_commit(repo_dir), installed_at=now,
            )

    return lock


def save_lock(lock: LockFile, path: str | Path = DEFAULT_LOCK_FILE) -> None:
    data: dict = {"generated_at": lock.generated_at, "platform": lock.platform}
    if lock.toolchains:
        data["toolchains"] = {
            n: {"version": e.version, "sha256": e.sha256, "installed_at": e.installed_at}
            for n, e in lock.toolchains.items()
        }
    if lock.libraries:
        data["libraries"] = {
            n: {"version": e.version, "sha256": e.sha256, "installed_at": e.installed_at}
            for n, e in lock.libraries.items()
        }
    if lock.repos:
        data["repos"] = {
            n: {"ref": e.ref, "commit": e.commit, "installed_at": e.installed_at}
            for n, e in lock.repos.items()
        }
    with open(Path(path), "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)


def load_lock(path: str | Path = DEFAULT_LOCK_FILE) -> LockFile | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    lock = LockFile(generated_at=data.get("generated_at", ""), platform=data.get("platform", ""))
    for section, target in [
        ("toolchains", lock.toolchains),
        ("libraries", lock.libraries),
        ("repos", lock.repos),
    ]:
        for name, raw in data.get(section, {}).items():
            target[name] = LockEntry(
                name=name, version=raw.get("version", ""), sha256=raw.get("sha256", ""),
                ref=raw.get("ref", ""), commit=raw.get("commit", ""),
                installed_at=raw.get("installed_at", ""),
            )
    return lock
