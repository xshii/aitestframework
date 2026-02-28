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

# Table-driven serialization: section -> fields to extract from LockEntry
_LOCK_FIELDS: dict[str, tuple[str, ...]] = {
    "toolchains": ("version", "sha256", "installed_at"),
    "libraries": ("version", "sha256", "installed_at"),
    "repos": ("ref", "commit", "installed_at"),
}


def generate_lock(cfg: DepsConfig, cache_dir: Path, repos_dir: Path) -> LockFile:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    lock = LockFile(generated_at=now, platform=detect_platform())

    # Toolchains & libraries — same pattern, different sha256 access
    for section, cfg_section in [("toolchains", cfg.toolchains), ("libraries", cfg.libraries)]:
        for name, dep in cfg_section.items():
            if (cache_dir / f"{name}-{dep.version}").is_dir():
                sha = dep.sha256.get(detect_platform(), "") if isinstance(dep.sha256, dict) else dep.sha256
                getattr(lock, section)[name] = LockEntry(
                    name=name, version=dep.version, sha256=sha, installed_at=now,
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
    for section, fields in _LOCK_FIELDS.items():
        entries = getattr(lock, section)
        if entries:
            data[section] = {
                n: {f: getattr(e, f) for f in fields}
                for n, e in entries.items()
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
