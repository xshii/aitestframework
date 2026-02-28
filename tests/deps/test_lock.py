"""Tests for deps.lock — lock file management."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aitf.deps.config import DepsConfig, load_deps_config
from aitf.deps.lock import generate_lock, load_lock, save_lock
from aitf.deps.types import LockEntry, LockFile


class TestLockFileIO:
    def test_save_and_load(self, tmp_path):
        lock = LockFile(
            generated_at="2026-02-28T10:00:00",
            platform="linux-x86_64",
        )
        lock.toolchains["cc"] = LockEntry(
            name="cc", version="1.0", sha256="abc123", installed_at="2026-02-28T10:00:00",
        )
        lock.libraries["lib"] = LockEntry(
            name="lib", version="2.0", sha256="def456", installed_at="2026-02-28T10:00:00",
        )
        lock.repos["repo"] = LockEntry(
            name="repo", ref="main", commit="a" * 40, installed_at="2026-02-28T10:00:00",
        )

        path = tmp_path / "deps.lock.yaml"
        save_lock(lock, path)
        assert path.exists()

        loaded = load_lock(path)
        assert loaded is not None
        assert loaded.platform == "linux-x86_64"
        assert "cc" in loaded.toolchains
        assert loaded.toolchains["cc"].version == "1.0"
        assert "lib" in loaded.libraries
        assert "repo" in loaded.repos
        assert loaded.repos["repo"].commit == "a" * 40

    def test_load_missing(self, tmp_path):
        assert load_lock(tmp_path / "missing.yaml") is None

    def test_save_empty(self, tmp_path):
        lock = LockFile()
        path = tmp_path / "empty.lock.yaml"
        save_lock(lock, path)
        loaded = load_lock(path)
        assert loaded is not None
        assert loaded.toolchains == {}


class TestGenerateLock:
    def test_generates_from_installed(self, deps_yaml, project_root):
        cfg = load_deps_config(deps_yaml)
        cache_dir = project_root / "build" / "cache"
        repos_dir = project_root / "build" / "repos"

        # Simulate installed toolchain
        (cache_dir / "npu-compiler-2.1.0").mkdir(parents=True)
        # Simulate installed library
        (cache_dir / "json-c-0.17").mkdir(parents=True)

        lock = generate_lock(cfg, cache_dir, repos_dir)
        assert lock.platform
        assert "npu-compiler" in lock.toolchains
        assert lock.toolchains["npu-compiler"].version == "2.1.0"
        assert "json-c" in lock.libraries

    def test_skips_uninstalled(self, deps_yaml, project_root):
        cfg = load_deps_config(deps_yaml)
        cache_dir = project_root / "build" / "cache"
        repos_dir = project_root / "build" / "repos"

        # Nothing installed
        lock = generate_lock(cfg, cache_dir, repos_dir)
        assert lock.toolchains == {}
        assert lock.libraries == {}
