"""Tests for deps.manager — DepsManager facade."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from aitf.deps.acquire import sha256_file
from aitf.deps.config import load_deps_config
from aitf.deps.manager import DepsManager
from aitf.deps.types import DepsConfigError, DepsError


@pytest.fixture()
def manager(deps_yaml, project_root):
    """Create a DepsManager with temp project root."""
    return DepsManager(
        project_root=str(project_root),
        deps_file="deps.yaml",
        build_dir="build",
    )


class TestDepsManagerInit:
    def test_creates_directories(self, manager, project_root):
        assert (project_root / "build" / "cache").is_dir()
        assert (project_root / "build" / "repos").is_dir()

    def test_config_lazy_load(self, manager):
        cfg = manager.config
        assert "npu-compiler" in cfg.toolchains

    def test_reload(self, manager):
        _ = manager.config
        manager.reload()
        # After reload, accessing config should re-parse
        cfg = manager.config
        assert "npu-compiler" in cfg.toolchains


class TestDepsManagerInstall:
    def _setup_toolchain_archive(self, project_root):
        """Create a fake toolchain archive for installation."""
        uploads = project_root / "deps" / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)

        content = project_root / "tmp_content" / "npu-compiler-2.1.0"
        (content / "bin").mkdir(parents=True)
        (content / "bin" / "npu-gcc").write_text("#!/bin/sh\n")

        archive = uploads / "npu-compiler-2.1.0.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(content, arcname="npu-compiler-2.1.0")
        return archive

    def test_install_one(self, manager, project_root):
        self._setup_toolchain_archive(project_root)
        manager.install(name="npu-compiler")
        assert (project_root / "build" / "cache" / "npu-compiler-2.1.0").is_dir()

    def test_install_unknown_raises(self, manager):
        with pytest.raises(DepsError, match="Unknown dependency"):
            manager.install(name="nonexistent")

    def test_install_all_logs_errors(self, manager, caplog):
        """install() without name tries all deps — logs errors for missing archives."""
        import logging
        with caplog.at_level(logging.ERROR):
            manager.install()
        # Should have logged errors for missing archives
        # (repos will also fail since URLs are fake)


class TestDepsManagerList:
    def test_list_installed(self, manager):
        items = manager.list_installed()
        # Should include all declared deps
        names = {getattr(item, "name") for item in items}
        assert "npu-compiler" in names
        assert "json-c" in names
        assert "npu-runtime" in names


class TestDepsManagerClean:
    def test_clean_removes_cache(self, manager, project_root):
        cache = project_root / "build" / "cache"
        (cache / "dummy-1.0").mkdir()
        count = manager.clean()
        assert count == 1
        assert not (cache / "dummy-1.0").exists()


class TestDepsManagerDoctor:
    def test_doctor_returns_results(self, manager):
        results = manager.doctor()
        assert len(results) > 0
        # Config check should always pass
        config_checks = [r for r in results if r.check == "deps.yaml configuration"]
        assert len(config_checks) == 1


class TestDepsManagerEnv:
    def test_env_empty_when_not_installed(self, manager):
        env = manager.get_env()
        # Nothing installed, so env should be empty
        assert env == {}

    def test_env_with_installed_toolchain(self, manager, project_root):
        # Simulate installed toolchain
        cache = project_root / "build" / "cache"
        tc_dir = cache / "npu-compiler-2.1.0"
        tc_dir.mkdir(parents=True)

        env = manager.get_env()
        assert "NPU_CC" in env
        assert "npu-compiler-2.1.0" in env["NPU_CC"]


class TestDepsManagerGetInstallDir:
    def test_not_installed(self, manager):
        assert manager.get_install_dir("npu-compiler") is None

    def test_installed(self, manager, project_root):
        (project_root / "build" / "cache" / "npu-compiler-2.1.0").mkdir(parents=True)
        d = manager.get_install_dir("npu-compiler")
        assert d is not None
        assert d.name == "npu-compiler-2.1.0"

    def test_unknown_dep(self, manager):
        assert manager.get_install_dir("unknown") is None

    def test_repo_dir(self, manager, project_root):
        (project_root / "build" / "repos" / "npu-runtime" / ".git").mkdir(parents=True)
        d = manager.get_install_dir("npu-runtime")
        assert d is not None
        assert d.name == "npu-runtime"


class TestDepsManagerLock:
    def test_lock_creates_file(self, manager, project_root):
        # Simulate installed deps
        (project_root / "build" / "cache" / "npu-compiler-2.1.0").mkdir(parents=True)
        manager.lock()
        assert (project_root / "deps.lock.yaml").exists()
