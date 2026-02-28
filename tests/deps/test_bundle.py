"""Tests for deps.bundle — BundleManager."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest
import yaml

from aitf.deps.bundle import BundleManager
from aitf.deps.manager import DepsManager
from aitf.deps.types import BundleError, BundleNotFoundError


@pytest.fixture()
def bundle_mgr(deps_yaml, project_root):
    """Create a BundleManager with temp project root."""
    dm = DepsManager(
        project_root=str(project_root),
        deps_file="deps.yaml",
        build_dir="build",
    )
    return BundleManager(dm, deps_file=deps_yaml)


class TestListBundles:
    def test_list_returns_all(self, bundle_mgr):
        bundles = bundle_mgr.list_bundles()
        names = {b.name for b in bundles}
        assert "npu-v2.1" in names
        assert "npu-v2.0" in names

    def test_list_empty_config(self, tmp_path):
        empty_cfg = tmp_path / "deps.yaml"
        with open(empty_cfg, "w") as fh:
            yaml.dump({}, fh)
        dm = DepsManager(
            project_root=str(tmp_path),
            deps_file="deps.yaml",
            build_dir="build",
        )
        bm = BundleManager(dm, deps_file=empty_cfg)
        assert bm.list_bundles() == []


class TestShow:
    def test_show_existing(self, bundle_mgr):
        b = bundle_mgr.show("npu-v2.1")
        assert b.name == "npu-v2.1"
        assert b.status == "verified"
        assert b.description == "NPU test env v2.1"

    def test_show_not_found(self, bundle_mgr):
        with pytest.raises(BundleNotFoundError):
            bundle_mgr.show("nonexistent")


class TestActive:
    def test_active_bundle(self, bundle_mgr):
        active = bundle_mgr.active()
        assert active is not None
        assert active.name == "npu-v2.1"

    def test_no_active(self, tmp_path):
        cfg = tmp_path / "deps.yaml"
        with open(cfg, "w") as fh:
            yaml.dump({"bundles": {"b1": {"description": "test", "status": "testing"}}}, fh)
        dm = DepsManager(project_root=str(tmp_path), deps_file="deps.yaml", build_dir="build")
        bm = BundleManager(dm, deps_file=cfg)
        assert bm.active() is None


class TestUse:
    def test_use_deprecated_without_force(self, bundle_mgr):
        with pytest.raises(BundleError, match="deprecated"):
            bundle_mgr.use("npu-v2.0")

    def test_use_deprecated_with_force(self, bundle_mgr):
        # Should not raise — but may log warnings about missing deps
        bundle_mgr.use("npu-v2.0", force=True)

    def test_use_nonexistent(self, bundle_mgr):
        with pytest.raises(BundleNotFoundError):
            bundle_mgr.use("nonexistent")

    def test_use_verified(self, bundle_mgr):
        # Will attempt to install deps (may log errors for missing archives)
        bundle_mgr.use("npu-v2.1")


class TestInstall:
    def test_install_nonexistent(self, bundle_mgr):
        with pytest.raises(BundleNotFoundError):
            bundle_mgr.install("nonexistent")


class TestExportImport:
    def test_export_creates_archive(self, bundle_mgr, project_root):
        # Create fake cached deps
        cache = project_root / "build" / "cache"
        tc_dir = cache / "npu-compiler-2.1.0"
        (tc_dir / "bin").mkdir(parents=True)
        (tc_dir / "bin" / "cc").write_text("dummy")

        lib_dir = cache / "json-c-0.17"
        (lib_dir / "include").mkdir(parents=True)
        (lib_dir / "include" / "json.h").write_text("dummy")

        output = project_root / "export" / "npu-v2.1-bundle.tar.gz"
        result = bundle_mgr.export_bundle("npu-v2.1", output)
        assert result.is_file()

        # Verify archive structure
        with tarfile.open(result, "r:gz") as tf:
            names = tf.getnames()
        assert any("bundle.yaml" in n for n in names)
        assert any("toolchains" in n for n in names)
        assert any("libraries" in n for n in names)

    def test_export_nonexistent_bundle(self, bundle_mgr, project_root):
        with pytest.raises(BundleNotFoundError):
            bundle_mgr.export_bundle("nonexistent", project_root / "out.tar.gz")

    def test_import_bundle(self, bundle_mgr, project_root):
        # First export
        cache = project_root / "build" / "cache"
        (cache / "npu-compiler-2.1.0" / "bin").mkdir(parents=True)
        (cache / "npu-compiler-2.1.0" / "bin" / "cc").write_text("dummy")

        archive = project_root / "npu-v2.1.tar.gz"
        bundle_mgr.export_bundle("npu-v2.1", archive)

        # Clean cache, then import
        import shutil
        shutil.rmtree(cache / "npu-compiler-2.1.0")

        name = bundle_mgr.import_bundle(archive)
        assert name == "npu-v2.1"

    def test_import_missing_file(self, bundle_mgr, project_root):
        with pytest.raises(BundleError, match="not found"):
            bundle_mgr.import_bundle(project_root / "nonexistent.tar.gz")


class TestGetBundleEnv:
    def test_env_from_active(self, bundle_mgr, project_root):
        env = bundle_mgr.get_bundle_env()
        assert "NPU_SDK_VERSION" in env
        assert env["NPU_SDK_VERSION"] == "2.1"

    def test_env_from_named(self, bundle_mgr):
        env = bundle_mgr.get_bundle_env("npu-v2.1")
        assert "NPU_SDK_VERSION" in env
