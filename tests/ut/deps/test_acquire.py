"""Tests for deps.acquire — dependency acquisition."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import pytest

from aitf.deps.acquire import (
    _find_archive,
    clean_cache,
    install_toolchain,
    is_installed,
)
from aitf.deps.types import AcquireConfig, AcquireError, ToolchainConfig


class TestFindArchive:
    def test_finds_generic(self, tmp_path):
        archive = tmp_path / "lib-2.0.tar.gz"
        archive.write_bytes(b"fake")
        assert _find_archive(tmp_path, "lib", "2.0", "linux-x86_64") == archive

    def test_finds_platform_specific(self, tmp_path):
        generic = tmp_path / "lib-2.0.tar.gz"
        generic.write_bytes(b"generic")
        specific = tmp_path / "lib-2.0-linux-x86_64.tar.gz"
        specific.write_bytes(b"specific")
        # Platform-specific should be preferred
        assert _find_archive(tmp_path, "lib", "2.0", "linux-x86_64") == specific

    def test_not_found(self, tmp_path):
        assert _find_archive(tmp_path, "missing", "1.0", "linux-x86_64") is None


def _make_tarball(project_root: Path, name: str, version: str,
                  filename: str | None = None) -> Path:
    uploads = project_root / "deps" / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    content_dir = project_root / "tmp_tc_content" / f"{name}-{version}"
    (content_dir / "bin").mkdir(parents=True, exist_ok=True)
    (content_dir / "bin" / "compiler").write_text("#!/bin/sh\necho ok\n")

    archive = uploads / (filename or f"{name}-{version}.tar.gz")
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(content_dir, arcname=f"{name}-{version}")
    return archive


class TestInstallToolchain:
    def test_install_from_local(self, project_root):
        _make_tarball(project_root, "test-cc", "1.0")
        tc = ToolchainConfig(
            name="test-cc", version="1.0",
            acquire=AcquireConfig(local_dir="deps/uploads/"),
        )
        cache_dir = project_root / "build" / "cache"
        result = install_toolchain(tc, cache_dir=cache_dir, project_root=project_root)
        assert result.is_dir()
        assert result.name == "test-cc"

    def test_already_cached(self, project_root):
        cache_dir = project_root / "build" / "cache"
        cached = cache_dir / "test-cc"
        cached.mkdir(parents=True)
        (cached / "marker").write_text("x")  # non-empty

        tc = ToolchainConfig(name="test-cc", version="1.0")
        result = install_toolchain(tc, cache_dir=cache_dir, project_root=project_root)
        assert result.is_dir()

    def test_no_archive_raises(self, project_root):
        tc = ToolchainConfig(
            name="missing-cc", version="9.9",
            acquire=AcquireConfig(local_dir="deps/uploads/"),
        )
        cache_dir = project_root / "build" / "cache"
        with pytest.raises(AcquireError, match="Could not find"):
            install_toolchain(tc, cache_dir=cache_dir, project_root=project_root)

    def test_srcpkg_explicit_filename(self, project_root):
        _make_tarball(project_root, "test-cc", "1.0", filename="custom-pkg.tar.gz")
        tc = ToolchainConfig(
            name="test-cc", version="1.0",
            acquire=AcquireConfig(
                local_dir="deps/uploads/",
                srcpkg="custom-pkg.tar.gz",
            ),
        )
        cache_dir = project_root / "build" / "cache"
        result = install_toolchain(tc, cache_dir=cache_dir, project_root=project_root)
        assert result.is_dir()

    def test_zip_archive(self, project_root):
        uploads = project_root / "deps" / "uploads"
        archive = uploads / "test-zip-1.0.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("test-zip-1.0/file.txt", "hello")
        tc = ToolchainConfig(
            name="test-zip", version="1.0",
            acquire=AcquireConfig(local_dir="deps/uploads/"),
        )
        cache_dir = project_root / "build" / "cache"
        result = install_toolchain(tc, cache_dir=cache_dir, project_root=project_root)
        assert result.is_dir()
        assert (result / "test-zip-1.0" / "file.txt").read_text() == "hello"

    def test_custom_install_dir(self, project_root):
        _make_tarball(project_root, "test-cc", "1.0")
        tc = ToolchainConfig(
            name="test-cc", version="1.0",
            acquire=AcquireConfig(local_dir="deps/uploads/"),
        )
        cache_dir = project_root / "build" / "cache"
        custom = project_root / "my_install"
        result = install_toolchain(
            tc, cache_dir=cache_dir, project_root=project_root, install_dir=custom,
        )
        assert result == custom
        assert result.is_dir()


class TestIsInstalled:
    def test_installed(self, tmp_path):
        (tmp_path / "cc").mkdir()
        (tmp_path / "cc" / "marker").write_text("x")
        assert is_installed("cc", "1.0", tmp_path) is True

    def test_empty_dir_not_installed(self, tmp_path):
        (tmp_path / "cc").mkdir()
        assert is_installed("cc", "1.0", tmp_path) is False

    def test_not_installed(self, tmp_path):
        assert is_installed("cc", "1.0", tmp_path) is False


class TestCleanCache:
    def test_clean(self, tmp_path):
        (tmp_path / "cc").mkdir()
        (tmp_path / "lib").mkdir()
        count = clean_cache(tmp_path)
        assert count == 2
        assert list(tmp_path.iterdir()) == []

    def test_clean_empty(self, tmp_path):
        assert clean_cache(tmp_path) == 0

    def test_clean_nonexistent(self, tmp_path):
        assert clean_cache(tmp_path / "no_dir") == 0
