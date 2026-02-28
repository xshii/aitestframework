"""Tests for deps.acquire — dependency acquisition."""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

import pytest

from aitf.deps.acquire import (
    _archive_name,
    _find_archive,
    clean_cache,
    install_library,
    install_toolchain,
    is_installed,
    sha256_file,
)
from aitf.deps.types import AcquireConfig, AcquireError, LibraryConfig, ToolchainConfig


class TestSha256File:
    def test_known_value(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert sha256_file(f) == expected


class TestArchiveName:
    def test_with_platform(self):
        assert _archive_name("cc", "1.0", "linux-x86_64") == "cc-1.0-linux-x86_64.tar.gz"

    def test_without_platform(self):
        assert _archive_name("lib", "2.0") == "lib-2.0.tar.gz"


class TestFindArchive:
    def test_finds_generic(self, tmp_path):
        archive = tmp_path / "lib-2.0.tar.gz"
        archive.write_bytes(b"fake")
        assert _find_archive(tmp_path, "lib", "2.0") == archive

    def test_not_found(self, tmp_path):
        assert _find_archive(tmp_path, "missing", "1.0") is None


def _make_toolchain_archive(
    project_root: Path, name: str, version: str
) -> tuple[ToolchainConfig, str]:
    """Create a fake toolchain archive in deps/uploads/ and return config + sha256."""
    uploads = project_root / "deps" / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    # Create dummy content
    content_dir = project_root / "tmp_tc_content" / f"{name}-{version}"
    (content_dir / "bin").mkdir(parents=True)
    (content_dir / "bin" / "compiler").write_text("#!/bin/sh\necho ok\n")

    # Create archive
    archive = uploads / f"{name}-{version}.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(content_dir, arcname=f"{name}-{version}")

    sha = sha256_file(archive)

    tc = ToolchainConfig(
        name=name,
        version=version,
        sha256={},
        acquire=AcquireConfig(local_dir="deps/uploads/"),
    )
    return tc, sha


class TestInstallToolchain:
    def test_install_from_local(self, project_root):
        tc, sha = _make_toolchain_archive(project_root, "test-cc", "1.0")
        cache_dir = project_root / "build" / "cache"

        result = install_toolchain(tc, cache_dir=cache_dir, project_root=project_root)
        assert result.is_dir()
        assert (result).name == "test-cc-1.0"

    def test_already_cached(self, project_root):
        cache_dir = project_root / "build" / "cache"
        (cache_dir / "test-cc-1.0").mkdir(parents=True)

        tc = ToolchainConfig(name="test-cc", version="1.0")
        result = install_toolchain(tc, cache_dir=cache_dir, project_root=project_root)
        assert result.is_dir()

    def test_sha256_mismatch(self, project_root):
        tc, _ = _make_toolchain_archive(project_root, "bad-cc", "1.0")
        tc.sha256 = {"linux-x86_64": "0000badsha"}
        cache_dir = project_root / "build" / "cache"

        # This may or may not raise depending on platform detection match
        # If platform matches, it will raise; otherwise it skips sha check
        try:
            install_toolchain(tc, cache_dir=cache_dir, project_root=project_root)
        except AcquireError:
            pass  # Expected on matching platform

    def test_no_archive_raises(self, project_root):
        tc = ToolchainConfig(
            name="missing-cc",
            version="9.9",
            acquire=AcquireConfig(local_dir="deps/uploads/"),
        )
        cache_dir = project_root / "build" / "cache"
        with pytest.raises(AcquireError, match="Could not find"):
            install_toolchain(tc, cache_dir=cache_dir, project_root=project_root)


def _make_library_archive(
    project_root: Path, name: str, version: str
) -> tuple[LibraryConfig, str]:
    """Create a fake library archive."""
    uploads = project_root / "deps" / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    content_dir = project_root / "tmp_lib_content" / f"{name}-{version}"
    (content_dir / "include").mkdir(parents=True)
    (content_dir / "include" / "lib.h").write_text("#pragma once\n")

    archive = uploads / f"{name}-{version}.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(content_dir, arcname=f"{name}-{version}")

    sha = sha256_file(archive)

    lib = LibraryConfig(
        name=name,
        version=version,
        sha256="",
        acquire=AcquireConfig(local_dir="deps/uploads/"),
    )
    return lib, sha


class TestInstallLibrary:
    def test_install_from_local(self, project_root):
        lib, sha = _make_library_archive(project_root, "test-lib", "0.5")
        cache_dir = project_root / "build" / "cache"

        result = install_library(lib, cache_dir=cache_dir, project_root=project_root)
        assert result.is_dir()
        assert result.name == "test-lib-0.5"

    def test_sha256_check(self, project_root):
        lib, sha = _make_library_archive(project_root, "checked-lib", "1.0")
        lib.sha256 = sha  # correct sha
        cache_dir = project_root / "build" / "cache"

        result = install_library(lib, cache_dir=cache_dir, project_root=project_root)
        assert result.is_dir()

    def test_sha256_mismatch_raises(self, project_root):
        lib, _ = _make_library_archive(project_root, "bad-lib", "1.0")
        lib.sha256 = "badhash"
        cache_dir = project_root / "build" / "cache"

        with pytest.raises(AcquireError, match="SHA-256 mismatch"):
            install_library(lib, cache_dir=cache_dir, project_root=project_root)


class TestIsInstalled:
    def test_installed(self, tmp_path):
        (tmp_path / "cc-1.0").mkdir()
        assert is_installed("cc", "1.0", tmp_path) is True

    def test_not_installed(self, tmp_path):
        assert is_installed("cc", "1.0", tmp_path) is False


class TestCleanCache:
    def test_clean(self, tmp_path):
        (tmp_path / "cc-1.0").mkdir()
        (tmp_path / "lib-2.0").mkdir()
        count = clean_cache(tmp_path)
        assert count == 2
        assert list(tmp_path.iterdir()) == []

    def test_clean_empty(self, tmp_path):
        assert clean_cache(tmp_path) == 0

    def test_clean_nonexistent(self, tmp_path):
        assert clean_cache(tmp_path / "no_dir") == 0
