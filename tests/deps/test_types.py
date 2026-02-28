"""Tests for deps.types data classes."""

from __future__ import annotations

from aitf.deps.types import (
    AcquireConfig,
    BundleConfig,
    BundleError,
    BundleNotFoundError,
    BundleStatus,
    DepStatus,
    DepsConfigError,
    DepsError,
    DiagLevel,
    DiagResult,
    LibraryConfig,
    LockEntry,
    LockFile,
    RepoConfig,
    ToolchainConfig,
)


class TestDataClasses:
    def test_toolchain_config_defaults(self):
        tc = ToolchainConfig(name="cc", version="1.0")
        assert tc.name == "cc"
        assert tc.version == "1.0"
        assert tc.sha256 == {}
        assert tc.bin_dir is None
        assert tc.env == {}

    def test_library_config_defaults(self):
        lib = LibraryConfig(name="libc", version="0.1")
        assert lib.build_system == "cmake"
        assert lib.cmake_args == []
        assert lib.build_script is None

    def test_repo_config_defaults(self):
        rc = RepoConfig(name="repo", url="git@host:repo.git")
        assert rc.ref == "main"
        assert rc.depth is None
        assert rc.sparse_checkout == []

    def test_bundle_config_defaults(self):
        bc = BundleConfig(name="b1")
        assert bc.status == "testing"
        assert bc.toolchains == {}
        assert bc.libraries == {}

    def test_diag_result(self):
        r = DiagResult(check="test", level=DiagLevel.PASS, message="ok")
        assert r.level == DiagLevel.PASS

    def test_lock_file_defaults(self):
        lf = LockFile()
        assert lf.toolchains == {}
        assert lf.libraries == {}
        assert lf.repos == {}


class TestExceptions:
    def test_deps_error_hierarchy(self):
        assert issubclass(DepsConfigError, DepsError)
        assert issubclass(BundleError, DepsError)
        assert issubclass(BundleNotFoundError, BundleError)
        assert issubclass(BundleNotFoundError, KeyError)

    def test_bundle_status_enum(self):
        assert BundleStatus.VERIFIED == "verified"
        assert BundleStatus.DEPRECATED == "deprecated"

    def test_dep_status_enum(self):
        assert DepStatus.INSTALLED == "installed"
        assert DepStatus.NOT_INSTALLED == "not_installed"
