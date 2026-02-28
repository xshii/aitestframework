"""Tests for deps.types data classes."""

from __future__ import annotations

from aitf.deps.types import (
    AcquireConfig,
    BundleConfig,
    BundleError,
    BundleNotFoundError,
    BundleStatus,
    DepsConfigError,
    DepsError,
    DiagResult,
    LibraryConfig,
    LockEntry,
    LockFile,
    RemoteDepotConfig,
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
        r = DiagResult(check="test", ok=True, message="ok")
        assert r.ok is True

    def test_lock_file_defaults(self):
        lf = LockFile()
        assert lf.toolchains == {}
        assert lf.libraries == {}
        assert lf.repos == {}

    def test_acquire_config_remote_flag(self):
        acq = AcquireConfig(remote=True)
        assert acq.remote is True
        assert acq.local_dir is None
        assert acq.script is None

    def test_remote_depot_config(self):
        rdc = RemoteDepotConfig(host="10.0.0.1", user="deploy", path="/data/deps")
        assert rdc.host == "10.0.0.1"
        assert rdc.port == 22
        assert rdc.key_file is None


class TestExceptions:
    def test_deps_error_hierarchy(self):
        assert issubclass(DepsConfigError, DepsError)
        assert issubclass(BundleError, DepsError)
        assert issubclass(BundleNotFoundError, BundleError)
        assert issubclass(BundleNotFoundError, KeyError)

    def test_bundle_status_enum(self):
        assert BundleStatus.VERIFIED == "verified"
        assert BundleStatus.DEPRECATED == "deprecated"
