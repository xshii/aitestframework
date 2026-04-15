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
    LockFile,
    RepoConfig,
    ToolchainConfig,
)


class TestDataClasses:
    def test_toolchain_config_defaults(self):
        tc = ToolchainConfig(name="cc", version="1.0")
        assert tc.name == "cc"
        assert tc.version == "1.0"
        assert tc.order == 0.0
        assert tc.acquire.local_dir is None

    def test_repo_config_defaults(self):
        rc = RepoConfig(name="repo", url="git@host:repo.git", branch="main")
        assert rc.resolved_ref == "main"
        assert rc.is_commit is False
        assert rc.depth is None
        assert rc.sparse_checkout == []

    def test_repo_ref_priority_commit_over_tag_over_branch(self):
        rc = RepoConfig(name="r", url="u", branch="main", tag="v1", commitno="abcdef1")
        assert rc.resolved_ref == "abcdef1"
        assert rc.is_commit is True

        rc = RepoConfig(name="r", url="u", branch="main", tag="v1")
        assert rc.resolved_ref == "v1"
        assert rc.is_commit is False

        rc = RepoConfig(name="r", url="u", branch="feat")
        assert rc.resolved_ref == "feat"

    def test_bundle_config_defaults(self):
        bc = BundleConfig(name="b1")
        assert bc.status == "testing"
        assert bc.toolchains == {}
        assert bc.repos == {}

    def test_diag_result(self):
        r = DiagResult(check="test", ok=True, message="ok")
        assert r.ok is True

    def test_lock_file_defaults(self):
        lf = LockFile()
        assert lf.toolchains == {}
        assert lf.repos == {}

    def test_acquire_config_defaults(self):
        acq = AcquireConfig()
        assert acq.local_dir is None
        assert acq.script is None
        assert acq.srcpkg is None
        assert acq.install_dir is None


class TestExceptions:
    def test_deps_error_hierarchy(self):
        assert issubclass(DepsConfigError, DepsError)
        assert issubclass(BundleError, DepsError)
        assert issubclass(BundleNotFoundError, BundleError)
        assert issubclass(BundleNotFoundError, KeyError)

    def test_bundle_status_enum(self):
        assert BundleStatus.VERIFIED == "verified"
        assert BundleStatus.DEPRECATED == "deprecated"
