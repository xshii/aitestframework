"""Tests for deps.doctor — dependency diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest

from aitf.deps.config import load_deps_config
from aitf.deps.doctor import run_diagnostics
from aitf.deps.types import DiagLevel


class TestRunDiagnostics:
    def test_all_missing(self, deps_yaml, project_root):
        cfg = load_deps_config(deps_yaml)
        results = run_diagnostics(
            cfg,
            cache_dir=project_root / "build" / "cache",
            repos_dir=project_root / "build" / "repos",
            project_root=project_root,
        )
        # Config check should pass
        config_check = [r for r in results if r.check == "deps.yaml configuration"]
        assert len(config_check) == 1
        assert config_check[0].level == DiagLevel.PASS

        # Toolchains and libraries should fail (not installed)
        tc_checks = [r for r in results if r.check.startswith("toolchain:")]
        assert all(r.level == DiagLevel.FAIL for r in tc_checks)

        lib_checks = [r for r in results if r.check.startswith("library:")]
        assert all(r.level == DiagLevel.FAIL for r in lib_checks)

    def test_installed_toolchain_passes(self, deps_yaml, project_root):
        cfg = load_deps_config(deps_yaml)
        cache = project_root / "build" / "cache"
        (cache / "npu-compiler-2.1.0").mkdir(parents=True)

        results = run_diagnostics(
            cfg,
            cache_dir=cache,
            repos_dir=project_root / "build" / "repos",
            project_root=project_root,
        )
        tc_checks = [r for r in results if r.check == "toolchain:npu-compiler"]
        assert len(tc_checks) == 1
        assert tc_checks[0].level == DiagLevel.PASS

    def test_build_tools_check(self, deps_yaml, project_root):
        cfg = load_deps_config(deps_yaml)
        results = run_diagnostics(
            cfg,
            cache_dir=project_root / "build" / "cache",
            repos_dir=project_root / "build" / "repos",
            project_root=project_root,
        )
        tool_checks = [r for r in results if r.check.startswith("tool:")]
        # git should be available in test environment
        git_check = [r for r in tool_checks if r.check == "tool:git"]
        assert len(git_check) == 1
        assert git_check[0].level == DiagLevel.PASS

    def test_lock_missing(self, deps_yaml, project_root):
        cfg = load_deps_config(deps_yaml)
        results = run_diagnostics(
            cfg,
            cache_dir=project_root / "build" / "cache",
            repos_dir=project_root / "build" / "repos",
            project_root=project_root,
            lock_path=project_root / "deps.lock.yaml",
        )
        lock_checks = [r for r in results if r.check == "lock_file"]
        assert any(r.level == DiagLevel.FAIL for r in lock_checks)

    def test_script_checks(self, deps_yaml, project_root):
        """Scripts referenced in config should be checked for existence."""
        cfg = load_deps_config(deps_yaml)
        # Our sample config has no scripts, so this list should be empty
        results = run_diagnostics(
            cfg,
            cache_dir=project_root / "build" / "cache",
            repos_dir=project_root / "build" / "repos",
            project_root=project_root,
        )
        script_checks = [r for r in results if r.check.startswith("script:")]
        # No scripts in fixture config, so no script checks
        assert len(script_checks) == 0
