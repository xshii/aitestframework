"""Smoke tests: verify shared library existence, loadability, and symbol exports."""

import ctypes
import os

import pytest


class TestBuildSanity:
    """Verify the stub_runner shared library is built correctly."""

    def test_so_exists(self, libstub_runner_path):
        assert libstub_runner_path.exists(), (
            f"Shared library not found: {libstub_runner_path}"
        )

    def test_so_non_empty(self, libstub_runner_path):
        assert libstub_runner_path.stat().st_size > 0, (
            "Shared library is empty"
        )

    def test_ctypes_loadable(self, libstub_runner_path):
        if not libstub_runner_path.exists():
            pytest.skip("shared library not found")
        lib = ctypes.CDLL(str(libstub_runner_path))
        assert lib is not None

    @pytest.mark.parametrize("symbol", ["stub_entry", "stub_exit"])
    def test_symbol_exported(self, libstub_runner_path, symbol):
        if not libstub_runner_path.exists():
            pytest.skip("shared library not found")
        lib = ctypes.CDLL(str(libstub_runner_path))
        fn = getattr(lib, symbol, None)
        assert fn is not None, f"Symbol '{symbol}' not exported"
