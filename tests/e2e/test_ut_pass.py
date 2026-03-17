"""Smoke test: verify all ctest unit tests pass."""

import subprocess

import pytest


class TestUnitTests:
    """Run ctest inside the build directory and verify all tests pass."""

    def test_ctest_all_pass(self, build_dir):
        if not build_dir.exists():
            pytest.skip("build directory not found")

        result = subprocess.run(
            ["ctest", "--test-dir", str(build_dir), "--output-on-failure"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"ctest failed (rc={result.returncode}):\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    def test_ctest_reports_four_tests(self, build_dir):
        """Verify the expected number of tests were discovered."""
        if not build_dir.exists():
            pytest.skip("build directory not found")

        result = subprocess.run(
            ["ctest", "--test-dir", str(build_dir), "-N"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        # ctest -N prints "Total Tests: N"
        for line in result.stdout.splitlines():
            if "Total Tests:" in line:
                count = int(line.split(":")[-1].strip())
                assert count >= 4, f"Expected >= 4 tests, got {count}"
                return
        pytest.fail("Could not parse ctest -N output")
