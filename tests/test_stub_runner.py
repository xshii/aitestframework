"""Smoke test: run func_sim_main and verify basic behaviour."""

import subprocess

import pytest


@pytest.fixture(scope="session")
def func_sim_main_path(build_dir):
    """Locate the func_sim_main executable in the build tree."""
    candidate = build_dir / "func_sim_main"
    if candidate.exists():
        return candidate
    # Some CMake generators place binaries in subdirectories
    for child in build_dir.rglob("func_sim_main"):
        if child.is_file():
            return child
    return candidate  # return default path even if missing


class TestStubRunner:
    """Integration tests exercising func_sim_main."""

    def test_list_models(self, func_sim_main_path):
        if not func_sim_main_path.exists():
            pytest.skip("func_sim_main not found")

        result = subprocess.run(
            [str(func_sim_main_path), "--list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"--list failed (rc={result.returncode}): {result.stderr}"
        )
        output = result.stdout.lower()
        assert "tdd" in output, "tdd model not listed"
        assert "fdd" in output, "fdd model not listed"

    def test_run_all_models(self, func_sim_main_path):
        if not func_sim_main_path.exists():
            pytest.skip("func_sim_main not found")

        result = subprocess.run(
            [str(func_sim_main_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"func_sim_main failed (rc={result.returncode}):\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    def test_run_single_model(self, func_sim_main_path):
        if not func_sim_main_path.exists():
            pytest.skip("func_sim_main not found")

        result = subprocess.run(
            [str(func_sim_main_path), "--models", "tdd"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"func_sim_main --models tdd failed (rc={result.returncode}):\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
