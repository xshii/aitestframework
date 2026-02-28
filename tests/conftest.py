"""Pytest fixtures for stub framework smoke tests."""

import pathlib

import pytest
import yaml


def pytest_addoption(parser):
    parser.addoption(
        "--platform",
        default="func_sim",
        help="Hook platform used for the build (default: func_sim)",
    )
    parser.addoption(
        "--deps-config",
        default=None,
        help="Path to a deps_config YAML file (optional)",
    )


@pytest.fixture(scope="session")
def platform(request):
    return request.config.getoption("--platform")


@pytest.fixture(scope="session")
def deps_config(request):
    path = request.config.getoption("--deps-config")
    if path is None:
        return {}
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


@pytest.fixture(scope="session")
def project_root():
    return pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def build_dir(project_root, platform):
    return project_root / "build" / platform


@pytest.fixture(scope="session")
def libstub_runner_path(build_dir):
    """Return the path to libstub_runner.so (or .dylib on macOS)."""
    for suffix in (".so", ".dylib"):
        candidate = build_dir / f"libstub_runner{suffix}"
        if candidate.exists():
            return candidate
    return build_dir / "libstub_runner.so"
