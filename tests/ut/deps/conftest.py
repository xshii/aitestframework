"""Shared fixtures for deps module tests."""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

import pytest
import yaml


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    """Create a minimal project tree with deps.yaml and directories."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "build" / "cache").mkdir(parents=True)
    (root / "build" / "repos").mkdir(parents=True)
    (root / "deps" / "uploads").mkdir(parents=True)
    return root


@pytest.fixture()
def deps_yaml(project_root: Path) -> Path:
    """Write a sample deps.yaml and return its path."""
    cfg = {
        "toolchains": {
            "npu-compiler": {
                "version": "2.1.0",
                "sha256": {},
                "bin_dir": "bin",
                "env": {"NPU_CC": "{install_dir}/bin/npu-gcc"},
                "acquire": {"local_dir": "deps/uploads/"},
            },
        },
        "libraries": {
            "json-c": {
                "version": "0.17",
                "sha256": "",
                "build_system": "cmake",
                "acquire": {"local_dir": "deps/uploads/"},
            },
        },
        "repos": {
            "npu-runtime": {
                "url": "git@10.0.0.1:hw/npu-runtime.git",
                "ref": "main",
                "depth": 1,
            },
        },
        "bundles": {
            "npu-v2.1": {
                "description": "NPU test env v2.1",
                "status": "verified",
                "toolchains": {"npu-compiler": "2.1.0"},
                "libraries": {"json-c": "0.17"},
                "repos": {"npu-runtime": "main"},
                "env": {"NPU_SDK_VERSION": "2.1"},
            },
            "npu-v2.0": {
                "description": "NPU test env v2.0 (legacy)",
                "status": "deprecated",
                "toolchains": {"npu-compiler": "2.0.3"},
            },
        },
        "active": "npu-v2.1",
    }
    p = project_root / "deps.yaml"
    with open(p, "w") as fh:
        yaml.dump(cfg, fh)
    return p


def make_tar_gz(directory: Path, archive_path: Path) -> str:
    """Create a .tar.gz from *directory* and return its SHA-256."""
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(directory, arcname=directory.name)
    h = hashlib.sha256()
    with open(archive_path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()
