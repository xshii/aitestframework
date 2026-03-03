"""Tests for aitf.config — unified config.yaml loading and mode detection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from aitf.config import AitfConfig, Mode, _determine_mode, is_local_ip, load_config


# ---------------------------------------------------------------------------
# is_local_ip
# ---------------------------------------------------------------------------

class TestIsLocalIp:
    def test_localhost_literal(self):
        assert is_local_ip("127.0.0.1") is True

    def test_localhost_name(self):
        assert is_local_ip("localhost") is True

    def test_ipv6_loopback(self):
        assert is_local_ip("::1") is True

    def test_test_net_is_remote(self):
        # 192.0.2.0/24 (TEST-NET-1) should never be a local address.
        assert is_local_ip("192.0.2.1") is False


# ---------------------------------------------------------------------------
# _determine_mode
# ---------------------------------------------------------------------------

class TestDetermineMode:
    def test_empty_string_standalone(self):
        assert _determine_mode("") == Mode.STANDALONE

    def test_remote_ip_client(self):
        assert _determine_mode("192.0.2.1") == Mode.CLIENT

    def test_local_ip_server(self):
        with patch("aitf.config.is_local_ip", return_value=True):
            assert _determine_mode("10.0.0.100") == Mode.SERVER


# ---------------------------------------------------------------------------
# AitfConfig properties
# ---------------------------------------------------------------------------

class TestAitfConfig:
    def test_default_standalone(self):
        cfg = AitfConfig()
        assert cfg.mode == Mode.STANDALONE
        assert cfg.bind_host == "127.0.0.1"
        assert cfg.server_url is None

    def test_server_mode(self):
        with patch("aitf.config.is_local_ip", return_value=True):
            cfg = AitfConfig(server="10.0.0.100", port=8080)
        assert cfg.mode == Mode.SERVER
        assert cfg.bind_host == "0.0.0.0"
        assert cfg.server_url is None

    def test_client_mode(self):
        cfg = AitfConfig(server="192.0.2.1", port=5000)
        assert cfg.mode == Mode.CLIENT
        assert cfg.bind_host == "127.0.0.1"
        assert cfg.server_url == "http://192.0.2.1:5000"

    def test_project_root_default(self):
        cfg = AitfConfig()
        assert cfg.project_root == Path(".").resolve()

    def test_build_root_default(self, tmp_path):
        cfg = AitfConfig(project_root=tmp_path)
        assert cfg.build_root == tmp_path / "build"

    def test_build_root_explicit(self, tmp_path):
        cfg = AitfConfig(project_root=tmp_path, _build_root=Path("/opt/build"))
        assert cfg.build_root == Path("/opt/build")

    def test_datastore_dir_default(self, tmp_path):
        cfg = AitfConfig(project_root=tmp_path)
        assert cfg.datastore_dir == tmp_path / "datastore"

    def test_datastore_dir_explicit(self, tmp_path):
        cfg = AitfConfig(project_root=tmp_path, _datastore_dir=Path("/data/golden"))
        assert cfg.datastore_dir == Path("/data/golden")


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_missing_file_returns_standalone(self, tmp_path):
        cfg = load_config(path=tmp_path / "nonexistent.yaml", project_root=tmp_path)
        assert cfg.mode == Mode.STANDALONE
        assert cfg.port == 5000

    def test_empty_file_returns_standalone(self, tmp_path):
        (tmp_path / "config.yaml").write_text("")
        cfg = load_config(project_root=tmp_path)
        assert cfg.mode == Mode.STANDALONE

    def test_loads_server_and_port(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            yaml.dump({"server": "192.0.2.1", "port": 8080})
        )
        cfg = load_config(project_root=tmp_path)
        assert cfg.server == "192.0.2.1"
        assert cfg.port == 8080
        assert cfg.mode == Mode.CLIENT

    def test_server_mode_via_mock(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            yaml.dump({"server": "10.0.0.100"})
        )
        with patch("aitf.config.is_local_ip", return_value=True):
            cfg = load_config(project_root=tmp_path)
        assert cfg.mode == Mode.SERVER

    def test_project_root_resolved(self, tmp_path):
        cfg = load_config(project_root=tmp_path)
        assert cfg.project_root == tmp_path.resolve()

    def test_invalid_yaml_returns_standalone(self, tmp_path):
        (tmp_path / "config.yaml").write_text(": bad: yaml: [")
        cfg = load_config(project_root=tmp_path)
        assert cfg.mode == Mode.STANDALONE

    def test_non_mapping_yaml_returns_standalone(self, tmp_path):
        (tmp_path / "config.yaml").write_text("- list\n- item\n")
        cfg = load_config(project_root=tmp_path)
        assert cfg.mode == Mode.STANDALONE

    def test_build_root_absolute(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            yaml.dump({"build_root": "/opt/aitf/build"})
        )
        cfg = load_config(project_root=tmp_path)
        assert cfg.build_root == Path("/opt/aitf/build")

    def test_build_root_relative(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            yaml.dump({"build_root": "out/build"})
        )
        cfg = load_config(project_root=tmp_path)
        assert cfg.build_root == (tmp_path / "out/build").resolve()

    def test_datastore_dir_from_yaml(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            yaml.dump({"datastore_dir": "/data/golden"})
        )
        cfg = load_config(project_root=tmp_path)
        assert cfg.datastore_dir == Path("/data/golden")

    def test_paths_default_when_unset(self, tmp_path):
        (tmp_path / "config.yaml").write_text(yaml.dump({"port": 8080}))
        cfg = load_config(project_root=tmp_path)
        assert cfg.build_root == tmp_path / "build"
        assert cfg.datastore_dir == tmp_path / "datastore"
