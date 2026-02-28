"""Tests for datastore.sftp — SftpClient with mocked paramiko."""

from __future__ import annotations

import stat as stat_mod
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aitf.ds.sftp import SftpClient
from aitf.ds.types import RemoteConfig, SyncError


@pytest.fixture()
def remote_config():
    return RemoteConfig(
        name="test-server",
        host="10.0.0.1",
        user="ci",
        path="/data/store",
        port=22,
    )


@pytest.fixture()
def mock_sftp():
    """Return a mock SFTPClient."""
    return MagicMock()


@pytest.fixture()
def client(remote_config, mock_sftp):
    """Return an SftpClient with a pre-injected mock SFTP channel."""
    c = SftpClient(remote_config)
    c._sftp = mock_sftp
    c._transport = MagicMock()
    return c


class TestConnect:
    @patch("aitf.ds.sftp.paramiko")
    def test_connect_with_password(self, mock_paramiko, remote_config):
        client = SftpClient(remote_config)
        client.connect()
        mock_paramiko.Transport.assert_called_once_with(("10.0.0.1", 22))
        mock_paramiko.SFTPClient.from_transport.assert_called_once()

    @patch("aitf.ds.sftp.paramiko")
    def test_connect_with_key(self, mock_paramiko):
        cfg = RemoteConfig(
            name="lab", host="10.0.0.1", user="ci", path="/data",
            ssh_key="/home/ci/.ssh/id_rsa",
        )
        client = SftpClient(cfg)
        client.connect()
        mock_paramiko.RSAKey.from_private_key_file.assert_called_once_with(
            "/home/ci/.ssh/id_rsa"
        )

    @patch("aitf.ds.sftp.paramiko")
    def test_context_manager(self, mock_paramiko, remote_config):
        with SftpClient(remote_config) as c:
            assert c._sftp is not None
        # close should have been called
        mock_paramiko.Transport.return_value.close.assert_called_once()


class TestPullCase:
    def test_pull_downloads_files(self, client, mock_sftp, tmp_path):
        # Set up remote listing
        entry = MagicMock()
        entry.filename = "w.bin"
        entry.st_mode = stat_mod.S_IFREG
        entry.st_size = 100
        mock_sftp.listdir_attr.return_value = [entry]

        # Mock get to actually create the file
        def fake_get(remote, local):
            Path(local).parent.mkdir(parents=True, exist_ok=True)
            Path(local).write_bytes(b"\x00" * 100)

        mock_sftp.get.side_effect = fake_get

        store = tmp_path / "store"
        result = client.pull_case("npu/tdd/fp32", str(store), "/data/store")
        assert result.direction == "pull"
        # Mock returns the same file for all 3 default pull types (weights, inputs, golden)
        assert result.files_transferred == 3
        assert result.ok

    def test_pull_skips_existing(self, client, mock_sftp, tmp_path):
        # Pre-create the local file with matching size
        store = tmp_path / "store"
        local_file = store / "npu" / "tdd" / "fp32" / "weights" / "w.bin"
        local_file.parent.mkdir(parents=True)
        local_file.write_bytes(b"\x00" * 100)

        entry = MagicMock()
        entry.filename = "w.bin"
        entry.st_mode = stat_mod.S_IFREG
        entry.st_size = 100
        mock_sftp.listdir_attr.return_value = [entry]

        result = client.pull_case(
            "npu/tdd/fp32", str(store), "/data/store", data_types=("weights",),
        )
        assert result.files_skipped == 1
        assert result.files_transferred == 0

    def test_pull_missing_remote_dir(self, client, mock_sftp, tmp_path):
        mock_sftp.listdir_attr.side_effect = FileNotFoundError
        store = tmp_path / "store"
        result = client.pull_case("npu/tdd/fp32", str(store), "/data/store")
        assert result.files_transferred == 0
        assert result.ok


class TestPushCase:
    def test_push_uploads_files(self, client, mock_sftp, tmp_path):
        # Create local files
        store = tmp_path / "store"
        case_dir = store / "npu" / "tdd" / "fp32" / "weights"
        case_dir.mkdir(parents=True)
        (case_dir / "w.bin").write_bytes(b"\x01" * 50)

        mock_sftp.stat.side_effect = FileNotFoundError  # remote dirs don't exist

        result = client.push_case("npu/tdd/fp32", str(store), "/data/store")
        assert result.direction == "push"
        assert result.files_transferred == 1
        assert result.ok


class TestPushArtifacts:
    def test_push_artifacts_only(self, client, mock_sftp, tmp_path):
        store = tmp_path / "store"
        art_dir = store / "npu" / "tdd" / "fp32" / "artifacts"
        art_dir.mkdir(parents=True)
        (art_dir / "result.bin").write_bytes(b"\xFF" * 20)

        mock_sftp.stat.side_effect = FileNotFoundError

        result = client.push_case(
            "npu/tdd/fp32", str(store), "/data/store",
            data_types=("artifacts",),
        )
        assert result.files_transferred == 1
        assert result.ok

    def test_push_artifacts_custom_dir(self, client, mock_sftp, tmp_path):
        custom = tmp_path / "custom_artifacts"
        custom.mkdir()
        (custom / "out.bin").write_bytes(b"\xAA" * 10)

        mock_sftp.stat.side_effect = FileNotFoundError

        result = client.push_case(
            "npu/tdd/fp32", str(tmp_path / "store"), "/data/store",
            data_types=("artifacts",),
            src_override=str(custom),
        )
        assert result.files_transferred == 1


class TestRetry:
    @patch("aitf.ds.sftp._RETRY_DELAY", 0)
    def test_retry_on_failure(self, client, mock_sftp, tmp_path):
        store = tmp_path / "store"
        case_dir = store / "npu" / "tdd" / "fp32" / "weights"
        case_dir.mkdir(parents=True)
        (case_dir / "w.bin").write_bytes(b"\x01" * 10)

        # Fail twice, succeed on third try
        call_count = 0

        def flaky_put(local, remote):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("connection reset")

        mock_sftp.put.side_effect = flaky_put
        mock_sftp.stat.side_effect = FileNotFoundError

        result = client.push_case("npu/tdd/fp32", str(store), "/data/store")
        assert result.files_transferred == 1
        assert call_count == 3
