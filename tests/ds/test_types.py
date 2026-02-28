"""Tests for datastore.types — dataclasses and exceptions."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aitf.ds.types import (
    CaseData,
    CaseNotFoundError,
    DataStoreError,
    FileEntry,
    IntegrityError,
    RemoteConfig,
    SyncError,
    SyncResult,
    VerifyResult,
)


class TestFileEntry:
    def test_defaults(self):
        e = FileEntry(path="weights/w.bin")
        assert e.size == 0
        assert e.checksum == ""

    def test_all_fields(self):
        e = FileEntry(path="golden/out.bin", size=1024, checksum="sha256:abc")
        assert e.path == "golden/out.bin"
        assert e.size == 1024
        assert e.checksum == "sha256:abc"


class TestCaseData:
    def test_defaults(self):
        c = CaseData(case_id="npu/tdd/basic")
        assert c.name == ""
        assert c.version == "v1"
        assert c.files == {}
        assert c.source == "local"
        assert isinstance(c.created_at, datetime)

    def test_full_construction(self, sample_case):
        assert sample_case.platform == "npu"
        assert len(sample_case.files["weights"]) == 2
        assert sample_case.files["golden"][0].path == "golden/output.bin"


class TestRemoteConfig:
    def test_defaults(self):
        r = RemoteConfig(name="lab", host="10.0.0.1", user="ci", path="/data")
        assert r.port == 22
        assert r.ssh_key is None

    def test_with_key(self):
        r = RemoteConfig(
            name="lab", host="10.0.0.1", user="ci", path="/data",
            port=2222, ssh_key="/home/ci/.ssh/id_rsa",
        )
        assert r.port == 2222
        assert r.ssh_key == "/home/ci/.ssh/id_rsa"


class TestSyncResult:
    def test_ok_property_true(self):
        r = SyncResult(case_id="a/b/c", direction="pull", files_transferred=3)
        assert r.ok is True

    def test_ok_property_false(self):
        r = SyncResult(case_id="a/b/c", direction="push", files_failed=1)
        assert r.ok is False


class TestExceptions:
    def test_case_not_found_is_key_error(self):
        with pytest.raises(KeyError):
            raise CaseNotFoundError("npu/tdd/basic")

    def test_case_not_found_is_datastore_error(self):
        with pytest.raises(DataStoreError):
            raise CaseNotFoundError("npu/tdd/basic")

    def test_integrity_error(self):
        with pytest.raises(DataStoreError):
            raise IntegrityError("bad checksum")

    def test_sync_error(self):
        with pytest.raises(DataStoreError):
            raise SyncError("connection refused")
