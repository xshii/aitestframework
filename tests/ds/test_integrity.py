"""Tests for datastore.integrity — SHA-256, directory scan, verification."""

from __future__ import annotations

from pathlib import Path

import pytest

from aitf.ds.integrity import compute_sha256, scan_directory, verify_case
from aitf.ds.types import CaseData, FileEntry


class TestComputeSha256:
    def test_small_file(self, tmp_path):
        p = tmp_path / "small.bin"
        p.write_bytes(b"hello world")
        result = compute_sha256(p)
        assert result.startswith("sha256:")
        assert len(result) == len("sha256:") + 64

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        result = compute_sha256(p)
        # SHA-256 of empty input is well-known
        assert result == "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_deterministic(self, tmp_path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"\x00" * 10000)
        assert compute_sha256(p) == compute_sha256(p)

    def test_large_file(self, tmp_path):
        """Ensure chunked reading works for files larger than the chunk size."""
        p = tmp_path / "large.bin"
        # Write 9 MB (exceeds the 8 MB chunk)
        p.write_bytes(b"\xAB" * (9 * 1024 * 1024))
        result = compute_sha256(p)
        assert result.startswith("sha256:")


class TestScanDirectory:
    def test_scan_weights(self, store_dir):
        case_dir = store_dir / "npu" / "tdd" / "fp32_basic"
        entries = scan_directory(case_dir, "weights")
        assert len(entries) == 2
        paths = {e.path for e in entries}
        assert "weights/conv_w.bin" in paths
        assert "weights/conv_b.bin" in paths
        for e in entries:
            assert e.size > 0
            assert e.checksum.startswith("sha256:")

    def test_scan_missing_dir(self, tmp_path):
        entries = scan_directory(tmp_path / "nonexistent", "weights")
        assert entries == []

    def test_scan_empty_dir(self, tmp_path):
        (tmp_path / "inputs").mkdir()
        entries = scan_directory(tmp_path, "inputs")
        assert entries == []


class TestVerifyCase:
    def test_all_pass(self, store_dir):
        case_dir = store_dir / "npu" / "tdd" / "fp32_basic"
        # Build a CaseData from actual on-disk checksums
        weights = scan_directory(case_dir, "weights")
        golden = scan_directory(case_dir, "golden")
        case = CaseData(
            case_id="npu/tdd/fp32_basic",
            files={"weights": weights, "golden": golden},
        )
        results = verify_case(case_dir, case)
        assert all(r.ok for r in results)
        assert len(results) == 3

    def test_checksum_mismatch(self, store_dir):
        case_dir = store_dir / "npu" / "tdd" / "fp32_basic"
        case = CaseData(
            case_id="npu/tdd/fp32_basic",
            files={
                "weights": [
                    FileEntry(path="weights/conv_w.bin", size=64, checksum="sha256:wrong"),
                ],
            },
        )
        results = verify_case(case_dir, case)
        assert len(results) == 1
        assert results[0].ok is False

    def test_missing_file(self, store_dir):
        case_dir = store_dir / "npu" / "tdd" / "fp32_basic"
        case = CaseData(
            case_id="npu/tdd/fp32_basic",
            files={
                "weights": [
                    FileEntry(path="weights/nonexistent.bin", checksum="sha256:xxx"),
                ],
            },
        )
        results = verify_case(case_dir, case)
        assert len(results) == 1
        assert results[0].ok is False
        assert results[0].error == "file not found"
