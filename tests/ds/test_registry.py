"""Tests for datastore.registry — YAML-based case registry."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aitf.ds.registry import (
    case_to_dict,
    delete_case,
    dict_to_case,
    load_all_cases,
    load_case,
    save_case,
)
from aitf.ds.types import CaseData, FileEntry


class TestSerialization:
    def test_round_trip(self, sample_case):
        d = case_to_dict(sample_case)
        restored = dict_to_case(sample_case.case_id, d)
        assert restored.case_id == sample_case.case_id
        assert restored.platform == sample_case.platform
        assert restored.model == sample_case.model
        assert restored.variant == sample_case.variant
        assert restored.version == sample_case.version
        assert len(restored.files["weights"]) == 2
        assert restored.files["weights"][0].checksum == "sha256:aaa"

    def test_datetime_serialization(self, sample_case):
        d = case_to_dict(sample_case)
        assert isinstance(d["created_at"], str)
        restored = dict_to_case(sample_case.case_id, d)
        assert isinstance(restored.created_at, datetime)

    def test_missing_fields_use_defaults(self):
        c = dict_to_case("x/y/z", {})
        assert c.name == ""
        assert c.version == "v1"
        assert c.files == {}

    def test_none_files(self):
        c = dict_to_case("x/y/z", {"files": None})
        assert c.files == {}


class TestSaveAndLoad:
    def test_save_creates_yaml(self, registry_dir, sample_case):
        save_case(registry_dir, sample_case)
        yf = registry_dir / "npu_tdd.yaml"
        assert yf.exists()

    def test_load_case(self, registry_dir, sample_case):
        save_case(registry_dir, sample_case)
        loaded = load_case(registry_dir, sample_case.case_id)
        assert loaded is not None
        assert loaded.case_id == sample_case.case_id

    def test_load_case_not_found(self, registry_dir):
        result = load_case(registry_dir, "does/not/exist")
        assert result is None

    def test_load_all_cases(self, registry_dir, sample_case):
        save_case(registry_dir, sample_case)
        # Add a second case in a different file
        case2 = CaseData(case_id="gpu/fdd/int8")
        save_case(registry_dir, case2)
        cases = load_all_cases(registry_dir)
        assert len(cases) == 2
        ids = {c.case_id for c in cases}
        assert "npu/tdd/fp32_basic" in ids
        assert "gpu/fdd/int8" in ids

    def test_load_all_empty_dir(self, registry_dir):
        cases = load_all_cases(registry_dir)
        assert cases == []

    def test_load_all_nonexistent_dir(self, tmp_path):
        cases = load_all_cases(tmp_path / "nope")
        assert cases == []

    def test_save_overwrites(self, registry_dir, sample_case):
        save_case(registry_dir, sample_case)
        sample_case.version = "v2"
        save_case(registry_dir, sample_case)
        loaded = load_case(registry_dir, sample_case.case_id)
        assert loaded is not None
        assert loaded.version == "v2"


class TestDeleteCase:
    def test_delete_existing(self, registry_dir, sample_case):
        save_case(registry_dir, sample_case)
        assert delete_case(registry_dir, sample_case.case_id) is True
        assert load_case(registry_dir, sample_case.case_id) is None

    def test_delete_removes_empty_yaml(self, registry_dir, sample_case):
        save_case(registry_dir, sample_case)
        delete_case(registry_dir, sample_case.case_id)
        yf = registry_dir / "npu_tdd.yaml"
        assert not yf.exists()

    def test_delete_nonexistent(self, registry_dir):
        assert delete_case(registry_dir, "no/such/case") is False

    def test_delete_preserves_siblings(self, registry_dir):
        c1 = CaseData(case_id="npu/tdd/a")
        c2 = CaseData(case_id="npu/tdd/b")
        save_case(registry_dir, c1)
        save_case(registry_dir, c2)
        delete_case(registry_dir, "npu/tdd/a")
        assert load_case(registry_dir, "npu/tdd/a") is None
        assert load_case(registry_dir, "npu/tdd/b") is not None
