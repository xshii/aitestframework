"""Tests for datastore.cache — SQLite cache layer."""

from __future__ import annotations

import pytest

from aitf.ds.cache import (
    delete_case,
    query_case,
    query_cases,
    rebuild_cache,
    upsert_case,
)
from aitf.ds.registry import save_case as save_yaml
from aitf.ds.types import CaseData, FileEntry


class TestUpsertAndQuery:
    def test_upsert_then_query(self, session, sample_case):
        upsert_case(session, sample_case)
        loaded = query_case(session, sample_case.case_id)
        assert loaded is not None
        assert loaded.case_id == sample_case.case_id
        assert loaded.platform == "npu"
        assert len(loaded.files["weights"]) == 2

    def test_query_missing(self, session):
        assert query_case(session, "no/such/case") is None

    def test_upsert_replaces(self, session, sample_case):
        upsert_case(session, sample_case)
        sample_case.version = "v99"
        sample_case.files = {}
        upsert_case(session, sample_case)
        loaded = query_case(session, sample_case.case_id)
        assert loaded is not None
        assert loaded.version == "v99"
        assert loaded.files == {}


class TestQueryCases:
    def test_filter_by_platform(self, session):
        c1 = CaseData(case_id="npu/a/b")
        c2 = CaseData(case_id="gpu/a/b")
        upsert_case(session, c1)
        upsert_case(session, c2)
        results = query_cases(session, platform="npu")
        assert len(results) == 1
        assert results[0].case_id == "npu/a/b"

    def test_filter_by_model(self, session):
        c1 = CaseData(case_id="npu/tdd/a")
        c2 = CaseData(case_id="npu/fdd/a")
        upsert_case(session, c1)
        upsert_case(session, c2)
        results = query_cases(session, model="fdd")
        assert len(results) == 1
        assert results[0].model == "fdd"

    def test_no_filter(self, session):
        c1 = CaseData(case_id="a/b/c")
        c2 = CaseData(case_id="d/e/f")
        upsert_case(session, c1)
        upsert_case(session, c2)
        results = query_cases(session)
        assert len(results) == 2

    def test_combined_filter(self, session):
        c1 = CaseData(case_id="npu/tdd/a")
        c2 = CaseData(case_id="npu/fdd/b")
        c3 = CaseData(case_id="gpu/tdd/c")
        upsert_case(session, c1)
        upsert_case(session, c2)
        upsert_case(session, c3)
        results = query_cases(session, platform="npu", model="tdd")
        assert len(results) == 1
        assert results[0].case_id == "npu/tdd/a"


class TestDeleteCase:
    def test_delete_existing(self, session, sample_case):
        upsert_case(session, sample_case)
        delete_case(session, sample_case.case_id)
        assert query_case(session, sample_case.case_id) is None

    def test_delete_cascades_files(self, session, sample_case):
        upsert_case(session, sample_case)
        delete_case(session, sample_case.case_id)
        from aitf.comm.models import FileEntryRow

        remaining = session.query(FileEntryRow).filter_by(case_id=sample_case.case_id).all()
        assert remaining == []

    def test_delete_nonexistent(self, session):
        # Should not raise
        delete_case(session, "no/such/case")


class TestRebuildCache:
    def test_rebuild_from_yaml(self, session, registry_dir, sample_case):
        save_yaml(registry_dir, sample_case)
        count = rebuild_cache(session, registry_dir)
        assert count == 1
        loaded = query_case(session, sample_case.case_id)
        assert loaded is not None
        assert loaded.platform == "npu"

    def test_rebuild_clears_stale(self, session, registry_dir, sample_case):
        # Insert directly, then rebuild from empty YAML
        upsert_case(session, sample_case)
        count = rebuild_cache(session, registry_dir)
        assert count == 0
        assert query_case(session, sample_case.case_id) is None
