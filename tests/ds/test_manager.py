"""Tests for datastore.manager — DataStoreManager integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aitf.comm import db as core_db
from aitf.ds.manager import DataStoreManager
from aitf.ds.types import CaseNotFoundError


@pytest.fixture()
def manager(tmp_path):
    """Create a DataStoreManager with temp directories."""
    core_db.reset()
    base = tmp_path / "datastore"
    base.mkdir()
    db_path = str(tmp_path / "test.db")
    mgr = DataStoreManager(base_dir=str(base), db_path=db_path)
    yield mgr
    core_db.reset()


@pytest.fixture()
def case_dir(tmp_path):
    """Create a case directory with sample files on disk."""
    d = tmp_path / "local_case"
    (d / "weights").mkdir(parents=True)
    (d / "weights" / "w.bin").write_bytes(b"\x01" * 64)
    (d / "golden").mkdir()
    (d / "golden" / "out.bin").write_bytes(b"\x02" * 32)
    return d


class TestRegister:
    def test_register_and_get(self, manager, case_dir):
        case = manager.register("npu/tdd/fp32", str(case_dir))
        assert case.case_id == "npu/tdd/fp32"
        assert case.platform == "npu"
        assert case.model == "tdd"
        assert case.variant == "fp32"
        assert "weights" in case.files
        assert "golden" in case.files

        fetched = manager.get("npu/tdd/fp32")
        assert fetched.case_id == case.case_id

    def test_register_bad_id(self, manager, case_dir):
        with pytest.raises(ValueError, match="case_id must be"):
            manager.register("bad_id", str(case_dir))

    def test_register_no_data(self, manager, tmp_path):
        empty = tmp_path / "empty_case"
        empty.mkdir()
        case = manager.register("npu/tdd/empty", str(empty))
        assert case.files == {}


class TestGetAndDelete:
    def test_get_not_found(self, manager):
        with pytest.raises(CaseNotFoundError):
            manager.get("no/such/case")

    def test_delete(self, manager, case_dir):
        manager.register("npu/tdd/fp32", str(case_dir))
        manager.delete("npu/tdd/fp32")
        with pytest.raises(CaseNotFoundError):
            manager.get("npu/tdd/fp32")

    def test_delete_not_found(self, manager):
        with pytest.raises(CaseNotFoundError):
            manager.delete("no/such/case")


class TestList:
    def test_list_all(self, manager, case_dir):
        manager.register("npu/tdd/a", str(case_dir))
        manager.register("npu/fdd/b", str(case_dir))
        cases = manager.list()
        assert len(cases) == 2

    def test_list_filter_platform(self, manager, case_dir):
        manager.register("npu/tdd/a", str(case_dir))
        manager.register("gpu/tdd/b", str(case_dir))
        cases = manager.list(platform="npu")
        assert len(cases) == 1
        assert cases[0].platform == "npu"

    def test_list_filter_model(self, manager, case_dir):
        manager.register("npu/tdd/a", str(case_dir))
        manager.register("npu/fdd/b", str(case_dir))
        cases = manager.list(model="fdd")
        assert len(cases) == 1
        assert cases[0].model == "fdd"

    def test_list_empty(self, manager):
        assert manager.list() == []


class TestVerify:
    def test_verify_case(self, manager, case_dir):
        # Register then verify — files should match because checksums
        # were computed from the same data on disk
        manager.register("npu/tdd/fp32", str(case_dir))

        # Copy files to the store so verify can find them
        store = Path(manager._store_dir) / "npu" / "tdd" / "fp32"
        store.mkdir(parents=True, exist_ok=True)
        import shutil

        for dtype in ("weights", "golden"):
            src = case_dir / dtype
            dst = store / dtype
            if src.exists():
                shutil.copytree(src, dst, dirs_exist_ok=True)

        results = manager.verify(case_id="npu/tdd/fp32")
        assert all(r.ok for r in results)


class TestPathHelpers:
    @pytest.mark.parametrize("dtype", ["weights", "inputs", "golden", "artifacts"])
    def test_get_dir(self, manager, dtype):
        p = manager.get_dir("npu/tdd/fp32", dtype)
        assert str(p).endswith(f"store/npu/tdd/fp32/{dtype}")


class TestRebuildCache:
    def test_rebuild(self, manager, case_dir):
        manager.register("npu/tdd/fp32", str(case_dir))
        count = manager.rebuild_cache()
        assert count == 1
        # Case should still be accessible
        assert manager.get("npu/tdd/fp32").case_id == "npu/tdd/fp32"
