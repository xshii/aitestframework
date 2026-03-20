"""Tests for report generation — JUnit XML."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from aitf.tc.report import generate_junit_xml


class TestJunitXml:
    def test_generates_valid_xml(self):
        detail = {
            "id": "test-001",
            "started_at": "2026-01-01T10:00:00",
            "finished_at": "2026-01-01T10:01:00",
            "total": 3,
            "passed": 1,
            "failed": 1,
            "errored": 1,
            "cases": [
                {"suite_class": "TestA", "case_method": "test_pass",
                 "status": "PASS", "duration_s": 1.0,
                 "failure_reason": None, "stdout": "ok", "stderr": None},
                {"suite_class": "TestA", "case_method": "test_fail",
                 "status": "FAIL", "duration_s": 2.0,
                 "failure_reason": "AssertionError: 1 != 2",
                 "stdout": None, "stderr": None},
                {"suite_class": "TestA", "case_method": "test_error",
                 "status": "ERROR", "duration_s": 0.5,
                 "failure_reason": "RuntimeError: boom",
                 "stdout": None, "stderr": "traceback"},
            ],
        }

        with patch("aitf.tc.report.store") as mock_store:
            mock_store.get_execution_detail.return_value = detail
            with tempfile.TemporaryDirectory() as td:
                path = generate_junit_xml("test-001", td)
                assert path.exists()
                content = path.read_text()
                assert '<?xml version' in content
                assert 'testsuites' in content
                assert 'test_pass' in content
                assert 'test_fail' in content
                assert '<failure' in content
                assert '<error' in content
                assert '<system-out>' in content
                assert '<system-err>' in content
