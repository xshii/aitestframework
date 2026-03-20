"""Tests for runner helper functions — truncation, report paths."""

from __future__ import annotations

from aitf.tc.runner import _truncate


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello") == "hello"

    def test_long_text_truncated(self):
        text = "x" * 60_000
        result = _truncate(text, limit=100)
        assert len(result) < 200
        assert result.startswith("x" * 100)
        assert "truncated" in result
        assert "60000" in result

    def test_exact_limit_unchanged(self):
        text = "a" * 50_000
        assert _truncate(text, limit=50_000) == text

    def test_empty_string(self):
        assert _truncate("") == ""
