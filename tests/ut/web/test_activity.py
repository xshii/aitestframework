"""Tests for activity log — IP capture, ring buffer."""

from __future__ import annotations

from aitf.web.activity import _get_client_ip, get_activity, log_activity


class TestClientIP:
    def test_no_flask_context(self):
        """Outside Flask request context, returns empty string."""
        assert _get_client_ip() == ""

    def test_explicit_ip_kwarg(self):
        """ip= kwarg takes precedence."""
        log_activity("test", "detail", ip="1.2.3.4")
        entries = get_activity(1)
        assert entries[0]["extra"]["ip"] == "1.2.3.4"


class TestRingBuffer:
    def test_entries_newest_first(self):
        log_activity("a", "first", ip="t1")
        log_activity("b", "second", ip="t2")
        entries = get_activity(2)
        assert entries[0]["action"] == "b"
        assert entries[1]["action"] == "a"
