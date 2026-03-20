"""Tests for schedule policy — time window matching, load/save."""

from __future__ import annotations

from datetime import datetime

import pytest

from aitf.runner.schedule import (
    ScheduleRule,
    in_time_window,
    load_schedules,
    save_schedules,
)


class TestTimeWindow:
    def test_empty_hours_always_matches(self):
        assert in_time_window("") is True

    def test_current_time_basic(self):
        """in_time_window with a range covering the full day always matches."""
        assert in_time_window("00:00-23:59") is True

    def test_overnight_range_format(self):
        """Overnight range '22:00-06:00' is valid and doesn't crash."""
        # Just check it returns a bool without error
        result = in_time_window("22:00-06:00")
        assert isinstance(result, bool)

    def test_invalid_hours_returns_false(self):
        assert in_time_window("invalid") is False

    def test_day_filter_format(self):
        """Day filter with valid day names doesn't crash."""
        result = in_time_window("", "mon,tue,wed,thu,fri,sat,sun")
        assert result is True  # today is always in this list


class TestLoadSave:
    def test_roundtrip(self, tmp_path):
        rules = [
            ScheduleRule(name="test", hours="09:00-21:00",
                         targets=["local"], idle_timeout_min=10,
                         cooldown_min=30, plan="smoke.yaml"),
        ]
        save_schedules(tmp_path, rules)
        loaded = load_schedules(tmp_path)
        assert len(loaded) == 1
        assert loaded[0].name == "test"
        assert loaded[0].hours == "09:00-21:00"
        assert loaded[0].plan == "smoke.yaml"

    def test_load_empty(self, tmp_path):
        assert load_schedules(tmp_path) == []

    def test_once_and_expires_preserved(self, tmp_path):
        rules = [ScheduleRule(name="once", once=True, expires="2026-12-31")]
        save_schedules(tmp_path, rules)
        loaded = load_schedules(tmp_path)
        assert loaded[0].once is True
        assert loaded[0].expires == "2026-12-31"
