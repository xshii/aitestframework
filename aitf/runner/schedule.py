"""Schedule policy — UI-configurable idle auto-run rules.

Stored in ``runner/schedules.yaml``. Each rule defines:
- Time window (hours)
- Target environments or pools
- Idle timeout + cooldown
- Test plan to execute

The :class:`ExecutionQueue` idle watcher reads these rules instead of
(or in addition to) the per-target ``idle`` config in targets.yaml.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ScheduleRule:
    """A single schedule policy rule."""

    name: str = ""
    hours: str = ""                     # e.g. "09:00-21:00"
    days: str = ""                      # e.g. "mon,tue,wed,thu,fri"
    targets: list[str] = field(default_factory=list)  # target names or pool names
    idle_timeout_min: int = 30
    cooldown_min: int = 60
    plan: str = ""                      # testplan filename
    cases: list[str] = field(default_factory=list)    # fallback: direct case paths
    enabled: bool = True
    once: bool = False                  # True = auto-disable after first execution
    expires: str = ""                   # ISO date (YYYY-MM-DD), auto-disable after this day

    def to_dict(self) -> dict:
        return asdict(self)


def _schedules_path(project_root: str | Path) -> Path:
    root = Path(project_root)
    primary = root / "data" / "config" / "schedules.yaml"
    if primary.is_file():
        return primary
    legacy = root / "runner" / "schedules.yaml"
    if legacy.is_file():
        return legacy
    # Default: new location
    primary.parent.mkdir(parents=True, exist_ok=True)
    return primary


def load_schedules(project_root: str | Path) -> list[ScheduleRule]:
    """Load schedule rules from runner/schedules.yaml."""
    p = _schedules_path(project_root)
    if not p.is_file():
        return []
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load schedules.yaml")
        return []
    if not isinstance(data, list):
        return []

    _fields = ScheduleRule.__dataclass_fields__
    return [
        ScheduleRule(**{k: item[k] for k in _fields if k in item})
        for item in data
    ]


def save_schedules(project_root: str | Path,
                   rules: list[ScheduleRule]) -> None:
    """Save schedule rules to runner/schedules.yaml."""
    p = _schedules_path(project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [r.to_dict() for r in rules]
    p.write_text(
        yaml.dump(data, default_flow_style=False,
                  allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def in_time_window(hours: str, days: str = "") -> bool:
    """Check if current time matches a schedule's time window."""
    from datetime import datetime
    now = datetime.now()

    # Day filter
    if days:
        day_abbr = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"][now.weekday()]
        allowed = [d.strip().lower() for d in days.split(",")]
        if day_abbr not in allowed:
            return False

    # Hour range
    if not hours:
        return True
    try:
        start_s, end_s = hours.split("-")
        sh, sm = map(int, start_s.strip().split(":"))
        eh, em = map(int, end_s.strip().split(":"))
        start_min = sh * 60 + sm
        end_min = eh * 60 + em
        cur_min = now.hour * 60 + now.minute
        # Overnight range: "22:00-06:00"
        if start_min <= end_min:
            return start_min <= cur_min < end_min
        return cur_min >= start_min or cur_min < end_min
    except (ValueError, AttributeError):
        return False
