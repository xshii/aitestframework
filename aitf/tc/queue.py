"""Execution queue — per-target FIFO to prevent environment contention.

Only one execution runs per target at a time. Additional submissions
are queued and processed in order when the current execution finishes.

Pool support: when target is a pool name, the queue automatically picks
the least-busy target in the pool (fewest queued + running items).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class QueueStatus:
    """Queue item status constants."""
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

    ACTIVE = (QUEUED, RUNNING)
    FINISHED = (DONE, FAILED, CANCELLED)


class Trigger:
    """Queue item trigger source."""
    MANUAL = "manual"
    IDLE = "idle"


@dataclass
class QueueItem:
    """A single queued or running execution request."""

    id: str
    target: str                     # resolved concrete target
    pool: str = ""                  # original pool name (if submitted to pool)
    trigger: str = Trigger.MANUAL
    paths: list[str] = field(default_factory=list)
    run_kwargs: dict = field(default_factory=dict)
    status: str = QueueStatus.QUEUED
    execution_id: str = ""          # filled when execution starts
    submitted_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""
    label: str = ""                 # human-readable summary

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("run_kwargs", None)  # internal, not exposed to frontend
        return d


@dataclass
class IdleSchedule:
    """A time-window override for idle auto-run."""
    hours: str = ""            # e.g. "22:00-06:00"
    days: str = ""             # e.g. "sat,sun"
    cases: list[str] = field(default_factory=list)
    cooldown_min: int = 0      # 0 = no cooldown (continuous)


@dataclass
class IdleConfig:
    """Per-target idle auto-run configuration."""
    target: str
    timeout_min: int = 10
    cooldown_min: int = 30
    cases: list[str] = field(default_factory=list)
    schedule: list[IdleSchedule] = field(default_factory=list)


def _load_targets_yaml(project_root: str) -> dict:
    """Read targets.yaml raw dict, checking data/config/ and legacy paths."""
    root = Path(project_root)
    for candidate in [root / "data" / "config" / "targets.yaml",
                      root / "runner" / "targets.yaml",
                      root / "targets.yaml"]:
        if candidate.is_file():
            try:
                return yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            except Exception:
                return {}
    return {}


def _load_idle_configs(project_root: str) -> list[IdleConfig]:
    """Extract idle auto-run configs from targets.yaml."""
    data = _load_targets_yaml(project_root)
    configs = []
    for name, cfg in data.items():
        idle = cfg.get("idle")
        if not idle or not idle.get("cases"):
            continue
        _fields = IdleSchedule.__dataclass_fields__
        schedules = [
            IdleSchedule(**{k: s[k] for k in _fields if k in s})
            for s in idle.get("schedule", [])
        ]
        configs.append(IdleConfig(
            target=name,
            timeout_min=idle.get("timeout_min", 10),
            cooldown_min=idle.get("cooldown_min", 30),
            cases=idle["cases"],
            schedule=schedules,
        ))
    return configs


def _match_schedule(schedules: list[IdleSchedule]) -> IdleSchedule | None:
    """Return the first matching schedule for current time, or None."""
    from aitf.runner.schedule import in_time_window
    for s in schedules:
        if in_time_window(s.hours, s.days):
            return s
    return None


def _load_pool_map(project_root: str) -> dict[str, list[str]]:
    """Read targets.yaml → {pool_name: [target_name, ...]}."""
    data = _load_targets_yaml(project_root)
    pools: dict[str, list[str]] = {}
    for name, cfg in data.items():
        pool = cfg.get("pool", "")
        if pool:
            pools.setdefault(pool, []).append(name)
    return pools


class ExecutionQueue:
    """Per-target execution queue with pool-based auto-allocation.

    Usage::

        q = ExecutionQueue(cases_dir, db_path, project_root)
        # Submit to specific target
        q.submit("fpga-01", ["test_conv2d.py"])
        # Submit to pool — auto-picks least-busy target
        q.submit("fpga_pool", ["test_conv2d.py"])
    """

    def __init__(self, cases_dir: str, db_path: str,
                 project_root: str = "."):
        self._cases_dir = cases_dir
        self._db_path = db_path
        self._project_root = project_root
        self._lock = threading.Lock()
        self._items: list[QueueItem] = []
        self._max_history = 50
        self._idle_stop = threading.Event()
        self._state_file = Path(project_root) / "data" / "build" / "queue_state.json"
        self._load_state()

    def _resolve_target(self, target: str) -> tuple[str, str]:
        """Resolve target to (concrete_target, pool_name).

        If target is a pool name, pick the least-busy member.
        Returns (target, pool) where pool is "" if not a pool submission.
        """
        pools = _load_pool_map(self._project_root)
        if target not in pools:
            return target, ""

        # Pool submission: pick member with fewest active items
        members = pools[target]
        if not members:
            return target, target

        def _load(t: str) -> int:
            return sum(1 for i in self._items
                       if i.target == t and i.status in QueueStatus.ACTIVE)

        best = min(members, key=_load)
        logger.info("Pool %s: assigned to %s (load: %d)", target, best, _load(best))
        return best, target

    def submit(self, target: str, paths: list[str],
               label: str = "", trigger: str = Trigger.MANUAL,
               **run_kwargs) -> QueueItem:
        """Add an execution to the queue. Returns the QueueItem.

        If *target* is a pool name, auto-selects the least-busy member.
        *trigger* is "manual" (user) or "idle" (auto-triggered).
        """
        with self._lock:
            concrete, pool = self._resolve_target(target or "local")
            item = QueueItem(
                id=uuid.uuid4().hex[:8],
                target=concrete,
                pool=pool,
                trigger=trigger,
                paths=paths,
                run_kwargs=run_kwargs,
                submitted_at=time.time(),
                label=label or ", ".join(paths[:3]) + ("..." if len(paths) > 3 else ""),
            )
            self._items.append(item)
            self._trim_history()
            self._save_state()
        self._maybe_start(item.target)
        return item

    def cancel(self, item_id: str) -> bool:
        """Cancel a queued (not yet running) item."""
        with self._lock:
            for item in self._items:
                if item.id == item_id and item.status == QueueStatus.QUEUED:
                    item.status = QueueStatus.CANCELLED
                    item.finished_at = time.time()
                    self._save_state()
                    return True
        return False

    def get_status(self) -> dict:
        """Return per-target queue status."""
        with self._lock:
            active = [i for i in self._items if i.status in QueueStatus.ACTIVE]
            recent = [i for i in self._items
                      if i.status in QueueStatus.FINISHED][-20:]

        # Group by target
        targets: dict[str, dict] = {}
        for item in active:
            t = item.target
            if t not in targets:
                targets[t] = {"running": None, "queued": []}
            if item.status == QueueStatus.RUNNING:
                targets[t]["running"] = item.to_dict()
            else:
                targets[t]["queued"].append(item.to_dict())

        return {
            "targets": targets,
            "recent": [i.to_dict() for i in reversed(recent)],
        }

    def _maybe_start(self, target: str) -> None:
        """If nothing is running for this target, start the next queued item."""
        with self._lock:
            running = any(i.target == target and i.status == QueueStatus.RUNNING
                          for i in self._items)
            if running:
                return
            # Find next queued item for this target
            next_item = None
            for i in self._items:
                if i.target == target and i.status == QueueStatus.QUEUED:
                    next_item = i
                    break
            if not next_item:
                return
            next_item.status = QueueStatus.RUNNING
            next_item.started_at = time.time()
            self._save_state()

        # Run in background thread (with Flask app context if available)
        item = next_item
        app = self._get_flask_app()

        def _worker():
            ctx = app.app_context() if app else None
            if ctx:
                ctx.push()
            try:
                from aitf.tc.runner import run_tests
                from aitf.tc.testplan import RunConfig

                kw = item.run_kwargs
                config = RunConfig(
                    tests=item.paths,
                    filter_k=kw.get("filter_k"),
                    bundle=kw.get("bundle"),
                    target=item.target if item.target != "local" else None,
                    golden_model=kw.get("golden_model"),
                    golden_version=kw.get("golden_version"),
                    params=kw.get("params") or {},
                    test_timeout=kw.get("test_timeout", 300),
                    retry=kw.get("retry", 0),
                )
                eid, passed = run_tests(
                    self._cases_dir, self._db_path, config=config)
                item.execution_id = eid
                item.status = QueueStatus.DONE if passed else QueueStatus.FAILED
            except Exception as exc:
                logger.exception("Queue item %s failed", item.id)
                item.status = QueueStatus.FAILED
                item.error = str(exc)
            finally:
                item.finished_at = time.time()
                with self._lock:
                    self._save_state()
                if ctx:
                    ctx.pop()
                # Process next in queue for this target
                self._maybe_start(item.target)

        t = threading.Thread(target=_worker,
                             name=f"aitf-queue-{item.id}", daemon=True)
        t.start()

    @staticmethod
    def _get_flask_app():
        """Get Flask app for creating app context in worker threads."""
        try:
            from flask import current_app
            return current_app._get_current_object()
        except (ImportError, RuntimeError):
            return None

    # -- idle auto-run -------------------------------------------------------

    def start_idle_watcher(self) -> None:
        """Start a background thread that checks for idle environments."""
        self._idle_stop = threading.Event()
        t = threading.Thread(target=self._idle_loop,
                             name="aitf-idle-watcher", daemon=True)
        t.start()
        logger.info("Idle watcher started")

    def _idle_loop(self) -> None:
        """Check every 60s for idle environments that should auto-run."""
        while not self._idle_stop.wait(60):
            try:
                self._check_idle_targets()
            except Exception:
                logger.exception("Idle watcher error")

    def _check_idle_targets(self) -> None:
        from datetime import date

        from aitf.runner.schedule import (
            ScheduleRule, in_time_window, load_schedules,
        )

        # 1. Load schedule rules (UI-configured, primary source)
        rules = load_schedules(self._project_root)

        # 2. Also load legacy per-target idle configs from targets.yaml
        legacy = _load_idle_configs(self._project_root)
        for ic in legacy:
            sched = _match_schedule(ic.schedule)
            rules.append(ScheduleRule(
                name=f'legacy:{ic.target}',
                hours=sched.hours if sched else '',
                days=sched.days if sched else '',
                targets=[ic.target],
                idle_timeout_min=ic.timeout_min,
                cooldown_min=sched.cooldown_min if sched else ic.cooldown_min,
                cases=sched.cases if sched and sched.cases else ic.cases,
            ))

        if not rules:
            return

        now = time.time()
        pools = _load_pool_map(self._project_root)

        dirty = False  # track if we need to save schedule changes
        for rule in rules:
            if not getattr(rule, 'enabled', True):
                continue

            # Check expiry (YYYY-MM-DD)
            expires = getattr(rule, 'expires', '')
            if expires:
                try:
                    if date.fromisoformat(expires) < date.today():
                        rule.enabled = False
                        dirty = True
                        continue
                except ValueError:
                    pass

            if not in_time_window(rule.hours, getattr(rule, 'days', '')):
                continue

            # Resolve targets (expand pools)
            concrete_targets = []
            for t in rule.targets:
                if t in pools:
                    concrete_targets.extend(pools[t])
                else:
                    concrete_targets.append(t)

            # Determine what to run
            cases = getattr(rule, 'cases', [])
            plan = getattr(rule, 'plan', '')
            if plan:
                cases = self._resolve_plan_cases(plan)
            if not cases:
                continue

            submitted = False
            for target in concrete_targets:
                if self._try_idle_submit(
                    target, cases, rule.idle_timeout_min,
                    rule.cooldown_min, rule.name, now,
                ):
                    submitted = True

            # once=True → disable after first successful submission
            if submitted and getattr(rule, 'once', False):
                rule.enabled = False
                dirty = True

        # Persist changes (expired/once rules disabled)
        if dirty:
            try:
                from aitf.runner.schedule import ScheduleRule, save_schedules
                # Only save UI-configured rules (not legacy from targets.yaml)
                saveable = [r for r in rules
                            if isinstance(r, ScheduleRule)
                            and not r.name.startswith("legacy:")]
                save_schedules(self._project_root, saveable)
            except Exception:
                logger.debug("Failed to save schedule updates", exc_info=True)

    def _resolve_plan_cases(self, plan_filename: str) -> list[str]:
        """Load test paths from a testplan file."""
        try:
            from aitf.tc.testplan import load_testplan
            cfg_root = Path(self._project_root)
            # Check data/testplans/, then legacy locations
            for candidate in [cfg_root / "data" / "testplans" / plan_filename,
                              cfg_root / "testplans" / plan_filename,
                              cfg_root / plan_filename]:
                if candidate.is_file():
                    plan_path = candidate
                    break
            else:
                plan_path = cfg_root / plan_filename
            if not plan_path.is_file():
                return []
            plan = load_testplan(plan_path)
            paths = []
            for c in plan.configs:
                paths.extend(c.tests or [])
            return paths
        except Exception:
            logger.debug("Failed to load plan %s", plan_filename, exc_info=True)
            return []

    def _try_idle_submit(self, target: str, cases: list[str],
                         timeout_min: int, cooldown_min: int,
                         rule_name: str, now: float) -> bool:
        """Check idle conditions for a single target and submit if met.

        Returns True if a job was submitted.
        """
        with self._lock:
            if any(i.target == target and i.status in QueueStatus.ACTIVE
                   for i in self._items):
                return False

            finished = [i for i in self._items
                        if i.target == target
                        and i.status in (QueueStatus.DONE, QueueStatus.FAILED)
                        and i.finished_at > 0]
            if not finished:
                return False

            last = max(finished, key=lambda i: i.finished_at)
            idle_sec = now - last.finished_at
            if idle_sec < timeout_min * 60:
                return False

            if cooldown_min > 0:
                idle_items = [i for i in finished if i.trigger == Trigger.IDLE]
                if idle_items:
                    last_idle_finish = max(i.finished_at for i in idle_items)
                    if now - last_idle_finish < cooldown_min * 60:
                        return False

        logger.info("Schedule '%s': auto-run on %s (idle %.0fs)",
                    rule_name, target, idle_sec)
        self.submit(
            target, cases,
            label=f"[自动] {rule_name}",
            trigger=Trigger.IDLE,
        )
        return True

    # -- persistence -----------------------------------------------------------

    def _save_state(self) -> None:
        """Persist queue state to JSON file (call inside _lock)."""
        try:
            import json
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            data = [asdict(i) for i in self._items]
            self._state_file.write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception:
            logger.debug("Failed to save queue state", exc_info=True)

    def _load_state(self) -> None:
        """Restore queue state from JSON file on startup."""
        if not self._state_file.is_file():
            return
        try:
            import json
            data = json.loads(self._state_file.read_text(encoding="utf-8"))
            _fields = QueueItem.__dataclass_fields__
            for d in data:
                # Mark interrupted running items as failed
                if d.get("status") == QueueStatus.RUNNING:
                    d["status"] = QueueStatus.FAILED
                    d["error"] = "interrupted by server restart"
                    d["finished_at"] = time.time()
                item = QueueItem(**{k: d[k] for k in _fields if k in d})
                self._items.append(item)
            # Re-start any queued items
            targets = {i.target for i in self._items if i.status == QueueStatus.QUEUED}
            for t in targets:
                self._maybe_start(t)
            logger.info("Restored %d queue items from state file", len(self._items))
        except Exception:
            logger.debug("Failed to load queue state", exc_info=True)

    def _trim_history(self) -> None:
        """Remove old completed items to prevent unbounded growth."""
        done = [i for i in self._items if i.status in QueueStatus.FINISHED]
        if len(done) > self._max_history:
            remove = set(id(i) for i in done[:-self._max_history])
            self._items = [i for i in self._items if id(i) not in remove]
