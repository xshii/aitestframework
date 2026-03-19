"""Test-case store — business logic for suite discovery and execution queries."""

from __future__ import annotations

import ast
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from aitf.tc.db import get_session
from aitf.tc.models import CaseResult, CaseStatus, Execution, SuiteInfo

logger = logging.getLogger(__name__)


@dataclass
class ScannedSuite:
    """A test suite discovered by AST scanning (not yet in DB)."""

    module_path: str
    class_name: str
    docstring: str | None
    platform: str | None
    category: str | None
    case_count: int
    case_names: str  # JSON encoded list


# ---------------------------------------------------------------------------
# Suite discovery
# ---------------------------------------------------------------------------

def scan_cases_dir(cases_dir: str | Path) -> list[ScannedSuite]:
    """Scan cases/ directory for unittest.TestCase subclasses.

    Uses AST parsing (no import side-effects).
    """
    cases_dir = Path(cases_dir)
    if not cases_dir.is_dir():
        return []

    results: list[ScannedSuite] = []
    for py_file in sorted(cases_dir.rglob("test_*.py")):
        rel = py_file.relative_to(cases_dir)
        module_path = str(rel)

        # Infer platform / category from directory structure
        parts = rel.parts  # e.g. ("npu", "operators", "test_conv2d.py")
        platform = parts[0] if len(parts) > 1 else None
        category = parts[1] if len(parts) > 2 else None

        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except SyntaxError:
            logger.warning("scan: syntax error in %s, skipping", py_file)
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            # Check if it inherits from something with "TestCase" in the name
            is_testcase = any(
                (isinstance(b, ast.Attribute) and b.attr == "TestCase")
                or (isinstance(b, ast.Name) and "TestCase" in b.id)
                for b in node.bases
            )
            if not is_testcase:
                continue

            methods = [
                n.name for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name.startswith("test_")
            ]

            results.append(ScannedSuite(
                module_path=module_path,
                class_name=node.name,
                docstring=ast.get_docstring(node),
                platform=platform,
                category=category,
                case_count=len(methods),
                case_names=json.dumps(methods),
            ))

    return results


def refresh_suites(cases_dir: str | Path) -> int:
    """Scan cases/ and upsert SuiteInfo rows. Returns count of suites found."""
    from dataclasses import asdict

    discovered = scan_cases_dir(cases_dir)
    now = datetime.now(UTC)

    with get_session() as session:
        # Build lookup of existing suites
        existing = {
            s.module_path + "::" + s.class_name: s
            for s in session.execute(select(SuiteInfo)).scalars()
        }

        _update_attrs = ("docstring", "platform", "category",
                         "case_count", "case_names")

        for scanned in discovered:
            key = scanned.module_path + "::" + scanned.class_name
            if key in existing:
                suite = existing.pop(key)
                for attr in _update_attrs:
                    setattr(suite, attr, getattr(scanned, attr))
                suite.scanned_at = now
            else:
                session.add(SuiteInfo(**asdict(scanned), scanned_at=now))

        # Mark removed suites — don't delete (keep history linkage)
        for suite in existing.values():
            suite.case_count = 0
            suite.case_names = "[]"
            suite.scanned_at = now

        session.commit()

    logger.info("refresh_suites: found %d suites", len(discovered))
    return len(discovered)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def list_suites() -> list[dict]:
    """Return all suites as dicts."""
    with get_session() as session:
        rows = session.execute(
            select(SuiteInfo).order_by(SuiteInfo.platform, SuiteInfo.module_path)
        ).scalars().all()
        return [r.to_dict() for r in rows]


def list_executions(limit: int = 50) -> list[dict]:
    """Return recent executions, newest first."""
    with get_session() as session:
        rows = session.execute(
            select(Execution).order_by(Execution.started_at.desc()).limit(limit)
        ).scalars().all()
        return [r.to_dict() for r in rows]


def get_execution_detail(execution_id: str) -> dict | None:
    """Return execution with all case results."""
    with get_session() as session:
        exe = session.get(Execution, execution_id)
        if exe is None:
            return None
        cases = session.execute(
            select(CaseResult)
            .where(CaseResult.execution_id == execution_id)
            .order_by(CaseResult.suite_class, CaseResult.id)
        ).scalars().all()
        d = exe.to_dict()
        d["cases"] = [c.to_dict() for c in cases]
        return d


def create_execution(
    execution_id: str, *, bundle: str | None = None,
    target: str | None = None, platform: str | None = None,
    golden_model: str | None = None, golden_version: str | None = None,
    plan_name: str | None = None,
    pipeline_id: str | None = None, pipeline_stage: int | None = None,
    git_commit: str | None = None, trigger: str = "manual",
    suite_cases: list[tuple[str, list[str]]] | None = None,
) -> str:
    """Create a new execution batch with PENDING case results.

    suite_cases: list of (suite_class, [method_names])
    Returns execution_id.
    """
    now = datetime.now(UTC)
    total = sum(len(methods) for _, methods in (suite_cases or []))

    with get_session() as session:
        exe = Execution(
            id=execution_id, started_at=now,
            bundle=bundle, target=target, platform=platform,
            golden_model=golden_model, golden_version=golden_version,
            plan_name=plan_name,
            pipeline_id=pipeline_id, pipeline_stage=pipeline_stage,
            git_commit=git_commit, trigger=trigger, total=total,
        )
        session.add(exe)

        for suite_class, methods in (suite_cases or []):
            for method in methods:
                session.add(CaseResult(
                    execution_id=execution_id,
                    suite_class=suite_class,
                    case_method=method,
                    status=CaseStatus.PENDING,
                ))

        session.commit()

    return execution_id


def update_case_status(
    execution_id: str, suite_class: str, case_method: str,
    status: str, *, session: Session | None = None, **kwargs,
) -> None:
    """Update a single case result status and optional fields.

    If *session* is provided it is reused (caller manages lifecycle);
    otherwise a throwaway session is created and committed immediately.
    """
    own_session = session is None
    if own_session:
        session = get_session()

    try:
        row = session.execute(
            select(CaseResult).where(
                CaseResult.execution_id == execution_id,
                CaseResult.suite_class == suite_class,
                CaseResult.case_method == case_method,
            )
        ).scalar_one_or_none()
        if row is None:
            logger.warning("update_case_status: case not found %s::%s in %s",
                           suite_class, case_method, execution_id)
            return

        row.status = status
        if status == CaseStatus.RUNNING and row.started_at is None:
            row.started_at = datetime.now(UTC)
        if status in CaseStatus.TERMINAL:
            row.finished_at = datetime.now(UTC)
            if row.started_at:
                # SQLite drops timezone info; strip tzinfo before subtracting
                fin = row.finished_at.replace(tzinfo=None)
                sta = row.started_at.replace(tzinfo=None)
                row.duration_s = (fin - sta).total_seconds()

        for key in ("failure_reason", "compare_detail", "stdout", "stderr"):
            if key in kwargs:
                val = kwargs[key]
                if key == "compare_detail" and not isinstance(val, str):
                    val = json.dumps(val)
                setattr(row, key, val)

        session.commit()
    finally:
        if own_session:
            session.close()


def finish_execution(execution_id: str) -> None:
    """Finalise an execution: compute summary counts and update suite info."""
    with get_session() as session:
        exe = session.get(Execution, execution_id)
        if exe is None:
            return

        cases = session.execute(
            select(CaseResult).where(CaseResult.execution_id == execution_id)
        ).scalars().all()

        counts: dict[str, int] = {}
        suite_summary: dict[str, dict] = {}
        for c in cases:
            counts[c.status] = counts.get(c.status, 0) + 1
            ss = suite_summary.setdefault(c.suite_class, {})
            ss[c.status] = ss.get(c.status, 0) + 1

        exe.finished_at = datetime.now(UTC)
        exe.total = len(cases)
        exe.passed = counts.get(CaseStatus.PASS, 0)
        exe.failed = counts.get(CaseStatus.FAIL, 0)
        exe.timeout = counts.get(CaseStatus.TIMEOUT, 0)
        exe.crashed = counts.get(CaseStatus.CRASH, 0)
        exe.skipped = counts.get(CaseStatus.SKIP, 0)
        exe.errored = counts.get(CaseStatus.ERROR, 0)
        exe.pass_rate = exe.passed / exe.total if exe.total else 0.0

        # Update suite_info with last execution summary
        for suite_class, summary in suite_summary.items():
            suites = session.execute(
                select(SuiteInfo).where(SuiteInfo.class_name == suite_class)
            ).scalars().all()
            for suite in suites:
                suite.last_execution_id = execution_id
                suite.last_status_summary = json.dumps(summary)

        session.commit()


def generate_execution_id() -> str:
    """Generate a unique execution ID based on timestamp."""
    import uuid
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:6]}"


def import_execution_from_dict(data: dict, *, trigger: str = "sync") -> str:
    """Import an execution (with cases) from a dict into the local DB.

    Used by both sync receive (server-side) and sync pull (client-side).
    Returns the execution ID.
    """
    eid = data["id"]

    with get_session() as session:
        # Deduplicate
        existing = session.get(Execution, eid)
        if existing:
            return eid

        exe = Execution(
            id=eid,
            bundle=data.get("bundle"),
            target=data.get("target"),
            golden_model=data.get("golden_model"),
            golden_version=data.get("golden_version"),
            plan_name=data.get("plan_name"),
            platform=data.get("platform"),
            git_commit=data.get("git_commit"),
            trigger=data.get("trigger", trigger),
            total=data.get("total", 0),
            passed=data.get("passed", 0),
            failed=data.get("failed", 0),
            timeout=data.get("timeout", 0),
            crashed=data.get("crashed", 0),
            skipped=data.get("skipped", 0),
            errored=data.get("errored", 0),
            pass_rate=data.get("pass_rate", 0.0),
        )
        for ts_field in ("started_at", "finished_at"):
            val = data.get(ts_field)
            if val:
                try:
                    setattr(exe, ts_field, datetime.fromisoformat(val)
                            if isinstance(val, str) else val)
                except (ValueError, TypeError):
                    pass
        session.add(exe)

        for c in data.get("cases", []):
            cr = CaseResult(
                execution_id=eid,
                suite_class=c.get("suite_class", ""),
                case_method=c.get("case_method", ""),
                status=c.get("status", CaseStatus.PENDING),
                failure_reason=c.get("failure_reason"),
            )
            if c.get("duration_s") is not None:
                cr.duration_s = c["duration_s"]
            if c.get("stdout"):
                cr.stdout = c["stdout"]
            if c.get("stderr"):
                cr.stderr = c["stderr"]
            if c.get("compare_detail"):
                cr.compare_detail = (
                    json.dumps(c["compare_detail"])
                    if not isinstance(c["compare_detail"], str)
                    else c["compare_detail"]
                )
            session.add(cr)

        session.commit()

    logger.info("Imported execution %s (%d cases)",
                eid, len(data.get("cases", [])))
    return eid
