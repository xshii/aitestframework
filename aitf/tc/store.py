"""Test-case store — business logic for suite discovery and execution queries."""

from __future__ import annotations

import ast
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from aitf.tc.db import get_session
from aitf.tc.models import CaseResult, Execution, SuiteInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Suite discovery
# ---------------------------------------------------------------------------

def scan_cases_dir(cases_dir: str | Path) -> list[dict]:
    """Scan cases/ directory for unittest.TestCase subclasses.

    Uses AST parsing (no import side-effects). Returns list of dicts with
    keys: module_path, class_name, docstring, platform, category,
    case_count, case_names.
    """
    cases_dir = Path(cases_dir)
    if not cases_dir.is_dir():
        return []

    results: list[dict] = []
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
            docstring = ast.get_docstring(node)

            results.append({
                "module_path": module_path,
                "class_name": node.name,
                "docstring": docstring,
                "platform": platform,
                "category": category,
                "case_count": len(methods),
                "case_names": json.dumps(methods),
            })

    return results


def refresh_suites(cases_dir: str | Path) -> int:
    """Scan cases/ and upsert SuiteInfo rows. Returns count of suites found."""
    discovered = scan_cases_dir(cases_dir)
    now = datetime.utcnow()

    with get_session() as session:
        # Build lookup of existing suites
        existing = {
            s.module_path + "::" + s.class_name: s
            for s in session.execute(select(SuiteInfo)).scalars()
        }

        for d in discovered:
            key = d["module_path"] + "::" + d["class_name"]
            if key in existing:
                suite = existing.pop(key)
                for attr in ("docstring", "platform", "category",
                             "case_count", "case_names"):
                    setattr(suite, attr, d[attr])
                suite.scanned_at = now
            else:
                session.add(SuiteInfo(**d, scanned_at=now))

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
        return [
            {
                "id": r.id,
                "module_path": r.module_path,
                "class_name": r.class_name,
                "docstring": r.docstring,
                "platform": r.platform,
                "category": r.category,
                "case_count": r.case_count,
                "case_names": json.loads(r.case_names or "[]"),
                "scanned_at": r.scanned_at.isoformat() if r.scanned_at else None,
                "last_execution_id": r.last_execution_id,
                "last_status_summary": json.loads(r.last_status_summary or "{}"),
            }
            for r in rows
        ]


def list_executions(limit: int = 50) -> list[dict]:
    """Return recent executions, newest first."""
    with get_session() as session:
        rows = session.execute(
            select(Execution).order_by(Execution.started_at.desc()).limit(limit)
        ).scalars().all()
        return [
            {
                "id": r.id,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                "bundle": r.bundle,
                "target": r.target,
                "platform": r.platform,
                "git_commit": r.git_commit,
                "trigger": r.trigger,
                "total": r.total,
                "passed": r.passed,
                "failed": r.failed,
                "timeout": r.timeout,
                "crashed": r.crashed,
                "skipped": r.skipped,
                "errored": r.errored,
                "pass_rate": r.pass_rate,
            }
            for r in rows
        ]


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
        return {
            "id": exe.id,
            "started_at": exe.started_at.isoformat() if exe.started_at else None,
            "finished_at": exe.finished_at.isoformat() if exe.finished_at else None,
            "bundle": exe.bundle,
            "target": exe.target,
            "platform": exe.platform,
            "git_commit": exe.git_commit,
            "trigger": exe.trigger,
            "total": exe.total,
            "passed": exe.passed,
            "failed": exe.failed,
            "timeout": exe.timeout,
            "crashed": exe.crashed,
            "skipped": exe.skipped,
            "errored": exe.errored,
            "pass_rate": exe.pass_rate,
            "cases": [
                {
                    "id": c.id,
                    "suite_class": c.suite_class,
                    "case_method": c.case_method,
                    "status": c.status,
                    "duration_s": c.duration_s,
                    "started_at": c.started_at.isoformat() if c.started_at else None,
                    "finished_at": c.finished_at.isoformat() if c.finished_at else None,
                    "failure_reason": c.failure_reason,
                    "compare_detail": json.loads(c.compare_detail or "null"),
                    "stdout_path": c.stdout_path,
                    "stderr_path": c.stderr_path,
                }
                for c in cases
            ],
        }


def create_execution(
    execution_id: str, *, bundle: str | None = None,
    target: str | None = None, platform: str | None = None,
    git_commit: str | None = None, trigger: str = "manual",
    suite_cases: list[tuple[str, list[str]]] | None = None,
) -> str:
    """Create a new execution batch with PENDING case results.

    suite_cases: list of (suite_class, [method_names])
    Returns execution_id.
    """
    now = datetime.utcnow()
    total = sum(len(methods) for _, methods in (suite_cases or []))

    with get_session() as session:
        exe = Execution(
            id=execution_id, started_at=now,
            bundle=bundle, target=target, platform=platform,
            git_commit=git_commit, trigger=trigger, total=total,
        )
        session.add(exe)

        for suite_class, methods in (suite_cases or []):
            for method in methods:
                session.add(CaseResult(
                    execution_id=execution_id,
                    suite_class=suite_class,
                    case_method=method,
                    status="PENDING",
                ))

        session.commit()

    return execution_id


def update_case_status(
    execution_id: str, suite_class: str, case_method: str,
    status: str, **kwargs,
) -> None:
    """Update a single case result status and optional fields."""
    with get_session() as session:
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
        if status == "RUNNING" and row.started_at is None:
            row.started_at = datetime.utcnow()
        if status in ("PASS", "FAIL", "TIMEOUT", "CRASH", "SKIP", "ERROR"):
            row.finished_at = datetime.utcnow()
            if row.started_at:
                row.duration_s = (row.finished_at - row.started_at).total_seconds()

        for key in ("failure_reason", "compare_detail", "stdout_path", "stderr_path"):
            if key in kwargs:
                val = kwargs[key]
                if key == "compare_detail" and not isinstance(val, str):
                    val = json.dumps(val)
                setattr(row, key, val)

        session.commit()


def finish_execution(execution_id: str) -> None:
    """Finalise an execution: compute summary counts and update suite info."""
    with get_session() as session:
        exe = session.get(Execution, execution_id)
        if exe is None:
            return

        cases = session.execute(
            select(CaseResult).where(CaseResult.execution_id == execution_id)
        ).scalars().all()

        counts = {"PASS": 0, "FAIL": 0, "TIMEOUT": 0, "CRASH": 0,
                  "SKIP": 0, "ERROR": 0}
        suite_summary: dict[str, dict] = {}
        for c in cases:
            counts[c.status] = counts.get(c.status, 0) + 1
            ss = suite_summary.setdefault(c.suite_class, {})
            ss[c.status] = ss.get(c.status, 0) + 1

        exe.finished_at = datetime.utcnow()
        exe.total = len(cases)
        exe.passed = counts["PASS"]
        exe.failed = counts["FAIL"]
        exe.timeout = counts["TIMEOUT"]
        exe.crashed = counts["CRASH"]
        exe.skipped = counts["SKIP"]
        exe.errored = counts["ERROR"]
        exe.pass_rate = exe.passed / exe.total if exe.total else 0.0

        # Update suite_info with last execution summary
        for suite_class, summary in suite_summary.items():
            suite = session.execute(
                select(SuiteInfo).where(SuiteInfo.class_name == suite_class)
            ).scalar_one_or_none()
            if suite:
                suite.last_execution_id = execution_id
                suite.last_status_summary = json.dumps(summary)

        session.commit()


def generate_execution_id() -> str:
    """Generate a unique execution ID based on timestamp."""
    import hashlib
    ts = time.strftime("%Y%m%d-%H%M%S")
    h = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
    return f"{ts}-{h}"
