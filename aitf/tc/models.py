"""SQLAlchemy models for test-case management (REQ-4 + REQ-7)."""

from __future__ import annotations

import json
import re
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship

# -- shared constants -------------------------------------------------------

SAFE_FILENAME_RE = re.compile(r'^[a-zA-Z0-9_\-\u4e00-\u9fff]+$')
"""Regex for safe filenames (letters, digits, underscore, hyphen, CJK)."""


class CaseStatus:
    """Test case status constants."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASS = "PASS"
    FAIL = "FAIL"
    TIMEOUT = "TIMEOUT"
    CRASH = "CRASH"
    SKIP = "SKIP"
    ERROR = "ERROR"

    TERMINAL = (PASS, FAIL, TIMEOUT, CRASH, SKIP, ERROR)
    """Statuses that indicate the case has finished."""

    FAILURE = (FAIL, ERROR, TIMEOUT, CRASH)
    """Statuses that count as failures."""


def _ts(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


class Base(DeclarativeBase):
    pass


class SuiteInfo(Base):
    """Test suite discovered by scanning cases/ directory."""

    __tablename__ = "suite_info"

    id = Column(Integer, primary_key=True, autoincrement=True)
    module_path = Column(String, nullable=False)                # cases/npu/operators/test_conv2d.py
    class_name = Column(String, nullable=False)                 # TestConv2dNPU

    __table_args__ = (
        UniqueConstraint("module_path", "class_name", name="uq_suite_module_class"),
    )
    docstring = Column(String, nullable=True)
    platform = Column(String, nullable=True)                    # npu / gpu / cpu
    category = Column(String, nullable=True)                    # operators / models / preprocess
    case_count = Column(Integer, default=0)
    case_names = Column(Text, nullable=True)                    # JSON: ["test_fp32_3x3_basic", ...]
    scanned_at = Column(DateTime, default=func.now())
    last_execution_id = Column(String, nullable=True)
    last_status_summary = Column(Text, nullable=True)           # JSON: {"pass":3,"fail":1}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "module_path": self.module_path,
            "class_name": self.class_name,
            "docstring": self.docstring,
            "platform": self.platform,
            "category": self.category,
            "case_count": self.case_count,
            "case_names": json.loads(self.case_names or "[]"),
            "scanned_at": _ts(self.scanned_at),
            "last_execution_id": self.last_execution_id,
            "last_status_summary": json.loads(self.last_status_summary or "{}"),
        }


class Execution(Base):
    """A test execution batch."""

    __tablename__ = "executions"

    id = Column(String, primary_key=True)                       # e.g. 20260227-143000-abc123
    started_at = Column(DateTime, default=func.now())
    finished_at = Column(DateTime, nullable=True)
    bundle = Column(String, nullable=True)                      # deps bundle used
    target = Column(String, nullable=True)                      # execution target
    golden_model = Column(String, nullable=True)                # golden data model name
    golden_version = Column(String, nullable=True)              # golden data version
    plan_name = Column(String, nullable=True)                   # testplan name (if from plan)
    platform = Column(String, nullable=True)                    # pipeline stage platform
    pipeline_id = Column(String, nullable=True)                 # groups executions in a pipeline run
    pipeline_stage = Column(Integer, nullable=True)             # 0-based stage index in pipeline
    git_commit = Column(String, nullable=True)
    trigger = Column(String, default="manual")                  # manual / jenkins
    jenkins_job = Column(String, nullable=True)
    jenkins_build = Column(Integer, nullable=True)
    jenkins_url = Column(String, nullable=True)
    total = Column(Integer, default=0)
    passed = Column(Integer, default=0)
    failed = Column(Integer, default=0)
    timeout = Column(Integer, default=0)
    crashed = Column(Integer, default=0)
    skipped = Column(Integer, default=0)
    errored = Column(Integer, default=0)
    pass_rate = Column(Float, default=0.0)
    report_json_path = Column(String, nullable=True)

    cases = relationship("CaseResult", back_populates="execution",
                         cascade="all, delete-orphan")

    @property
    def failed_total(self) -> int:
        """All failure-category counts combined (failed + errored + timeout + crashed)."""
        return (self.failed or 0) + (self.errored or 0) + (self.timeout or 0) + (self.crashed or 0)

    @property
    def not_run(self) -> int:
        """Cases not yet executed (total - passed - failed_total)."""
        return max(0, (self.total or 0) - (self.passed or 0) - self.failed_total)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "started_at": _ts(self.started_at),
            "finished_at": _ts(self.finished_at),
            "bundle": self.bundle,
            "target": self.target,
            "golden_model": self.golden_model,
            "golden_version": self.golden_version,
            "plan_name": self.plan_name,
            "platform": self.platform,
            "pipeline_id": self.pipeline_id,
            "pipeline_stage": self.pipeline_stage,
            "git_commit": self.git_commit,
            "trigger": self.trigger,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "timeout": self.timeout,
            "crashed": self.crashed,
            "skipped": self.skipped,
            "errored": self.errored,
            "failed_total": self.failed_total,
            "not_run": self.not_run,
            "pass_rate": self.pass_rate,
        }


class CaseResult(Base):
    """Result of a single test case within an execution."""

    __tablename__ = "case_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    execution_id = Column(String, ForeignKey("executions.id"), nullable=False, index=True)
    suite_class = Column(String, nullable=False)                # TestConv2dNPU
    case_method = Column(String, nullable=False)                # test_fp32_3x3_basic
    status = Column(String, default="PENDING")                  # PENDING/RUNNING/PASS/FAIL/TIMEOUT/CRASH/SKIP/ERROR
    duration_s = Column(Float, nullable=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    failure_reason = Column(String, nullable=True)
    compare_detail = Column(Text, nullable=True)                # JSON
    stdout = Column(Text, nullable=True)                        # captured stdout
    stderr = Column(Text, nullable=True)                        # captured stderr

    execution = relationship("Execution", back_populates="cases")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "suite_class": self.suite_class,
            "case_method": self.case_method,
            "status": self.status,
            "duration_s": self.duration_s,
            "started_at": _ts(self.started_at),
            "finished_at": _ts(self.finished_at),
            "failure_reason": self.failure_reason,
            "compare_detail": json.loads(self.compare_detail or "null"),
            "stdout": self.stdout,
            "stderr": self.stderr,
        }
