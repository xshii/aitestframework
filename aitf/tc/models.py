"""SQLAlchemy models for test-case management (REQ-4 + REQ-7).

Uses :class:`DictMixin` to auto-generate ``to_dict()`` from SQLAlchemy
column definitions — no manual field-by-field mapping needed.  Frontend
field names always match column names exactly.

Special handling:
- ``DateTime`` columns → ISO-8601 string (or None)
- Columns with ``info={"json": True}`` → ``json.loads()`` on output
- ``_extra_fields``: class-level tuple of ``@property`` names to include
"""

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


class DictMixin:
    """Auto-serialize SQLAlchemy model to dict via column introspection.

    Rules:
    - DateTime columns → ``isoformat()`` or None
    - Columns with ``info={"json": True}`` → ``json.loads()``
    - Extra ``@property`` names listed in ``_extra_fields`` are appended
    """

    _extra_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d: dict = {}
        for col in self.__table__.columns:
            val = getattr(self, col.name)
            if isinstance(val, datetime):
                val = val.isoformat() if val else None
            elif col.info.get("json"):
                val = json.loads(val) if val else col.info.get("json_default")
            d[col.name] = val
        for prop in self._extra_fields:
            d[prop] = getattr(self, prop)
        return d


class Base(DeclarativeBase):
    pass


class SuiteInfo(DictMixin, Base):
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
    case_names = Column(Text, info={"json": True, "json_default": []})
    scanned_at = Column(DateTime, default=func.now())
    last_execution_id = Column(String, nullable=True)
    last_status_summary = Column(Text, info={"json": True, "json_default": {}})


class Execution(DictMixin, Base):
    """A test execution batch."""

    __tablename__ = "executions"
    _extra_fields = ("failed_total", "not_run")

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


class CaseResult(DictMixin, Base):
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
    compare_detail = Column(Text, info={"json": True, "json_default": None})
    stdout = Column(Text, nullable=True)                        # captured stdout
    stderr = Column(Text, nullable=True)                        # captured stderr

    execution = relationship("Execution", back_populates="cases")
