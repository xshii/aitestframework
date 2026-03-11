"""SQLAlchemy models for test-case management (REQ-4 + REQ-7)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class SuiteInfo(Base):
    """Test suite discovered by scanning cases/ directory."""

    __tablename__ = "suite_info"

    id = Column(Integer, primary_key=True, autoincrement=True)
    module_path = Column(String, unique=True, nullable=False)   # cases/npu/operators/test_conv2d.py
    class_name = Column(String, nullable=False)                 # TestConv2dNPU
    docstring = Column(String, nullable=True)
    platform = Column(String, nullable=True)                    # npu / gpu / cpu
    category = Column(String, nullable=True)                    # operators / models / preprocess
    case_count = Column(Integer, default=0)
    case_names = Column(Text, nullable=True)                    # JSON: ["test_fp32_3x3_basic", ...]
    scanned_at = Column(DateTime, default=func.now())
    last_execution_id = Column(String, nullable=True)
    last_status_summary = Column(Text, nullable=True)           # JSON: {"pass":3,"fail":1}


class Execution(Base):
    """A test execution batch."""

    __tablename__ = "executions"

    id = Column(String, primary_key=True)                       # e.g. 20260227-143000-abc123
    started_at = Column(DateTime, default=func.now())
    finished_at = Column(DateTime, nullable=True)
    bundle = Column(String, nullable=True)                      # deps bundle used
    target = Column(String, nullable=True)                      # execution target
    platform = Column(String, nullable=True)
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
    stdout_path = Column(String, nullable=True)
    stderr_path = Column(String, nullable=True)

    execution = relationship("Execution", back_populates="cases")
