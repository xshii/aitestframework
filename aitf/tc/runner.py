"""Test runner with custom TestResult that writes to SQLite (REQ-4.3 / REQ-6)."""

from __future__ import annotations

import importlib.util
import logging
import sys
import threading
import traceback
import unittest
from datetime import datetime, timezone
from pathlib import Path

from aitf.tc import store
from aitf.tc.db import init_db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom TestResult — pushes status to SQLite
# ---------------------------------------------------------------------------

class AitfTestResult(unittest.TestResult):
    """Writes case status to the database as tests run."""

    def __init__(self, execution_id: str, **kwargs):
        super().__init__(**kwargs)
        self.execution_id = execution_id

    def _case_id(self, test: unittest.TestCase) -> tuple[str, str]:
        """Return (suite_class, case_method)."""
        return test.__class__.__name__, test._testMethodName

    def startTest(self, test):
        super().startTest(test)
        suite_class, case_method = self._case_id(test)
        store.update_case_status(
            self.execution_id, suite_class, case_method, "RUNNING",
        )

    def addSuccess(self, test):
        super().addSuccess(test)
        suite_class, case_method = self._case_id(test)
        store.update_case_status(
            self.execution_id, suite_class, case_method, "PASS",
        )

    def addFailure(self, test, err):
        super().addFailure(test, err)
        suite_class, case_method = self._case_id(test)
        reason = "".join(traceback.format_exception(*err))
        store.update_case_status(
            self.execution_id, suite_class, case_method, "FAIL",
            failure_reason=reason,
        )

    def addError(self, test, err):
        super().addError(test, err)
        suite_class, case_method = self._case_id(test)
        reason = "".join(traceback.format_exception(*err))
        # Distinguish timeout from other errors
        status = "ERROR"
        if err[0] is TimeoutError:
            status = "TIMEOUT"
        elif err[0] is SystemExit or err[0] is KeyboardInterrupt:
            status = "CRASH"
        store.update_case_status(
            self.execution_id, suite_class, case_method, status,
            failure_reason=reason,
        )

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        suite_class, case_method = self._case_id(test)
        store.update_case_status(
            self.execution_id, suite_class, case_method, "SKIP",
            failure_reason=reason,
        )


# ---------------------------------------------------------------------------
# Suite loading
# ---------------------------------------------------------------------------

def _load_suites(
    cases_dir: Path,
    paths: list[str] | None = None,
    filter_k: str | None = None,
) -> unittest.TestSuite:
    """Discover and load test suites from cases/ directory.

    paths: specific files/directories to run (relative to cases_dir)
    filter_k: substring match on test name (like pytest -k)
    """
    loader = unittest.TestLoader()

    if paths:
        suite = unittest.TestSuite()
        for p in paths:
            target = cases_dir / p
            if target.is_file() and target.suffix == ".py":
                suite.addTests(_load_from_file(loader, target, cases_dir))
            elif target.is_dir():
                suite.addTests(loader.discover(str(target), pattern="test_*.py",
                                               top_level_dir=str(cases_dir)))
            else:
                logger.warning("skip unknown path: %s", p)
    else:
        suite = loader.discover(str(cases_dir), pattern="test_*.py",
                                top_level_dir=str(cases_dir))

    if filter_k:
        suite = _filter_suite(suite, filter_k)

    return suite


def _load_from_file(
    loader: unittest.TestLoader, filepath: Path, top_dir: Path,
) -> unittest.TestSuite:
    """Load tests from a single .py file."""
    rel = filepath.relative_to(top_dir)
    module_name = str(rel.with_suffix("")).replace("/", ".").replace("\\", ".")
    spec = importlib.util.spec_from_file_location(module_name, filepath,
                                                   submodule_search_locations=[])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return loader.loadTestsFromModule(mod)


def _filter_suite(suite: unittest.TestSuite, k: str) -> unittest.TestSuite:
    """Filter tests by substring match on full test ID."""
    filtered = unittest.TestSuite()
    for test in _iter_tests(suite):
        if k.lower() in str(test).lower():
            filtered.addTest(test)
    return filtered


def _iter_tests(suite):
    """Recursively yield individual test cases from a suite."""
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_tests(item)
        else:
            yield item


# ---------------------------------------------------------------------------
# Collect suite/case info for creating execution
# ---------------------------------------------------------------------------

def _collect_suite_cases(suite: unittest.TestSuite) -> list[tuple[str, list[str]]]:
    """Extract (class_name, [method_names]) from a loaded test suite."""
    mapping: dict[str, list[str]] = {}
    for test in _iter_tests(suite):
        cls_name = test.__class__.__name__
        method = test._testMethodName
        mapping.setdefault(cls_name, []).append(method)
    return list(mapping.items())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_tests(
    cases_dir: str | Path,
    db_path: str | Path,
    paths: list[str] | None = None,
    filter_k: str | None = None,
    bundle: str | None = None,
    target: str | None = None,
    verbosity: int = 1,
) -> tuple[str, bool]:
    """Run tests and record results in SQLite.

    Returns (execution_id, all_passed).
    """
    cases_dir = Path(cases_dir)
    init_db(db_path)

    # Ensure cases_dir is on sys.path for imports
    cases_str = str(cases_dir)
    if cases_str not in sys.path:
        sys.path.insert(0, cases_str)

    # Discover
    suite = _load_suites(cases_dir, paths, filter_k)
    suite_cases = _collect_suite_cases(suite)
    total = sum(len(ms) for _, ms in suite_cases)

    if total == 0:
        logger.warning("No tests found")
        return "", True

    # Create execution
    execution_id = store.generate_execution_id()
    store.create_execution(
        execution_id, bundle=bundle, target=target,
        suite_cases=suite_cases,
    )

    logger.info("execution %s: %d tests in %d suites", execution_id, total, len(suite_cases))

    # Run with custom result
    result = AitfTestResult(execution_id)
    runner = unittest.TextTestRunner(verbosity=verbosity, resultclass=None)
    # We use our own result object instead of the runner's
    suite.run(result)

    # Finalize
    store.finish_execution(execution_id)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Execution: {execution_id}")
    print(f"Total: {result.testsRun}  "
          f"Pass: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}  "
          f"Fail: {len(result.failures)}  Error: {len(result.errors)}  Skip: {len(result.skipped)}")
    print(f"{'='*60}")

    all_passed = len(result.failures) == 0 and len(result.errors) == 0
    return execution_id, all_passed


def run_tests_async(
    cases_dir: str | Path,
    db_path: str | Path,
    **kwargs,
) -> str:
    """Run tests in a background thread. Returns execution_id immediately.

    Used by the Web API to trigger execution without blocking.
    """
    cases_dir = Path(cases_dir)
    init_db(db_path)

    cases_str = str(cases_dir)
    if cases_str not in sys.path:
        sys.path.insert(0, cases_str)

    suite = _load_suites(cases_dir, kwargs.get("paths"), kwargs.get("filter_k"))
    suite_cases = _collect_suite_cases(suite)
    total = sum(len(ms) for _, ms in suite_cases)

    if total == 0:
        return ""

    execution_id = store.generate_execution_id()
    store.create_execution(
        execution_id,
        bundle=kwargs.get("bundle"),
        target=kwargs.get("target"),
        suite_cases=suite_cases,
    )

    def _worker():
        result = AitfTestResult(execution_id)
        try:
            suite.run(result)
        except Exception:
            logger.exception("execution %s failed", execution_id)
        finally:
            store.finish_execution(execution_id)

    t = threading.Thread(target=_worker, name=f"aitf-run-{execution_id}", daemon=True)
    t.start()
    return execution_id
