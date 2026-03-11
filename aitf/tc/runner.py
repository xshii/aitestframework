"""Test runner with custom TestResult that writes to SQLite (REQ-4.3 / REQ-6)."""

from __future__ import annotations

import importlib.util
import logging
import sys
import threading
import traceback
import unittest
from datetime import datetime
from pathlib import Path

from aitf.tc import store
from aitf.tc.db import init_db
from aitf.tc.testplan import RunConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run context — thread-local, accessible by test cases via api module
# ---------------------------------------------------------------------------

_run_context = threading.local()


def get_run_context() -> RunConfig | None:
    """Return the current RunConfig (set during test execution)."""
    return getattr(_run_context, "config", None)


def _set_run_context(cfg: RunConfig | None) -> None:
    _run_context.config = cfg


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
    """Discover and load test suites from cases/ directory."""
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
    rel = filepath.relative_to(top_dir)
    module_name = str(rel.with_suffix("")).replace("/", ".").replace("\\", ".")
    spec = importlib.util.spec_from_file_location(module_name, filepath,
                                                   submodule_search_locations=[])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return loader.loadTestsFromModule(mod)


def _filter_suite(suite: unittest.TestSuite, k: str) -> unittest.TestSuite:
    filtered = unittest.TestSuite()
    for test in _iter_tests(suite):
        if k.lower() in str(test).lower():
            filtered.addTest(test)
    return filtered


def _iter_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_tests(item)
        else:
            yield item


def _collect_suite_cases(suite: unittest.TestSuite) -> list[tuple[str, list[str]]]:
    mapping: dict[str, list[str]] = {}
    for test in _iter_tests(suite):
        cls_name = test.__class__.__name__
        method = test._testMethodName
        mapping.setdefault(cls_name, []).append(method)
    return list(mapping.items())


# ---------------------------------------------------------------------------
# Run a single RunConfig
# ---------------------------------------------------------------------------

def _run_one_config(
    cases_dir: Path, config: RunConfig, verbosity: int = 1,
) -> tuple[str, bool]:
    """Run tests for one RunConfig. Returns (execution_id, all_passed)."""
    cases_str = str(cases_dir)
    if cases_str not in sys.path:
        sys.path.insert(0, cases_str)

    suite = _load_suites(cases_dir, config.tests or None, config.filter_k)
    suite_cases = _collect_suite_cases(suite)
    total = sum(len(ms) for _, ms in suite_cases)

    if total == 0:
        logger.warning("No tests found for config: %s", config.name)
        return "", True

    execution_id = store.generate_execution_id()
    store.create_execution(
        execution_id,
        bundle=config.bundle, target=config.target,
        golden_model=config.golden_model, golden_version=config.golden_version,
        plan_name=config.name or None,
        suite_cases=suite_cases,
    )

    logger.info("execution %s [%s]: %d tests", execution_id, config.name, total)

    # Set thread-local context so tests can access RunConfig
    _set_run_context(config)
    try:
        result = AitfTestResult(execution_id)
        suite.run(result)
    finally:
        _set_run_context(None)

    store.finish_execution(execution_id)

    passed = len(result.failures) == 0 and len(result.errors) == 0
    return execution_id, passed


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
    golden_model: str | None = None,
    golden_version: str | None = None,
    verbosity: int = 1,
) -> tuple[str, bool]:
    """Run tests with a single config. Returns (execution_id, all_passed)."""
    cases_dir = Path(cases_dir)
    init_db(db_path)

    config = RunConfig(
        tests=paths or [],
        filter_k=filter_k,
        bundle=bundle, target=target,
        golden_model=golden_model, golden_version=golden_version,
    )

    eid, passed = _run_one_config(cases_dir, config, verbosity)

    if not eid:
        return "", True

    detail = store.get_execution_detail(eid)
    print(f"\n{'='*60}")
    print(f"Execution: {eid}")
    print(f"Total: {detail['total']}  Pass: {detail['passed']}  "
          f"Fail: {detail['failed']}  Error: {detail['errored']}  "
          f"Skip: {detail['skipped']}")
    print(f"{'='*60}")

    return eid, passed


def run_testplan(
    cases_dir: str | Path,
    db_path: str | Path,
    plan_path: str | Path,
    verbosity: int = 1,
) -> tuple[list[str], bool]:
    """Run all configs in a testplan. Returns (execution_ids, all_passed)."""
    from aitf.tc.testplan import load_testplan

    cases_dir = Path(cases_dir)
    init_db(db_path)

    plan = load_testplan(plan_path)
    all_passed = True
    execution_ids: list[str] = []

    print(f"Test Plan: {plan.name} ({len(plan.configs)} configs)")
    print(f"{'='*60}")

    for i, config in enumerate(plan.configs, 1):
        label = config.name or f"config-{i}"
        print(f"\n[{i}/{len(plan.configs)}] {label}")
        if config.bundle:
            print(f"  bundle: {config.bundle}")
        if config.golden_model:
            print(f"  golden: {config.golden_model}/{config.golden_version}")

        eid, passed = _run_one_config(cases_dir, config, verbosity)
        if eid:
            execution_ids.append(eid)
            detail = store.get_execution_detail(eid)
            print(f"  result: {detail['passed']}/{detail['total']} passed "
                  f"({detail['pass_rate']*100:.1f}%)")
            if not passed:
                all_passed = False
        else:
            print("  (no tests)")

    print(f"\n{'='*60}")
    print(f"Plan complete: {len(execution_ids)} executions, "
          f"{'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"{'='*60}")

    return execution_ids, all_passed


def run_tests_async(
    cases_dir: str | Path,
    db_path: str | Path,
    **kwargs,
) -> str:
    """Run tests in a background thread. Returns execution_id immediately."""
    cases_dir = Path(cases_dir)
    init_db(db_path)

    config = RunConfig(
        tests=kwargs.get("paths") or [],
        filter_k=kwargs.get("filter_k"),
        bundle=kwargs.get("bundle"),
        target=kwargs.get("target"),
        golden_model=kwargs.get("golden_model"),
        golden_version=kwargs.get("golden_version"),
    )

    cases_str = str(cases_dir)
    if cases_str not in sys.path:
        sys.path.insert(0, cases_str)

    suite = _load_suites(cases_dir, config.tests or None, config.filter_k)
    suite_cases = _collect_suite_cases(suite)
    total = sum(len(ms) for _, ms in suite_cases)

    if total == 0:
        return ""

    execution_id = store.generate_execution_id()
    store.create_execution(
        execution_id,
        bundle=config.bundle, target=config.target,
        golden_model=config.golden_model, golden_version=config.golden_version,
        suite_cases=suite_cases,
    )

    def _worker():
        _set_run_context(config)
        result = AitfTestResult(execution_id)
        try:
            suite.run(result)
        except Exception:
            logger.exception("execution %s failed", execution_id)
        finally:
            _set_run_context(None)
            store.finish_execution(execution_id)

    t = threading.Thread(target=_worker, name=f"aitf-run-{execution_id}", daemon=True)
    t.start()
    return execution_id


def run_testplan_async(
    cases_dir: str | Path,
    db_path: str | Path,
    plan_path: str | Path,
) -> list[str]:
    """Run all configs in a testplan via background threads.
    Returns list of execution_ids immediately."""
    from aitf.tc.testplan import load_testplan

    cases_dir = Path(cases_dir)
    init_db(db_path)

    plan = load_testplan(plan_path)
    execution_ids: list[str] = []

    for config in plan.configs:
        cases_str = str(cases_dir)
        if cases_str not in sys.path:
            sys.path.insert(0, cases_str)

        suite = _load_suites(cases_dir, config.tests or None, config.filter_k)
        suite_cases = _collect_suite_cases(suite)
        total = sum(len(ms) for _, ms in suite_cases)

        if total == 0:
            continue

        execution_id = store.generate_execution_id()
        store.create_execution(
            execution_id,
            bundle=config.bundle, target=config.target,
            golden_model=config.golden_model, golden_version=config.golden_version,
            plan_name=config.name or plan.name,
            suite_cases=suite_cases,
        )
        execution_ids.append(execution_id)

        def _worker(eid=execution_id, s=suite, c=config):
            _set_run_context(c)
            result = AitfTestResult(eid)
            try:
                s.run(result)
            except Exception:
                logger.exception("execution %s failed", eid)
            finally:
                _set_run_context(None)
                store.finish_execution(eid)

        t = threading.Thread(target=_worker, name=f"aitf-run-{execution_id}", daemon=True)
        t.start()

    return execution_ids
