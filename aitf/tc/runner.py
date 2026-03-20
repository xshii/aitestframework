"""Test runner with custom TestResult that writes to SQLite (REQ-4.3 / REQ-6)."""

from __future__ import annotations

import ctypes
import importlib.util
import json
import logging
import sys
import threading
import traceback
import unittest
from pathlib import Path

# Max captured stdout/stderr per test case (prevent DB bloat)
_MAX_OUTPUT_CHARS = 50_000


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    """Truncate text to limit chars, appending a note if truncated."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated, {len(text)} chars total)"

from aitf.tc import store
from aitf.tc.db import get_session, init_db
from aitf.tc.models import CaseStatus
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

class _PerTestLogHandler(logging.Handler):
    """Temporary handler attached to root logger during a single test.

    Captures all log records regardless of existing handler configuration.
    Unlike redirecting sys.stderr, this works because we add ourselves
    directly to the logger — we don't depend on any stream reference.
    """

    def __init__(self):
        super().__init__(logging.DEBUG)
        self._records: list[str] = []
        self.setFormatter(logging.Formatter(
            "%(levelname)-5s [%(name)s] %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._records.append(self.format(record))
        except Exception:
            pass

    def get_output(self) -> str:
        return "\n".join(self._records)


class AitfTestResult(unittest.TestResult):
    """Writes case status to the database as tests run.

    Capture strategy (per test):
      1. **stdout/stderr** — ``buffer = True`` delegates to unittest's own
         ``_setupStdout`` / ``_restoreStdout`` which swap sys.stdout/stderr
         with per-test StringIO buffers.  Captures ``print()`` output.
      2. **logging** — a ``_PerTestLogHandler`` is added to the root logger
         so we capture log records even when StreamHandlers hold references
         to the *original* stderr (which our redirect cannot reach).

    Lifecycle:
      - ``startTest``:  DB → RUNNING, attach log handler
        (super sets up stdout/stderr buffer)
      - ``addSuccess / addFailure / …``:  stash status + failure_reason
      - ``stopTest``:   read captured output, write final result to DB,
        detach log handler  (super restores stdout/stderr)
    """

    def __init__(self, execution_id: str, *,
                 test_timeout: int = 300, **kwargs):
        super().__init__(**kwargs)
        self.execution_id = execution_id
        self.buffer = True  # let unittest manage stdout/stderr capture
        self._session = get_session()
        self._test_timeout = test_timeout
        self._timeout_timer: threading.Timer | None = None
        self._test_thread_id: int | None = None
        # Per-test state (set in startTest, consumed in stopTest)
        self._log_handler: _PerTestLogHandler | None = None
        self._pending_status: str | None = None
        self._pending_kwargs: dict = {}

    def close(self) -> None:
        """Close the underlying DB session."""
        self._session.close()

    def _update(self, test, status, **kwargs):
        suite_class, method = test.__class__.__name__, test._testMethodName
        store.update_case_status(
            self.execution_id, suite_class, method, status,
            session=self._session, **kwargs,
        )

    # -- lifecycle -------------------------------------------------------------

    def startTest(self, test):
        super().startTest(test)          # sets up stdout/stderr buffer
        self._pending_status = None
        self._pending_kwargs = {}
        self._update(test, CaseStatus.RUNNING)
        # Attach per-test log handler
        self._log_handler = _PerTestLogHandler()
        logging.getLogger().addHandler(self._log_handler)
        # Start timeout watchdog
        self._test_thread_id = threading.current_thread().ident
        if self._test_timeout > 0:
            self._timeout_timer = threading.Timer(
                self._test_timeout, self._on_timeout, [test])
            self._timeout_timer.daemon = True
            self._timeout_timer.start()

    def _on_timeout(self, test):
        """Raise TimeoutError in the test thread when timeout expires."""
        tid = self._test_thread_id
        if tid is None:
            return
        logger.warning("Test %s timed out after %ds", test, self._test_timeout)
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(tid), ctypes.py_object(TimeoutError))

    def stopTest(self, test):
        # Cancel timeout timer
        if self._timeout_timer:
            self._timeout_timer.cancel()
            self._timeout_timer = None
        self._test_thread_id = None
        # 1. Collect logging output
        log_output = ""
        if self._log_handler:
            log_output = self._log_handler.get_output()
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None

        # 2. Collect stdout/stderr from unittest's buffer
        #    (must read BEFORE super().stopTest restores the streams)
        stdout_text = ""
        stderr_text = ""
        if self._stdout_buffer is not None:
            stdout_text = self._stdout_buffer.getvalue()
        if self._stderr_buffer is not None:
            stderr_text = self._stderr_buffer.getvalue()

        # 3. Merge logging into stdout (logging is informational output)
        if log_output:
            if stdout_text:
                stdout_text += "\n"
            stdout_text += log_output

        # 4. Truncate and write final status + captured output to DB
        if self._pending_status:
            kw = self._pending_kwargs
            if stdout_text:
                kw["stdout"] = _truncate(stdout_text)
            if stderr_text:
                kw["stderr"] = _truncate(stderr_text)
            self._update(test, self._pending_status, **kw)

        self._pending_status = None
        self._pending_kwargs = {}
        super().stopTest(test)           # restores stdout/stderr

    # -- result callbacks (just stash, actual DB write in stopTest) ------------

    def addSuccess(self, test):
        super().addSuccess(test)
        self._pending_status = CaseStatus.PASS

    def addFailure(self, test, err):
        super().addFailure(test, err)
        reason = "".join(traceback.format_exception(*err))
        self._pending_status = CaseStatus.FAIL
        self._pending_kwargs = {"failure_reason": reason}

    def addError(self, test, err):
        super().addError(test, err)
        reason = "".join(traceback.format_exception(*err))
        status = CaseStatus.ERROR
        if err[0] is TimeoutError:
            status = CaseStatus.TIMEOUT
        elif err[0] is SystemExit or err[0] is KeyboardInterrupt:
            status = CaseStatus.CRASH
        self._pending_status = status
        self._pending_kwargs = {"failure_reason": reason}

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._pending_status = CaseStatus.SKIP
        self._pending_kwargs = {"failure_reason": reason}


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
    keywords = [kw.strip().lower() for kw in k.split("|") if kw.strip()]
    for test in _iter_tests(suite):
        test_str = str(test).lower()
        if any(kw in test_str for kw in keywords):
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
# Shared helpers: prepare execution + run suite
# ---------------------------------------------------------------------------

def _ensure_sys_path(cases_dir: Path) -> None:
    """Ensure cases_dir is on sys.path for unittest discovery."""
    cases_str = str(cases_dir)
    if cases_str not in sys.path:
        sys.path.insert(0, cases_str)


def _prepare_execution(
    cases_dir: Path, config: RunConfig, *, plan_name: str | None = None,
    platform: str | None = None,
    pipeline_id: str | None = None, pipeline_stage: int | None = None,
) -> tuple[str, unittest.TestSuite | None]:
    """Load suites and create execution row. Returns (execution_id, suite).

    Returns ("", None) if no tests are found.
    """
    _ensure_sys_path(cases_dir)

    suite = _load_suites(cases_dir, config.tests or None, config.filter_k)
    suite_cases = _collect_suite_cases(suite)
    total = sum(len(ms) for _, ms in suite_cases)

    if total == 0:
        logger.warning("No tests found for config: %s", config.name)
        return "", None

    execution_id = store.generate_execution_id()
    store.create_execution(
        execution_id,
        bundle=config.bundle, target=config.target,
        platform=platform,
        golden_model=config.golden_model, golden_version=config.golden_version,
        plan_name=plan_name or config.name or None,
        pipeline_id=pipeline_id, pipeline_stage=pipeline_stage,
        suite_cases=suite_cases,
    )
    return execution_id, suite


def _execute_suite(
    execution_id: str, suite: unittest.TestSuite, config: RunConfig,
) -> bool:
    """Run suite with context tracking. Returns True if all passed."""
    _set_run_context(config)
    result = AitfTestResult(execution_id,
                            test_timeout=config.test_timeout)
    try:
        suite.run(result)

        # Retry failed tests if configured
        if config.retry > 0 and (result.failures or result.errors):
            _retry_failed(result, suite, config)
    finally:
        result.close()
        _set_run_context(None)

    store.finish_execution(execution_id)

    # Save results to files for archiving (REQ-6.6)
    _save_report_files(execution_id)

    # Sync results to remote server (if running in CLIENT mode)
    _enqueue_sync(execution_id)

    return len(result.failures) == 0 and len(result.errors) == 0


def _retry_failed(result: AitfTestResult, suite: unittest.TestSuite,
                  config: RunConfig) -> None:
    """Re-run failed tests up to config.retry times."""
    for attempt in range(config.retry):
        # Collect currently failed test IDs
        failed_ids = {str(t) for t, _ in result.failures + result.errors}
        if not failed_ids:
            break

        # Build a sub-suite of only the failed tests
        retry_suite = unittest.TestSuite()
        for test in _iter_tests(suite):
            if str(test) in failed_ids:
                retry_suite.addTest(test)

        if not retry_suite.countTestCases():
            break

        logger.info("Retry attempt %d/%d: %d failed tests",
                    attempt + 1, config.retry, len(failed_ids))

        # Clear previous failures for these tests so re-run overwrites
        result.failures = [(t, tb) for t, tb in result.failures
                           if str(t) not in failed_ids]
        result.errors = [(t, tb) for t, tb in result.errors
                         if str(t) not in failed_ids]

        retry_suite.run(result)


# ---------------------------------------------------------------------------
# Report files — save per-case stdout/stderr to build/reports/ (REQ-6.6)
# ---------------------------------------------------------------------------

def _report_root() -> Path:
    """Resolve report output directory from Flask config or fallback."""
    try:
        from flask import current_app
        cfg = current_app.config.get("AITF_CONFIG")
        if cfg:
            return cfg.build_root / "reports"
    except (ImportError, RuntimeError):
        pass
    return Path("data") / "build" / "reports"


def _save_report_files(execution_id: str) -> None:
    """Write per-case log files and JSON report to build/reports/<eid>/."""
    detail = store.get_execution_detail(execution_id)
    if not detail:
        return

    try:
        report_dir = _report_root() / execution_id
        report_dir.mkdir(parents=True, exist_ok=True)

        # JSON summary
        (report_dir / "result.json").write_text(
            json.dumps(detail, indent=2, ensure_ascii=False), encoding="utf-8")

        # JUnit XML for Jenkins
        try:
            from aitf.tc.report import generate_junit_xml
            generate_junit_xml(execution_id, report_dir.parent)
        except Exception:
            logger.debug("JUnit XML generation failed", exc_info=True)

        # Per-case log files
        for case in detail.get("cases", []):
            suite = case.get("suite_class", "Unknown")
            method = case.get("case_method", "unknown")
            case_dir = report_dir / "logs" / suite / method
            case_dir.mkdir(parents=True, exist_ok=True)

            if case.get("stdout"):
                (case_dir / "stdout.log").write_text(case["stdout"], encoding="utf-8")
            if case.get("stderr"):
                (case_dir / "stderr.log").write_text(case["stderr"], encoding="utf-8")
            if case.get("failure_reason"):
                (case_dir / "traceback.txt").write_text(case["failure_reason"], encoding="utf-8")

        logger.info("Report files saved: %s", report_dir)
    except Exception:
        logger.exception("Failed to save report files for %s", execution_id)


# ---------------------------------------------------------------------------
# Sync — enqueue results for upload to remote server
# ---------------------------------------------------------------------------

def _enqueue_sync(execution_id: str) -> None:
    """If a SyncWorker is active, enqueue the execution for upload."""
    try:
        from flask import current_app
        sw = current_app.config.get("SYNC_WORKER")
        if sw is None:
            return
    except (ImportError, RuntimeError):
        # Not running inside Flask, or no app context
        return

    detail = store.get_execution_detail(execution_id)
    if detail:
        sw.enqueue(detail)
        logger.info("Enqueued execution %s for sync", execution_id)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_tests(
    cases_dir: str | Path,
    db_path: str | Path,
    config: RunConfig | None = None,
    *,
    paths: list[str] | None = None,
    filter_k: str | None = None,
    bundle: str | None = None,
    target: str | None = None,
    golden_model: str | None = None,
    golden_version: str | None = None,
    params: dict | None = None,
    verbosity: int = 1,
) -> tuple[str, bool]:
    """Run tests with a single config. Returns (execution_id, all_passed).

    Accepts either a ``RunConfig`` object or individual keyword arguments.
    """
    cases_dir = Path(cases_dir)
    init_db(db_path)

    if config is None:
        config = RunConfig(
            tests=paths or [],
            filter_k=filter_k,
            bundle=bundle, target=target,
            golden_model=golden_model, golden_version=golden_version,
            params=params or {},
        )

    eid, suite = _prepare_execution(cases_dir, config)
    if not eid:
        return "", True

    passed = _execute_suite(eid, suite, config)

    detail = store.get_execution_detail(eid)
    print(f"\n{'='*60}")
    print(f"Execution: {eid}")
    print(f"Total: {detail['total']}  Pass: {detail['passed']}  "
          f"Fail: {detail['failed_total']}  "
          f"Not run: {detail['not_run']}")
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

        eid, suite = _prepare_execution(cases_dir, config, plan_name=config.name or plan.name)
        if not eid:
            print("  (no tests)")
            continue

        passed = _execute_suite(eid, suite, config)
        execution_ids.append(eid)
        detail = store.get_execution_detail(eid)
        print(f"  result: {detail['passed']}/{detail['total']} passed, "
              f"{detail['failed_total']} failed "
              f"({detail['pass_rate']*100:.1f}%)")
        if not passed:
            all_passed = False

    print(f"\n{'='*60}")
    print(f"Plan complete: {len(execution_ids)} executions, "
          f"{'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"{'='*60}")

    return execution_ids, all_passed


def run_tests_async(
    cases_dir: str | Path,
    db_path: str | Path,
    config: RunConfig | None = None,
    **kwargs,
) -> str:
    """Run tests in a background thread. Returns execution_id immediately.

    Accepts either a ``RunConfig`` object or individual keyword arguments.
    """
    cases_dir = Path(cases_dir)
    init_db(db_path)

    if config is None:
        config = RunConfig(
            tests=kwargs.get("paths") or [],
            filter_k=kwargs.get("filter_k"),
            bundle=kwargs.get("bundle"),
            target=kwargs.get("target"),
            golden_model=kwargs.get("golden_model"),
            golden_version=kwargs.get("golden_version"),
            params=kwargs.get("params") or {},
        )

    eid, suite = _prepare_execution(cases_dir, config)
    if not eid:
        return ""

    def _worker():
        try:
            _execute_suite(eid, suite, config)
        except Exception:
            logger.exception("execution %s failed", eid)

    t = threading.Thread(target=_worker, name=f"aitf-run-{eid}", daemon=True)
    t.start()
    return eid


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
        eid, suite = _prepare_execution(
            cases_dir, config, plan_name=config.name or plan.name,
        )
        if not eid:
            continue

        execution_ids.append(eid)

        def _worker(eid=eid, s=suite, c=config):
            try:
                _execute_suite(eid, s, c)
            except Exception:
                logger.exception("execution %s failed", eid)

        t = threading.Thread(target=_worker, name=f"aitf-run-{eid}", daemon=True)
        t.start()

    return execution_ids


def run_pipeline_async(
    cases_dir: str | Path,
    db_path: str | Path,
    plan_path: str | Path,
) -> str:
    """Run a testplan as a platform pipeline in a single background thread.

    Pipeline stages execute sequentially. All configs run within each stage;
    if any config fails, the pipeline stops and remaining stages are skipped.

    Returns pipeline_id immediately.
    """
    import uuid
    from aitf.tc.testplan import load_testplan

    cases_dir = Path(cases_dir)
    init_db(db_path)

    plan = load_testplan(plan_path)
    pipeline_id = f"pipe-{store.generate_execution_id()}-{uuid.uuid4().hex[:4]}"

    if not plan.pipeline:
        logger.warning("No pipeline defined in testplan %s", plan.name)
        return ""

    def _pipeline_worker():
        for stage_idx, platform in enumerate(plan.pipeline):
            logger.info("Pipeline %s stage %d/%d: %s",
                        pipeline_id, stage_idx + 1, len(plan.pipeline), platform)

            stage_passed = True
            for config in plan.configs:
                plan_label = f"{config.name or plan.name} [{platform}]"
                eid, suite = _prepare_execution(
                    cases_dir, config,
                    plan_name=plan_label,
                    platform=platform,
                    pipeline_id=pipeline_id,
                    pipeline_stage=stage_idx,
                )
                if not eid:
                    continue

                try:
                    passed = _execute_suite(eid, suite, config)
                    if not passed:
                        stage_passed = False
                except Exception:
                    logger.exception("pipeline %s stage %s execution %s failed",
                                     pipeline_id, platform, eid)
                    stage_passed = False

            if not stage_passed:
                logger.warning("Pipeline %s stopped at stage %s (failed)",
                               pipeline_id, platform)
                break

        logger.info("Pipeline %s completed", pipeline_id)

    t = threading.Thread(target=_pipeline_worker,
                         name=f"aitf-pipeline-{pipeline_id}", daemon=True)
    t.start()
    return pipeline_id


def run_targets_pipeline_async(
    cases_dir: str | Path,
    db_path: str | Path,
    targets: list[str],
    **kwargs,
) -> str:
    """Run the same tests sequentially on multiple targets (pipeline by target).

    Each target is a stage. If a stage fails, remaining stages are skipped.
    Returns pipeline_id immediately.
    """
    import uuid

    cases_dir = Path(cases_dir)
    init_db(db_path)

    pipeline_id = f"pipe-{store.generate_execution_id()}-{uuid.uuid4().hex[:4]}"

    base_config = RunConfig(
        tests=kwargs.get("paths") or [],
        filter_k=kwargs.get("filter_k"),
        bundle=kwargs.get("bundle"),
        golden_model=kwargs.get("golden_model"),
        golden_version=kwargs.get("golden_version"),
        params=kwargs.get("params") or {},
    )

    def _worker():
        for stage_idx, target_name in enumerate(targets):
            logger.info("Pipeline %s stage %d/%d: target=%s",
                        pipeline_id, stage_idx + 1, len(targets), target_name)

            config = RunConfig(
                name=base_config.name,
                tests=base_config.tests,
                filter_k=base_config.filter_k,
                bundle=base_config.bundle,
                target=target_name,
                golden_model=base_config.golden_model,
                golden_version=base_config.golden_version,
                params=base_config.params,
            )

            eid, suite = _prepare_execution(
                cases_dir, config,
                plan_name=target_name,
                platform=target_name,
                pipeline_id=pipeline_id,
                pipeline_stage=stage_idx,
            )
            if not eid:
                logger.warning("Pipeline %s stage %s: no tests found, stopping",
                               pipeline_id, target_name)
                break

            try:
                passed = _execute_suite(eid, suite, config)
                if not passed:
                    logger.warning("Pipeline %s stopped at stage %s (failed)",
                                   pipeline_id, target_name)
                    break
            except Exception:
                logger.exception("Pipeline %s stage %s failed",
                                 pipeline_id, target_name)
                break

        logger.info("Pipeline %s completed", pipeline_id)

    t = threading.Thread(target=_worker,
                         name=f"aitf-pipeline-{pipeline_id}", daemon=True)
    t.start()
    return pipeline_id
