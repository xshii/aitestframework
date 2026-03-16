"""LocalExecutor — run commands on the local machine via subprocess."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time

from .executor import ExecuteResult, TargetConfig

logger = logging.getLogger(__name__)


class LocalExecutor:
    """Execute commands locally using :func:`subprocess.run`."""

    def __init__(self, config: TargetConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def run(
        self,
        command: str,
        timeout: int = 300,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> ExecuteResult:
        """Run *command* in a shell and return the result.

        The process environment is built by merging the current
        ``os.environ`` with ``config.env`` and the caller-supplied *env*
        (later values win).  If the process exceeds *timeout* seconds it
        receives SIGTERM; if it has not exited after 5 more seconds it
        receives SIGKILL.
        """
        merged_env = self._build_env(env)
        work_dir = cwd or self.config.build_dir

        logger.debug("LocalExecutor.run: %s (cwd=%s, timeout=%ds)", command, work_dir, timeout)

        start = time.monotonic()
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=merged_env,
                cwd=work_dir,
                start_new_session=True,
            )
            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("Command timed out after %ds, sending SIGTERM: %s", timeout, command)
                self._terminate_gracefully(proc)
                stdout_bytes, stderr_bytes = proc.communicate()
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error("Failed to execute command: %s", exc)
            return ExecuteResult(
                returncode=-1,
                stdout="",
                stderr=str(exc),
                duration_s=elapsed,
            )

        elapsed = time.monotonic() - start

        result = ExecuteResult(
            returncode=proc.returncode,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            duration_s=round(elapsed, 3),
        )
        logger.debug(
            "LocalExecutor.run finished: rc=%d duration=%.3fs",
            result.returncode,
            result.duration_s,
        )
        return result

    def close(self) -> None:
        """No-op for local executor."""

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _build_env(self, extra: dict[str, str] | None) -> dict[str, str]:
        merged = os.environ.copy()
        merged.update(self.config.env)
        if extra:
            merged.update(extra)
        return merged

    @staticmethod
    def _terminate_gracefully(proc: subprocess.Popen, grace: float = 5.0) -> None:
        """Send SIGTERM, wait up to *grace* seconds, then SIGKILL."""
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:
            pass
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            logger.warning("Process did not exit after SIGTERM; sending SIGKILL")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                pass
