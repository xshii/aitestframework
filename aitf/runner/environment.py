"""Environment — unified wrapper around local and remote executors."""

from __future__ import annotations

import logging

from .config import ExecuteResult, TargetConfig

logger = logging.getLogger(__name__)


class Environment:
    """High-level facade that delegates to a :class:`LocalExecutor` or
    :class:`RemoteExecutor` based on the target configuration.

    Usage::

        env = Environment("sim-server", config)
        result = env.run("./run_test --op conv2d")
        env.cleanup()
    """

    def __init__(self, name: str, config: TargetConfig) -> None:
        self.name = name
        self.config = config

        if not config.is_local:
            from .remote import RemoteExecutor

            self._executor = RemoteExecutor(config)
            self._executor.connect()
            logger.info("Environment %r initialised (remote: %s)", name, config.primary_host)
        else:
            from .local import LocalExecutor

            self._executor = LocalExecutor(config)
            logger.info("Environment %r initialised (local)", name)

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def run(
        self,
        command: str,
        timeout: int = 300,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> ExecuteResult:
        """Execute *command* on the target and return the result."""
        return self._executor.run(command, timeout=timeout, env=env, cwd=cwd)

    # ------------------------------------------------------------------
    # File transfer (remote targets only)
    # ------------------------------------------------------------------

    def upload(self, local_path: str, remote_path: str) -> None:
        from .remote import RemoteExecutor

        if isinstance(self._executor, RemoteExecutor):
            self._executor.upload(local_path, remote_path)
        else:
            raise NotImplementedError(
                f"upload() is not supported on local target {self.name!r}"
            )

    def download(self, remote_path: str, local_path: str) -> None:
        from .remote import RemoteExecutor

        if isinstance(self._executor, RemoteExecutor):
            self._executor.download(remote_path, local_path)
        else:
            raise NotImplementedError(
                f"download() is not supported on local target {self.name!r}"
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release resources held by the underlying executor."""
        logger.info("Cleaning up environment %r", self.name)
        self._executor.close()
