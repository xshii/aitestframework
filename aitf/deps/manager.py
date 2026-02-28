"""DepsManager — unified facade for dependency management."""

from __future__ import annotations

import logging
from pathlib import Path

from aitf.deps import acquire, doctor, repo
from aitf.deps.config import DepsConfig, load_deps_config
from aitf.deps.lock import generate_lock, save_lock
from aitf.deps.types import (
    DepsError,
    DiagResult,
    LibraryConfig,
    RepoConfig,
    ToolchainConfig,
)

logger = logging.getLogger(__name__)


class DepsManager:
    """Central facade for dependency operations (REQ-3).

    Args:
        project_root: Project root directory (contains deps.yaml).
        deps_file: Path to deps.yaml (relative to *project_root*).
        build_dir: Build output directory (default ``build/``).
    """

    def __init__(
        self,
        project_root: str | Path = ".",
        deps_file: str = "deps.yaml",
        build_dir: str = "build",
    ) -> None:
        self._root = Path(project_root).resolve()
        self._deps_file = self._root / deps_file
        self._build_dir = self._root / build_dir
        self._cache_dir = self._build_dir / "cache"
        self._repos_dir = self._build_dir / "repos"
        self._lock_path = self._root / "deps.lock.yaml"

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._repos_dir.mkdir(parents=True, exist_ok=True)

        self._cfg: DepsConfig | None = None

    @property
    def config(self) -> DepsConfig:
        """Lazily load and cache the deps configuration."""
        if self._cfg is None:
            self._cfg = load_deps_config(self._deps_file)
        return self._cfg

    def reload(self) -> None:
        """Force-reload deps.yaml from disk."""
        self._cfg = None

    # -- install -------------------------------------------------------------

    def install(
        self,
        name: str | None = None,
        *,
        locked: bool = False,
    ) -> None:
        """Install dependencies.

        Args:
            name: Install only the named dependency. If ``None``, install all.
            locked: If ``True``, use deps.lock.yaml for exact versions (CI mode).
        """
        cfg = self.config

        if name:
            self._install_one(name, cfg)
        else:
            self._install_all(cfg)

        # Auto-update lock file after install
        lock = generate_lock(cfg, self._cache_dir, self._repos_dir)
        save_lock(lock, self._lock_path)

    def _install_one(self, name: str, cfg: DepsConfig) -> None:
        """Install a single named dependency."""
        if name in cfg.toolchains:
            acquire.install_toolchain(
                cfg.toolchains[name],
                cache_dir=self._cache_dir,
                project_root=self._root,
            )
            return

        if name in cfg.libraries:
            acquire.install_library(
                cfg.libraries[name],
                cache_dir=self._cache_dir,
                project_root=self._root,
            )
            return

        if name in cfg.repos:
            rc = cfg.repos[name]
            repo_dir = repo.clone_repo(rc, self._repos_dir)
            repo.build_repo(rc, repo_dir, repo_dir, project_root=self._root)
            return

        raise DepsError(f"Unknown dependency: {name}")

    def _install_all(self, cfg: DepsConfig) -> None:
        """Install all dependencies declared in deps.yaml."""
        for tc in cfg.toolchains.values():
            try:
                acquire.install_toolchain(
                    tc, cache_dir=self._cache_dir, project_root=self._root,
                )
            except Exception as exc:
                logger.error("Failed to install toolchain %s: %s", tc.name, exc)

        for lib in cfg.libraries.values():
            try:
                acquire.install_library(
                    lib, cache_dir=self._cache_dir, project_root=self._root,
                )
            except Exception as exc:
                logger.error("Failed to install library %s: %s", lib.name, exc)

        for rc in cfg.repos.values():
            try:
                repo_dir = repo.clone_repo(rc, self._repos_dir)
                repo.build_repo(rc, repo_dir, repo_dir, project_root=self._root)
            except Exception as exc:
                logger.error("Failed to install repo %s: %s", rc.name, exc)

    # -- list ----------------------------------------------------------------

    def list_installed(self) -> list[ToolchainConfig | LibraryConfig | RepoConfig]:
        """Return all dependencies with their installation status."""
        cfg = self.config
        items: list[ToolchainConfig | LibraryConfig | RepoConfig] = []
        items.extend(cfg.toolchains.values())
        items.extend(cfg.libraries.values())
        items.extend(cfg.repos.values())
        return items

    # -- lock ----------------------------------------------------------------

    def lock(self) -> None:
        """Generate / update ``deps.lock.yaml``."""
        cfg = self.config
        lf = generate_lock(cfg, self._cache_dir, self._repos_dir)
        save_lock(lf, self._lock_path)

    # -- clean ---------------------------------------------------------------

    def clean(self) -> int:
        """Remove all cached dependencies. Returns number of dirs removed."""
        return acquire.clean_cache(self._cache_dir)

    # -- doctor --------------------------------------------------------------

    def doctor(self) -> list[DiagResult]:
        """Run diagnostic checks."""
        cfg = self.config
        return doctor.run_diagnostics(
            cfg,
            cache_dir=self._cache_dir,
            repos_dir=self._repos_dir,
            project_root=self._root,
            lock_path=self._lock_path if self._lock_path.exists() else None,
        )

    # -- env -----------------------------------------------------------------

    def get_env(self) -> dict[str, str]:
        """Collect environment variables from all installed dependencies.

        Template variables like ``{install_dir}`` are resolved to actual paths.
        """
        cfg = self.config
        env: dict[str, str] = {}

        for name, tc in cfg.toolchains.items():
            install_dir = self._cache_dir / f"{name}-{tc.version}"
            if install_dir.is_dir():
                for key, val in tc.env.items():
                    env[key] = val.replace("{install_dir}", str(install_dir))

        for name, rc in cfg.repos.items():
            repo_dir = self._repos_dir / name
            if repo_dir.is_dir():
                for key, val in rc.env.items():
                    env[key] = val.replace("{install_dir}", str(repo_dir))

        return env

    # -- path helpers --------------------------------------------------------

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def repos_dir(self) -> Path:
        return self._repos_dir

    def get_install_dir(self, name: str) -> Path | None:
        """Return the install directory for a named dependency, or ``None``."""
        cfg = self.config
        if name in cfg.toolchains:
            d = self._cache_dir / f"{name}-{cfg.toolchains[name].version}"
            return d if d.is_dir() else None
        if name in cfg.libraries:
            d = self._cache_dir / f"{name}-{cfg.libraries[name].version}"
            return d if d.is_dir() else None
        if name in cfg.repos:
            d = self._repos_dir / name
            return d if d.is_dir() else None
        return None
