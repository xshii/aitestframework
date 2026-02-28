"""DepsManager — unified facade for dependency management."""

from __future__ import annotations

import logging
from pathlib import Path

from aitf.deps import acquire, doctor, repo
from aitf.deps.config import DepsConfig, load_deps_config
from aitf.deps.lock import generate_lock, save_lock
from aitf.deps.types import DepsError, DiagResult

logger = logging.getLogger(__name__)


class DepsManager:
    """Central facade for dependency operations (REQ-3).

    Args:
        project_root: Project root directory (contains deps.yaml).
        deps_file: Path to deps.yaml (relative to *project_root*).
        build_dir: Build output directory (default ``build/``).
    """

    def __init__(
        self, project_root: str | Path = ".",
        deps_file: str = "deps.yaml", build_dir: str = "build",
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
        if self._cfg is None:
            self._cfg = load_deps_config(self._deps_file)
        return self._cfg

    def reload(self) -> None:
        self._cfg = None

    # -- install -------------------------------------------------------------

    def install(self, name: str | None = None, *, locked: bool = False) -> None:
        cfg = self.config
        if name:
            self._install_one(name, cfg)
        else:
            for tc in cfg.toolchains.values():
                self._try(lambda t=tc: acquire.install_toolchain(
                    t, cache_dir=self._cache_dir, project_root=self._root,
                    remote=cfg.remote,
                ), f"toolchain {tc.name}")
            for lib in cfg.libraries.values():
                self._try(lambda l=lib: acquire.install_library(
                    l, cache_dir=self._cache_dir, project_root=self._root,
                    remote=cfg.remote,
                ), f"library {lib.name}")
            for rc in cfg.repos.values():
                self._try(lambda r=rc: self._clone_and_build(r), f"repo {rc.name}")

        lock = generate_lock(cfg, self._cache_dir, self._repos_dir)
        save_lock(lock, self._lock_path)

    def _install_one(self, name: str, cfg: DepsConfig) -> None:
        if name in cfg.toolchains:
            acquire.install_toolchain(
                cfg.toolchains[name], cache_dir=self._cache_dir,
                project_root=self._root, remote=cfg.remote,
            )
        elif name in cfg.libraries:
            acquire.install_library(
                cfg.libraries[name], cache_dir=self._cache_dir,
                project_root=self._root, remote=cfg.remote,
            )
        elif name in cfg.repos:
            self._clone_and_build(cfg.repos[name])
        else:
            raise DepsError(f"Unknown dependency: {name}")

    def _clone_and_build(self, rc: object) -> None:
        from aitf.deps.types import RepoConfig
        assert isinstance(rc, RepoConfig)
        repo_dir = repo.clone_repo(rc, self._repos_dir)
        repo.build_repo(rc, repo_dir, repo_dir, project_root=self._root)

    @staticmethod
    def _try(fn: object, label: str) -> None:
        try:
            fn()  # type: ignore[operator]
        except Exception as exc:
            logger.error("Failed to install %s: %s", label, exc)

    # -- list / lock / clean / doctor / env ----------------------------------

    def list_installed(self) -> list:
        cfg = self.config
        return [*cfg.toolchains.values(), *cfg.libraries.values(), *cfg.repos.values()]

    def lock(self) -> None:
        lf = generate_lock(self.config, self._cache_dir, self._repos_dir)
        save_lock(lf, self._lock_path)

    def clean(self) -> int:
        return acquire.clean_cache(self._cache_dir)

    def doctor(self) -> list[DiagResult]:
        return doctor.run_diagnostics(
            self.config, cache_dir=self._cache_dir, repos_dir=self._repos_dir,
            project_root=self._root,
            lock_path=self._lock_path if self._lock_path.exists() else None,
        )

    def get_env(self) -> dict[str, str]:
        cfg = self.config
        env: dict[str, str] = {}
        for name, tc in cfg.toolchains.items():
            d = self._cache_dir / f"{name}-{tc.version}"
            if d.is_dir():
                for k, v in tc.env.items():
                    env[k] = v.replace("{install_dir}", str(d))
        for name, rc in cfg.repos.items():
            d = self._repos_dir / name
            if d.is_dir():
                for k, v in rc.env.items():
                    env[k] = v.replace("{install_dir}", str(d))
        return env

    # -- path helpers --------------------------------------------------------

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def repos_dir(self) -> Path:
        return self._repos_dir

    def get_install_dir(self, name: str) -> Path | None:
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
