"""DepsManager — unified facade for dependency management."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from aitf.deps import acquire, doctor, repo
from aitf.deps.config import DepsConfig, load_deps_config
from aitf.deps.lock import generate_lock, save_lock
from aitf.deps.types import DepsError, DiagResult, RepoConfig, ToolchainConfig

logger = logging.getLogger(__name__)


class DepsManager:
    """Central facade for dependency operations."""

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

    def install(
        self, name: str | None = None, *,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> None:
        cfg = self.config
        if name:
            self._install_one(name, cfg)
            if on_progress:
                on_progress(1, 1, name)
        else:
            acquire.run_ordered(self._build_steps(cfg), on_progress=on_progress)

        lock = generate_lock(cfg, self._cache_dir, self._repos_dir,
                             build_dir=self._build_dir)
        save_lock(lock, self._lock_path)

    def _build_steps(self, cfg: DepsConfig) -> list[acquire.Step]:
        return [
            *(
                (tc.order, f"toolchain {tc.name}",
                 lambda t=tc: self._install_toolchain(t))
                for tc in cfg.toolchains.values()
            ),
            *(
                (rc.order, f"repo {rc.name}",
                 lambda r=rc: self._clone_and_build(r))
                for rc in cfg.repos.values()
            ),
        ]

    def _install_one(self, name: str, cfg: DepsConfig) -> None:
        if name in cfg.toolchains:
            self._install_toolchain(cfg.toolchains[name])
        elif name in cfg.repos:
            self._clone_and_build(cfg.repos[name])
        else:
            raise DepsError(f"Unknown dependency: {name}")

    def _install_toolchain(self, tc: ToolchainConfig) -> None:
        acquire.install_toolchain(
            tc, cache_dir=self._cache_dir, project_root=self._root,
            install_dir=tc.install_path(self._cache_dir, self._build_dir),
        )

    def _clone_and_build(self, rc: RepoConfig) -> None:
        target = rc.install_path(self._repos_dir, self._build_dir)
        repo_dir = repo.clone_repo(rc, self._repos_dir, repo_dir=target)
        repo.build_repo(rc, repo_dir, repo_dir, project_root=self._root)

    # -- list / lock / clean / doctor ----------------------------------------

    def list_installed(self) -> list:
        cfg = self.config
        return [*cfg.toolchains.values(), *cfg.repos.values()]

    def lock(self) -> None:
        lf = generate_lock(self.config, self._cache_dir, self._repos_dir,
                           build_dir=self._build_dir)
        save_lock(lf, self._lock_path)

    def clean(self) -> int:
        return acquire.clean_cache(self._cache_dir)

    def doctor(self) -> list[DiagResult]:
        return doctor.run_diagnostics(
            self.config, cache_dir=self._cache_dir, repos_dir=self._repos_dir,
            project_root=self._root, build_dir=self._build_dir,
            lock_path=self._lock_path if self._lock_path.exists() else None,
        )

    # -- path helpers --------------------------------------------------------

    @property
    def project_root(self) -> Path:
        return self._root

    @property
    def deps_file(self) -> Path:
        return self._deps_file

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def repos_dir(self) -> Path:
        return self._repos_dir

    def get_install_dir(self, name: str) -> Path | None:
        cfg = self.config
        if name in cfg.toolchains:
            d = cfg.toolchains[name].install_path(self._cache_dir, self._build_dir)
        elif name in cfg.repos:
            d = cfg.repos[name].install_path(self._repos_dir, self._build_dir)
        else:
            return None
        return d if d.is_dir() else None
