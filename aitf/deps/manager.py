"""DepsManager — unified facade for dependency management."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from aitf.deps import acquire, doctor, repo
from aitf.deps.config import DepsConfig, load_deps_config
from aitf.deps.lock import generate_lock, save_lock
from aitf.deps.types import DepsError, DiagResult, RepoConfig

logger = logging.getLogger(__name__)


class DepsManager:
    """Central facade for dependency operations (REQ-3)."""

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

    def _resolve_dir(self, dep, default_base: Path) -> Path | None:
        """Resolve custom ``install_dir`` (relative to build_root) or *None*."""
        if dep.install_dir:
            p = Path(dep.install_dir)
            return p if p.is_absolute() else (self._build_dir / p)
        return None

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
            kw = dict(cache_dir=self._cache_dir, project_root=self._root)
            steps: list[tuple[int, str, Callable[[], None]]] = [
                *(
                    (tc.order, f"toolchain {tc.name}",
                     lambda t=tc: acquire.install_toolchain(
                         t, **kw, install_dir=self._resolve_dir(t, self._cache_dir)))
                    for tc in cfg.toolchains.values()
                ),
                *(
                    (lib.order, f"library {lib.name}",
                     lambda l=lib: acquire.install_library(
                         l, **kw, install_dir=self._resolve_dir(l, self._cache_dir)))
                    for lib in cfg.libraries.values()
                ),
                *(
                    (rc.order, f"repo {rc.name}",
                     lambda r=rc: self._clone_and_build(r))
                    for rc in cfg.repos.values()
                ),
            ]
            steps.sort(key=lambda s: s[0])
            for i, (_, label, fn) in enumerate(steps):
                if on_progress:
                    on_progress(i + 1, len(steps), label)
                self._try(fn, label)

        lock = generate_lock(
            cfg, self._cache_dir, self._repos_dir,
            dir_overrides=self._dir_overrides(cfg),
        )
        save_lock(lock, self._lock_path)

    def _install_one(self, name: str, cfg: DepsConfig) -> None:
        kw = dict(cache_dir=self._cache_dir, project_root=self._root)
        if name in cfg.toolchains:
            tc = cfg.toolchains[name]
            acquire.install_toolchain(tc, **kw, install_dir=self._resolve_dir(tc, self._cache_dir))
        elif name in cfg.libraries:
            lib = cfg.libraries[name]
            acquire.install_library(lib, **kw, install_dir=self._resolve_dir(lib, self._cache_dir))
        elif name in cfg.repos:
            self._clone_and_build(cfg.repos[name])
        else:
            raise DepsError(f"Unknown dependency: {name}")

    def _clone_and_build(self, rc: RepoConfig) -> None:
        custom = self._resolve_dir(rc, self._repos_dir)
        repo_dir = repo.clone_repo(rc, self._repos_dir, repo_dir=custom)
        repo.build_repo(rc, repo_dir, repo_dir, project_root=self._root)

    def _dir_overrides(self, cfg: DepsConfig) -> dict[str, Path]:
        """Build a name→Path mapping for deps with custom install_dir."""
        out: dict[str, Path] = {}
        for name, dep in {**cfg.toolchains, **cfg.libraries}.items():
            d = self._resolve_dir(dep, self._cache_dir)
            if d:
                out[name] = d
        for name, dep in cfg.repos.items():
            d = self._resolve_dir(dep, self._repos_dir)
            if d:
                out[name] = d
        return out

    @staticmethod
    def _try(fn: Callable[[], None], label: str) -> None:
        try:
            fn()
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
            dir_overrides=self._dir_overrides(self.config),
        )

    def get_env(self) -> dict[str, str]:
        cfg = self.config
        env: dict[str, str] = {}
        entries = [
            *((self._resolve_dir(tc, self._cache_dir) or self._cache_dir / n, tc.env)
              for n, tc in cfg.toolchains.items()),
            *((self._resolve_dir(rc, self._repos_dir) or self._repos_dir / n, rc.env)
              for n, rc in cfg.repos.items()),
        ]
        for d, dep_env in entries:
            if d.is_dir():
                for k, v in dep_env.items():
                    env[k] = v.replace("{install_dir}", str(d))
        return env

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
        for section, base in [(cfg.toolchains, self._cache_dir), (cfg.libraries, self._cache_dir)]:
            if name in section:
                d = self._resolve_dir(section[name], base) or base / name
                return d if d.is_dir() else None
        if name in cfg.repos:
            dep = cfg.repos[name]
            d = self._resolve_dir(dep, self._repos_dir) or self._repos_dir / name
            return d if d.is_dir() else None
        return None
