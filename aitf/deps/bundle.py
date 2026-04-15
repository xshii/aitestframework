"""BundleManager — configuration set (bundle) management."""

from __future__ import annotations

import logging
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path

import yaml

from aitf.deps.config import save_deps_config
from aitf.deps.manager import DepsManager
from aitf.deps.types import BundleConfig, BundleError, BundleNotFoundError, BundleStatus

logger = logging.getLogger(__name__)


def _tar_directory(src: Path, dest: Path) -> None:
    with tarfile.open(dest, "w:gz") as tf:
        tf.add(src, arcname=src.name)


class BundleManager:
    """Manage configuration bundles."""

    def __init__(self, deps_mgr: DepsManager, deps_file: str | Path = "deps.yaml") -> None:
        self._mgr = deps_mgr
        self._deps_file = Path(deps_file)

    # -- query ---------------------------------------------------------------

    def list_bundles(self) -> list[BundleConfig]:
        return list(self._mgr.config.bundles.values())

    def show(self, name: str) -> BundleConfig:
        b = self._mgr.config.bundles.get(name)
        if b is None:
            raise BundleNotFoundError(f"Bundle not found: {name}")
        return b

    def active(self) -> BundleConfig | None:
        name = self._mgr.config.active_bundle
        return self._mgr.config.bundles.get(name) if name else None

    # -- use / install -------------------------------------------------------

    def use(
        self, name: str, *, force: bool = False,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> None:
        bundle = self.show(name)
        if bundle.status == BundleStatus.DEPRECATED and not force:
            raise BundleError(f"Bundle '{name}' is deprecated. Use --force to override.")

        self.install(name, on_progress=on_progress)
        self._mgr.config.active_bundle = name
        save_deps_config(self._mgr.config, self._deps_file)

    def install(
        self, name: str, *,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> None:
        bundle = self.show(name)
        cfg = self._mgr.config

        dep_names = [
            *(n for n in bundle.toolchains if n in cfg.toolchains),
            *(n for n in bundle.repos if n in cfg.repos),
        ]
        for i, dep_name in enumerate(dep_names):
            if on_progress:
                on_progress(i + 1, len(dep_names), dep_name)
            try:
                self._mgr._install_one(dep_name, cfg)
            except Exception as exc:
                logger.warning("Failed to install %s: %s", dep_name, exc)

    # -- export / import -----------------------------------------------------

    def export_bundle(self, name: str, output: str | Path) -> Path:
        bundle = self.show(name)
        cfg = self._mgr.config
        output_path = Path(output)

        sections = (
            ("toolchains", bundle.toolchains, cfg.toolchains),
            ("repos",      bundle.repos,      cfg.repos),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir) / name
            staging.mkdir()

            with open(staging / "bundle.yaml", "w", encoding="utf-8") as fh:
                yaml.dump({
                    "name": bundle.name, "description": bundle.description,
                    "status": bundle.status, "toolchains": bundle.toolchains,
                    "repos": bundle.repos,
                }, fh, default_flow_style=False, allow_unicode=True)

            for section, selected, known in sections:
                sec_dir = staging / section
                sec_dir.mkdir()
                for dep_name, dep_ver in selected.items():
                    if dep_name not in known:
                        continue
                    src = self._mgr.get_install_dir(dep_name)
                    if src and src.is_dir():
                        _tar_directory(src, sec_dir / f"{dep_name}-{dep_ver}.tar.gz")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(output_path, "w:gz") as tf:
                tf.add(staging, arcname=name)

        return output_path

    def import_bundle(self, path: str | Path) -> str:
        archive = Path(path)
        if not archive.is_file():
            raise BundleError(f"Archive not found: {archive}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with tarfile.open(archive, "r:gz") as tf:
                tf.extractall(tmp, filter="data")

            dirs = [d for d in tmp.iterdir() if d.is_dir()]
            if len(dirs) != 1:
                raise BundleError("Expected exactly one top-level directory in bundle archive")
            bundle_dir = dirs[0]

            meta_file = bundle_dir / "bundle.yaml"
            if not meta_file.exists():
                raise BundleError("bundle.yaml not found in archive")
            with open(meta_file, encoding="utf-8") as fh:
                meta = yaml.safe_load(fh)
            bundle_name = meta.get("name", bundle_dir.name)

            for section, dest_base in [
                ("toolchains", self._mgr.cache_dir),
                ("repos", self._mgr.repos_dir),
            ]:
                sec_dir = bundle_dir / section
                if not sec_dir.is_dir():
                    continue
                for arc in sec_dir.glob("*.tar.gz"):
                    dest_name = arc.stem.replace(".tar", "")
                    dest = dest_base / dest_name
                    if not dest.is_dir():
                        dest.mkdir(parents=True, exist_ok=True)
                        with tarfile.open(arc, "r:gz") as tf:
                            tf.extractall(dest, filter="data")

        return bundle_name
