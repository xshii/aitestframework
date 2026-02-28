"""BundleManager — configuration set (bundle) management."""

from __future__ import annotations

import logging
import shutil
import tarfile
import tempfile
from pathlib import Path

import yaml

from aitf.deps.config import DepsConfig, load_deps_config, save_deps_config
from aitf.deps.manager import DepsManager
from aitf.deps.types import (
    BundleConfig,
    BundleError,
    BundleNotFoundError,
    BundleStatus,
)

logger = logging.getLogger(__name__)


class BundleManager:
    """Manage configuration bundles (REQ-3.5).

    A bundle is a named, versioned set of toolchain + library + repo versions
    that have been tested together.

    Args:
        deps_mgr: The :class:`DepsManager` to delegate install operations to.
        deps_file: Path to deps.yaml.
    """

    def __init__(
        self,
        deps_mgr: DepsManager,
        deps_file: str | Path = "deps.yaml",
    ) -> None:
        self._mgr = deps_mgr
        self._deps_file = Path(deps_file)

    @property
    def _cfg(self) -> DepsConfig:
        return self._mgr.config

    # -- list / show ---------------------------------------------------------

    def list_bundles(self) -> list[BundleConfig]:
        """Return all defined bundles."""
        return list(self._cfg.bundles.values())

    def show(self, name: str) -> BundleConfig:
        """Return details of a named bundle.

        Raises:
            BundleNotFoundError: If the bundle does not exist.
        """
        b = self._cfg.bundles.get(name)
        if b is None:
            raise BundleNotFoundError(f"Bundle not found: {name}")
        return b

    def active(self) -> BundleConfig | None:
        """Return the currently active bundle, or ``None``."""
        name = self._cfg.active_bundle
        if not name:
            return None
        return self._cfg.bundles.get(name)

    # -- use / switch --------------------------------------------------------

    def use(self, name: str, *, force: bool = False) -> None:
        """Switch to a named bundle.

        This sets the ``active`` field in deps.yaml and installs any missing
        dependencies declared by the bundle.

        Args:
            name: Bundle name to activate.
            force: If ``True``, allow switching to deprecated bundles.

        Raises:
            BundleNotFoundError: If the bundle does not exist.
            BundleError: If the bundle is deprecated and *force* is ``False``.
        """
        bundle = self.show(name)

        if bundle.status == BundleStatus.DEPRECATED and not force:
            raise BundleError(
                f"Bundle '{name}' is deprecated. Use --force to override."
            )

        # Install missing deps
        self.install(name)

        # Update active in config and persist
        self._cfg.active_bundle = name
        save_deps_config(self._cfg, self._deps_file)
        logger.info("Switched to bundle: %s", name)

    # -- install -------------------------------------------------------------

    def install(self, name: str) -> None:
        """Install all dependencies required by a bundle.

        Missing archives or network errors are logged as warnings rather than
        aborting the entire operation.

        Raises:
            BundleNotFoundError: If the bundle does not exist.
        """
        bundle = self.show(name)

        for tc_name in bundle.toolchains:
            if tc_name in self._cfg.toolchains:
                try:
                    self._mgr._install_one(tc_name, self._cfg)
                except Exception as exc:
                    logger.warning("Failed to install toolchain %s: %s", tc_name, exc)
            else:
                logger.warning(
                    "Bundle %s references unknown toolchain: %s", name, tc_name,
                )

        for lib_name in bundle.libraries:
            if lib_name in self._cfg.libraries:
                try:
                    self._mgr._install_one(lib_name, self._cfg)
                except Exception as exc:
                    logger.warning("Failed to install library %s: %s", lib_name, exc)
            else:
                logger.warning(
                    "Bundle %s references unknown library: %s", name, lib_name,
                )

        for repo_name in bundle.repos:
            if repo_name in self._cfg.repos:
                try:
                    self._mgr._install_one(repo_name, self._cfg)
                except Exception as exc:
                    logger.warning("Failed to install repo %s: %s", repo_name, exc)
            else:
                logger.warning(
                    "Bundle %s references unknown repo: %s", name, repo_name,
                )

    # -- export / import -----------------------------------------------------

    def export_bundle(self, name: str, output: str | Path) -> Path:
        """Export a bundle as an offline ``.tar.gz`` archive.

        The archive includes bundle metadata and all cached archives / repo
        snapshots so the bundle can be deployed on another machine.

        Args:
            name: Bundle name to export.
            output: Output file path.

        Returns:
            Path to the created archive.
        """
        bundle = self.show(name)
        output_path = Path(output)

        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir) / name
            staging.mkdir()

            # Write bundle metadata
            meta = {
                "name": bundle.name,
                "description": bundle.description,
                "status": bundle.status,
                "toolchains": bundle.toolchains,
                "libraries": bundle.libraries,
                "repos": bundle.repos,
                "env": bundle.env,
            }
            with open(staging / "bundle.yaml", "w", encoding="utf-8") as fh:
                yaml.dump(meta, fh, default_flow_style=False, allow_unicode=True)

            # Copy toolchain archives
            tc_dir = staging / "toolchains"
            tc_dir.mkdir()
            for tc_name, tc_ver in bundle.toolchains.items():
                src = self._mgr.cache_dir / f"{tc_name}-{tc_ver}"
                if src.is_dir():
                    _tar_directory(src, tc_dir / f"{tc_name}-{tc_ver}.tar.gz")

            # Copy library archives
            lib_dir = staging / "libraries"
            lib_dir.mkdir()
            for lib_name, lib_ver in bundle.libraries.items():
                src = self._mgr.cache_dir / f"{lib_name}-{lib_ver}"
                if src.is_dir():
                    _tar_directory(src, lib_dir / f"{lib_name}-{lib_ver}.tar.gz")

            # Snapshot repos
            repo_dir = staging / "repos"
            repo_dir.mkdir()
            for repo_name, repo_ref in bundle.repos.items():
                src = self._mgr.repos_dir / repo_name
                if src.is_dir():
                    _tar_directory(src, repo_dir / f"{repo_name}-{repo_ref}.tar.gz")

            # Create final archive
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(output_path, "w:gz") as tf:
                tf.add(staging, arcname=name)

        logger.info("Exported bundle '%s' -> %s", name, output_path)
        return output_path

    def import_bundle(self, path: str | Path) -> str:
        """Import a bundle from an offline archive.

        Extracts toolchains and libraries into the cache, repos into repos_dir.

        Args:
            path: Path to the ``.tar.gz`` bundle archive.

        Returns:
            Name of the imported bundle.

        Raises:
            BundleError: If the archive is invalid.
        """
        archive = Path(path)
        if not archive.is_file():
            raise BundleError(f"Archive not found: {archive}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with tarfile.open(archive, "r:gz") as tf:
                tf.extractall(tmp, filter="data")

            # Find the bundle directory (top-level dir in archive)
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

            # Extract toolchains
            tc_dir = bundle_dir / "toolchains"
            if tc_dir.is_dir():
                for arc in tc_dir.glob("*.tar.gz"):
                    dest_name = arc.stem.replace(".tar", "")
                    dest = self._mgr.cache_dir / dest_name
                    if not dest.is_dir():
                        dest.mkdir(parents=True, exist_ok=True)
                        with tarfile.open(arc, "r:gz") as tf:
                            tf.extractall(dest, filter="data")

            # Extract libraries
            lib_dir = bundle_dir / "libraries"
            if lib_dir.is_dir():
                for arc in lib_dir.glob("*.tar.gz"):
                    dest_name = arc.stem.replace(".tar", "")
                    dest = self._mgr.cache_dir / dest_name
                    if not dest.is_dir():
                        dest.mkdir(parents=True, exist_ok=True)
                        with tarfile.open(arc, "r:gz") as tf:
                            tf.extractall(dest, filter="data")

            # Extract repos
            repo_dir_src = bundle_dir / "repos"
            if repo_dir_src.is_dir():
                for arc in repo_dir_src.glob("*.tar.gz"):
                    # Extract repo name from filename: <name>-<ref>.tar.gz
                    stem = arc.stem.replace(".tar", "")
                    # Repo name is everything before the last hyphen-separated ref
                    dest = self._mgr.repos_dir / stem
                    if not dest.is_dir():
                        dest.mkdir(parents=True, exist_ok=True)
                        with tarfile.open(arc, "r:gz") as tf:
                            tf.extractall(dest, filter="data")

        logger.info("Imported bundle: %s", bundle_name)
        return bundle_name

    # -- env helpers ---------------------------------------------------------

    def get_bundle_env(self, name: str | None = None) -> dict[str, str]:
        """Collect env vars from the active (or named) bundle.

        Merges the bundle's own ``env`` with env vars from its dependencies.
        """
        if name is None:
            b = self.active()
            if b is None:
                return {}
        else:
            b = self.show(name)

        env = dict(self._mgr.get_env())
        env.update(b.env)
        return env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tar_directory(src: Path, dest: Path) -> None:
    """Create a .tar.gz archive of *src* directory."""
    with tarfile.open(dest, "w:gz") as tf:
        tf.add(src, arcname=src.name)
