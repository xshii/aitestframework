"""Git repository dependency management."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from aitf.deps.types import RepoConfig, RepoError

logger = logging.getLogger(__name__)


def _git(args: list[str], *, cwd: str | None = None, timeout: int = 300) -> str:
    """Run a git command, returning stdout. Raises :class:`RepoError` on failure."""
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=cwd, timeout=timeout,
    )
    if result.returncode != 0:
        raise RepoError(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def clone_repo(repo: RepoConfig, dest: Path) -> Path:
    """Clone or update a repository into *dest/<name>*."""
    repo_dir = dest / repo.name
    if repo_dir.is_dir():
        return update_repo(repo, repo_dir)

    dest.mkdir(parents=True, exist_ok=True)

    clone_args = ["clone"]
    if repo.depth:
        clone_args += ["--depth", str(repo.depth)]
    if repo.sparse_checkout:
        clone_args += ["--filter=blob:none", "--sparse"]
    clone_args += [repo.url, str(repo_dir)]

    _git(clone_args, timeout=600)

    if repo.sparse_checkout:
        _git(["sparse-checkout", "set", *repo.sparse_checkout], cwd=str(repo_dir))

    _checkout_ref(repo, repo_dir)
    return repo_dir


def update_repo(repo: RepoConfig, repo_dir: Path) -> Path:
    """Fetch latest changes and checkout the configured ref."""
    fetch_args = ["fetch"]
    if repo.depth:
        fetch_args += ["--depth", str(repo.depth)]
    fetch_args += ["origin"]

    _git(fetch_args, cwd=str(repo_dir), timeout=300)
    _checkout_ref(repo, repo_dir)
    return repo_dir


def _checkout_ref(repo: RepoConfig, repo_dir: Path) -> None:
    ref = repo.ref
    if repo.depth and _looks_like_commit(ref):
        try:
            _git(["fetch", "--depth", str(repo.depth), "origin", ref], cwd=str(repo_dir))
        except RepoError:
            pass

    try:
        _git(["checkout", ref], cwd=str(repo_dir))
    except RepoError:
        try:
            _git(["checkout", "-b", ref, f"origin/{ref}"], cwd=str(repo_dir))
        except RepoError as exc:
            raise RepoError(f"Cannot checkout ref '{ref}' in {repo.name}") from exc


def _looks_like_commit(ref: str) -> bool:
    return len(ref) >= 7 and all(c in "0123456789abcdefABCDEF" for c in ref)


def get_head_commit(repo_dir: Path) -> str:
    return _git(["rev-parse", "HEAD"], cwd=str(repo_dir))


def is_cloned(name: str, repos_dir: Path) -> bool:
    repo_dir = repos_dir / name
    return repo_dir.is_dir() and (repo_dir / ".git").exists()


def build_repo(repo: RepoConfig, repo_dir: Path, install_dir: Path, *, project_root: Path) -> None:
    """Run the repo's build script if configured."""
    if not repo.build_script:
        return
    from aitf.deps.acquire import run_script
    run_script(repo.build_script, [str(repo_dir), str(install_dir)],
               project_root=project_root, timeout=1800)
