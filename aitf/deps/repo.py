"""Git repository dependency management."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from aitf.deps.types import RepoConfig, RepoError

logger = logging.getLogger(__name__)


def _run_git(args: list[str], *, cwd: str | None = None, timeout: int = 300) -> str:
    """Run a git command, returning stdout on success.

    Raises:
        RepoError: If the command exits with a non-zero code.
    """
    cmd = ["git", *args]
    logger.debug("git %s", " ".join(args))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RepoError(f"git {' '.join(args)} failed:\n{stderr}")
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Clone / update
# ---------------------------------------------------------------------------

def clone_repo(repo: RepoConfig, dest: Path) -> Path:
    """Clone a repository into *dest/<name>*.

    Supports shallow clone (``depth``), sparse checkout, and arbitrary ref
    (branch, tag, or commit hash).

    If the repo already exists locally, it is updated instead.

    Returns:
        Path to the cloned/updated repository.
    """
    repo_dir = dest / repo.name
    if repo_dir.is_dir():
        return update_repo(repo, repo_dir)

    dest.mkdir(parents=True, exist_ok=True)

    # Build clone arguments
    clone_args = ["clone"]
    if repo.depth:
        clone_args += ["--depth", str(repo.depth)]
    if repo.sparse_checkout:
        clone_args += ["--filter=blob:none", "--sparse"]

    clone_args += [repo.url, str(repo_dir)]

    _run_git(clone_args, timeout=600)

    # Sparse checkout setup
    if repo.sparse_checkout:
        _run_git(
            ["sparse-checkout", "set", *repo.sparse_checkout],
            cwd=str(repo_dir),
        )

    # Checkout the desired ref
    _checkout_ref(repo, repo_dir)

    logger.info("Cloned %s -> %s (ref=%s)", repo.url, repo_dir, repo.ref)
    return repo_dir


def update_repo(repo: RepoConfig, repo_dir: Path) -> Path:
    """Fetch latest changes and checkout the configured ref."""
    fetch_args = ["fetch"]
    if repo.depth:
        fetch_args += ["--depth", str(repo.depth)]
    fetch_args += ["origin"]

    _run_git(fetch_args, cwd=str(repo_dir), timeout=300)
    _checkout_ref(repo, repo_dir)

    logger.info("Updated %s (ref=%s)", repo.name, repo.ref)
    return repo_dir


def _checkout_ref(repo: RepoConfig, repo_dir: Path) -> None:
    """Checkout the correct ref (branch, tag, or commit)."""
    ref = repo.ref

    # For shallow clones with a specific commit, fetch it first
    if repo.depth and _looks_like_commit(ref):
        try:
            _run_git(
                ["fetch", "--depth", str(repo.depth), "origin", ref],
                cwd=str(repo_dir),
            )
        except RepoError:
            # May already be present
            pass

    try:
        _run_git(["checkout", ref], cwd=str(repo_dir))
    except RepoError:
        # Try as remote branch
        try:
            _run_git(["checkout", "-b", ref, f"origin/{ref}"], cwd=str(repo_dir))
        except RepoError as exc:
            raise RepoError(f"Cannot checkout ref '{ref}' in {repo.name}") from exc


def _looks_like_commit(ref: str) -> bool:
    """Heuristic: a hex string of 7+ chars is likely a commit hash."""
    return len(ref) >= 7 and all(c in "0123456789abcdefABCDEF" for c in ref)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_head_commit(repo_dir: Path) -> str:
    """Return the full HEAD commit hash of a local repo."""
    return _run_git(["rev-parse", "HEAD"], cwd=str(repo_dir))


def is_cloned(name: str, repos_dir: Path) -> bool:
    """Check whether a repository has been cloned."""
    repo_dir = repos_dir / name
    return repo_dir.is_dir() and (repo_dir / ".git").exists()


def build_repo(repo: RepoConfig, repo_dir: Path, install_dir: Path, *, project_root: Path) -> None:
    """Run the repo's build script if configured.

    Script interface: ``bash <script> <repo_dir> <install_dir>``
    """
    if not repo.build_script:
        return
    script_path = project_root / repo.build_script
    if not script_path.is_file():
        raise RepoError(f"Build script not found: {script_path}")

    install_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Building %s with %s", repo.name, repo.build_script)

    result = subprocess.run(
        ["bash", str(script_path), str(repo_dir), str(install_dir)],
        capture_output=True,
        text=True,
        cwd=str(project_root),
        timeout=1800,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RepoError(
            f"Build script for {repo.name} failed (exit {result.returncode}):\n{stderr}"
        )
