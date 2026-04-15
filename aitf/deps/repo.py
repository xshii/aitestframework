"""Git repository dependency management."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from aitf.deps.types import RepoConfig, RepoError

logger = logging.getLogger(__name__)


def _git(args: list[str], *, cwd: str | None = None, timeout: int = 300) -> str:
    """Run a git command, returning stdout. Raises :class:`RepoError` on failure."""
    logger.info("git %s%s", " ".join(args), f"  (cwd={cwd})" if cwd else "")
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=cwd, timeout=timeout,
    )
    if result.returncode != 0:
        raise RepoError(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def clone_repo(repo: RepoConfig, dest: Path,
               repo_dir: Path | None = None) -> Path:
    """Clone or update a repository into *repo_dir* (default: *dest/<name>*).

    Cloning strategy (single round-trip in all cases):

    - **branch / tag**: ``git clone --branch <ref> --single-branch [--depth N]``
      lands HEAD on the right ref directly; no extra checkout needed.
    - **commit SHA**: ``git clone --filter=blob:none`` (partial clone) gets
      the full commit graph without file blobs, then ``git checkout <sha>``
      lazy-fetches just the blobs that commit references. Requires git ≥ 2.19
      on the client; works against all modern servers without special config.
    """
    repo_dir = repo_dir if repo_dir else dest / repo.name
    if repo_dir.is_dir():
        return update_repo(repo, repo_dir)

    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    is_commit = _looks_like_commit(repo.ref)
    clone_args = ["clone"]

    if is_commit:
        # Partial clone: commit graph only, blobs lazy-fetched on checkout.
        # --branch can't accept raw SHAs, so we don't pass it here.
        clone_args += ["--filter=blob:none"]
    else:
        # Branch or tag: clone exactly that ref, optionally shallow.
        if repo.depth:
            clone_args += ["--depth", str(repo.depth)]
        clone_args += ["--branch", repo.ref, "--single-branch"]

    if repo.sparse_checkout:
        # sparse_checkout already implies --filter=blob:none above is fine;
        # add --sparse so git only materialises the configured paths.
        if "--filter=blob:none" not in clone_args:
            clone_args += ["--filter=blob:none"]
        clone_args += ["--sparse"]
    clone_args += [repo.url, str(repo_dir)]

    _git(clone_args, timeout=600)

    if repo.sparse_checkout:
        _git(["sparse-checkout", "set", *repo.sparse_checkout], cwd=str(repo_dir))

    if is_commit:
        _git(["checkout", repo.ref], cwd=str(repo_dir))
    # For branch/tag, --branch already left HEAD on the right ref.
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
        except RepoError as exc:
            # Best-effort: server may not allow fetching arbitrary SHAs
            # (uploadpack.allowReachableSHA1InWant). Fall through to checkout
            # which will succeed if the SHA is already reachable, fail clearly
            # otherwise.
            logger.debug("shallow SHA fetch failed for %s, falling through: %s", ref, exc)

    try:
        _git(["checkout", ref], cwd=str(repo_dir))
    except RepoError as exc:
        logger.debug("direct checkout of %s failed, trying tracking branch: %s", ref, exc)
        try:
            _git(["checkout", "-b", ref, f"origin/{ref}"], cwd=str(repo_dir))
        except RepoError as exc2:
            raise RepoError(f"Cannot checkout ref '{ref}' in {repo.name}") from exc2


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
