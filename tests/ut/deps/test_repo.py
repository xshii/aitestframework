"""Tests for deps.repo — git repository management."""

from __future__ import annotations

from pathlib import Path

import pytest

from aitf.deps.repo import (
    _looks_like_commit,
    clone_repo,
    get_head_commit,
    is_cloned,
    update_repo,
)
from aitf.deps.types import RepoConfig, RepoError


class TestLooksLikeCommit:
    def test_short_hash(self):
        assert _looks_like_commit("abc1234") is True

    def test_full_hash(self):
        assert _looks_like_commit("a" * 40) is True

    def test_branch_name(self):
        assert _looks_like_commit("main") is False

    def test_tag(self):
        assert _looks_like_commit("v1.2.3") is False

    def test_too_short(self):
        assert _looks_like_commit("abc") is False


class TestIsCloned:
    def test_not_cloned(self, tmp_path):
        assert is_cloned("repo", tmp_path) is False

    def test_dir_without_git(self, tmp_path):
        (tmp_path / "repo").mkdir()
        assert is_cloned("repo", tmp_path) is False

    def test_cloned(self, tmp_path):
        (tmp_path / "repo" / ".git").mkdir(parents=True)
        assert is_cloned("repo", tmp_path) is True


class TestCloneRepo:
    """Integration tests that use real git commands on a local bare repo."""

    @pytest.fixture()
    def local_bare_repo(self, tmp_path):
        """Create a local bare git repo to clone from."""
        import subprocess

        bare = tmp_path / "bare_repo.git"
        bare.mkdir()
        subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)

        # Create a working copy, add a commit, push to bare
        work = tmp_path / "work"
        work.mkdir()
        subprocess.run(["git", "init", str(work)], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(work), "remote", "add", "origin", str(bare)],
            check=True, capture_output=True,
        )
        (work / "README.md").write_text("hello\n")
        subprocess.run(
            ["git", "-C", str(work), "add", "."], check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(work), "commit", "-m", "initial"],
            check=True, capture_output=True,
            env={"GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
                 "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t",
                 "HOME": str(tmp_path), "PATH": "/usr/bin:/bin:/usr/local/bin"},
        )
        # Determine default branch name
        result = subprocess.run(
            ["git", "-C", str(work), "branch", "--show-current"],
            check=True, capture_output=True, text=True,
        )
        branch = result.stdout.strip()
        subprocess.run(
            ["git", "-C", str(work), "push", "origin", branch],
            check=True, capture_output=True,
        )
        return bare, branch

    def test_clone_basic(self, tmp_path, local_bare_repo):
        bare, branch = local_bare_repo
        dest = tmp_path / "repos"
        rc = RepoConfig(name="test-repo", url=str(bare), ref=branch)
        repo_dir = clone_repo(rc, dest)
        assert repo_dir.is_dir()
        assert (repo_dir / "README.md").is_file()
        assert is_cloned("test-repo", dest)

    def test_clone_already_exists_updates(self, tmp_path, local_bare_repo):
        bare, branch = local_bare_repo
        dest = tmp_path / "repos"
        rc = RepoConfig(name="test-repo", url=str(bare), ref=branch)
        clone_repo(rc, dest)
        # Clone again — should update instead
        repo_dir = clone_repo(rc, dest)
        assert repo_dir.is_dir()

    def test_clone_with_depth(self, tmp_path, local_bare_repo):
        bare, branch = local_bare_repo
        dest = tmp_path / "repos"
        rc = RepoConfig(name="shallow", url=str(bare), ref=branch, depth=1)
        repo_dir = clone_repo(rc, dest)
        assert repo_dir.is_dir()

    def test_get_head_commit(self, tmp_path, local_bare_repo):
        bare, branch = local_bare_repo
        dest = tmp_path / "repos"
        rc = RepoConfig(name="test-repo", url=str(bare), ref=branch)
        repo_dir = clone_repo(rc, dest)
        commit = get_head_commit(repo_dir)
        assert len(commit) == 40
        assert all(c in "0123456789abcdef" for c in commit)

    def test_clone_bad_url(self, tmp_path):
        dest = tmp_path / "repos"
        rc = RepoConfig(name="bad-repo", url="/nonexistent/path.git", ref="main")
        with pytest.raises(RepoError):
            clone_repo(rc, dest)
