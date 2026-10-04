# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19267: the drift guard reads the NAMED omnimarket clone, never the
repository an inherited ``GIT_DIR`` / ``GIT_WORK_TREE`` points at.

Measured on h201 and h202 (lane omn19267-cb1b-2a21, 2026-10-04): the lane
process exports ``GIT_DIR`` and ``GIT_WORK_TREE`` for its own worktree, and
``git -C <clone>`` does not override either. The guard therefore inspected the
lane's detached worktree and refused with "canonical omnimarket clone is on a
DETACHED HEAD ... it tracks no branch" although the clone was attached to dev.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from omnibase_infra.cli import omnimarket_drift_guard as guard

pytestmark = pytest.mark.integration


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
            "HOME": str(cwd),
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
        },
        timeout=30,
    ).stdout.strip()


def _repo(path: Path, *, detach: bool) -> str:
    path.mkdir()
    _git(path, "init", "-q", "-b", "dev")
    (path / "f.txt").write_text(f"{path.name}\n", encoding="utf-8")
    _git(path, "add", "f.txt")
    _git(path, "commit", "-q", "-m", f"commit in {path.name}")
    sha = _git(path, "rev-parse", "HEAD")
    if detach:
        _git(path, "checkout", "-q", "--detach")
    return sha


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    """An ``$OMNI_HOME`` with an ATTACHED omnimarket clone, while the process
    environment points git at a different, DETACHED repository (the lane)."""
    home = tmp_path / "home"
    home.mkdir()
    clone_sha = _repo(home / "omnimarket", detach=False)
    lane = tmp_path / "lane"
    _repo(lane, detach=True)
    monkeypatch.setenv("GIT_DIR", str(lane / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(lane))
    monkeypatch.setenv("GIT_INDEX_FILE", str(lane / ".git" / "index"))
    return home, clone_sha


def test_attachment_reads_the_named_clone_not_the_inherited_git_dir(
    workspace: tuple[Path, str],
) -> None:
    home, _ = workspace

    assert (
        guard.canonical_clone_attachment(str(home))
        is guard.CanonicalCloneAttachment.ATTACHED
    ), "an inherited GIT_DIR made the guard read the lane's detached worktree"


def test_clone_commit_reads_the_named_clone_not_the_inherited_git_dir(
    workspace: tuple[Path, str],
) -> None:
    home, clone_sha = workspace

    assert guard.canonical_local_omnimarket_commit(str(home)) == clone_sha


def test_clean_git_env_drops_repository_selecting_variables_only(
    workspace: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_AUTHOR_NAME", "kept")

    env = guard._clean_git_env()

    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        assert name not in env
    assert env["GIT_AUTHOR_NAME"] == "kept"
    assert "PATH" in env
