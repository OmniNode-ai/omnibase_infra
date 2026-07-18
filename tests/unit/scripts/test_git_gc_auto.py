# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for scripts/git-gc-auto.sh (OMN-14760 / F-21).

The helper runs a non-destructive, worktree-aware ``git gc --auto`` across the
canonical clones under a root, but SKIPS any clone with an active git operation
(index.lock or in-progress rebase/merge/cherry-pick/revert/bisect) in the main
clone OR any linked worktree.

All git is driven in disposable tmp repos with GIT_DIR/GIT_INDEX_FILE/
GIT_WORK_TREE stripped from the environment, per the OMN-14746/14744
worktree-safety lesson — a leaked GIT_DIR would redirect every ``git -C`` at the
real repository.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER = REPO_ROOT / "scripts" / "git-gc-auto.sh"

_HERMETIC_ENV = {
    k: v
    for k, v in os.environ.items()
    if k not in {"GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE", "OMNI_HOME"}
}


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=_HERMETIC_ENV,
        capture_output=True,
        text=True,
        check=True,
    )


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(["init", "-q"], path)
    _git(["config", "user.email", "t@t.com"], path)
    _git(["config", "user.name", "T"], path)
    # Force any gc to run inline (default autoDetach backgrounds it) so the
    # helper's effect is observable synchronously in tests. Do NOT set a low
    # gc.auto here — that would make each commit auto-pack (default is 6700), and
    # we need loose objects to accumulate for the --execute test to observe.
    _git(["config", "gc.autoDetach", "false"], path)
    (path / "README.md").write_text("hello\n")
    _git(["add", "-A"], path)
    _git(["commit", "-qm", "init"], path)
    return path


def _loose_count(path: Path) -> int:
    out = _git(["count-objects"], path).stdout
    return int(out.split()[0])


def _populate_loose_objects_including_17(path: Path, batches: int = 12) -> None:
    """Create many *reachable* loose objects until objects/17 is non-empty.

    ``git gc --auto`` estimates the loose-object total by sampling the
    ``objects/17`` fan-out directory (count * 256); guaranteeing an entry there
    (with gc.auto=1) makes the auto-gc decision deterministic. The objects must
    be *reachable* (committed) — ``git gc`` packs reachable objects regardless of
    age, but keeps freshly-created *unreachable* objects loose until the 2-week
    prune grace, so unreachable blobs would never drop within a test.
    """
    obj17 = path / ".git" / "objects" / "17"
    for b in range(batches):
        d = path / f"data{b}"
        d.mkdir()
        for i in range(600):
            (d / f"f{i}.txt").write_text(f"content-{b}-{i}\n")
        _git(["add", "-A"], path)
        _git(["commit", "-qm", f"batch {b}"], path)
        if obj17.is_dir() and any(obj17.iterdir()):
            return
    raise AssertionError(
        f"objects/17 not populated after {batches} batches (test setup failure)"
    )


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(HELPER), *args],
        env=_HERMETIC_ENV,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_helper_exists_and_executable() -> None:
    assert HELPER.is_file()
    assert os.access(HELPER, os.X_OK)


def test_no_hardcoded_absolute_paths() -> None:
    text = HELPER.read_text()
    assert "/Users/" not in text
    assert "/Volumes/" not in text


def test_missing_root_fails_fast(tmp_path: Path) -> None:
    """No --root and no OMNI_HOME → fail fast (rule #8), not a wrong default."""
    result = _run()  # OMNI_HOME stripped from _HERMETIC_ENV
    assert result.returncode != 0
    assert "OMNI_HOME" in (result.stdout + result.stderr)


def test_dry_run_mutates_nothing(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repo = _make_repo(root / "repo")
    _populate_loose_objects_including_17(repo)
    # Simulate "over the loose-object threshold" (real OCC clone: ~11.8k > 6700).
    _git(["config", "gc.auto", "1"], repo)
    before = _loose_count(repo)
    result = _run("--root", str(root))  # dry-run is default
    assert result.returncode == 0
    assert "[would gc] repo" in result.stdout
    assert _loose_count(repo) == before, "dry-run must not mutate the object store"


def test_execute_packs_loose_and_preserves_refs(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repo = _make_repo(root / "repo")
    _populate_loose_objects_including_17(repo)
    # Simulate "over the loose-object threshold" (real OCC clone: ~11.8k > 6700).
    _git(["config", "gc.auto", "1"], repo)
    before = _loose_count(repo)
    head_before = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    log_before = _git(["log", "--oneline"], repo).stdout

    result = _run("--execute", "--root", str(root))
    assert result.returncode == 0, result.stderr
    assert "[gc] repo" in result.stdout

    after = _loose_count(repo)
    assert after < before, f"expected loose objects to drop ({before} -> {after})"
    # Refs are preserved — gc is non-destructive to reachable history.
    assert _git(["rev-parse", "HEAD"], repo).stdout.strip() == head_before
    assert _git(["log", "--oneline"], repo).stdout == log_before


def test_skip_when_index_lock_present(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repo = _make_repo(root / "repo")
    (repo / ".git" / "index.lock").write_text("")
    result = _run("--execute", "--root", str(root))
    assert result.returncode == 0
    assert "[skip] repo" in result.stdout
    assert "index.lock" in result.stdout


def test_skip_when_rebase_in_progress(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repo = _make_repo(root / "repo")
    (repo / ".git" / "rebase-merge").mkdir()
    result = _run("--execute", "--root", str(root))
    assert result.returncode == 0
    assert "[skip] repo" in result.stdout
    assert "rebase-merge" in result.stdout


def test_skip_when_linked_worktree_has_active_op(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repo = _make_repo(root / "repo")
    wt = tmp_path / "wt"
    _git(["worktree", "add", "-q", str(wt)], repo)
    # Find the worktree admin dir and inject an active-op marker there.
    admin_root = repo / ".git" / "worktrees"
    admin_dirs = list(admin_root.iterdir())
    assert admin_dirs, "expected a linked-worktree admin dir"
    (admin_dirs[0] / "index.lock").write_text("")
    result = _run("--execute", "--root", str(root))
    assert result.returncode == 0
    assert "[skip] repo" in result.stdout
    assert "worktree" in result.stdout


def test_dot_git_file_worktree_is_not_a_clone(tmp_path: Path) -> None:
    """A dir whose .git is a FILE (a linked worktree) is not gc'd directly."""
    root = tmp_path / "root"
    root.mkdir()
    main = _make_repo(root / "main")
    # A linked worktree placed under the root has a .git FILE, not a dir.
    _git(["worktree", "add", "-q", str(root / "linked")], main)
    assert (root / "linked" / ".git").is_file()
    result = _run("--root", str(root))
    assert result.returncode == 0
    assert "[would gc] main" in result.stdout
    assert "[would gc] linked" not in result.stdout
    assert "1 clone(s)" in result.stdout
