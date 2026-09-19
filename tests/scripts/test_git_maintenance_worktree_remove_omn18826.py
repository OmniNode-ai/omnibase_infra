# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Worktree-removal reporting in `scripts/git-maintenance.sh` (OMN-18826).

`git worktree remove` without `--force` is not atomic. It deletes the admin
directory under `<clone>/.git/worktrees/<id>` BEFORE it walks the working tree,
and the walk continues past every unlink failure. A failure partway through
therefore leaves a destroyed tree with no admin directory, and a later removal
over that debris fails with "contains modified or untracked files" -- a message
about the tree's contents, pointing at the wrong cause entirely. That is how 148
worktree directories were damaged on 2026-09-19 and then reported as a mystery.

Every test here drives the REAL script against a throwaway registry and asserts
on its stdout, because the defect being fixed was a reporting defect: the script
discarded git's stderr and printed a hardcoded guess in its place.

`test_control_*` are the positive controls. Without them a script that reported
every removal as a failure, or that never removed anything, would satisfy the
failure assertions here.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "git-maintenance.sh"


def _env(registry: Path, worktrees: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env["GIT_AUTHOR_NAME"] = "OMN-18826 Test"
    env["GIT_AUTHOR_EMAIL"] = "omn18826@example.invalid"
    env["GIT_COMMITTER_NAME"] = env["GIT_AUTHOR_NAME"]
    env["GIT_COMMITTER_EMAIL"] = env["GIT_AUTHOR_EMAIL"]
    env["OMNI_HOME"] = str(registry)
    env["WORKTREE_ROOT"] = str(worktrees)
    env.pop("GIT_PREFIX", None)
    # git exports the repo-scoping variables into every hook process and they
    # OVERRIDE both `cwd=` and `git -C`, so a fixture that misses one rewrites
    # the real worktree this test runs from (OMN-14891 / OMN-18434).
    return scrub_git_location_env(env)


def _git(
    *args: str, cwd: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=scrub_git_location_env(env),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"git {args}: {result.stderr}"
    return result


class World:
    """A registry with one clone and a worktrees root, ready for worktrees.

    The clone has no remote on purpose: the script's branch phase fetches first
    and skips a repo whose fetch fails, so phase 1 is inert here and every
    assertion below is about the worktree phase.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.registry = tmp_path / "omni_home"
        self.worktrees = tmp_path / "omni_worktrees"
        self.registry.mkdir()
        self.worktrees.mkdir()
        self.env = _env(self.registry, self.worktrees)
        self.clone = self.registry / "fixture_repo"
        self.clone.mkdir()
        _git("init", "-q", "-b", "main", cwd=self.clone, env=self.env)
        keep = self.clone / "keep"
        keep.mkdir()
        (keep / "tracked.txt").write_text("tracked\n", encoding="utf-8")
        (self.clone / "root.txt").write_text("root\n", encoding="utf-8")
        _git("add", "-A", cwd=self.clone, env=self.env)
        _git("commit", "-q", "-m", "seed", cwd=self.clone, env=self.env)

    def add_worktree(self, ticket: str, name: str = "fixture_repo") -> Path:
        path = self.worktrees / ticket / name
        _git(
            "worktree",
            "add",
            "-q",
            str(path),
            "-b",
            f"wt/{ticket}/{name}",
            "HEAD",
            cwd=self.clone,
            env=self.env,
        )
        return path

    def admin_dir(self, worktree: Path) -> Path:
        pointer = (worktree / ".git").read_text(encoding="utf-8")
        line = next(x for x in pointer.splitlines() if x.startswith("gitdir: "))
        return Path(line.removeprefix("gitdir: ").strip())

    def run(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT), "--execute", "--prune-worktrees"],
            env=scrub_git_location_env(self.env),
            capture_output=True,
            text=True,
            check=False,
        )


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


def test_failed_removal_prints_git_stderr_verbatim_and_classifies_half_removed(
    world: World,
) -> None:
    """The RED test: a clean worktree holding one unwritable subdirectory.

    git passes its dirty-tree check, deletes the admin directory, then fails to
    unlink inside the unwritable directory. The script must print git's own
    error line including its errno, and must classify the survivor as debris.
    """
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions, so the unlink never fails")

    worktree = world.add_worktree("OMN-FAIL")
    admin = world.admin_dir(worktree)
    assert admin.is_dir()
    (worktree / "keep").chmod(0o500)
    try:
        result = world.run()
        out = result.stdout

        # git's own words, not the script's guess.
        assert "[remove-failed]" in out, out
        assert "failed to delete" in out, out
        assert "Permission denied" in out, out

        # The classification is read back off the tree, and it names the tree
        # as debris so the next pass does not mistake it for peer work.
        assert "classification: half-removed: admin dir deleted," in out, out
        assert "files remain" in out, out
        assert "this tree is debris, not peer work" in out, out

        # The reason the script used to print was invented. It is gone.
        assert "may have untracked files" not in out, out

        # A failed removal is no longer announced as a removal.
        assert "[incomplete] OMN-FAIL" in out, out
        assert "[removed] OMN-FAIL" not in out, out

        # And the classification matches reality: git really did delete the
        # admin directory while leaving the tree on disk.
        assert not admin.exists(), "fixture did not reproduce the half-removal"
        assert worktree.exists()
    finally:
        (worktree / "keep").chmod(0o700)


def test_failed_removal_records_the_head_oid_before_destroying_the_tree(
    world: World,
) -> None:
    """A half-removed tree has no index and no admin directory, so the only way
    to reconstruct it is an oid captured before the removal ran."""
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions, so the unlink never fails")

    worktree = world.add_worktree("OMN-OID")
    head = _git("rev-parse", "HEAD", cwd=worktree, env=world.env).stdout.strip()
    (worktree / "keep").chmod(0o500)
    try:
        out = world.run().stdout
        assert f"head {head}" in out, out
        assert f"reconstruct from head {head}" in out, out
    finally:
        (worktree / "keep").chmod(0o700)


def test_debris_tree_is_named_as_debris_not_as_a_lane_with_uncommitted_changes(
    world: World,
) -> None:
    """The second-order defect: a tree left behind by an earlier failed removal.

    Its signature is a `.git` pointer whose target is gone. `git status` cannot
    report it -- with no admin directory status fails and prints nothing, so the
    tree reads as clean -- and a removal over it fails talking about the tree's
    contents.
    """
    worktree = world.add_worktree("OMN-DEBRIS")
    admin = world.admin_dir(worktree)
    subprocess.run(["rm", "-rf", str(admin)], check=True)
    (worktree / "root.txt").unlink()

    out = world.run().stdout
    assert "[debris]" in out, out
    assert ".git points at a missing admin dir" in out, out
    assert str(admin) in out, out
    assert "half-removed tree, not peer work" in out, out
    assert "[skip] OMN-DEBRIS (half-removed worktree present)" in out, out
    assert "has uncommitted changes" not in out, out
    assert worktree.exists(), "debris must be reported, not deleted"


def test_control_clean_worktree_is_removed_and_reported_as_removed(
    world: World,
) -> None:
    """Positive control for every assertion above: the ordinary path still works
    and still says so."""
    worktree = world.add_worktree("OMN-OK")
    ticket_dir = worktree.parent

    result = world.run()
    out = result.stdout

    assert result.returncode == 0, result.stderr
    assert f"[removed] {worktree}" in out, out
    assert "[removed] OMN-OK" in out, out
    assert "[remove-failed]" not in out, out
    assert "classification:" not in out, out
    assert "[debris]" not in out, out
    assert not ticket_dir.exists(), "a complete removal takes the ticket dir with it"


def test_control_clean_registry_reports_no_debris(world: World) -> None:
    """AC3's positive control: the debris scan returns zero on a healthy tree,
    so a zero finding means something."""
    world.add_worktree("OMN-HEALTHY-A")
    world.add_worktree("OMN-HEALTHY-B")
    out = world.run().stdout
    assert "[debris]" not in out, out
    assert "missing admin dir" not in out, out


def test_removal_does_not_suppress_stderr_in_source(world: World) -> None:
    """AC2's falsifier, pinned mechanically: reintroducing the suppression is a
    red test rather than something a reviewer has to notice."""
    lines = [
        line
        for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if "worktree remove" in line and not line.lstrip().startswith("#")
    ]
    assert lines, "the removal call disappeared; this test no longer guards anything"
    for line in lines:
        assert "2>/dev/null" not in line, line
        assert "--force" not in line, line
