# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the worktree-collision fix in refresh_stability_lane.sh (OMN-14889).

Real bug found by the release-train canary: refresh_stability_lane.sh's step 3
("Refresh the omnibase_infra ambient clone itself to --ref") used to do

    git -C "${INFRA_CLONE}" checkout dev
    git -C "${INFRA_CLONE}" reset --hard "${REF}"

which HARD-FAILS on .201 whenever another worktree on the same host (e.g.
deploy-agent's runtime-sync-worktrees/OMN-12618) already has the local `dev`
branch checked out -- git refuses to check the same branch out into two
worktrees at once ("already checked out at <path>").

The fix resolves --ref to a commit SHA and checks it out DETACHED
(`git checkout --force --detach <sha>`), which carries no branch identity and
therefore can never collide with a sibling worktree, regardless of which
branch that worktree holds.

These tests reproduce the exact collision with real git worktrees (not a
simulation): RED proves `git checkout dev` fails when `dev` is busy elsewhere;
GREEN proves the detached-by-SHA approach succeeds under the identical
condition. A third test statically guards the actual script source so the
collision-prone pattern cannot silently come back.

Per reference_git_env_vars_override_c_and_cwd: strip GIT_DIR/GIT_INDEX_FILE/
GIT_WORK_TREE from the subprocess env so an inherited pre-push hook export
cannot redirect these git operations onto the real worktree.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "refresh_stability_lane.sh"

_HERMETIC_ENV = {
    k: v
    for k, v in os.environ.items()
    if k not in {"GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE"}
}


def _git(
    args: list[str], cwd: Path, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=_HERMETIC_ENV,
        capture_output=True,
        text=True,
        check=check,
    )


def _build_busy_dev_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    """Build origin + a main clone whose local `dev` is checked out in a
    SIBLING worktree, reproducing the .201 deploy-agent collision.

    Returns (main_clone, sibling_worktree, dev_sha).
    """
    origin = tmp_path / "origin.git"
    _git(["init", "--bare", str(origin)], cwd=tmp_path)
    # Bare repos default HEAD to whatever init.defaultBranch happens to be
    # (main/master), which never gets a ref pushed to it here -- point HEAD at
    # `dev` explicitly so `git clone` checks out a local `dev` branch instead
    # of leaving the clone with only a dangling/unresolvable HEAD.
    _git(["symbolic-ref", "HEAD", "refs/heads/dev"], cwd=origin)

    seed = tmp_path / "seed"
    _git(["init", "-b", "dev", str(seed)], cwd=tmp_path)
    (seed / "f.txt").write_text("one\n")
    _git(["add", "f.txt"], cwd=seed)
    _git(
        ["-c", "user.email=t@t.com", "-c", "user.name=t", "commit", "-m", "c1"],
        cwd=seed,
    )
    (seed / "f.txt").write_text("two\n")
    _git(["add", "f.txt"], cwd=seed)
    _git(
        ["-c", "user.email=t@t.com", "-c", "user.name=t", "commit", "-m", "c2"],
        cwd=seed,
    )
    _git(["remote", "add", "origin", str(origin)], cwd=seed)
    _git(["push", "origin", "dev"], cwd=seed)

    main_clone = tmp_path / "main_clone"
    _git(["clone", str(origin), str(main_clone)], cwd=tmp_path)
    dev_sha = _git(["rev-parse", "dev"], cwd=main_clone).stdout.strip()

    # Free `dev` in the main clone (a real refresh run leaves this clone
    # detached/mid-refresh between invocations -- it is never guaranteed to
    # still be sitting on `dev`), then check `dev` out in a SIBLING worktree,
    # mirroring deploy-agent's runtime-sync-worktrees/OMN-12618.
    _git(["checkout", "--detach", "HEAD"], cwd=main_clone)
    sibling = tmp_path / "runtime-sync-worktrees" / "OMN-12618"
    sibling.parent.mkdir(parents=True, exist_ok=True)
    _git(["worktree", "add", str(sibling), "dev"], cwd=main_clone)

    return main_clone, sibling, dev_sha


def test_branch_checkout_fails_when_dev_busy_in_sibling_worktree(
    tmp_path: Path,
) -> None:
    """RED: reproduces the exact live .201 failure mode.

    `git checkout dev` in the ambient clone must fail while a sibling
    worktree already holds `dev` checked out -- this is the bug the canary
    found, not a hypothesized one.
    """
    main_clone, sibling, _dev_sha = _build_busy_dev_fixture(tmp_path)
    assert sibling.is_dir()

    result = _git(["checkout", "dev"], cwd=main_clone, check=False)

    assert result.returncode != 0, (
        "expected `git checkout dev` to fail while `dev` is checked out in a "
        f"sibling worktree; got rc=0 stdout={result.stdout!r}"
    )
    combined = (result.stdout + result.stderr).lower()
    assert "already" in combined or "already checked out" in combined, (
        f"unexpected git error message, no longer matches the known collision "
        f"signature: {result.stdout!r} {result.stderr!r}"
    )


def test_detached_checkout_by_sha_succeeds_when_dev_busy_in_sibling_worktree(
    tmp_path: Path,
) -> None:
    """GREEN: the fix -- resolve --ref to a SHA, checkout --force --detach --
    succeeds under the identical busy-`dev` condition that fails above.
    """
    main_clone, sibling, dev_sha = _build_busy_dev_fixture(tmp_path)
    assert sibling.is_dir()

    resolved = _git(["rev-parse", "dev^{commit}"], cwd=main_clone).stdout.strip()
    assert resolved == dev_sha

    result = _git(
        ["checkout", "--force", "--detach", resolved], cwd=main_clone, check=False
    )

    assert result.returncode == 0, (
        f"detached-by-SHA checkout must succeed even while `dev` is busy in a "
        f"sibling worktree: {result.stdout!r} {result.stderr!r}"
    )
    head = _git(["rev-parse", "HEAD"], cwd=main_clone).stdout.strip()
    assert head == dev_sha

    # And the sibling worktree's checkout of `dev` is completely undisturbed.
    sibling_head = _git(["rev-parse", "HEAD"], cwd=sibling).stdout.strip()
    assert sibling_head == dev_sha
    sibling_branch = _git(
        ["rev-parse", "--abbrev-ref", "HEAD"], cwd=sibling
    ).stdout.strip()
    assert sibling_branch == "dev"


def test_script_uses_detached_sha_checkout_not_bare_branch_checkout() -> None:
    """Static regression guard: the collision-prone pattern must not come back.

    Asserts the actual shipped script resolves --ref to a commit before
    checkout and uses `checkout --force --detach`, and no longer contains a
    bare `git checkout dev` / `git checkout "${REF}"` branch-checkout call.
    """
    assert SCRIPT.is_file()
    text = SCRIPT.read_text()

    assert 'checkout "dev"' not in text
    assert "checkout dev" not in text
    assert 'checkout "${REF}"' not in text

    assert "checkout --force --detach" in text
    assert 'rev-parse "${REF}^{commit}"' in text
