# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-20266: the clone reconciler keeps ``refs/remotes/origin/HEAD`` current.

``reconcile`` puts every canonical clone on ``dev`` with ``git checkout --force
-B dev <sha>``, but the ``origin/HEAD`` symref is a local copy nothing refreshed:
on h105 it still named ``main`` six months after the remote default moved to
``dev``. The remote-lane clone parity check reads that symref as the default
branch, so every omnimarket lane placed on h105 was refused with ``checked out
dev, not the default branch main`` while the clone sat on the right branch.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SOURCE_REF = REPO_ROOT / "scripts" / "runtime_build" / "deploy_source_ref.py"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={**scrub_git_location_env(os.environ), "GIT_TERMINAL_PROMPT": "0"},
    )
    return result.stdout.strip()


def _commit(repo: Path, name: str) -> None:
    (repo / f"{name}.txt").write_text(f"{name}\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", name)


def _origin_whose_default_moved_to_dev(tmp_path: Path) -> tuple[Path, Path]:
    """An origin that was cloned while ``main`` was its default, then moved to ``dev``."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "t@t.t")
    _git(origin, "config", "user.name", "t")
    _commit(origin, "init")
    _git(origin, "branch", "dev")
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(origin), str(clone)],
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    )
    _git(origin, "checkout", "-q", "dev")
    _commit(origin, "dev-advance")
    return origin, clone


def _reconcile(clone: Path, receipt: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python3",
            str(DEPLOY_SOURCE_REF),
            "reconcile",
            "--branch",
            "dev",
            "--output",
            str(receipt),
            "--repo",
            f"clone={clone}",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )


def test_reconcile_refreshes_a_stale_origin_head_symref(tmp_path: Path) -> None:
    origin, clone = _origin_whose_default_moved_to_dev(tmp_path)
    assert _git(clone, "symbolic-ref", "refs/remotes/origin/HEAD") == (
        "refs/remotes/origin/main"
    )

    done = _reconcile(clone, tmp_path / "receipt.json")

    assert done.returncode == 0, done.stderr
    assert _git(clone, "symbolic-ref", "--short", "HEAD") == "dev"
    assert _git(clone, "rev-parse", "HEAD") == _git(origin, "rev-parse", "dev")
    assert _git(clone, "symbolic-ref", "refs/remotes/origin/HEAD") == (
        "refs/remotes/origin/dev"
    )


def test_reconcile_still_syncs_when_the_symref_refresh_cannot_run(
    tmp_path: Path,
) -> None:
    """A failed refresh is not a failed sync: the clone still lands on the tip."""
    origin, clone = _origin_whose_default_moved_to_dev(tmp_path)
    # ``set-head --auto`` asks the remote for its HEAD; a remote with a detached
    # HEAD at a commit that no branch names leaves the refresh nothing to set.
    # Detaching at dev itself is insufficient: Git can still infer dev.
    _git(origin, "checkout", "-q", "--detach")
    _commit(origin, "detached-only")

    done = _reconcile(clone, tmp_path / "receipt.json")

    assert done.returncode == 0, done.stderr
    assert "origin/HEAD not refreshed" in done.stderr
    assert _git(clone, "rev-parse", "HEAD") == _git(origin, "rev-parse", "dev")
