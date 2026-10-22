# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""refresh_dev_lane.sh refuses a --ref not on dev, before the lane lock (OMN-21272).

On 2026-10-21 an out-of-band ``refresh_dev_lane.sh --ref 30baa7158ea2`` -- a
feature-branch merge commit not on dev -- held the dev compose lane lock from
10:51:31Z. The deploy agent's job for the next dev merge ended
``lane_lock_contended`` and the post-merge guard wrote a FAIL about a deploy
that never happened. The script drives only the dev compose project, so it now
refuses, with exit 64 and before it takes the lane lock, a ``--ref`` whose
commit is not contained in ``origin/dev``.

Each case drives the real script against a temp git repository standing in for
the ambient omnibase_infra clone (its ``origin`` is a local bare repository, so
the fetch never leaves the machine) and a ``docker`` shim that never reaches a
daemon. The lane lock is held by THIS test process with ``--lock-timeout 0``,
so a script that reached the lock would exit 2 naming the holder; a refusal
exits 64 without it. An accepted ref goes on to the compose-env preflight.

Falsifier (OMN-21272): ``-k refresh_dev_lane_refuses_ref_not_on_dev``.
"""

from __future__ import annotations

import fcntl
import os
import stat
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from tests.helpers.lane_census_fixtures import NEUTRAL_CONFIG_ROOT, NEUTRAL_HOSTNAME

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "runtime_build" / "refresh_dev_lane.sh"
_LOCK_PY = _REPO / "scripts" / "runtime_build" / "lane_lock.py"
_COMPOSE_PROJECT = "omnibase-infra"

_EXIT_USAGE = 64

_DOCKER_SHIM = """#!/bin/sh
echo "docker $*" >> "$SHIM_CALLS"
exit 0
"""


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "init.defaultBranch=dev", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
        env={
            **scrub_git_location_env(os.environ),
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
        },
    ).stdout.strip()


class _Lane:
    """An OMNI_HOME whose omnibase_infra clone tracks a local bare ``origin``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.workspace = root / "workspace"
        self.clone = self.workspace / "omnibase_infra"
        origin = root / "origin.git"
        seed = root / "seed"
        seed.mkdir()
        _git(seed, "init", "-q")
        (seed / "f").write_text("1\n")
        _git(seed, "add", "f")
        _git(seed, "commit", "-q", "-m", "base")
        self.dev_parent = _git(seed, "rev-parse", "HEAD")
        (seed / "f").write_text("2\n")
        _git(seed, "commit", "-q", "-am", "dev tip")
        self.dev_tip = _git(seed, "rev-parse", "HEAD")
        _git(seed, "checkout", "-q", "-b", "feature", self.dev_parent)
        (seed / "g").write_text("feature\n")
        _git(seed, "add", "g")
        _git(seed, "commit", "-q", "-m", "feature")
        self.feature = _git(seed, "rev-parse", "HEAD")
        _git(root, "clone", "-q", "--bare", str(seed), str(origin))
        self.workspace.mkdir()
        _git(self.workspace, "clone", "-q", str(origin), str(self.clone))
        # A commit the clone first learns of by the script's own fetch.
        (seed / "g").write_text("later\n")
        _git(seed, "commit", "-q", "-am", "feature later")
        self.feature_later = _git(seed, "rev-parse", "HEAD")
        _git(seed, "push", "-q", str(origin), "feature")

        bin_dir = root / "bin"
        bin_dir.mkdir()
        shim = bin_dir / "docker"
        shim.write_text(_DOCKER_SHIM)
        shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        # Resolved before the lock; never run in these cases.
        deploy = bin_dir / "onex-runtime-deploy"
        deploy.write_text(_DOCKER_SHIM)
        deploy.chmod(deploy.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        self.calls = root / "calls.log"
        self.calls.write_text("")
        self.lock_dir = root / "locks"

        # Scrubbed: under a git hook GIT_DIR would point the script's own git
        # calls at this repository instead of the temp clone.
        env = scrub_git_location_env(os.environ)
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
        env["HOME"] = str(root)
        env["OMNI_HOME"] = str(self.workspace)
        env["ONEX_WORKSPACE_CONFIG_ROOT"] = str(NEUTRAL_CONFIG_ROOT)
        env["LANE_CENSUS_HOST"] = NEUTRAL_HOSTNAME
        env["LANE_CENSUS_PYTHON"] = sys.executable
        env["LANE_CENSUS_DOCKER_SOCKET"] = "/nonexistent/refresh-dev-lane-test.sock"
        env["ONEX_LANE_LOCK_DIR"] = str(self.lock_dir)
        env["SHIM_CALLS"] = str(self.calls)
        for name in (
            "LANE_MANIFEST",
            "OMNIBASE_INFRA_COMPOSE_PROJECT",
            "ONEX_LANE_LOCK_HELD",
            "LANE_MEMORY_BROKER_CONTAINER",
        ):
            env.pop(name, None)
        self.env = env

    def run(self, ref: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(_SCRIPT), "--ref", ref, "--execute", "--lock-timeout", "0"],
            capture_output=True,
            text=True,
            env=self.env,
            cwd=str(_REPO),
            timeout=120,
            check=False,
        )


@pytest.fixture
def lane(tmp_path: Path) -> Iterator[_Lane]:
    built = _Lane(tmp_path)
    lock_file = Path(
        subprocess.run(
            [
                sys.executable,
                str(_LOCK_PY),
                "path",
                "--compose-project",
                _COMPOSE_PROJECT,
            ],
            check=True,
            capture_output=True,
            text=True,
            env=built.env,
        ).stdout.strip()
    )
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("w") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield built


def _refused(proc: subprocess.CompletedProcess[str], ref: str, sha: str) -> None:
    assert proc.returncode == _EXIT_USAGE, proc.stderr
    assert "REF_NOT_ON_DEV" in proc.stderr
    assert ref in proc.stderr
    assert sha in proc.stderr
    # Refused before the lane lock: a script that reached it would have exited
    # 2 against the lock this test holds.
    assert "held by another process" not in proc.stderr


def test_refresh_dev_lane_refuses_ref_not_on_dev_feature_sha(lane: _Lane) -> None:
    _refused(lane.run(lane.feature), lane.feature, lane.feature)
    assert lane.calls.read_text() == ""


def test_refresh_dev_lane_refuses_ref_not_on_dev_short_sha_learnt_by_fetch(
    lane: _Lane,
) -> None:
    # The incident's shape: an abbreviated sha the clone only has after fetching.
    short = lane.feature_later[:12]
    _refused(lane.run(short), short, lane.feature_later)


def test_refresh_dev_lane_refuses_ref_not_on_dev_branch_name(lane: _Lane) -> None:
    _refused(lane.run("origin/feature"), "origin/feature", lane.feature_later)


def test_refresh_dev_lane_refuses_ref_not_on_dev_unresolvable(lane: _Lane) -> None:
    proc = lane.run("0" * 40)
    assert proc.returncode == _EXIT_USAGE, proc.stderr
    assert "REF_NOT_ON_DEV" in proc.stderr
    assert "held by another process" not in proc.stderr


@pytest.mark.parametrize("which", ["origin/dev", "dev_tip", "dev_parent"])
def test_refresh_dev_lane_refuses_ref_not_on_dev_accepts_refs_on_dev(
    lane: _Lane, which: str
) -> None:
    """Positive control: origin/dev and a sha on dev pass the guard.

    The run goes on to the next precondition, the compose-env preflight, which
    this sandbox has no compose files for; reaching it is what passing means.
    """
    ref = which if which == "origin/dev" else getattr(lane, which)
    proc = lane.run(ref)
    assert "REF_NOT_ON_DEV" not in proc.stderr
    assert "is contained in origin/dev" in proc.stderr
    assert "Required compose env preflight" in proc.stderr


def test_refresh_dev_lane_refuses_ref_not_on_dev_before_any_lock_call() -> None:
    """The guard sits before every lane_lock_acquire call in the script."""
    lines = _SCRIPT.read_text().splitlines()
    guard = next(i for i, line in enumerate(lines) if "REF_NOT_ON_DEV" in line)
    locks = [
        i
        for i, line in enumerate(lines)
        if "lane_lock_acquire " in line and not line.lstrip().startswith("#")
    ]
    assert locks, "the script no longer takes the lane lock"
    assert guard < min(locks)
