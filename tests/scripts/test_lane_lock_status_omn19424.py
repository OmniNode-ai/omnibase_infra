# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""`lane_lock.py status` reports whether a lane lock is held, and by whom [OMN-19424].

The runtime-train pre-merge check has to WAIT while a reconcile holds the target
lane. Until this subcommand the only read was `describe`, which prints a holder
sidecar as prose and says nothing about whether the flock is actually held: a
sidecar outlives a killed holder's lock only until the next `release`, and a lock
outlives a dead shell while a surviving child keeps the descriptor. `status`
answers the question a machine reader asks -- is the lane locked right now -- by
probing the flock itself, and returns the sidecar as JSON.

These tests drive the real `scripts/runtime_build/lane_lock.py` and the real
shell front end `lane_lock.sh`, with a holder in its own process group.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LANE_LOCK_PY = REPO_ROOT / "scripts" / "runtime_build" / "lane_lock.py"
LANE_LOCK_SH = REPO_ROOT / "scripts" / "runtime_build" / "lane_lock.sh"

PROJECT = "omnibase-infra"


def _env(lock_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["ONEX_LANE_LOCK_DIR"] = str(lock_dir)
    env.pop("ONEX_LANE_LOCK_HELD", None)
    return env


def _status(lock_dir: Path, project: str = PROJECT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(LANE_LOCK_PY),
            "status",
            "--compose-project",
            project,
        ],
        env=_env(lock_dir),
        capture_output=True,
        text=True,
        check=False,
    )


def _start_holder(lock_dir: Path, marker: Path) -> subprocess.Popen[str]:
    script = "\n".join(
        [
            "set -euo pipefail",
            f'source "{LANE_LOCK_SH}"',
            f'lane_lock_acquire "{PROJECT}" "stability-reconcile" "origin/dev" 30 "reconcile.sh"',
            f'printf held > "{marker}"',
            "sleep 30",
            "lane_lock_release",
        ]
    )
    return subprocess.Popen(
        ["bash", "-c", script],
        env=_env(lock_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _stop_holder(holder: subprocess.Popen[str]) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(holder.pid), signal.SIGKILL)
    holder.wait(timeout=10)


def _await_held(holder: subprocess.Popen[str], marker: Path) -> None:
    deadline = time.monotonic() + 20
    while not marker.exists():
        assert holder.poll() is None, "holder exited before taking the lock"
        assert time.monotonic() < deadline, "holder never signalled it held the lock"
        time.sleep(0.05)


@pytest.mark.unit
def test_status_of_a_lane_never_locked_reads_free(tmp_path: Path) -> None:
    result = _status(tmp_path / "locks")

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["compose_project"] == PROJECT
    assert report["held"] is False
    assert report["holder"] is None


@pytest.mark.unit
def test_status_of_a_held_lane_names_the_holder(tmp_path: Path) -> None:
    lock_dir = tmp_path / "locks"
    marker = tmp_path / "held"
    holder = _start_holder(lock_dir, marker)
    try:
        _await_held(holder, marker)
        result = _status(lock_dir)
    finally:
        _stop_holder(holder)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["held"] is True
    assert report["holder"]["lane"] == "stability-reconcile"
    assert report["holder"]["ref"] == "origin/dev"
    assert "reconcile.sh" in report["holder"]["argv"]
    assert report["holder"]["pid"] == holder.pid
    assert report["holder_alive"] is True


@pytest.mark.unit
def test_status_does_not_disturb_a_held_lane_or_leave_a_free_one_locked(
    tmp_path: Path,
) -> None:
    """The probe takes the flock on its own open file description and drops it.
    A second status must still read held while the holder lives, and a lane
    that was free must still be acquirable after the probe."""
    lock_dir = tmp_path / "locks"
    marker = tmp_path / "held"

    free = _status(lock_dir)
    assert json.loads(free.stdout)["held"] is False
    acquire = subprocess.run(
        [
            "bash",
            "-c",
            "\n".join(
                [
                    "set -euo pipefail",
                    f'source "{LANE_LOCK_SH}"',
                    f'lane_lock_acquire "{PROJECT}" "dev" "origin/dev" 2 "after-probe.sh"',
                    "lane_lock_release",
                ]
            ),
        ],
        env=_env(lock_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    assert acquire.returncode == 0, (
        f"a probe left the lane locked: stderr={acquire.stderr!r}"
    )

    holder = _start_holder(lock_dir, marker)
    try:
        _await_held(holder, marker)
        first = json.loads(_status(lock_dir).stdout)
        second = json.loads(_status(lock_dir).stdout)
    finally:
        _stop_holder(holder)

    assert first["held"] is True
    assert second["held"] is True


@pytest.mark.unit
def test_status_reads_free_once_the_holder_is_killed(tmp_path: Path) -> None:
    """A killed holder leaves its sidecar behind but not the lock; the lane reads
    free, and the leftover sidecar is reported as such rather than as a holder."""
    lock_dir = tmp_path / "locks"
    marker = tmp_path / "held"
    holder = _start_holder(lock_dir, marker)
    _await_held(holder, marker)
    _stop_holder(holder)

    report = json.loads(_status(lock_dir).stdout)

    assert report["held"] is False
    assert report["holder_alive"] is False


@pytest.mark.unit
def test_status_rejects_a_traversing_project_name(tmp_path: Path) -> None:
    result = _status(tmp_path / "locks", project="../../etc/passwd")

    assert result.returncode == 3
    assert "invalid compose project name" in result.stderr
