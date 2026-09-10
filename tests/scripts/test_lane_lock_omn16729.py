# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Per-compose-project lane lock [OMN-16729].

Defect this closes, measured on the .201 dev lane 2026-09-08T13:51Z (third
recorded occurrence of the class, after the 2026-09-05/06 prophylactic ones):
`scripts/runtime_build/refresh_dev_lane.sh`, `refresh_stability_lane.sh` and
`scripts/deploy-runtime.sh` all mutate ONE lane, and none of them took a
host-level lock over the whole critical section. A lane's own refresh completed
at 13:44:04Z and, at 13:51:14Z, WHILE its post-deploy readback was running, a
second sanctioned refresh of the same compose project recreated all four core
containers. The first lane's readback died mid-loop with
"container a8612839b451 is not running", and the lane was left carrying a
feature branch rather than the ref either lane intended.

`deploy-runtime.sh`'s pre-existing `.deploy.lock` did not and could not prevent
this: it is host-WIDE (unrelated lanes block each other) and it is scoped to
that one script, so it is already released while the wrapper is still capturing
pre-state, health-gating and reading back.

These tests drive the real `scripts/runtime_build/lane_lock.py` and the real
shell front end `lane_lock.sh` -- no surrogate. The contention test really does
run two shells against one lock file.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LANE_LOCK_PY = REPO_ROOT / "scripts" / "runtime_build" / "lane_lock.py"
LANE_LOCK_SH = REPO_ROOT / "scripts" / "runtime_build" / "lane_lock.sh"

PROJECT = "omnibase-infra"


def _env(lock_dir: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env["ONEX_LANE_LOCK_DIR"] = str(lock_dir)
    env.pop("ONEX_LANE_LOCK_HELD", None)
    env.update(extra)
    return env


def _holder_script(lock_dir: Path, hold_seconds: float, marker: Path) -> str:
    """A shell that takes the lane lock, signals it holds it, then sleeps."""
    return "\n".join(
        [
            "set -euo pipefail",
            f'source "{LANE_LOCK_SH}"',
            f'lane_lock_acquire "{PROJECT}" "dev" "origin/dev" 30 "holder.sh"',
            f'printf held > "{marker}"',
            f"sleep {hold_seconds}",
            "lane_lock_release",
        ]
    )


def _start_holder(
    lock_dir: Path, hold_seconds: float, marker: Path
) -> subprocess.Popen[str]:
    """Start a lock holder in its OWN process group.

    The holder's `sleep` is a grandchild of this test, and killing only the
    shell would orphan it (OMN-16995: that is how test_heavy_lock.py leaked a
    busy loop per run until 19 of them held 18.6 of `.200`'s 24 cores). Every
    Popen here is group-spawned and torn down with os.killpg.
    """
    return subprocess.Popen(
        ["bash", "-c", _holder_script(lock_dir, hold_seconds, marker)],
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
def test_second_acquirer_times_out_and_names_the_holder(tmp_path: Path) -> None:
    """AC1: a second acquisition of the SAME lane refuses within its bounded
    wait, exits non-zero, and names the holding pid/lane/argv rather than
    hanging anonymously or stealing the lock."""
    lock_dir = tmp_path / "locks"
    marker = tmp_path / "held"

    holder = _start_holder(lock_dir, 20, marker)
    try:
        deadline = time.monotonic() + 20
        while not marker.exists():
            assert holder.poll() is None, "holder exited before taking the lock"
            assert time.monotonic() < deadline, (
                "holder never signalled it held the lock"
            )
            time.sleep(0.05)

        started = time.monotonic()
        second = subprocess.run(
            [
                "bash",
                "-c",
                "\n".join(
                    [
                        "set -euo pipefail",
                        f'source "{LANE_LOCK_SH}"',
                        f'lane_lock_acquire "{PROJECT}" "dev" "origin/dev" 2 "second.sh"',
                    ]
                ),
            ],
            env=_env(lock_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        waited = time.monotonic() - started
    finally:
        _stop_holder(holder)

    assert second.returncode == 2, (
        "a contended lane acquisition must exit 2, not proceed. "
        f"stdout={second.stdout!r} stderr={second.stderr!r}"
    )
    assert 1.5 <= waited < 15, (
        f"the bounded wait must be honoured (asked for 2s, waited {waited:.1f}s)"
    )
    assert "CONTENDED" in second.stderr
    assert str(holder.pid) in second.stderr, (
        "the refusal must NAME the holding pid so a wedged holder is diagnosable "
        f"-- got: {second.stderr!r}"
    )
    assert "holder.sh" in second.stderr, "the refusal must name the holder's command"
    assert "NEVER stolen" in second.stderr


@pytest.mark.unit
def test_nested_reentry_succeeds_without_deadlock(tmp_path: Path) -> None:
    """AC2: a nested call (refresh_*_lane.sh -> deploy-runtime.sh) inherits the
    token and re-enters immediately instead of blocking on its own parent."""
    lock_dir = tmp_path / "locks"
    inner = "\n".join(
        [
            "set -euo pipefail",
            f'source "{LANE_LOCK_SH}"',
            # A 1-second timeout: if re-entry were NOT honoured this would fail
            # with exit 2 rather than block long enough to be mistaken for a pass.
            f'lane_lock_acquire "{PROJECT}" "dev" "origin/dev" 1 "inner.sh"',
            'printf "INNER_OK\\n"',
        ]
    )
    inner_path = tmp_path / "inner.sh"
    inner_path.write_text(inner + "\n", encoding="utf-8")
    outer = "\n".join(
        [
            "set -euo pipefail",
            f'source "{LANE_LOCK_SH}"',
            f'lane_lock_acquire "{PROJECT}" "dev" "origin/dev" 10 "outer.sh"',
            'printf "OUTER_OK\\n"',
            f'bash "{inner_path}"',
            "lane_lock_release",
        ]
    )
    result = subprocess.run(
        ["bash", "-c", outer],
        env=_env(lock_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OUTER_OK" in result.stdout
    assert "INNER_OK" in result.stdout, (
        "the nested call must not deadlock on its parent"
    )
    assert "re-entrant" in result.stderr


@pytest.mark.unit
def test_different_lanes_do_not_block_each_other(tmp_path: Path) -> None:
    """AC3: the lock is per compose project. A stability refresh must not be
    serialised behind a dev refresh -- the host-wide `.deploy.lock` this
    supplements did exactly that."""
    lock_dir = tmp_path / "locks"
    marker = tmp_path / "held"
    holder = _start_holder(lock_dir, 20, marker)
    try:
        _await_held(holder, marker)

        other = subprocess.run(
            [
                "bash",
                "-c",
                "\n".join(
                    [
                        "set -euo pipefail",
                        f'source "{LANE_LOCK_SH}"',
                        'lane_lock_acquire "omnibase-infra-stability-test" '
                        '"stability-test" "origin/dev" 2 "other.sh"',
                        'printf "OTHER_OK\\n"',
                        "lane_lock_release",
                    ]
                ),
            ],
            env=_env(lock_dir),
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        _stop_holder(holder)

    assert other.returncode == 0, f"stderr={other.stderr!r}"
    assert "OTHER_OK" in other.stdout


@pytest.mark.unit
def test_lock_is_released_when_the_holder_tree_is_killed(tmp_path: Path) -> None:
    """AC4: a killed holder never leaves the lane locked, which is why nothing
    here needs -- or has -- a lock-stealing branch. flock is released by the
    kernel on the last close of the open file description, so no stale-lock
    recovery (and therefore no wrong-guess mutation of a live lane) is possible."""
    lock_dir = tmp_path / "locks"
    marker = tmp_path / "held"
    holder = _start_holder(lock_dir, 60, marker)
    _await_held(holder, marker)
    # Kill the whole session: the descriptor is shared with children, so
    # releasing the lane means the holder TREE is gone, not just its shell.
    _stop_holder(holder)

    after = subprocess.run(
        [
            "bash",
            "-c",
            "\n".join(
                [
                    "set -euo pipefail",
                    f'source "{LANE_LOCK_SH}"',
                    f'lane_lock_acquire "{PROJECT}" "dev" "origin/dev" 5 "after.sh"',
                    'printf "AFTER_OK\\n"',
                    "lane_lock_release",
                ]
            ),
        ],
        env=_env(lock_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    assert after.returncode == 0, f"stderr={after.stderr!r}"
    assert "AFTER_OK" in after.stdout


@pytest.mark.unit
def test_surviving_child_keeps_the_lane_locked_and_says_so(tmp_path: Path) -> None:
    """The honest residual, asserted rather than implied. An fcntl lock lives on
    the open file description, which children inherit -- so a surviving child of
    an exited holder keeps the lane locked. That is the safe direction (an
    orphaned `docker compose up` is exactly when a second refresh must not
    start), and it must be DIAGNOSABLE: the refusal names the shell pid and
    reports it NOT RUNNING, which is the signal to look for the live child."""
    lock_dir = tmp_path / "locks"
    marker = tmp_path / "held"
    child_pid_file = tmp_path / "child.pid"
    # The shell takes the lock, forks a long-lived child that inherits the
    # descriptor, records both pids, and exits WITHOUT releasing.
    holder_script = "\n".join(
        [
            "set -euo pipefail",
            f'source "{LANE_LOCK_SH}"',
            f'lane_lock_acquire "{PROJECT}" "dev" "origin/dev" 30 "orphan-holder.sh"',
            # stdout/stderr go to /dev/null so the CHILD does not hold the
            # captured pipes open -- otherwise subprocess.run() below would
            # block until it exits, which is the same wait it is meant to prove.
            "sleep 120 >/dev/null 2>&1 &",
            f'printf "%s" "$!" > "{child_pid_file}"',
            f'printf held > "{marker}"',
        ]
    )
    holder = subprocess.run(
        ["bash", "-c", holder_script],
        env=_env(lock_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    assert holder.returncode == 0, holder.stderr
    assert marker.exists()
    child_pid = int(child_pid_file.read_text(encoding="utf-8"))

    try:
        second = subprocess.run(
            [
                "bash",
                "-c",
                "\n".join(
                    [
                        "set -euo pipefail",
                        f'source "{LANE_LOCK_SH}"',
                        f'lane_lock_acquire "{PROJECT}" "dev" "origin/dev" 2 "second.sh"',
                    ]
                ),
            ],
            env=_env(lock_dir),
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child_pid, signal.SIGKILL)

    assert second.returncode == 2, (
        "a lane whose descriptor is still held by a surviving child must stay "
        f"locked. stderr={second.stderr!r}"
    )
    assert "NOT RUNNING" in second.stderr, (
        "an orphaned holder must be reported as NOT RUNNING so the operator "
        "knows to look for a surviving child rather than a live deploy"
    )
    assert "NEVER stolen" in second.stderr


@pytest.mark.unit
def test_holder_sidecar_records_lane_ref_and_argv(tmp_path: Path) -> None:
    """The sidecar is the whole diagnosis surface for a contended lane -- assert
    it carries lane, compose project, ref and argv, not just a pid."""
    lock_dir = tmp_path / "locks"
    marker = tmp_path / "held"
    holder = _start_holder(lock_dir, 20, marker)
    try:
        _await_held(holder, marker)
        sidecar = json.loads(
            (lock_dir / f"{PROJECT}.lock.holder").read_text(encoding="utf-8")
        )
    finally:
        _stop_holder(holder)

    assert sidecar["compose_project"] == PROJECT
    assert sidecar["lane"] == "dev"
    assert sidecar["ref"] == "origin/dev"
    assert "holder.sh" in sidecar["argv"]
    assert sidecar["pid"] == holder.pid, (
        "the sidecar must name the SHELL that holds the lock, not the short-lived "
        "helper process that flocked the inherited fd"
    )


@pytest.mark.unit
def test_lock_path_rejects_a_traversing_project_name(tmp_path: Path) -> None:
    """A compose project string can never escape the lock directory."""
    result = subprocess.run(
        [
            "python3",
            str(LANE_LOCK_PY),
            "path",
            "--compose-project",
            "../../etc/passwd",
        ],
        env=_env(tmp_path / "locks"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 3
    assert "invalid compose project name" in result.stderr


@pytest.mark.unit
def test_lock_path_is_per_compose_project_under_the_fixed_root() -> None:
    """The operational path is fixed; only tests may relocate it."""
    env = dict(os.environ)
    env.pop("ONEX_LANE_LOCK_DIR", None)
    result = subprocess.run(
        ["python3", str(LANE_LOCK_PY), "path", "--compose-project", PROJECT],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == str(
        Path.home() / ".omnibase" / "state" / "lane-locks" / f"{PROJECT}.lock"
    )


@pytest.mark.unit
def test_every_lane_mutating_script_takes_the_lock() -> None:
    """Regression fence: the three scripts that mutate a lane must all source
    the helper and call lane_lock_acquire. A fourth entry point added later
    without a lock is exactly how this class recurred three times."""
    for rel in (
        "scripts/deploy-runtime.sh",
        "scripts/runtime_build/refresh_dev_lane.sh",
        "scripts/runtime_build/refresh_stability_lane.sh",
    ):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "lane_lock.sh" in text, f"{rel} does not source the lane lock helper"
        assert "lane_lock_acquire" in text, f"{rel} never acquires the lane lock"
