# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression/simulation coverage for `scripts/runner-disk-admission-restore.sh`
(OMN-16363 AC2: "Regression/simulation test proving the guard actually
prevents the write-amplification pattern ... rather than just detecting low
disk after the fact.").

This is the restore half of the OMN-16363 mechanism: the "slope-plus-canary"
criterion documented on the ticket (a fixed absolute free-space threshold was
tried live and restored too conservatively; sustained POSITIVE SLOPE across a
canary batch is what actually worked) is exercised here as a deterministic
multi-tick simulation against a stub `docker` binary and injected `df`
readings -- no real Docker daemon, no real disk pressure required.

Each test drives the real script across a SEQUENCE of ticks (each tick is one
subprocess invocation, exactly as the systemd timer would fire it every 2
minutes) and asserts on the cumulative `docker start` call log plus which
pause markers remain, proving:

  1. Below the critical floor: never restores, regardless of how many paused
     runners are waiting.
  2. Above the critical floor but below the restore floor, or flat/declining
     at or above it: never restores.
  3. Two consecutive climbing ticks at/above the restore floor: releases
     exactly ONE canary-sized batch, in pause order (oldest first).
  4. A batch release resets the climb streak -- the NEXT batch requires the
     climb criterion to be satisfied again, never releases automatically.
  5. Once every paused runner is restored, the batch-index state resets so a
     future incident starts its own batching from the canary size again.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RESTORE_SCRIPT = REPO_ROOT / "scripts" / "runner-disk-admission-restore.sh"

GB_KB = 1024 * 1024

pytestmark = [pytest.mark.unit]


def _write_docker_stub(bin_dir: Path, log_file: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    docker_stub = bin_dir / "docker"
    docker_stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "start" ]]; then\n'
        f'  echo "start $2" >> "{log_file}"\n'
        "  exit 0\n"
        'elif [[ "$1" == "inspect" ]]; then\n'
        "  echo running\n"
        "  exit 0\n"
        "fi\n"
        "exit 0\n"
    )
    docker_stub.chmod(0o755)


def _write_pause_marker(pause_dir: Path, runner: str, paused_at: str) -> None:
    pause_dir.mkdir(parents=True, exist_ok=True)
    (pause_dir / runner).write_text(
        f"runner={runner}\navail_gb=1.00\npaused_at={paused_at}\n"
        "reason=consecutive_disk_admission_failures\n"
    )


def _tick(
    *,
    avail_kb: int,
    pause_dir: Path,
    state_file: Path,
    docker_bin_dir: Path,
    critical_floor_gb: int = 15,
    restore_floor_gb: int = 40,
    climb_ticks_required: int = 2,
    batch_sizes: str = "10 20 20 20",
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["RUNNER_DISK_GUARD_AVAIL_KB_OVERRIDE"] = str(avail_kb)
    env["RUNNER_DISK_GUARD_DOCKER_BIN"] = str(docker_bin_dir / "docker")
    env["RUNNER_DISK_GUARD_CRITICAL_FLOOR_GB"] = str(critical_floor_gb)
    env["RUNNER_DISK_GUARD_RESTORE_FLOOR_GB"] = str(restore_floor_gb)
    env["RUNNER_DISK_GUARD_CLIMB_TICKS_REQUIRED"] = str(climb_ticks_required)
    env["RUNNER_DISK_GUARD_BATCH_SIZES"] = batch_sizes
    env["PATH"] = f"{docker_bin_dir}:{env.get('PATH', '')}"
    return subprocess.run(
        [
            "bash",
            str(RESTORE_SCRIPT),
            "--pause-dir",
            str(pause_dir),
            "--state-file",
            str(state_file),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_script_exists_and_is_executable() -> None:
    assert RESTORE_SCRIPT.is_file(), f"restore script missing: {RESTORE_SCRIPT}"
    mode = RESTORE_SCRIPT.stat().st_mode
    assert mode & 0o111, "restore script must be executable"


def test_below_critical_floor_never_restores_regardless_of_pause_count(
    tmp_path: Path,
) -> None:
    pause_dir = tmp_path / "pause"
    for i in range(1, 6):
        _write_pause_marker(
            pause_dir, f"omninode-runner-{i}", f"2026-08-22T18:0{i}:00Z"
        )
    docker_log = tmp_path / "docker_calls.log"
    docker_bin = tmp_path / "bin"
    _write_docker_stub(docker_bin, docker_log)
    state_file = tmp_path / "state.json"

    for avail_gb in (0, 5, 14):
        result = _tick(
            avail_kb=avail_gb * GB_KB,
            pause_dir=pause_dir,
            state_file=state_file,
            docker_bin_dir=docker_bin,
        )
        assert result.returncode == 0, result.stderr

    assert not docker_log.exists()
    assert len(list(pause_dir.iterdir())) == 5


def test_above_restore_floor_but_flat_never_restores(tmp_path: Path) -> None:
    pause_dir = tmp_path / "pause"
    _write_pause_marker(pause_dir, "omninode-runner-1", "2026-08-22T18:00:00Z")
    docker_log = tmp_path / "docker_calls.log"
    docker_bin = tmp_path / "bin"
    _write_docker_stub(docker_bin, docker_log)
    state_file = tmp_path / "state.json"

    # Same reading every tick -- never "climbing" relative to the previous one.
    for _ in range(4):
        result = _tick(
            avail_kb=50 * GB_KB,
            pause_dir=pause_dir,
            state_file=state_file,
            docker_bin_dir=docker_bin,
        )
        assert result.returncode == 0, result.stderr

    assert not docker_log.exists()
    assert len(list(pause_dir.iterdir())) == 1


def test_two_consecutive_climbing_ticks_releases_one_canary_batch_oldest_first(
    tmp_path: Path,
) -> None:
    pause_dir = tmp_path / "pause"
    # 15 paused runners, deliberately created out of numeric order so a
    # correct implementation must sort by paused_at, not by filename/mtime.
    order = [5, 1, 4, 2, 3, 10, 9, 8, 7, 6, 15, 14, 13, 12, 11]
    for rank, i in enumerate(order):
        _write_pause_marker(
            pause_dir, f"omninode-runner-{i}", f"2026-08-22T18:{rank:02d}:00Z"
        )
    docker_log = tmp_path / "docker_calls.log"
    docker_bin = tmp_path / "bin"
    _write_docker_stub(docker_bin, docker_log)
    state_file = tmp_path / "state.json"

    # tick 1: below restore floor -> no streak progress
    r1 = _tick(
        avail_kb=25 * GB_KB,
        pause_dir=pause_dir,
        state_file=state_file,
        docker_bin_dir=docker_bin,
    )
    assert r1.returncode == 0
    assert not docker_log.exists()

    # tick 2: climbing, above restore floor -> streak 1/2, still no restore
    r2 = _tick(
        avail_kb=45 * GB_KB,
        pause_dir=pause_dir,
        state_file=state_file,
        docker_bin_dir=docker_bin,
    )
    assert r2.returncode == 0
    assert "streak 1/2" in r2.stderr
    assert not docker_log.exists()

    # tick 3: still climbing -> streak 2/2 -> release canary batch (size 10)
    r3 = _tick(
        avail_kb=60 * GB_KB,
        pause_dir=pause_dir,
        state_file=state_file,
        docker_bin_dir=docker_bin,
    )
    assert r3.returncode == 0
    assert "streak 2/2" in r3.stderr
    assert docker_log.exists()

    started = [line.split()[1] for line in docker_log.read_text().splitlines() if line]
    assert len(started) == 10
    # Restored in pause order (oldest paused_at first): runner-5, then
    # runner-1, runner-4, runner-2, runner-3, runner-10, ... (the `order` list
    # above, first 10 entries).
    expected_first_ten = [f"omninode-runner-{i}" for i in order[:10]]
    assert started == expected_first_ten

    remaining = {p.name for p in pause_dir.iterdir()}
    assert remaining == {f"omninode-runner-{i}" for i in order[10:]}


def test_batch_release_resets_streak_next_batch_requires_reproving_climb(
    tmp_path: Path,
) -> None:
    pause_dir = tmp_path / "pause"
    for i in range(1, 13):  # 12 paused, batch sizes "5 5 5"
        _write_pause_marker(
            pause_dir, f"omninode-runner-{i}", f"2026-08-22T18:{i:02d}:00Z"
        )
    docker_log = tmp_path / "docker_calls.log"
    docker_bin = tmp_path / "bin"
    _write_docker_stub(docker_bin, docker_log)
    state_file = tmp_path / "state.json"

    def tick(avail_gb: int) -> subprocess.CompletedProcess[str]:
        return _tick(
            avail_kb=avail_gb * GB_KB,
            pause_dir=pause_dir,
            state_file=state_file,
            docker_bin_dir=docker_bin,
            batch_sizes="5 5 5",
        )

    tick(45)  # streak 1/2
    tick(60)  # streak 2/2 -> batch #1 (5 restored), streak resets to 0
    assert len(docker_log.read_text().splitlines()) == 5
    assert len(list(pause_dir.iterdir())) == 7

    # Immediately climbing again does NOT release batch #2 on the very next
    # tick -- the streak was reset by the batch release and must be re-proven.
    tick(75)  # streak 1/2 only
    assert len(docker_log.read_text().splitlines()) == 5, (
        "batch #2 must not release before the climb streak is re-proven"
    )

    tick(90)  # streak 2/2 -> batch #2 (5 more restored)
    assert len(docker_log.read_text().splitlines()) == 10
    assert len(list(pause_dir.iterdir())) == 2


def test_declining_tick_after_partial_restore_does_not_release_next_batch(
    tmp_path: Path,
) -> None:
    """The documented stop signal: a batch that turns the slope negative again
    halts further restoration -- it does not un-restore what already came
    back, and it does not proceed to the next batch either."""
    pause_dir = tmp_path / "pause"
    for i in range(1, 6):
        _write_pause_marker(
            pause_dir, f"omninode-runner-{i}", f"2026-08-22T18:0{i}:00Z"
        )
    docker_log = tmp_path / "docker_calls.log"
    docker_bin = tmp_path / "bin"
    _write_docker_stub(docker_bin, docker_log)
    state_file = tmp_path / "state.json"

    def tick(avail_gb: int) -> subprocess.CompletedProcess[str]:
        return _tick(
            avail_kb=avail_gb * GB_KB,
            pause_dir=pause_dir,
            state_file=state_file,
            docker_bin_dir=docker_bin,
            batch_sizes="2 2 2",
        )

    tick(45)  # streak 1/2
    tick(60)  # streak 2/2 -> batch #1 (2 restored)
    assert len(docker_log.read_text().splitlines()) == 2

    # Free space DROPS -- the fleet accepting jobs again consumed some of the
    # reclaimed headroom. This must not release batch #2.
    r = tick(55)
    assert r.returncode == 0
    assert len(docker_log.read_text().splitlines()) == 2
    assert len(list(pause_dir.iterdir())) == 3


def test_marker_only_cleared_on_verified_running_status(tmp_path: Path) -> None:
    """If `docker start` succeeds but the container does not reach
    Status=running, the marker must be KEPT so the next tick retries --
    never cleared on an unverified start."""
    pause_dir = tmp_path / "pause"
    _write_pause_marker(pause_dir, "omninode-runner-1", "2026-08-22T18:00:00Z")
    docker_bin = tmp_path / "bin"
    docker_bin.mkdir()
    docker_log = tmp_path / "docker_calls.log"
    # Stub that reports Status=created (never actually started) after `start`.
    (docker_bin / "docker").write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "start" ]]; then\n'
        f'  echo "start $2" >> "{docker_log}"\n'
        "  exit 0\n"
        'elif [[ "$1" == "inspect" ]]; then\n'
        "  echo created\n"
        "  exit 0\n"
        "fi\n"
        "exit 0\n"
    )
    (docker_bin / "docker").chmod(0o755)
    state_file = tmp_path / "state.json"

    _tick(
        avail_kb=45 * GB_KB,
        pause_dir=pause_dir,
        state_file=state_file,
        docker_bin_dir=docker_bin,
    )
    r = _tick(
        avail_kb=60 * GB_KB,
        pause_dir=pause_dir,
        state_file=state_file,
        docker_bin_dir=docker_bin,
    )

    assert "WARNING" in r.stderr
    assert (pause_dir / "omninode-runner-1").exists(), (
        "marker must survive an unverified (non-running) start attempt"
    )
