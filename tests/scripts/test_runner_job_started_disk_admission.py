# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression coverage for the OMN-16363 pre-job disk-admission gate in
`docker/runners/runner-job-started.sh`.

THE MECHANISM UNDER TEST. Per OMN-16360/OMN-16363: when a self-hosted runner's
free disk drops below a floor, GitHub still dispatches it a job (there is no
runner-side decline/requeue API), and today that job proceeds through
workspace reset, checkout, and dependency-install writes before finally dying
with ENOSPC -- so every rejected attempt still burns real disk. The gate this
file tests makes the hook fail INSTANTLY, before any of those writes, the
moment free space is below the admission floor -- capping each rejected
attempt's I/O cost at a single `df` call instead of however many megabytes a
partial checkout/cache-write burns before the kernel actually returns ENOSPC.

These are real-subprocess tests (the same style as
test_runner_job_started_root_owned_debris.py): they run the actual hook
script end to end via `bash`, with `DISK_ADMISSION_DF_OVERRIDE_KB` as the test
seam that lets a free-space reading be injected without needing a real
near-full filesystem.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_SCRIPT = REPO_ROOT / "docker" / "runners" / "runner-job-started.sh"

FIVE_GB_KB = 5 * 1024 * 1024


def _has_gnu_realpath_m() -> bool:
    try:
        result = subprocess.run(
            ["realpath", "-m", "."],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not _has_gnu_realpath_m(),
        reason="runner-job-started.sh requires GNU `realpath -m` (present on "
        "the Ubuntu 22.04 runner image and Linux CI; absent on BSD/macOS "
        "realpath) -- OMN-16363 shares the skip guard with OMN-15134's tests "
        "since it runs through the same main-body path.",
    ),
]


def _run_hook(
    runner_home: Path,
    workspace: Path,
    *,
    avail_kb: int | None,
    runner_name: str = "omninode-runner-1",
    backoff_n: int = 3,
    pause_dir: Path | None = None,
    extra_path: Path | None = None,
    min_free_gb: int = 5,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["RUNNER_HOME"] = str(runner_home)
    env["GITHUB_WORKSPACE"] = str(workspace)
    env["RUNNER_NAME"] = runner_name
    env["RUNNER_DISK_ADMISSION_BACKOFF_N"] = str(backoff_n)
    env["RUNNER_DISK_ADMISSION_MIN_FREE_GB"] = str(min_free_gb)
    if avail_kb is not None:
        env["DISK_ADMISSION_DF_OVERRIDE_KB"] = str(avail_kb)
    else:
        env.pop("DISK_ADMISSION_DF_OVERRIDE_KB", None)
    if pause_dir is not None:
        env["RUNNER_DISK_ADMISSION_PAUSE_DIR"] = str(pause_dir)
    else:
        env.pop("RUNNER_DISK_ADMISSION_PAUSE_DIR", None)
    if extra_path is not None:
        env["PATH"] = f"{extra_path}:{env.get('PATH', '')}"
    return subprocess.run(
        ["bash", str(HOOK_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_healthy_disk_proceeds_normally(tmp_path: Path) -> None:
    """Above the admission floor, the gate is a no-op and the hook still
    resets the workspace exactly as before OMN-16363 (regression guard)."""
    runner_home = tmp_path / "actions-runner"
    workspace = runner_home / "_work" / "omnibase_infra" / "omnibase_infra"
    workspace.mkdir(parents=True)
    (workspace / "stale.txt").write_text("leftover")

    result = _run_hook(runner_home, workspace, avail_kb=FIVE_GB_KB * 10)

    assert result.returncode == 0, result.stderr
    assert "RUNNER-DISK-ADMISSION" not in result.stdout + result.stderr
    assert workspace.is_dir()
    assert not (workspace / "stale.txt").exists()


def test_below_floor_fails_instantly_before_any_write(tmp_path: Path) -> None:
    """Below the admission floor, the hook must exit 1 BEFORE the workspace
    reset runs at all -- proving the zero-extra-write claim: the pre-existing
    stale file from a prior job is left completely untouched, not even
    removed and recreated."""
    runner_home = tmp_path / "actions-runner"
    workspace = runner_home / "_work" / "omnibase_infra" / "omnibase_infra"
    workspace.mkdir(parents=True)
    marker = workspace / "stale_from_prior_job.txt"
    marker.write_text("leftover")
    before_mtime = marker.stat().st_mtime_ns

    # Well below the 5 GB default floor.
    result = _run_hook(runner_home, workspace, avail_kb=1 * 1024 * 1024)

    assert result.returncode == 1
    assert "::error title=RUNNER-DISK-ADMISSION:" in result.stdout
    assert "below the 5 GB admission floor" in result.stdout
    # The workspace-reset rm -rf / mkdir never ran: the exact same file, with
    # the exact same mtime, is still there.
    assert marker.exists()
    assert marker.stat().st_mtime_ns == before_mtime
    assert marker.read_text() == "leftover"


def test_unreadable_disk_usage_fails_open(tmp_path: Path) -> None:
    """A `df` that cannot be read must never itself take the fleet down --
    the gate fails OPEN and the job proceeds (existing ENOSPC handling remains
    the backstop)."""
    runner_home = tmp_path / "actions-runner"
    workspace = runner_home / "_work" / "omnibase_infra" / "omnibase_infra"
    workspace.mkdir(parents=True)

    env = dict(os.environ)
    env["RUNNER_HOME"] = str(runner_home)
    env["GITHUB_WORKSPACE"] = str(workspace)
    env["RUNNER_NAME"] = "omninode-runner-1"
    # Point the mount at a path that cannot resolve via df, with no override set.
    env["RUNNER_DISK_ADMISSION_MOUNT"] = "/definitely/does/not/exist/omn16363"
    env.pop("DISK_ADMISSION_DF_OVERRIDE_KB", None)

    result = subprocess.run(
        ["bash", str(HOOK_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "gate fails open, job proceeds" in result.stdout
    assert workspace.is_dir()


def test_consecutive_failures_below_backoff_do_not_self_pause(
    tmp_path: Path,
) -> None:
    """Fewer than RUNNER_DISK_ADMISSION_BACKOFF_N consecutive admission
    failures must not trigger the self-pause escalation."""
    runner_home = tmp_path / "actions-runner"
    workspace = runner_home / "_work" / "omnibase_infra" / "omnibase_infra"
    workspace.mkdir(parents=True)
    pause_dir = tmp_path / "pause"
    pause_dir.mkdir()

    for _ in range(2):  # backoff_n defaults to 3 in this test
        result = _run_hook(
            runner_home,
            workspace,
            avail_kb=1 * 1024 * 1024,
            pause_dir=pause_dir,
            backoff_n=3,
        )
        assert result.returncode == 1

    assert list(pause_dir.iterdir()) == []


def test_backoff_threshold_writes_pause_marker_and_stops_self(
    tmp_path: Path,
) -> None:
    """Crossing the consecutive-failure backoff threshold must write a durable
    pause marker (for the host-side restore guard to find) and invoke
    `docker stop <own-container>` -- the only way a self-hosted runner can
    actually stop accepting new job assignments (fix direction #2)."""
    runner_home = tmp_path / "actions-runner"
    workspace = runner_home / "_work" / "omnibase_infra" / "omnibase_infra"
    workspace.mkdir(parents=True)
    pause_dir = tmp_path / "pause"
    pause_dir.mkdir()

    stub_bin = tmp_path / "stubbin"
    stub_bin.mkdir()
    docker_log = tmp_path / "docker_calls.log"
    (stub_bin / "docker").write_text(
        '#!/usr/bin/env bash\necho "$@" >> "' + str(docker_log) + '"\nexit 0\n'
    )
    (stub_bin / "docker").chmod(0o755)

    last_result: subprocess.CompletedProcess[str] | None = None
    for _ in range(3):  # backoff_n=3
        last_result = _run_hook(
            runner_home,
            workspace,
            avail_kb=1 * 1024 * 1024,
            pause_dir=pause_dir,
            backoff_n=3,
            extra_path=stub_bin,
        )
        assert last_result.returncode == 1

    assert last_result is not None
    assert "PAUSING omninode-runner-1" in last_result.stdout

    marker = pause_dir / "omninode-runner-1"
    assert marker.exists(), "expected a durable pause marker after backoff threshold"
    contents = marker.read_text()
    assert "runner=omninode-runner-1" in contents
    assert "reason=consecutive_disk_admission_failures" in contents

    # The self-stop is intentionally backgrounded (sleep 2 then docker stop) so
    # it runs after the hook itself has exited; wait for it here.
    import time

    deadline = time.time() + 10
    while time.time() < deadline and not docker_log.exists():
        time.sleep(0.2)
    assert docker_log.exists(), "expected a backgrounded `docker stop` invocation"
    assert "stop omninode-runner-1" in docker_log.read_text()


def test_self_pause_fails_open_without_pause_dir_mount(tmp_path: Path) -> None:
    """Before the fleet is recreated with the new compose volume, the
    pause-marker directory does not exist inside the container. Self-pause
    must fail open (skip, log, do nothing) -- the per-job admission gate
    above stays fully effective either way."""
    runner_home = tmp_path / "actions-runner"
    workspace = runner_home / "_work" / "omnibase_infra" / "omnibase_infra"
    workspace.mkdir(parents=True)
    missing_pause_dir = tmp_path / "does-not-exist" / "pause"

    last_result: subprocess.CompletedProcess[str] | None = None
    for _ in range(3):
        last_result = _run_hook(
            runner_home,
            workspace,
            avail_kb=1 * 1024 * 1024,
            pause_dir=missing_pause_dir,
            backoff_n=3,
        )
        assert last_result.returncode == 1

    assert last_result is not None
    assert "self-pause skipped" in last_result.stdout
    assert "not mounted" in last_result.stdout
    assert not missing_pause_dir.exists()


def test_recovery_clears_consecutive_failure_streak(tmp_path: Path) -> None:
    """A single healthy tick between two low-disk ticks must reset the
    consecutive-failure counter, so backoff only fires on a genuinely
    sustained low-disk condition, not intermittent noise."""
    runner_home = tmp_path / "actions-runner"
    workspace = runner_home / "_work" / "omnibase_infra" / "omnibase_infra"

    def _fresh_workspace() -> Path:
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    pause_dir = tmp_path / "pause"
    pause_dir.mkdir()

    # Two low-disk failures (below backoff_n=3)...
    for _ in range(2):
        _fresh_workspace()
        result = _run_hook(
            runner_home,
            workspace,
            avail_kb=1 * 1024 * 1024,
            pause_dir=pause_dir,
            backoff_n=3,
        )
        assert result.returncode == 1

    # ...then one healthy tick clears the streak...
    _fresh_workspace()
    healthy = _run_hook(
        runner_home, workspace, avail_kb=FIVE_GB_KB * 10, pause_dir=pause_dir
    )
    assert healthy.returncode == 0

    # ...so two MORE low-disk failures (still below backoff_n=3) must not pause.
    for _ in range(2):
        _fresh_workspace()
        result = _run_hook(
            runner_home,
            workspace,
            avail_kb=1 * 1024 * 1024,
            pause_dir=pause_dir,
            backoff_n=3,
        )
        assert result.returncode == 1

    assert list(pause_dir.iterdir()) == []
