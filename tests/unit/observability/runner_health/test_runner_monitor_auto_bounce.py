# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Behavioral tests for runner-monitor.sh auto_bounce() hardening (OMN-13947).

Root cause of the 2026-07-04 runner-fleet non-convergence incident: the
original ``auto_bounce()`` dispatched an unbounded, unlocked, fixed-120s-timeout
``docker compose up -d --force-recreate`` in the background. Under host load
this SIGTERM'd mid-batch, leaving containers ``Status=created`` but never
started; a second cron tick could race a still-running bounce (confirmed daemon
errors: "removal of container ... is already in progress"); and a straggler
left at ``Status=created`` matched none of the existing remediation-target
categories (crashloop/wedge), so it was never retried.

These tests drive the REAL shell script end-to-end against PATH-injected mock
binaries, proving:

  * a container stuck at Docker ``Created`` (never started) is flagged
    (``stuck_created_count >= 1``) and IS a remediation target — closing the
    "orphaned straggler" gap,
  * with auto-bounce enabled, a stuck-created target actually gets a
    ``docker compose ... --force-recreate`` dispatched, and the verify-retry
    loop explicitly ``docker start``s it if the compose call leaves it
    non-running,
  * a second monitor invocation started while a bounce still holds the
    ``AUTO_BOUNCE_LOCKFILE`` skips its own bounce instead of racing the daemon.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[4]
MONITOR_SCRIPT = REPO_ROOT / "docker" / "runners" / "runner-monitor.sh"

TEST_FLEET_COUNT = 2
PREFIX = "omninode-runner"
NOW = 1_750_000_000

pytestmark = pytest.mark.unit


def _require_tools() -> None:
    # `timeout` is mocked as a transparent passthrough in _make_mock_bin, so it
    # is not required on the host (macOS ships no GNU timeout by default).
    for tool in ("bash", "jq", "flock"):
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} not available; shell detection test requires it")


def _write_exec(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _write_fleet_config(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            f"""\
            version: "1.0"
            github_org: OmniNode-ai
            runner_host: 192.168.86.201
            runner_group: omnibase-ci
            runner_name_prefix: {PREFIX}
            expected_count: {TEST_FLEET_COUNT}
            burst_count: {TEST_FLEET_COUNT}
            """
        ),
        encoding="utf-8",
    )


def _runners_json(*, status: str = "online", busy: bool = False) -> str:
    runners = [
        {
            "name": f"{PREFIX}-{i}",
            "status": status,
            "busy": busy,
            "labels": [{"name": "self-hosted"}, {"name": "omnibase-ci"}],
        }
        for i in range(1, TEST_FLEET_COUNT + 1)
    ]
    return json.dumps({"total_count": TEST_FLEET_COUNT, "runners": runners})


def _make_mock_bin(
    bindir: Path,
    *,
    docker_status: str,
    start_marker_dir: Path,
) -> None:
    """Mock docker / curl / gh / date. `docker inspect --format
    '{{.State.Status}}' <name>` reports "created" until a marker file for that
    name exists (written by a mocked `docker start <name>`), at which point it
    reports "running" — simulating a straggler that heals on explicit start.
    """
    bindir.mkdir(parents=True, exist_ok=True)
    start_marker_dir.mkdir(parents=True, exist_ok=True)

    _write_exec(
        bindir / "docker",
        f"""\
        set -euo pipefail
        cmd="${{1:-}}"
        case "${{cmd}}" in
          ps)
            for i in $(seq 1 {TEST_FLEET_COUNT}); do
              printf '%s\\t%s\\n' "{PREFIX}-${{i}}" "{docker_status}"
            done
            ;;
          inspect)
            fmt="$*"
            name="${{fmt##* }}"
            if [[ "${{fmt}}" == *OOMKilled* ]]; then
              echo "false"
            elif [[ "${{fmt}}" == *RestartCount* ]]; then
              echo "0"
            elif [[ "${{fmt}}" == *State.Status* ]]; then
              if [[ -f "{start_marker_dir}/${{name}}" ]]; then
                echo "running"
              else
                echo "created"
              fi
            else
              echo "false"
            fi
            ;;
          logs)
            echo "[entrypoint] Starting runner (attempt 1)"
            ;;
          exec)
            echo "27.0.0"
            ;;
          start)
            name="${{2}}"
            touch "{start_marker_dir}/${{name}}"
            echo "start $*" >> "${{MOCK_DOCKER_CALLLOG:-/dev/null}}"
            ;;
          compose)
            echo "compose $*" >> "${{MOCK_DOCKER_CALLLOG:-/dev/null}}"
            ;;
          restart)
            echo "restart $*" >> "${{MOCK_DOCKER_CALLLOG:-/dev/null}}"
            ;;
          *)
            : ;;
        esac
        """,
    )

    _write_exec(
        bindir / "curl",
        f"""\
        set -euo pipefail
        url=""
        for a in "$@"; do url="${{a}}"; done
        if [[ "${{url}}" == *"/actions/runners?"* ]]; then
          cat <<'JSON'
{_runners_json()}
JSON
        elif [[ "${{url}}" == *"/actions/runs?status=queued"* ]]; then
          echo '{{"total_count":0,"workflow_runs":[]}}'
        elif [[ "${{url}}" == *"slack.com"* ]]; then
          echo '{{"ok":true}}'
        else
          echo ""
        fi
        """,
    )

    # OMN-13912 added github_api_get(), which prefers `gh api <path>` over curl
    # when `gh` is on PATH — so the mock must route by path exactly like the
    # curl mock above, not just mint a token unconditionally.
    _write_exec(
        bindir / "gh",
        f"""\
        set -euo pipefail
        path=""
        for a in "$@"; do
          if [[ "${{a}}" == /* ]]; then path="${{a}}"; fi
        done
        if [[ "$*" == *"registration-token"* ]]; then
          echo "mock-registration-token"
        elif [[ "${{path}}" == *"/actions/runners?"* ]]; then
          cat <<'JSON'
{_runners_json()}
JSON
        elif [[ "${{path}}" == *"/actions/runs?status=queued"* ]]; then
          echo '{{"total_count":0,"workflow_runs":[]}}'
        else
          echo '{{}}'
        fi
        """,
    )

    # macOS has no GNU `timeout` by default; make it a transparent passthrough
    # so these tests are portable without requiring coreutils on PATH (mirrors
    # OMN-13912's test harness pattern).
    _write_exec(
        bindir / "timeout",
        """\
        set -euo pipefail
        shift
        exec "$@"
        """,
    )

    _write_exec(
        bindir / "date",
        f"""\
        set -euo pipefail
        args="$*"
        if [[ "${{args}}" == *"-d "* ]]; then
          echo "{NOW}"
        elif [[ "${{args}}" == *"+%s"* ]]; then
          echo "{NOW}"
        elif [[ "${{args}}" == *"-Iseconds"* ]]; then
          echo "2026-07-05T00:00:00+00:00"
        else
          echo "00:00:00"
        fi
        """,
    )


def _run_monitor(
    tmp_path: Path,
    bindir: Path,
    *,
    docker_status: str,
    start_marker_dir: Path,
    extra_env: dict[str, str] | None = None,
) -> tuple[dict[str, object], Path]:
    """Run the real monitor script; return (parsed state file, call log path)."""
    state_file = tmp_path / "runner-monitor-state.json"
    fleet_config = tmp_path / "runner_fleet.yaml"
    _write_fleet_config(fleet_config)
    call_log = tmp_path / "docker-calls.log"
    _make_mock_bin(
        bindir, docker_status=docker_status, start_marker_dir=start_marker_dir
    )

    env = {
        "PATH": f"{bindir}:{os.environ.get('PATH', '')}",
        "HOME": str(tmp_path),
        "RUNNER_FLEET_CONFIG_PATH": str(fleet_config),
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_CHANNEL_ID": "C-test",
        "RUNNER_GITHUB_TOKEN": "ghp-test",
        "MOCK_DOCKER_CALLLOG": str(call_log),
        "WEDGE_WATCH_REPOS": "OmniNode-ai/omnibase_infra",
        # Fast, deterministic retry loop.
        "AUTO_BOUNCE_VERIFY_RETRY_COUNT": "2",
        "AUTO_BOUNCE_VERIFY_RETRY_SLEEP_SECONDS": "0",
        "AUTO_BOUNCE_PER_CONTAINER_BUDGET_SECONDS": "5",
        "AUTO_BOUNCE_HARD_LIMIT_SECONDS": "5",
        "AUTO_BOUNCE_LOCKFILE": str(tmp_path / "bounce.lock"),
    }
    if extra_env:
        env.update(extra_env)

    wrapper = tmp_path / "run.sh"
    wrapper.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            sed 's#^STATE_FILE=.*#STATE_FILE="{state_file}"#' "{MONITOR_SCRIPT}" > "{tmp_path}/monitor.sh"
            bash "{tmp_path}/monitor.sh"
            """
        ),
        encoding="utf-8",
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)

    result = subprocess.run(
        ["bash", str(wrapper)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, (
        f"monitor exited {result.returncode}\nSTDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert state_file.exists(), f"state file not written\nSTDOUT:\n{result.stdout}"
    state: dict[str, object] = json.loads(state_file.read_text(encoding="utf-8"))
    return state, call_log


def _poll_call_log(
    call_log: Path, *, expect_substring: str, timeout_s: float = 5.0
) -> str:
    """auto_bounce dispatches its compose+verify work in a detached background
    subshell, so the parent script can exit before that work finishes. Poll
    briefly for the expected call-log line rather than asserting immediately.
    """
    deadline = time.monotonic() + timeout_s
    text = ""
    while time.monotonic() < deadline:
        if call_log.exists():
            text = call_log.read_text(encoding="utf-8")
            if expect_substring in text:
                return text
        time.sleep(0.1)
    return text


def test_stuck_created_container_is_flagged(tmp_path: Path) -> None:
    """A container Docker reports as `Created` (never started) is the exact
    fingerprint of a force-recreate batch killed mid-flight. It must be flagged
    as its own category — this is captured synchronously in the main detection
    pass, before any backgrounding, so no polling is needed."""
    _require_tools()
    bindir = tmp_path / "bin"
    state, _ = _run_monitor(
        tmp_path,
        bindir,
        docker_status="Created 2 minutes ago",
        start_marker_dir=tmp_path / "markers",
        extra_env={"MONITOR_AUTO_BOUNCE": "0"},
    )
    assert state["stuck_created_count"] == TEST_FLEET_COUNT, state
    assert state["unhealthy_count"] == TEST_FLEET_COUNT, state


def test_healthy_fleet_has_no_stuck_created(tmp_path: Path) -> None:
    """Regression guard: a normal healthy fleet must not spuriously flag
    stuck_created."""
    _require_tools()
    bindir = tmp_path / "bin"
    state, _ = _run_monitor(
        tmp_path,
        bindir,
        docker_status="Up 6 hours (healthy)",
        start_marker_dir=tmp_path / "markers",
    )
    assert state["stuck_created_count"] == 0, state


def test_auto_bounce_recreates_and_verify_retries_stuck_created(tmp_path: Path) -> None:
    """With auto-bounce ON, a stuck-created runner must (a) get a
    `docker compose ... --force-recreate` dispatched naming it, and (b) get an
    explicit `docker start` from the verify-retry loop once the mock compose
    call leaves it non-running (compose itself is a no-op in the mock — only
    `docker start` flips the marker that makes inspect report "running")."""
    _require_tools()
    bindir = tmp_path / "bin"
    markers = tmp_path / "markers"
    state, call_log = _run_monitor(
        tmp_path,
        bindir,
        docker_status="Created 2 minutes ago",
        start_marker_dir=markers,
        extra_env={"MONITOR_AUTO_BOUNCE": "1"},
    )
    assert state["stuck_created_count"] == TEST_FLEET_COUNT, state

    calls = _poll_call_log(call_log, expect_substring="start")
    assert "compose" in calls, (
        f"auto_bounce never dispatched a force-recreate: {calls!r}"
    )
    assert f"start {PREFIX}-1" in calls or f"start {PREFIX}-2" in calls, (
        f"verify-retry never explicitly started a straggler left non-running: {calls!r}"
    )


def test_concurrent_bounce_is_skipped_not_raced(tmp_path: Path) -> None:
    """OMN-13947: the original auto_bounce had no mutex, so a cron tick firing
    while a prior bounce was still in flight raced the same containers,
    producing daemon-level 'removal ... already in progress' errors. Holding
    the lockfile externally must make the monitor's own bounce attempt skip
    cleanly rather than dispatch a second compose invocation."""
    _require_tools()
    bindir = tmp_path / "bin"
    lockfile = tmp_path / "bounce.lock"
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    lockfile.touch()

    # Hold the lock in a background flock process for the duration of the test.
    holder = subprocess.Popen(
        ["flock", str(lockfile), "sleep", "10"],
    )
    try:
        time.sleep(0.3)  # let the holder actually acquire the lock
        state, call_log = _run_monitor(
            tmp_path,
            bindir,
            docker_status="Created 2 minutes ago",
            start_marker_dir=tmp_path / "markers",
            extra_env={
                "MONITOR_AUTO_BOUNCE": "1",
                "AUTO_BOUNCE_LOCKFILE": str(lockfile),
            },
        )
        assert state["stuck_created_count"] == TEST_FLEET_COUNT, state
        # Give any (incorrect) background dispatch a moment to show up, then
        # assert it never did.
        time.sleep(1.0)
        calls = call_log.read_text(encoding="utf-8") if call_log.exists() else ""
        assert "compose" not in calls, (
            f"auto_bounce raced a held lock instead of skipping: {calls!r}"
        )
    finally:
        holder.terminate()
        holder.wait(timeout=5)
