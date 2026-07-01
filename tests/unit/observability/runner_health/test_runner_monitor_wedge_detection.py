# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Behavioral tests for runner-monitor.sh silent-wedge + crash-loop detection.

OMN-13109. The legacy monitor only checked "container Up (healthy) + runner
online", which PASSES a silently-wedged runner (online, registered, but not
pulling jobs) and a crash-looping runner (Up healthy in the window between
restart and the next config.sh exit). These tests drive the REAL shell script
end-to-end against PATH-injected mock binaries (docker / curl / gh / date) and
assert on the JSON state file the script writes, proving:

  * a wedged fleet (online, busy=0, job queued past threshold) is flagged
    (``wedge_count >= 1``) where the legacy container+registration logic was
    clean,
  * a crash-looping container (RestartCount past threshold OR repeated
    re-registration log markers) is flagged (``crashloop_count >= 1``),
  * a genuinely healthy fleet stays clean (``unhealthy_count == 0``),
  * the script never emits a ``docker restart`` or empty-filter bounce; the
    documented remediation is force-recreate of NAMED services only.

The script is exercised verbatim — no re-implementation of the detection logic
in Python — so a regression in the bash predicates fails these tests.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[4]
MONITOR_SCRIPT = REPO_ROOT / "docker" / "runners" / "runner-monitor.sh"
FLEET_CONFIG = REPO_ROOT / "config" / "runner_fleet.yaml"

# Small fleet for fast, deterministic tests. The script reads expected_count
# from a config file we generate per-test (NOT the live 48-runner config), so
# the loop is short.
TEST_FLEET_COUNT = 3
PREFIX = "omninode-runner"


pytestmark = pytest.mark.unit


def _require_tools() -> None:
    for tool in ("bash", "jq"):
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


def _runners_json(*, status: str, busy: bool, count: int) -> str:
    runners = [
        {
            "name": f"{PREFIX}-{i}",
            "status": status,
            "busy": busy,
            "labels": [{"name": "self-hosted"}, {"name": "omnibase-ci"}],
        }
        for i in range(1, count + 1)
    ]
    return json.dumps({"total_count": count, "runners": runners})


def _queued_runs_json(run_id: int | None) -> str:
    if run_id is None:
        return json.dumps({"total_count": 0, "workflow_runs": []})
    return json.dumps({"total_count": 1, "workflow_runs": [{"id": run_id}]})


def _queued_jobs_json(*, created_at: str) -> str:
    return json.dumps(
        {
            "total_count": 1,
            "jobs": [
                {
                    "status": "queued",
                    "labels": ["self-hosted", "omnibase-ci"],
                    "runner_name": "",
                    "created_at": created_at,
                    "started_at": created_at,
                }
            ],
        }
    )


def _make_mock_bin(
    bindir: Path,
    *,
    docker_status: str,
    docker_restart_count: int,
    docker_logs: str,
    runners_json: str,
    queued_run_id: int | None,
    queued_job_created_at: str,
    now_epoch: int,
    job_created_epoch: int,
) -> None:
    """Create mock docker / curl / gh / date executables for one scenario."""
    bindir.mkdir(parents=True, exist_ok=True)

    # --- docker mock: ps, inspect (.State.OOMKilled, .RestartCount), logs, exec
    _write_exec(
        bindir / "docker",
        f"""\
        set -euo pipefail
        cmd="${{1:-}}"
        case "${{cmd}}" in
          ps)
            # Emit one line per test runner with the configured status string.
            for i in $(seq 1 {TEST_FLEET_COUNT}); do
              printf '%s\\t%s\\n' "{PREFIX}-${{i}}" "{docker_status}"
            done
            ;;
          inspect)
            fmt="$*"
            if [[ "${{fmt}}" == *OOMKilled* ]]; then
              echo "false"
            elif [[ "${{fmt}}" == *RestartCount* ]]; then
              echo "{docker_restart_count}"
            else
              echo "false"
            fi
            ;;
          logs)
            cat <<'LOGS'
{docker_logs}
LOGS
            ;;
          exec)
            # `docker exec <name> docker info ...` socket probe — succeed.
            echo "27.0.0"
            ;;
          compose)
            # Record any compose invocation so the test can assert no unsafe
            # bounce ever ran (auto-bounce is OFF by default).
            echo "compose $*" >> "${{MOCK_DOCKER_CALLLOG:-/dev/null}}"
            ;;
          restart)
            # A `docker restart` against the fleet is the forbidden action.
            echo "restart $*" >> "${{MOCK_DOCKER_CALLLOG:-/dev/null}}"
            ;;
          *)
            : ;;
        esac
        """,
    )

    # --- curl mock: org runners API + per-repo queued runs/jobs API
    _write_exec(
        bindir / "curl",
        f"""\
        set -euo pipefail
        url=""
        for a in "$@"; do url="${{a}}"; done  # last arg is the URL
        if [[ "${{url}}" == *"/actions/runners?"* ]]; then
          cat <<'JSON'
{runners_json}
JSON
        elif [[ "${{url}}" == *"/actions/runs?status=queued"* ]]; then
          cat <<'JSON'
{_queued_runs_json(queued_run_id)}
JSON
        elif [[ "${{url}}" == *"/actions/runs/"*"/jobs"* ]]; then
          cat <<'JSON'
{_queued_jobs_json(created_at=queued_job_created_at)}
JSON
        elif [[ "${{url}}" == *"slack.com"* ]]; then
          echo '{{"ok":true}}'
        else
          echo ""
        fi
        """,
    )

    # --- gh mock: token mint (only used by auto-bounce, which stays OFF here)
    _write_exec(
        bindir / "gh",
        """\
        set -euo pipefail
        echo "mock-registration-token"
        """,
    )

    # --- date mock: deterministic clock + GNU-style `-d <iso>` parsing so the
    # test is portable to BSD date (macOS) while the .201 host uses GNU date.
    _write_exec(
        bindir / "date",
        f"""\
        set -euo pipefail
        args="$*"
        if [[ "${{args}}" == *"-d "* ]]; then
          # Parsing the queued job's created_at -> fixed epoch.
          echo "{job_created_epoch}"
        elif [[ "${{args}}" == *"+%s"* ]]; then
          echo "{now_epoch}"
        elif [[ "${{args}}" == *"-Iseconds"* ]]; then
          echo "2026-06-23T00:00:00+00:00"
        else
          # date '+%H:%M:%S' for log lines
          echo "00:00:00"
        fi
        """,
    )


def _run_monitor(
    tmp_path: Path,
    bindir: Path,
    *,
    extra_env: dict[str, str] | None = None,
) -> dict[str, object]:
    """Run the real monitor script with the mock PATH; return parsed state file."""
    state_file = tmp_path / "runner-monitor-state.json"
    fleet_config = tmp_path / "runner_fleet.yaml"
    _write_fleet_config(fleet_config)
    call_log = tmp_path / "docker-calls.log"

    env = {
        # Keep jq + bash real; everything else comes from the mock bindir first.
        "PATH": f"{bindir}:{os.environ.get('PATH', '')}",
        "HOME": str(tmp_path),
        "STATE_FILE": str(state_file),
        "RUNNER_FLEET_CONFIG_PATH": str(fleet_config),
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_CHANNEL_ID": "C-test",
        "RUNNER_GITHUB_TOKEN": "ghp-test",
        "MOCK_DOCKER_CALLLOG": str(call_log),
        # Wedge threshold low enough that the controlled queued-age trips it.
        "WEDGE_QUEUE_AGE_SECONDS": "600",
        "WEDGE_WATCH_REPOS": "OmniNode-ai/omnibase_infra",
        "CRASHLOOP_RESTART_THRESHOLD": "5",
        "CRASHLOOP_REREGISTER_MARKER_THRESHOLD": "3",
        # Auto-bounce MUST stay off — these tests prove detection only and must
        # never mutate anything.
        "MONITOR_AUTO_BOUNCE": "0",
    }
    if extra_env:
        env.update(extra_env)

    # The script reads STATE_FILE via a hardcoded default; override by editing
    # the env the script honors. The script defines STATE_FILE internally, so we
    # pass it through a wrapper that exports our value first.
    wrapper = tmp_path / "run.sh"
    wrapper.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            # The script hardcodes STATE_FILE; rewrite that one line on the fly
            # into a temp copy so the test controls the output path.
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
    # Attach the call log so callers can assert on bounce behavior.
    state["_docker_calls"] = (
        call_log.read_text(encoding="utf-8") if call_log.exists() else ""
    )
    return state


def _int(state: dict[str, object], key: str) -> int:
    """Extract an integer counter from the parsed state file (mypy-friendly)."""
    value = state[key]
    assert isinstance(value, int), f"{key} is not an int: {value!r}"
    return value


# ---------------------------------------------------------------------------
# Scenario fixtures
# ---------------------------------------------------------------------------

NOW = 1_750_000_000


def _scenario_bin(
    bindir: Path,
    *,
    status: str = "online",
    busy: bool = False,
    docker_status: str = "Up 6 hours (healthy)",
    restart_count: int = 0,
    docker_logs: str = "[entrypoint] Starting runner (attempt 1)",
    queued: bool = False,
    queued_age_seconds: int = 0,
) -> None:
    queued_run_id = 555 if queued else None
    job_created_epoch = NOW - queued_age_seconds
    _make_mock_bin(
        bindir,
        docker_status=docker_status,
        docker_restart_count=restart_count,
        docker_logs=docker_logs,
        runners_json=_runners_json(status=status, busy=busy, count=TEST_FLEET_COUNT),
        queued_run_id=queued_run_id,
        queued_job_created_at="2026-06-22T00:00:00Z",
        now_epoch=NOW,
        job_created_epoch=job_created_epoch,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_healthy_fleet_is_clean(tmp_path: Path) -> None:
    """All online, no queued work, restart 0 -> nothing flagged."""
    _require_tools()
    bindir = tmp_path / "bin"
    _scenario_bin(bindir, status="online", busy=False, queued=False)
    state = _run_monitor(tmp_path, bindir)

    assert state["unhealthy_count"] == 0, state
    assert state["wedge_count"] == 0, state
    assert state["crashloop_count"] == 0, state
    assert state["online"] == TEST_FLEET_COUNT, state


def test_silent_wedge_is_flagged_where_legacy_logic_passed(tmp_path: Path) -> None:
    """THE core proof: online + healthy + registered, but idle while a job has
    been queued past the threshold. Legacy container+registration logic sees
    only ``status=online`` and passes; the upgraded logic flags the wedge."""
    _require_tools()
    bindir = tmp_path / "bin"
    _scenario_bin(
        bindir,
        status="online",  # registration looks healthy — legacy logic's only signal
        busy=False,  # fleet is idle
        docker_status="Up 6 hours (healthy)",  # container looks healthy
        restart_count=0,
        queued=True,
        queued_age_seconds=3600,  # job queued an hour, well past 600s threshold
    )
    state = _run_monitor(tmp_path, bindir)

    # Legacy-equivalent assertion: container up + online would have been clean.
    # Upgraded assertion: the wedge is caught.
    assert _int(state, "wedge_count") >= 1, (
        "silent wedge (online+idle while job queued) not detected — this is the "
        f"exact state the legacy monitor missed. State: {state}"
    )
    assert _int(state, "unhealthy_count") >= 1, state
    assert state["busy"] == 0, state
    assert _int(state, "oldest_queued_job_age_seconds") >= 600, state


def test_wedge_not_flagged_when_a_runner_is_busy(tmp_path: Path) -> None:
    """Jobs queued but a runner IS busy -> work is flowing, not a wedge."""
    _require_tools()
    bindir = tmp_path / "bin"
    _scenario_bin(
        bindir,
        status="online",
        busy=True,  # at least one runner pulling work
        queued=True,
        queued_age_seconds=3600,
    )
    state = _run_monitor(tmp_path, bindir)
    assert state["wedge_count"] == 0, (
        f"busy fleet with queued backlog is draining, not wedged: {state}"
    )


def test_wedge_not_flagged_when_queue_is_young(tmp_path: Path) -> None:
    """Idle fleet but the queued job is younger than the threshold -> normal
    scheduling latency, not a wedge."""
    _require_tools()
    bindir = tmp_path / "bin"
    _scenario_bin(
        bindir,
        status="online",
        busy=False,
        queued=True,
        queued_age_seconds=60,  # < 600s threshold
    )
    state = _run_monitor(tmp_path, bindir)
    assert state["wedge_count"] == 0, state


def test_crashloop_flagged_on_high_restart_count(tmp_path: Path) -> None:
    """RestartCount past threshold -> crash-loop flagged even though the
    container momentarily reports Up (healthy)."""
    _require_tools()
    bindir = tmp_path / "bin"
    _scenario_bin(
        bindir,
        status="online",
        busy=False,
        docker_status="Up 10 seconds (healthy)",
        restart_count=42,  # > CRASHLOOP_RESTART_THRESHOLD=5
        queued=False,
    )
    state = _run_monitor(tmp_path, bindir)
    assert _int(state, "crashloop_count") >= 1, (
        f"climbing RestartCount not detected as crash-loop: {state}"
    )
    assert _int(state, "unhealthy_count") >= 1, state


def test_crashloop_flagged_on_reregistration_log_markers(tmp_path: Path) -> None:
    """Repeated re-registration markers in logs -> crash-loop flagged even with
    a low RestartCount (the config.sh 'already configured' loop)."""
    _require_tools()
    bindir = tmp_path / "bin"
    loop_logs = "\n".join(
        [
            "[entrypoint] Registration error detected.",
            "[entrypoint] Re-registering in 20s (retry 1/3)...",
            "[entrypoint] Re-registering in 40s (retry 2/3)...",
            "[entrypoint] Max retries (3) reached. Sleeping 5m before exit 1.",
        ]
    )
    _scenario_bin(
        bindir,
        status="online",
        busy=False,
        docker_status="Up 30 seconds (healthy)",
        restart_count=1,  # below the RestartCount threshold on purpose
        docker_logs=loop_logs,
        queued=False,
    )
    state = _run_monitor(tmp_path, bindir)
    assert _int(state, "crashloop_count") >= 1, (
        f"repeated re-registration markers not detected as crash-loop: {state}"
    )


def test_monitor_never_runs_docker_restart_or_empty_bounce(tmp_path: Path) -> None:
    """Safety invariant: with auto-bounce OFF (default) the monitor must never
    invoke `docker restart` or `docker compose up` against the fleet — it only
    DETECTS and renders the recipe into the alert."""
    _require_tools()
    bindir = tmp_path / "bin"
    _scenario_bin(
        bindir,
        status="online",
        busy=False,
        restart_count=99,  # crash-loop
        queued=True,
        queued_age_seconds=3600,  # wedge too
    )
    state = _run_monitor(tmp_path, bindir)
    calls = str(state["_docker_calls"])
    assert "restart" not in calls, f"forbidden `docker restart` invoked: {calls}"
    assert "compose" not in calls, (
        f"auto-bounce ran a compose recreate while MONITOR_AUTO_BOUNCE=0: {calls}"
    )


def test_script_documents_safe_bounce_and_forbids_docker_restart() -> None:
    """Static guarantee on the script text: the safe-bounce recipe uses
    force-recreate of NAMED services with a fresh token, and the script never
    bounces via `docker restart` or an empty service filter."""
    content = MONITOR_SCRIPT.read_text(encoding="utf-8")
    assert "--force-recreate" in content
    assert "registration-token" in content, "fresh-token mint recipe absent"
    assert "--no-deps" in content, "bounce must scope to named services only"
    # The recipe and prose must call out the forbidden action explicitly.
    assert "NEVER 'docker restart'" in content or "NEVER `docker restart`" in content
    # No actual `docker restart <container>` invocation anywhere in the script.
    # Every legitimate mention of the string is either a prohibition ("NEVER
    # ...docker restart...") in a comment or alert banner; none is an executed
    # command. Assert that there is no line whose first command token sequence
    # is `docker restart`.
    invocation_lines = [
        line
        for line in content.splitlines()
        if "docker restart" in line
        and "NEVER" not in line
        and "'docker restart'" not in line
        and "`docker restart`" not in line
        and not line.lstrip().startswith("#")
    ]
    assert not invocation_lines, (
        "script appears to invoke `docker restart` against the fleet: "
        f"{invocation_lines}"
    )
