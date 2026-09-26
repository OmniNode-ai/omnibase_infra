# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for runner-monitor.sh state recovery (OMN-19717)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.unit.observability.runner_health.test_runner_monitor_alert_dwell_omn19169 import (
    _Harness,
    _write_exec,
)

pytestmark = pytest.mark.unit


def _run_cycle(harness: _Harness) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": f"{harness.bindir}:{os.environ.get('PATH', '')}",
        "HOME": str(harness.tmp),
        "RUNNER_FLEET_CONFIG_PATH": str(harness.fleet_config),
        "SLACK_BOT_TOKEN": "xoxb-test",  # pragma: allowlist secret
        "SLACK_CHANNEL_ID": "C-test",
        "RUNNER_GITHUB_TOKEN": "ghp-test",  # pragma: allowlist secret
        "WEDGE_WATCH_REPOS": "OmniNode-ai/omnibase_infra",
        "RUNNER_MONITOR_ALERT_DWELL_CYCLES": str(harness.dwell),
    }
    return subprocess.run(
        [harness.modern_bash, str(harness.script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _assert_success(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        f"monitor exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _force_state_write_failure(harness: _Harness) -> None:
    real_jq = shutil.which("jq")
    assert real_jq is not None
    _write_exec(
        harness.bindir / "jq",
        f"""\
        set -euo pipefail
        if [[ "$*" == *"--argjson healthy"* ]]; then
          echo "forced state serialization failure" >&2
          exit 42
        fi
        exec "{real_jq}" "$@"
        """,
    )


def _seed_previous_state(harness: _Harness) -> str:
    previous = json.dumps(
        {
            "unhealthy_count": 0,
            "alert_count": 0,
            "announced_alert_count": 0,
            "pending_alert_count": 0,
            "pending_alert_streak": 0,
            "offline_first_seen": {},
            "sentinel": "preserve-me",
        },
        sort_keys=True,
    )
    harness.state_file.write_text(previous, encoding="utf-8")
    return previous


def test_empty_state_file_recovers_with_numeric_zero_defaults(tmp_path: Path) -> None:
    """AC1: jq's empty-success output cannot reach any --argjson input."""
    harness = _Harness(tmp_path)
    harness.state_file.touch()

    result = _run_cycle(harness)

    _assert_success(result)
    assert "jq: invalid JSON text passed to --argjson" not in result.stderr
    state = json.loads(harness.state_file.read_text(encoding="utf-8"))
    assert state["unhealthy_count"] == 0
    assert state["alert_count"] == 0
    assert state["announced_alert_count"] == 0
    assert state["pending_alert_count"] == 0
    assert state["pending_alert_streak"] == 0
    assert state["offline_first_seen"] == {}


def test_failed_state_write_preserves_previous_file(tmp_path: Path) -> None:
    """AC2: serialization failure cannot truncate the last good state."""
    harness = _Harness(tmp_path)
    previous = _seed_previous_state(harness)
    _force_state_write_failure(harness)

    result = _run_cycle(harness)

    _assert_success(result)
    assert "state write FAILED to serialize" in result.stdout
    assert harness.state_file.read_text(encoding="utf-8") == previous
    assert list(tmp_path.glob("runner-monitor-state.json.tmp.*")) == []


def test_failed_state_write_does_not_block_fleet_emit(tmp_path: Path) -> None:
    """AC3: the fleet publisher still runs after state serialization fails."""
    harness = _Harness(tmp_path)
    _seed_previous_state(harness)
    _force_state_write_failure(harness)

    builder = tmp_path / "fake_runner_fleet_event.py"
    builder.write_text(
        'import sys\nsys.stdin.read()\nprint(\'{"event_type": "test-fleet"}\')\n',
        encoding="utf-8",
    )
    source = harness.script.read_text(encoding="utf-8")
    source = "\n".join(
        f'RUNNER_FLEET_EVENT_BUILDER="{builder}"'
        if line.startswith("RUNNER_FLEET_EVENT_BUILDER=")
        else line
        for line in source.splitlines()
    )
    harness.script.write_text(source + "\n", encoding="utf-8")

    publisher_log = tmp_path / "published.jsonl"
    docker_real = harness.bindir / "docker-real"
    (harness.bindir / "docker").rename(docker_real)
    _write_exec(
        harness.bindir / "docker",
        f"""\
        set -euo pipefail
        if [[ "${{1:-}}" == "exec" && "${{2:-}}" == "-i" ]]; then
          cat >> "{publisher_log}"
          exit 0
        fi
        exec "{docker_real}" "$@"
        """,
    )

    result = _run_cycle(harness)

    _assert_success(result)
    assert "state write FAILED to serialize" in result.stdout
    assert "fleet observation published" in result.stdout
    assert json.loads(publisher_log.read_text(encoding="utf-8"))["event_type"] == (
        "test-fleet"
    )
