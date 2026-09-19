# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18768 AC1 — runner-monitor.sh publishes a fleet observation on the bus.

The unit tests beside this one pin the event BUILDER. These drive the REAL
shell script end-to-end against PATH-injected mocks and prove the wiring: that
the monitor actually calls the builder every cycle, actually produces to the
declared topic, and — the property that matters most operationally — that a
broken bus can never suppress a runner alert or a bounce.

Why the last one is a test and not a comment: this monitor is the only thing
standing between a wedged CI fleet and a silent outage. An emit bolted onto it
that can abort the run under `set -e`, or that swallows the alert path, would
trade an observability gap for an availability one.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

from tests.unit.observability.runner_health._resolve_modern_bash import (
    resolve_modern_bash,
)
from tests.unit.observability.runner_health.test_runner_monitor_wedge_detection import (
    PREFIX,
    TEST_FLEET_COUNT,
    _make_mock_bin,
    _require_tools,
    _runners_json,
    _write_exec,
    _write_fleet_config,
    _write_required_compose_overrides,
)

REPO_ROOT = Path(__file__).parents[4]
MONITOR_SCRIPT = REPO_ROOT / "docker" / "runners" / "runner-monitor.sh"
BUILDER = REPO_ROOT / "scripts" / "runner_fleet_event.py"
TOPIC = "onex.evt.infra.runner-fleet.v1"

# The general pool the detection loop watches, PLUS the classes that sit
# outside its `omninode-runner` prefix: two customer-plane runners and three
# verify runners, which the shared fixture always registers because the live
# org pool always carries them. The fleet emit covers all of them on purpose —
# see the prefix rationale in runner-monitor.sh. A count computed from the
# fixture rather than written as a literal is what keeps this honest when the
# fixture's out-of-pool set changes.
OUT_OF_POOL_COUNT = 5
POOL_COUNT = TEST_FLEET_COUNT + OUT_OF_POOL_COUNT

pytestmark = pytest.mark.unit


def _run(
    tmp_path: Path,
    *,
    runners_json: str,
    rpk_mode: str = "ok",
    builder_path: Path | None = BUILDER,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, object]]]:
    """Run the real monitor with a mock rpk; return the process and produced events."""
    _require_tools()
    bindir = tmp_path / "bin"
    _make_mock_bin(
        bindir,
        docker_status="Up (healthy)",
        docker_restart_count=0,
        docker_logs="Listening for Jobs",
        runners_json=runners_json,
        queued_run_id=None,
        queued_job_created_at="2026-09-18T23:00:00Z",
        now_epoch=1789000000,
        job_created_epoch=1789000000,
    )

    produced = tmp_path / "rpk-produced.jsonl"
    if rpk_mode == "ok":
        # Invocation is `rpk topic produce <topic> --brokers <b>`, record on stdin.
        _write_exec(
            bindir / "rpk",
            f"""\
            set -euo pipefail
            topic="${{3:-}}"
            payload="$(cat)"
            printf '%s\\t%s\\n' "${{topic}}" "${{payload}}" >> "{produced}"
            """,
        )
    elif rpk_mode == "failing":
        # A broker that refuses the produce. The monitor must survive it.
        _write_exec(
            bindir / "rpk",
            """\
            set -euo pipefail
            cat > /dev/null
            echo "broker unreachable" >&2
            exit 1
            """,
        )
    else:
        # No rpk at all on PATH and no broker address: the log-only path. An
        # rpk that happens to be installed on the developer's machine would
        # otherwise make this assertion pass or fail by accident of host setup.
        _write_exec(bindir / "rpk", "exit 127\n")

    state_file = tmp_path / "runner-monitor-state.json"
    fleet_config = tmp_path / "runner_fleet.yaml"
    _write_fleet_config(fleet_config)
    _write_required_compose_overrides(tmp_path)

    env = {
        "PATH": f"{bindir}:{os.environ.get('PATH', '')}",
        "HOME": str(tmp_path),
        "STATE_FILE": str(state_file),
        "RUNNER_FLEET_CONFIG_PATH": str(fleet_config),
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_CHANNEL_ID": "C-test",
        "RUNNER_GITHUB_TOKEN": "ghp-test",
        "MOCK_DOCKER_CALLLOG": str(tmp_path / "docker-calls.log"),
        "MOCK_SLACK_CALLLOG": str(tmp_path / "slack-calls.log"),
        "WEDGE_QUEUE_AGE_SECONDS": "600",
        "WEDGE_WATCH_REPOS": "OmniNode-ai/omnibase_infra",
        "MONITOR_AUTO_BOUNCE": "0",
        "AUTO_BOUNCE_LOCKFILE": str(tmp_path / "bounce.lock"),
        "AUTO_BOUNCE_BOUNCE_LOG": str(tmp_path / "bounce.log"),
        # The bus address is env-resolved, never defaulted (Operating Rule 8).
        "KAFKA_BOOTSTRAP_SERVERS": "mock-broker:9092",
    }
    if rpk_mode == "absent":
        env.pop("KAFKA_BOOTSTRAP_SERVERS")
    # The wrapper copies the script to a temp path, so BASH_SOURCE no longer
    # resolves the builder beside it -- name it explicitly, which is also the
    # seam a deployment with a relocated builder would use.
    if builder_path is not None:
        env["RUNNER_FLEET_EVENT_BUILDER"] = str(builder_path)
    if extra_env:
        env.update(extra_env)

    modern_bash = resolve_modern_bash()
    wrapper = tmp_path / "run.sh"
    wrapper.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            sed 's#^STATE_FILE=.*#STATE_FILE="{state_file}"#' "{MONITOR_SCRIPT}" > "{tmp_path}/monitor.sh"
            "{modern_bash}" "{tmp_path}/monitor.sh"
            """
        ),
        encoding="utf-8",
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)

    result = subprocess.run(
        [modern_bash, str(wrapper)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    events: list[dict[str, object]] = []
    if produced.exists():
        for line in produced.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            topic, _, payload = line.partition("\t")
            if topic == TOPIC:
                events.append(json.loads(payload))
    return result, events


def test_the_monitor_publishes_one_fleet_observation_per_cycle(tmp_path: Path) -> None:
    """AC1 — one contract-declared event per observation cycle, not one per runner."""
    result, events = _run(
        tmp_path,
        runners_json=_runners_json(status="online", busy=False, count=TEST_FLEET_COUNT),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(events) == 1, f"expected exactly one fleet event, got {len(events)}"
    event = events[0]
    assert event["event_type"] == "runner-fleet-observation"
    assert event["topic"] == TOPIC
    # The WHOLE pool, not only the general pool the detection loop scopes to.
    assert event["runner_count"] == POOL_COUNT
    assert event["host"]


def test_the_published_event_carries_per_runner_status_and_labels(
    tmp_path: Path,
) -> None:
    """AC1 falsifier — the message on the topic carries per-runner status and labels."""
    _, events = _run(
        tmp_path,
        runners_json=_runners_json(status="online", busy=False, count=TEST_FLEET_COUNT),
    )
    rows = events[0]["runners"]
    assert isinstance(rows, list) and len(rows) == POOL_COUNT
    by_name = {row["runner_name"]: row for row in rows}
    first = by_name[f"{PREFIX}-1"]
    assert first["status"] == "online"
    assert first["label_class"] == "omnibase-ci"
    assert "self-hosted" in first["labels"]
    assert first["observed_at"]
    # The out-of-pool classes are present and correctly classed: which class is
    # down is the operational question, and a general-pool-only observation
    # cannot answer it for the single-runner classes at all.
    assert by_name["omninode-verify-runner-1"]["label_class"] == "omnibase-verify"
    assert (
        by_name["omninode-customer-plane-runner-1"]["label_class"]
        == "omnibase-customer-plane"
    )


def test_an_offline_fleet_is_published_offline_not_published_empty(
    tmp_path: Path,
) -> None:
    """AC4 — an offline runner is reported offline rather than omitted.

    The failure this pins: a fleet outage rendering downstream as a smaller
    healthy fleet, which is indistinguishable from a deliberate scale-down.
    """
    _, events = _run(
        tmp_path,
        runners_json=_runners_json(
            status="offline", busy=False, count=TEST_FLEET_COUNT
        ),
    )
    event = events[0]
    assert event["runner_count"] == POOL_COUNT
    # The fixture always registers the out-of-pool classes as online, so the
    # offline count is the general pool exactly — which is the assertion that
    # would fail if an offline runner were silently dropped.
    assert event["offline_count"] == TEST_FLEET_COUNT
    assert event["online_count"] == OUT_OF_POOL_COUNT
    offline = [r for r in event["runners"] if r["status"] == "offline"]
    assert {r["runner_name"] for r in offline} == {
        f"{PREFIX}-{i}" for i in range(1, TEST_FLEET_COUNT + 1)
    }


def test_a_busy_fleet_reports_busy(tmp_path: Path) -> None:
    _, events = _run(
        tmp_path,
        runners_json=_runners_json(status="online", busy=True, count=TEST_FLEET_COUNT),
    )
    event = events[0]
    assert event["busy_count"] == TEST_FLEET_COUNT
    busy = [r for r in event["runners"] if r["status"] == "busy"]
    assert len(busy) == TEST_FLEET_COUNT
    # No job id was resolvable: NULL, never invented.
    assert all(row["current_job_id"] is None for row in busy)


def test_a_broken_bus_does_not_break_the_monitor(tmp_path: Path) -> None:
    """A publish failure must never abort the run under `set -e`.

    This monitor is the only thing between a wedged CI fleet and a silent
    outage. An emit that can take it down trades an observability gap for an
    availability one.
    """
    result, events = _run(
        tmp_path,
        runners_json=_runners_json(status="online", busy=False, count=TEST_FLEET_COUNT),
        rpk_mode="absent",  # no rpk, no broker, no publish URL -> log-only path
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert events == []
    # The event is still built and logged, so it stays replayable.
    assert "runner-fleet-observation" in result.stdout
    assert (tmp_path / "runner-monitor-state.json").exists()


def test_a_refusing_broker_does_not_break_the_monitor(tmp_path: Path) -> None:
    """A produce that FAILS is a different path from a broker that is absent."""
    result, events = _run(
        tmp_path,
        runners_json=_runners_json(status="online", busy=False, count=TEST_FLEET_COUNT),
        rpk_mode="failing",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert events == []
    assert "fleet emit via rpk FAILED" in result.stdout
    # Detection still ran to completion and the alert path still wrote state.
    assert (tmp_path / "runner-monitor-state.json").exists()


def test_a_missing_builder_is_a_logged_skip_not_a_failed_monitor(
    tmp_path: Path,
) -> None:
    """A deployment that did not sync the builder must still monitor the fleet."""
    result, events = _run(
        tmp_path,
        runners_json=_runners_json(status="online", busy=False, count=TEST_FLEET_COUNT),
        builder_path=tmp_path / "does-not-exist.py",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert events == []
    assert "event builder not found" in result.stdout
    assert (tmp_path / "runner-monitor-state.json").exists()


def test_the_emit_can_be_turned_off_without_touching_detection(tmp_path: Path) -> None:
    result, events = _run(
        tmp_path,
        runners_json=_runners_json(status="online", busy=False, count=TEST_FLEET_COUNT),
        extra_env={"RUNNER_FLEET_EMIT": "false"},
    )
    assert result.returncode == 0
    assert events == []
    assert (tmp_path / "runner-monitor-state.json").exists()


def test_the_builder_is_declared_in_the_deploy_sync_set(tmp_path: Path) -> None:
    """The builder is resolved at ../../scripts/ from the monitor's own
    deployed location, and must be rsynced to that path.

    A builder present in the repo and absent from deploy-runners.sh's sync set
    is a fleet that is silently unobservable in production while every test
    here passes.
    """
    deploy = (REPO_ROOT / "scripts" / "deploy-runners.sh").read_text(encoding="utf-8")
    assert "scripts/runner_fleet_event.py" in deploy
    # And it must be in the rsync invocation, not only the declarative list.
    assert deploy.count("scripts/runner_fleet_event.py") >= 2
    # The monitor must resolve it relative to itself, never from an absolute
    # path (Operating Rule 6) and never from PATH.
    monitor = MONITOR_SCRIPT.read_text(encoding="utf-8")
    assert "${_MONITOR_DIR}/../../scripts/runner_fleet_event.py" in monitor
