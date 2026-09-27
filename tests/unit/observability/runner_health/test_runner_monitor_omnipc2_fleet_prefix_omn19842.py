# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19842 — the fleet emit's default name-prefix covers every declared host.

`config/runner_fleet.yaml`'s `hosts:` inventory (OMN-17477) declares a
`runner_name_prefix` per lab host. The monitor's fleet-emit default used to be
the single literal `"omninode-"`, which happens to be a prefix of four of the
five declared hosts and is NOT a prefix of `.202`'s `omnipc2-verify-runner` —
so that host's runners were silently absent from every fleet observation,
forever, however many of the org's `.202` runners were actually up.

These tests drive the REAL shell script end-to-end (no re-implementation of
the awk/env wiring in Python) against a `hosts:`-bearing config, and pin two
properties: the multi-host config now surfaces the `.202` runner, and a
pre-OMN-17477 scalar-only config (no `hosts:` block at all) still behaves
exactly as it always has.
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
    _make_mock_bin,
    _require_tools,
    _write_exec,
    _write_required_compose_overrides,
)

REPO_ROOT = Path(__file__).parents[4]
MONITOR_SCRIPT = REPO_ROOT / "docker" / "runners" / "runner-monitor.sh"
BUILDER = REPO_ROOT / "scripts" / "runner_fleet_event.py"
TOPIC = "onex.evt.omnibase-infra.runner-fleet.v1"
PREFIX = "omninode-runner"
TEST_FLEET_COUNT = 2

pytestmark = pytest.mark.unit


def _write_multi_host_fleet_config(path: Path) -> None:
    """The declared shape post-OMN-17477: scalars for the PRIMARY host plus a
    `hosts:` list naming every host's own prefix, including `.202`'s."""
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
            hosts:
              - host: 192.168.86.201
                arch: amd64
                runner_name_prefix: {PREFIX}
                expected_count: {TEST_FLEET_COUNT}
                classes: [action]
              - host: 192.168.86.202
                arch: amd64
                runner_name_prefix: omnipc2-verify-runner
                expected_count: 1
                classes: [verify]
            """
        ),
        encoding="utf-8",
    )


def _write_scalar_only_fleet_config(path: Path) -> None:
    """The pre-OMN-17477 shape: no `hosts:` block at all."""
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


def _runners_json_with_omnipc2(*, count: int) -> str:
    runners = [
        {
            "name": f"{PREFIX}-{i}",
            "status": "online",
            "busy": False,
            "labels": [{"name": "self-hosted"}, {"name": "omnibase-ci"}],
        }
        for i in range(1, count + 1)
    ]
    runners.append(
        {
            "name": "omnipc2-verify-runner-1",
            "status": "online",
            "busy": False,
            "labels": [
                {"name": "self-hosted"},
                {"name": "omnibase-verify"},
                {"name": "host-202"},
            ],
        }
    )
    # A runner outside every declared prefix must stay excluded.
    runners.append(
        {
            "name": "omnipc2-customer-1",
            "status": "online",
            "busy": False,
            "labels": [{"name": "self-hosted"}],
        }
    )
    return json.dumps({"total_count": len(runners), "runners": runners})


def _run(
    tmp_path: Path,
    *,
    write_fleet_config,
    runners_json: str,
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, object]]]:
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
    _write_exec(
        bindir / "rpk",
        f"""\
        set -euo pipefail
        topic="${{3:-}}"
        payload="$(cat)"
        printf '%s\\t%s\\n' "${{topic}}" "${{payload}}" >> "{produced}"
        """,
    )

    state_file = tmp_path / "runner-monitor-state.json"
    fleet_config = tmp_path / "runner_fleet.yaml"
    write_fleet_config(fleet_config)
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
        "KAFKA_BOOTSTRAP_SERVERS": "mock-broker:9092",
        "RUNNER_FLEET_BROKER_CONTAINER": "",
        "RUNNER_FLEET_EVENT_BUILDER": str(BUILDER),
    }

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


def test_a_declared_202_host_runner_is_observed(tmp_path: Path) -> None:
    """AC1/AC4 — the .202 runner appears once its prefix is in `hosts:`."""
    result, events = _run(
        tmp_path,
        write_fleet_config=_write_multi_host_fleet_config,
        runners_json=_runners_json_with_omnipc2(count=TEST_FLEET_COUNT),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(events) == 1
    names = {row["runner_name"] for row in events[0]["runners"]}
    assert "omnipc2-verify-runner-1" in names, (
        "the .202 runner is absent from the fleet event even though its "
        f"prefix is declared under hosts: — got {names}"
    )


def test_a_runner_outside_every_declared_prefix_stays_excluded(
    tmp_path: Path,
) -> None:
    """AC3 — widening to the exact declared set must not admit the sibling
    `omnipc2-customer` runner the config explicitly scopes out."""
    _, events = _run(
        tmp_path,
        write_fleet_config=_write_multi_host_fleet_config,
        runners_json=_runners_json_with_omnipc2(count=TEST_FLEET_COUNT),
    )
    names = {row["runner_name"] for row in events[0]["runners"]}
    assert "omnipc2-customer-1" not in names


def test_a_scalar_only_config_keeps_the_pre_existing_default(
    tmp_path: Path,
) -> None:
    """AC2 — a config with no `hosts:` block (pre-OMN-17477) is unaffected:
    the omninode- fallback still applies and the .202 runner (which this
    config never declares) is absent, exactly as it always has been."""
    result, events = _run(
        tmp_path,
        write_fleet_config=_write_scalar_only_fleet_config,
        runners_json=_runners_json_with_omnipc2(count=TEST_FLEET_COUNT),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(events) == 1
    names = {row["runner_name"] for row in events[0]["runners"]}
    assert names == {f"{PREFIX}-{i}" for i in range(1, TEST_FLEET_COUNT + 1)}
    assert "omnipc2-verify-runner-1" not in names
