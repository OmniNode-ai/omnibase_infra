# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Behavioral tests for runner-monitor.sh's customer-plane alerting (OMN-18396).

OMN-18392 added two credential-free runners, omninode-customer-plane-runner-1
and -2, deliberately OUTSIDE the RUNNER_NAME_PREFIX/EXPECTED_RUNNERS family the
monitor's main detection loop scopes to (``^omninode-runner-[0-9]+$``). Its own
docstring recorded the residual: this pair was invisible to the monitor's
alerting and auto-bounce.

These tests drive the REAL shell script end-to-end (PATH-injected mock
binaries, same pattern as the sibling auto-bounce/wedge-detection modules) and
prove:

  * a positive control -- everything healthy, including the customer-plane
    pair -- produces zero alert-worthy findings (proves the widened check does
    not fire vacuously),
  * a Docker-level outage of a customer-plane runner is reported into
    ``unhealthy_names`` in the state file AND actually pages (the state file's
    ``alert_count`` becomes nonzero, which is what the transition logic below
    gates the Slack ALERT message on -- ``unhealthy_list`` membership alone
    would NOT have paged, since ``current_alert_count`` is computed from
    ``remediation_target_count`` plus a handful of named booleans, not from
    raw unhealthy count),
  * a GitHub-registration-only outage (Docker healthy, GitHub not "online") is
    reported the same way,
  * with ``MONITOR_AUTO_BOUNCE=1`` and a customer-plane runner unhealthy, no
    ``docker compose ... --force-recreate`` naming either customer-plane
    runner is ever dispatched -- the widened check is alert-only, by two
    independent guards: it is never added to any of the four
    ``collect_remediation_targets()`` input lists, and that function's own
    service-name filter still hard-matches
    ``^${RUNNER_NAME_PREFIX}-[0-9]+$`` (``omninode-runner-N``), which a
    customer-plane name can never satisfy.
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

from tests.unit.observability.runner_health._resolve_modern_bash import (
    resolve_modern_bash,
)

REPO_ROOT = Path(__file__).parents[4]
MONITOR_SCRIPT = REPO_ROOT / "docker" / "runners" / "runner-monitor.sh"

PREFIX = "omninode-runner"
TEST_FLEET_COUNT = 1  # one general-pool runner is enough to prove no regression
CP1 = "omninode-customer-plane-runner-1"
CP2 = "omninode-customer-plane-runner-2"

pytestmark = pytest.mark.unit


def _require_tools() -> None:
    resolve_modern_bash()
    for tool in ("bash", "jq", "flock"):
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} not available; shell detection test requires it")


def _write_exec(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _write_required_compose_overrides(home: Path) -> None:
    compose_dir = home / ".omnibase" / "runners" / "docker"
    compose_dir.mkdir(parents=True, exist_ok=True)
    (compose_dir / "compose-overrides.list").write_text(
        "docker-compose.model-review-canary.yml\n", encoding="utf-8"
    )
    (compose_dir / "docker-compose.model-review-canary.yml").write_text(
        "services: {}\n", encoding="utf-8"
    )


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


def _runners_json(*, cp1_status: str | None, cp2_status: str | None) -> str:
    """The general pool (TEST_FLEET_COUNT healthy+online) plus whichever
    customer-plane entries the case wants -- omitting a name models it being
    absent from the org's registration list entirely (the 'missing' path)."""
    runners = [
        {
            "name": f"{PREFIX}-{i}",
            "status": "online",
            "busy": False,
            "labels": [{"name": "self-hosted"}, {"name": "omnibase-ci"}],
        }
        for i in range(1, TEST_FLEET_COUNT + 1)
    ]
    if cp1_status is not None:
        runners.append(
            {
                "name": CP1,
                "status": cp1_status,
                "busy": False,
                "labels": [
                    {"name": "self-hosted"},
                    {"name": "omnibase-customer-plane"},
                ],
            }
        )
    if cp2_status is not None:
        runners.append(
            {
                "name": CP2,
                "status": cp2_status,
                "busy": False,
                "labels": [
                    {"name": "self-hosted"},
                    {"name": "omnibase-customer-plane"},
                ],
            }
        )
    return json.dumps({"total_count": len(runners), "runners": runners})


def _make_mock_bin(
    bindir: Path,
    *,
    cp1_docker_status: str | None,
    cp2_docker_status: str | None,
    cp1_gh_status: str | None,
    cp2_gh_status: str | None,
    call_log: Path,
    slack_log: Path,
) -> None:
    """Mock docker / gh / timeout / date.

    The general pool is always ``Up (healthy)`` / online -- these tests are
    about the customer-plane pair, not the general-pool loop the sibling
    modules already cover. ``docker ps --filter name=X`` IS respected (unlike
    a same-value-regardless-of-filter mock), because this monitor now issues
    two DIFFERENT ``docker ps`` queries (general pool, then customer-plane) and
    a filter-blind mock would make the customer-plane query return whatever
    the general-pool query would have, defeating the whole point of the test.
    """
    bindir.mkdir(parents=True, exist_ok=True)

    cp_ps_lines = ""
    if cp1_docker_status is not None:
        cp_ps_lines += (
            f'printf "%s\\t%s\\n" "{CP1}" "{cp1_docker_status}"\n            '
        )
    if cp2_docker_status is not None:
        cp_ps_lines += (
            f'printf "%s\\t%s\\n" "{CP2}" "{cp2_docker_status}"\n            '
        )

    _write_exec(
        bindir / "docker",
        f"""\
        set -euo pipefail
        cmd="${{1:-}}"
        case "${{cmd}}" in
          ps)
            filter=""
            for a in "$@"; do
              case "${{a}}" in
                name=*) filter="${{a#name=}}" ;;
              esac
            done
            if [[ "${{filter}}" == "omninode-customer-plane-runner-" ]]; then
              {cp_ps_lines if cp_ps_lines else ":"}
            else
              for i in $(seq 1 {TEST_FLEET_COUNT}); do
                printf '%s\\t%s\\n' "{PREFIX}-${{i}}" "Up (healthy)"
              done
            fi
            ;;
          inspect)
            fmt="$*"
            if [[ "${{fmt}}" == *OOMKilled* ]]; then
              echo "false"
            elif [[ "${{fmt}}" == *RestartCount* ]]; then
              echo "0"
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
          compose)
            if [[ "$*" == *" config"* ]]; then
              if [[ "$*" == *"--services"* ]]; then
                echo "omninode-deploy-runner"
                for i in $(seq 1 {TEST_FLEET_COUNT}); do
                  echo "{PREFIX}-${{i}}"
                done
                echo "{CP1}"
                echo "{CP2}"
              fi
              exit 0
            fi
            echo "compose $*" >> "{call_log}"
            ;;
          start)
            echo "start $*" >> "{call_log}"
            ;;
          restart)
            echo "restart $*" >> "{call_log}"
            ;;
          *)
            : ;;
        esac
        """,
    )

    runners_json = _runners_json(cp1_status=cp1_gh_status, cp2_status=cp2_gh_status)
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
{runners_json}
JSON
        elif [[ "${{path}}" == *"/actions/runs?status=queued"* ]]; then
          echo '{{"total_count":0,"workflow_runs":[]}}'
        else
          echo '{{}}'
        fi
        """,
    )

    # slack_post() always shells out to curl directly (never gh), so a real
    # curl on PATH would make a live network call to slack.com in a test.
    # Capture the `-d` payload instead of sending it -- this is what lets the
    # ALERT test assert the finding actually rode the rendered message, not
    # merely the stdout log line.
    _write_exec(
        bindir / "curl",
        f"""\
        set -euo pipefail
        payload=""
        take_next=0
        for a in "$@"; do
          if [[ "${{take_next}}" == 1 ]]; then
            payload="${{a}}"
            take_next=0
          elif [[ "${{a}}" == "-d" ]]; then
            take_next=1
          fi
        done
        if [[ -n "${{payload}}" ]]; then
          printf '%s\\n' "${{payload}}" >> "{slack_log}"
        fi
        echo '{{"ok":true}}'
        """,
    )

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
        """\
        set -euo pipefail
        args="$*"
        if [[ "${args}" == *"-d "* ]]; then
          echo "1750000000"
        elif [[ "${args}" == *"+%s"* ]]; then
          echo "1750000000"
        elif [[ "${args}" == *"-Iseconds"* ]]; then
          echo "2026-09-15T00:00:00+00:00"
        else
          echo "00:00:00"
        fi
        """,
    )


def _run_monitor(
    tmp_path: Path,
    *,
    cp1_docker_status: str | None = "Up (healthy)",
    cp2_docker_status: str | None = "Up (healthy)",
    cp1_gh_status: str | None = "online",
    cp2_gh_status: str | None = "online",
    auto_bounce: bool = False,
) -> tuple[dict[str, object], str, Path]:
    """Run the real monitor script; return (parsed state, stdout, tmp_path).

    Slack alert text is read back from ``tmp_path / "slack-messages.log"``
    (the mocked curl's captured `-d` payloads), and compose call attempts from
    ``tmp_path / "docker-calls.log"``.
    """
    _require_tools()
    bindir = tmp_path / "bin"
    state_file = tmp_path / "runner-monitor-state.json"
    fleet_config = tmp_path / "runner_fleet.yaml"
    call_log = tmp_path / "docker-calls.log"
    slack_log = tmp_path / "slack-messages.log"
    _write_fleet_config(fleet_config)
    _write_required_compose_overrides(tmp_path)
    _make_mock_bin(
        bindir,
        cp1_docker_status=cp1_docker_status,
        cp2_docker_status=cp2_docker_status,
        cp1_gh_status=cp1_gh_status,
        cp2_gh_status=cp2_gh_status,
        call_log=call_log,
        slack_log=slack_log,
    )

    env = {
        "PATH": f"{bindir}:{os.environ.get('PATH', '')}",
        "HOME": str(tmp_path),
        "RUNNER_FLEET_CONFIG_PATH": str(fleet_config),
        "SLACK_BOT_TOKEN": "xoxb-test",  # pragma: allowlist secret
        "SLACK_CHANNEL_ID": "C-test",
        "RUNNER_GITHUB_TOKEN": "ghp-test",  # pragma: allowlist secret
        "WEDGE_WATCH_REPOS": "OmniNode-ai/omnibase_infra",
        "AUTO_BOUNCE_LOCKFILE": str(tmp_path / "bounce.lock"),
        "AUTO_BOUNCE_BOUNCE_LOG": str(tmp_path / "bounce.log"),
        "AUTO_BOUNCE_VERIFY_RETRY_COUNT": "1",
        "AUTO_BOUNCE_VERIFY_RETRY_SLEEP_SECONDS": "0",
        "AUTO_BOUNCE_PER_CONTAINER_BUDGET_SECONDS": "5",
        "AUTO_BOUNCE_HARD_LIMIT_SECONDS": "5",
    }
    if auto_bounce:
        env["MONITOR_AUTO_BOUNCE"] = "1"

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
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, (
        f"monitor script exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    state = json.loads(state_file.read_text(encoding="utf-8"))
    call_log_text = call_log.read_text(encoding="utf-8") if call_log.exists() else ""
    return state, result.stdout, tmp_path


class TestCustomerPlaneAlertingPositiveControl:
    """Everything healthy, including the customer-plane pair: zero findings.

    Without this, a false 'the check fires' reading in the cases below would
    be unfalsifiable -- there would be no proof the healthy path is actually
    silent."""

    def test_all_healthy_produces_no_customer_plane_finding(
        self, tmp_path: Path
    ) -> None:
        state, _stdout, _ = _run_monitor(tmp_path)
        assert state["customer_plane_alert_present"] is False
        assert state["alert_count"] == 0
        assert CP1 not in state["unhealthy_names"]
        assert CP2 not in state["unhealthy_names"]


class TestCustomerPlaneDockerOutageAlerts:
    def test_a_stopped_customer_plane_container_is_reported_and_pages(
        self, tmp_path: Path
    ) -> None:
        state, stdout, tmp = _run_monitor(
            tmp_path, cp1_docker_status="Exited (1) 2 minutes ago"
        )
        assert f"{CP1}: Docker Exited" in state["unhealthy_names"]
        assert "[customer-plane, alert-only]" in state["unhealthy_names"]
        # This is the assertion that matters: unhealthy_list membership alone
        # does NOT page (current_alert_count is not a function of raw
        # unhealthy count). Confirm the dedicated flag actually fired, the
        # stdout log line shows a real ALERT transition (not the silent "OK"
        # branch), and the finding rode the actual rendered Slack payload.
        assert state["customer_plane_alert_present"] is True
        assert state["alert_count"] >= 1
        assert "ALERT: 1 actionable issue(s)" in stdout
        slack_log = tmp / "slack-messages.log"
        assert slack_log.exists(), "no Slack message was ever sent"
        slack_payload = slack_log.read_text(encoding="utf-8")
        assert "RUNNER ALERT" in slack_payload
        assert CP1 in slack_payload

    def test_a_missing_customer_plane_container_is_reported_and_pages(
        self, tmp_path: Path
    ) -> None:
        state, _stdout, _ = _run_monitor(
            tmp_path, cp1_docker_status=None, cp2_docker_status="Up (healthy)"
        )
        assert f"{CP1}: Docker MISSING (no container)" in state["unhealthy_names"]
        assert state["customer_plane_alert_present"] is True
        assert state["alert_count"] >= 1


class TestCustomerPlaneGithubOutageAlerts:
    def test_offline_github_registration_with_healthy_docker_is_reported(
        self, tmp_path: Path
    ) -> None:
        state, _stdout, _ = _run_monitor(tmp_path, cp2_gh_status="offline")
        assert (
            f"{CP2}: GitHub offline while Docker Up (healthy)"
            in state["unhealthy_names"]
        )
        assert state["customer_plane_alert_present"] is True
        assert state["alert_count"] >= 1

    def test_a_healthy_general_pool_is_unaffected_by_a_customer_plane_outage(
        self, tmp_path: Path
    ) -> None:
        """Positive control for scope: the general-pool loop's own count is
        untouched by a customer-plane finding -- proves the two checks are
        independent, not that one masks the other."""
        state, _stdout, _ = _run_monitor(
            tmp_path, cp1_docker_status="Exited (1) 2 minutes ago"
        )
        assert state["healthy"] == TEST_FLEET_COUNT
        assert state["online"] == TEST_FLEET_COUNT


class TestCustomerPlaneNeverBounced:
    """AC2: the widened check must never make either runner a bounce target."""

    def test_auto_bounce_never_targets_a_customer_plane_runner(
        self, tmp_path: Path
    ) -> None:
        state, _stdout, tmp = _run_monitor(
            tmp_path,
            cp1_docker_status="Exited (1) 2 minutes ago",
            cp2_docker_status="Exited (1) 2 minutes ago",
            auto_bounce=True,
        )
        # The finding is real (proves this is not a vacuous pass because
        # nothing was unhealthy)...
        assert state["customer_plane_alert_present"] is True
        assert state["remediation_target_count"] == 0
        # ...and no compose --force-recreate call named either customer-plane
        # runner was ever dispatched.
        call_log = tmp / "docker-calls.log"
        call_log_text = (
            call_log.read_text(encoding="utf-8") if call_log.exists() else ""
        )
        assert CP1 not in call_log_text
        assert CP2 not in call_log_text
