# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Behavioral tests for runner-monitor.sh's Slack transition dwell (OMN-19169).

Measured on 2026-09-22 in the notifications channel: 46 of 165 message blocks
over 15 hours were this monitor alternating ``[RUNNER ALERT]`` /
``[RUNNER RECOVERED]``, every ALERT carrying ``wedge=1`` against a fleet whose
own log read ``60/60 healthy, 0 actionable``. Two stacked causes:

  * the transition arms are pure edge triggers -- one observation of a changed
    ``alert_count`` is enough to page -- so an input that oscillates on the
    monitor's own cadence pages twice per oscillation, and
  * two cron entries run this same script against one shared ``STATE_FILE``,
    so neither invocation sees a coherent previous state and one real
    transition can be announced twice.

The fix is one mechanism for both: the state file records what Slack has
already been TOLD (``announced_alert_count``), and a changed observation must
repeat for ``RUNNER_MONITOR_ALERT_DWELL_CYCLES`` consecutive observations
before it is announced. Because the dwell counter lives in the shared state
file, the two cron invocations cooperate on one announced state instead of
racing it.

These tests drive the REAL shell script end-to-end with PATH-injected mock
binaries -- the same pattern as the sibling auto-bounce / wedge-detection /
customer-plane modules -- and run it repeatedly against ONE persistent state
file, because a dwell is only observable across cycles.

The controlled input is fleet-count drift: ``docker compose config --services``
either agrees with ``expected_count`` (no finding) or declares one extra runner
service (exactly one finding, ``FLEET_COUNT_DRIFT``). That lever is used rather
than a wedge because it is deterministic and adds exactly 1 to
``current_alert_count``, which is the quantity the transition arms compare.
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
TEST_FLEET_COUNT = 1
CP1 = "omninode-customer-plane-runner-1"
CP2 = "omninode-customer-plane-runner-2"
VERIFY_RUNNERS = (
    "omninode-verify-runner-1",
    "omninode-verify-runner-2",
    "omninode-verify-runner-3",
)

DWELL = 3

pytestmark = pytest.mark.unit


def _require_tools() -> None:
    resolve_modern_bash()
    for tool in ("bash", "jq", "flock"):
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} not available; shell behavior test requires it")


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


def _write_required_compose_overrides(home: Path) -> None:
    compose_dir = home / ".omnibase" / "runners" / "docker"
    compose_dir.mkdir(parents=True, exist_ok=True)
    (compose_dir / "compose-overrides.list").write_text(
        "docker-compose.model-review-canary.yml\n", encoding="utf-8"
    )
    (compose_dir / "docker-compose.model-review-canary.yml").write_text(
        "services: {}\n", encoding="utf-8"
    )


def _runners_json() -> str:
    runners = [
        {
            "name": f"{PREFIX}-{i}",
            "status": "online",
            "busy": False,
            "labels": [{"name": "self-hosted"}, {"name": "omnibase-ci"}],
        }
        for i in range(1, TEST_FLEET_COUNT + 1)
    ]
    for name in (CP1, CP2, *VERIFY_RUNNERS):
        runners.append(
            {
                "name": name,
                "status": "online",
                "busy": False,
                "labels": [{"name": "self-hosted"}],
            }
        )
    return json.dumps({"total_count": len(runners), "runners": runners})


def _make_mock_bin(bindir: Path, *, drift_flag: Path, slack_log: Path) -> None:
    """Mock docker / gh / curl / timeout / date.

    Every runner in every family is staged healthy and online throughout, so
    the ONLY finding these tests can produce is fleet-count drift, and it is
    switched on and off between cycles by the presence of ``drift_flag``.
    """
    bindir.mkdir(parents=True, exist_ok=True)

    cp_ps = "\n            ".join(
        f'printf "%s\\t%s\\n" "{n}" "Up (healthy)"' for n in (CP1, CP2)
    )
    vr_ps = "\n            ".join(
        f'printf "%s\\t%s\\n" "{n}" "Up (healthy)"' for n in VERIFY_RUNNERS
    )
    extra_services = "\n                ".join(
        f'echo "{n}"' for n in (CP1, CP2, *VERIFY_RUNNERS)
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
              {cp_ps}
            elif [[ "${{filter}}" == "omninode-verify-runner-" ]]; then
              {vr_ps}
            else
              for i in $(seq 1 {TEST_FLEET_COUNT}); do
                printf '%s\\t%s\\n' "{PREFIX}-${{i}}" "Up (healthy)"
              done
            fi
            ;;
          inspect)
            fmt="$*"
            if [[ "${{fmt}}" == *RestartCount* ]]; then echo "0"; else echo "false"; fi
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
                {extra_services}
                if [[ -f "{drift_flag}" ]]; then
                  echo "{PREFIX}-$(( {TEST_FLEET_COUNT} + 1 ))"
                fi
              fi
              exit 0
            fi
            ;;
          *)
            : ;;
        esac
        """,
    )

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
          echo "2026-09-22T00:00:00+00:00"
        else
          echo "00:00:00"
        fi
        """,
    )


class _Harness:
    """One persistent STATE_FILE and slack sink across many monitor cycles."""

    def __init__(self, tmp_path: Path, dwell: int = DWELL) -> None:
        _require_tools()
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.tmp = tmp_path
        self.dwell = dwell
        self.bindir = tmp_path / "bin"
        self.state_file = tmp_path / "runner-monitor-state.json"
        self.slack_log = tmp_path / "slack-messages.log"
        self.drift_flag = tmp_path / "fleet-drift.on"
        self.fleet_config = tmp_path / "runner_fleet.yaml"
        _write_fleet_config(self.fleet_config)
        _write_required_compose_overrides(tmp_path)
        _make_mock_bin(
            self.bindir, drift_flag=self.drift_flag, slack_log=self.slack_log
        )
        self.modern_bash = resolve_modern_bash()
        self.script = tmp_path / "monitor.sh"
        source = MONITOR_SCRIPT.read_text(encoding="utf-8")
        patched = "\n".join(
            f'STATE_FILE="{self.state_file}"'
            if line.startswith("STATE_FILE=")
            else line
            for line in source.splitlines()
        )
        self.script.write_text(patched + "\n", encoding="utf-8")

    def cycle(self, *, drift: bool, invocations: int = 1) -> str:
        """Run the monitor ``invocations`` times for one simulated cycle."""
        if drift:
            self.drift_flag.write_text("1", encoding="utf-8")
        elif self.drift_flag.exists():
            self.drift_flag.unlink()
        stdout = ""
        for _ in range(invocations):
            env = {
                "PATH": f"{self.bindir}:{os.environ.get('PATH', '')}",
                "HOME": str(self.tmp),
                "RUNNER_FLEET_CONFIG_PATH": str(self.fleet_config),
                "SLACK_BOT_TOKEN": "xoxb-test",  # pragma: allowlist secret
                "SLACK_CHANNEL_ID": "C-test",
                "RUNNER_GITHUB_TOKEN": "ghp-test",  # pragma: allowlist secret
                "WEDGE_WATCH_REPOS": "OmniNode-ai/omnibase_infra",
                "RUNNER_MONITOR_ALERT_DWELL_CYCLES": str(self.dwell),
            }
            result = subprocess.run(
                [self.modern_bash, str(self.script)],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            assert result.returncode == 0, (
                f"monitor exited {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
            stdout += result.stdout
        return stdout

    @property
    def state(self) -> dict[str, object]:
        return json.loads(self.state_file.read_text(encoding="utf-8"))

    def _sink(self) -> str:
        if not self.slack_log.exists():
            return ""
        return self.slack_log.read_text(encoding="utf-8")

    def alerts(self) -> int:
        return self._sink().count("[RUNNER ALERT]")

    def recovered(self) -> int:
        return self._sink().count("[RUNNER RECOVERED]")

    def posts(self) -> int:
        """Slack payloads captured, not lines -- the mocked curl appends a
        pretty-printed JSON body per post, so a line count over-reports by an
        order of magnitude."""
        return self._sink().count('"channel"')


class TestDwellPositiveControl:
    """Without this, a silent monitor would satisfy every suppression test."""

    def test_a_sustained_finding_still_pages_once_the_dwell_is_met(
        self, tmp_path: Path
    ) -> None:
        h = _Harness(tmp_path)
        for _ in range(DWELL):
            h.cycle(drift=True)
        assert h.alerts() == 1, (
            "a finding held for the full dwell must page exactly once; "
            f"sink was:\n{h._sink()}"
        )
        assert h.state["alert_count"] >= 1

    def test_the_alert_lands_on_the_dwell_cycle_and_not_before(
        self, tmp_path: Path
    ) -> None:
        h = _Harness(tmp_path)
        for cycle in range(1, DWELL):
            h.cycle(drift=True)
            assert h.alerts() == 0, (
                f"paged on cycle {cycle} of a {DWELL}-cycle dwell; "
                f"sink was:\n{h._sink()}"
            )
        h.cycle(drift=True)
        assert h.alerts() == 1


class TestFlapSuppression:
    """AC1: a finding that appears and clears inside the dwell never pages."""

    def test_alternating_finding_produces_no_slack_post_at_all(
        self, tmp_path: Path
    ) -> None:
        h = _Harness(tmp_path)
        for _ in range(6):
            h.cycle(drift=True)
            h.cycle(drift=False)
        assert h.posts() == 0, (
            "six on/off oscillations must produce zero Slack posts; "
            f"sink was:\n{h._sink()}"
        )

    def test_suppression_is_recorded_in_the_log(self, tmp_path: Path) -> None:
        h = _Harness(tmp_path)
        stdout = h.cycle(drift=True)
        assert "DWELL" in stdout, (
            "a suppressed transition must say so on stdout, or the operator "
            f"cannot tell suppression from a dead monitor; stdout:\n{stdout}"
        )


class TestSustainedTransitionsAnnounceOnce:
    """AC2: one ALERT on the way in, one RECOVERED on the way out."""

    def test_four_on_then_four_off_posts_exactly_two_messages(
        self, tmp_path: Path
    ) -> None:
        h = _Harness(tmp_path)
        for _ in range(4):
            h.cycle(drift=True)
        for _ in range(4):
            h.cycle(drift=False)
        assert h.alerts() == 1, f"sink was:\n{h._sink()}"
        assert h.recovered() == 1, f"sink was:\n{h._sink()}"
        assert h.posts() == 2, f"sink was:\n{h._sink()}"


class TestInterleavedInvocationsShareOneAnnouncedState:
    """AC3: the two cron entries cannot announce the same transition twice."""

    def test_two_invocations_per_cycle_do_not_double_announce(
        self, tmp_path: Path
    ) -> None:
        solo = _Harness(tmp_path / "solo")
        for _ in range(6):
            solo.cycle(drift=True)
        for _ in range(6):
            solo.cycle(drift=False)

        paired = _Harness(tmp_path / "paired")
        for _ in range(6):
            paired.cycle(drift=True, invocations=2)
        for _ in range(6):
            paired.cycle(drift=False, invocations=2)

        assert paired.posts() == solo.posts(), (
            "a second interleaved invocation per cycle must not add posts; "
            f"solo sink:\n{solo._sink()}\npaired sink:\n{paired._sink()}"
        )
        assert paired.alerts() == 1
        assert paired.recovered() == 1
