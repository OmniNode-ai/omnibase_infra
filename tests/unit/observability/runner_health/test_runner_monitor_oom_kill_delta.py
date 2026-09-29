# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19958: a runner OOM kill is actionable in the pass that sees it.

The runner monitor's only OOM signal was Docker's ``State.OOMKilled``. That
flag is sticky -- it stays true until the container restarts -- and it records
only that the container's INIT process was killed. A memcg kill of a CI job's
child process inside a runner (the 2026-09-28 .202 incident: 12 kills, every
one a pre-commit fan-out inside ``omnipc2-ci-runner-13`` or ``-15``) leaves the
flag false, the runner healthy, and a job that the 137 auto-rerun can turn
green. Nothing recorded that the kill happened.

The kernel does record it: every cgroup v2 ``memory.events`` file carries an
``oom_kill`` counter. The monitor now reads that counter for every runner
container, keeps the last value per container id, and:

  * a counter that ROSE since the previous pass is an ``OOM_KILL`` line and a
    Slack post in the same pass -- it bypasses the OMN-19169 announcement
    dwell, because a kill is an event, not a state that can flap;
  * a FLAT counter is quiet, so the same kill is never announced twice (the
    dedup key is host, container id and counter total);
  * an UNREADABLE counter is an error finding, never a zero (FAIL-not-WARN);
  * a container id seen for the first time is a baseline, read and not
    alerted, so a recreate does not replay an old container's history;
  * the fleet observation carries ``oom_kill_total`` and ``oom_kill_delta``
    per runner.

These tests drive the REAL monitor end to end, with PATH-injected mock
binaries and a fixture cgroup tree, across several passes against ONE state
directory, because a delta only exists between passes.
"""

from __future__ import annotations

import hashlib
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
BUILDER = REPO_ROOT / "scripts" / "runner_fleet_event.py"

PREFIX = "omninode-runner"
RUNNER = f"{PREFIX}-1"
CP_RUNNERS = ("omninode-customer-plane-runner-1", "omninode-customer-plane-runner-2")
VERIFY_RUNNERS = (
    "omninode-verify-runner-1",
    "omninode-verify-runner-2",
    "omninode-verify-runner-3",
)
CGROUP_PARENT = "omnirunners.slice"
FIRST_ID = "a" * 64
RECREATED_ID = "b" * 64
# The alert-only families are out of this test's scope; each gets a fixed id
# and a flat zero counter so only the general-pool runner can move.
ALERT_ONLY_IDS = {
    name: hashlib.sha256(name.encode()).hexdigest()
    for name in (*CP_RUNNERS, *VERIFY_RUNNERS)
}

pytestmark = pytest.mark.unit


def _write_exec(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _runners_json() -> str:
    runners = [
        {
            "id": 1,
            "name": RUNNER,
            "status": "online",
            "busy": False,
            "labels": [{"name": "self-hosted"}, {"name": "omnibase-ci"}],
        }
    ]
    for offset, name in enumerate((*CP_RUNNERS, *VERIFY_RUNNERS), start=2):
        runners.append(
            {
                "id": offset,
                "name": name,
                "status": "online",
                "busy": False,
                "labels": [{"name": "self-hosted"}],
            }
        )
    return json.dumps({"total_count": len(runners), "runners": runners})


class _Harness:
    """One persistent state directory, cgroup tree and sinks across passes."""

    def __init__(self, tmp_path: Path) -> None:
        resolve_modern_bash()
        for tool in ("bash", "jq", "flock", "python3"):
            if shutil.which(tool) is None:
                pytest.skip(f"{tool} not available; shell behavior test requires it")
        self.tmp = tmp_path
        self.modern_bash = resolve_modern_bash()
        self.bindir = tmp_path / "bin"
        self.bindir.mkdir()
        self.cgroup_root = tmp_path / "cgroup"
        self.state_file = tmp_path / "state" / "runner-monitor-state.json"
        self.state_file.parent.mkdir()
        self.slack_log = tmp_path / "slack.log"
        self.produced = tmp_path / "rpk-produced.jsonl"
        self.container_id_file = tmp_path / "container-id"
        self.fleet_config = tmp_path / "runner_fleet.yaml"
        self.container_id = FIRST_ID
        self.extra_env: dict[str, str] = {}
        self._write_fleet_config()
        self._write_compose_overrides()
        self._write_mocks()
        source = MONITOR_SCRIPT.read_text(encoding="utf-8")
        patched = "\n".join(
            f'STATE_FILE="{self.state_file}"'
            if line.startswith("STATE_FILE=")
            else line
            for line in source.splitlines()
        )
        self.script = tmp_path / "monitor.sh"
        self.script.write_text(patched + "\n", encoding="utf-8")

    # -- fixture state -----------------------------------------------------

    def _write_fleet_config(self) -> None:
        self.fleet_config.write_text(
            textwrap.dedent(
                f"""\
                version: "1.0"
                github_org: OmniNode-ai
                runner_host: 192.168.86.202
                runner_group: omnibase-ci
                runner_name_prefix: {PREFIX}
                expected_count: 1
                burst_count: 1
                """
            ),
            encoding="utf-8",
        )

    def _write_compose_overrides(self) -> None:
        compose_dir = self.tmp / ".omnibase" / "runners" / "docker"
        compose_dir.mkdir(parents=True)
        (compose_dir / "compose-overrides.list").write_text(
            "docker-compose.model-review-canary.yml\n", encoding="utf-8"
        )
        (compose_dir / "docker-compose.model-review-canary.yml").write_text(
            "services: {}\n", encoding="utf-8"
        )

    def events_file(self, container_id: str | None = None) -> Path:
        cid = container_id or self.container_id
        return (
            self.cgroup_root / CGROUP_PARENT / f"docker-{cid}.scope" / "memory.events"
        )

    def set_oom_kill(self, total: int, container_id: str | None = None) -> None:
        path = self.events_file(container_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"low 0\nhigh 118303\nmax 42\noom {total}\noom_kill {total}\n"
            "oom_group_kill 0\n",
            encoding="utf-8",
        )

    def remove_counter(self) -> None:
        self.events_file().unlink()

    def recreate(self, container_id: str) -> None:
        self.container_id = container_id

    def _write_mocks(self) -> None:
        cp_and_verify = " ".join((*CP_RUNNERS, *VERIFY_RUNNERS))
        alert_only_cases = "\n                    ".join(
            f"{name}) printf '%s|%s\\n' '{cid}' '{CGROUP_PARENT}' ;;"
            for name, cid in ALERT_ONLY_IDS.items()
        )
        _write_exec(
            self.bindir / "docker",
            f"""\
            set -euo pipefail
            cmd="${{1:-}}"
            case "${{cmd}}" in
              ps)
                filter=""
                for a in "$@"; do
                  case "${{a}}" in name=*) filter="${{a#name=}}" ;; esac
                done
                for n in {RUNNER} {cp_and_verify}; do
                  [[ "${{n}}" == *"${{filter}}"* ]] || continue
                  if [[ "$*" == *".Status"* ]]; then
                    printf '%s\\t%s\\n' "${{n}}" "Up (healthy)"
                  else
                    printf '%s\\n' "${{n}}"
                  fi
                done
                ;;
              inspect)
                fmt="$*"
                target="${{!#}}"
                if [[ "${{fmt}}" == *RestartCount* ]]; then
                  echo "0"
                elif [[ "${{fmt}}" == *CgroupParent* ]]; then
                  case "${{target}}" in
                    {RUNNER}) printf '%s|%s\\n' "$(cat "{self.container_id_file}")" "{CGROUP_PARENT}" ;;
                    {alert_only_cases}
                    *) echo "Error: No such object: ${{target}}" >&2; exit 1 ;;
                  esac
                else
                  echo "false"
                fi
                ;;
              logs) echo "Listening for Jobs" ;;
              exec) echo "27.0.0" ;;
              compose)
                if [[ "$*" == *" config"* ]]; then
                  if [[ "$*" == *"--services"* ]]; then
                    echo "{RUNNER}"
                  fi
                fi
                exit 0
                ;;
              *) : ;;
            esac
            """,
        )
        _write_exec(
            self.bindir / "gh",
            f"""\
            set -euo pipefail
            path=""
            for a in "$@"; do
              if [[ "${{a}}" == /* ]]; then path="${{a}}"; fi
            done
            if [[ "${{path}}" == *"/actions/runners?"* ]]; then
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
            self.bindir / "curl",
            f"""\
            set -euo pipefail
            payload=""
            take_next=0
            for a in "$@"; do
              if [[ "${{take_next}}" == 1 ]]; then payload="${{a}}"; take_next=0
              elif [[ "${{a}}" == "-d" ]]; then take_next=1; fi
            done
            [[ -n "${{payload}}" ]] && printf '%s\\n' "${{payload}}" >> "{self.slack_log}"
            echo '{{"ok":true}}'
            """,
        )
        _write_exec(
            self.bindir / "rpk",
            f"""\
            set -euo pipefail
            payload="$(cat)"
            printf '%s\\n' "${{payload}}" >> "{self.produced}"
            """,
        )
        _write_exec(
            self.bindir / "timeout",
            """\
            set -euo pipefail
            shift
            exec "$@"
            """,
        )

    # -- running -----------------------------------------------------------

    def _seed_alert_only_counters(self) -> None:
        for cid in ALERT_ONLY_IDS.values():
            path = self.events_file(cid)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("oom 0\noom_kill 0\n", encoding="utf-8")

    def run_pass(self) -> str:
        self.container_id_file.write_text(self.container_id, encoding="utf-8")
        self._seed_alert_only_counters()
        env = {
            "PATH": f"{self.bindir}:{os.environ.get('PATH', '')}",
            "HOME": str(self.tmp),
            "RUNNER_FLEET_CONFIG_PATH": str(self.fleet_config),
            "SLACK_BOT_TOKEN": "xoxb-test",  # pragma: allowlist secret
            "SLACK_CHANNEL_ID": "C-test",
            "RUNNER_GITHUB_TOKEN": "ghp-test",  # pragma: allowlist secret
            "WEDGE_WATCH_REPOS": "OmniNode-ai/omnibase_infra",
            "MONITOR_AUTO_BOUNCE": "0",
            "AUTO_BOUNCE_LOCKFILE": str(self.tmp / "bounce.lock"),
            "AUTO_BOUNCE_BOUNCE_LOG": str(self.tmp / "bounce.log"),
            "RUNNER_MONITOR_ALERT_DWELL_CYCLES": "3",
            "RUNNER_MONITOR_CGROUP_ROOT": str(self.cgroup_root),
            "RUNNER_FLEET_BROKER_CONTAINER": "",
            "KAFKA_BOOTSTRAP_SERVERS": "mock-broker:9092",
            "RUNNER_FLEET_EVENT_BUILDER": str(BUILDER),
            **self.extra_env,
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
        return result.stdout + result.stderr

    # -- reading -----------------------------------------------------------

    def oom_posts(self) -> int:
        if not self.slack_log.exists():
            return 0
        return self.slack_log.read_text(encoding="utf-8").count("[RUNNER OOM-KILL]")

    def last_row(self) -> dict[str, object]:
        lines = [
            line
            for line in self.produced.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert lines, "the monitor published no fleet observation"
        event = json.loads(lines[-1])
        rows = [row for row in event["runners"] if row["runner_name"] == RUNNER]
        assert len(rows) == 1, event
        return rows[0]

    def state(self) -> dict[str, object]:
        return json.loads(self.state_file.read_text(encoding="utf-8"))


@pytest.fixture
def harness(tmp_path: Path) -> _Harness:
    h = _Harness(tmp_path)
    h.set_oom_kill(6)
    return h


class TestCounterRiseIsActionableInOnePass:
    """AC1 -- a counter rise is actionable in the pass that sees it."""

    def test_first_sighting_is_a_baseline_not_an_alert(self, harness: _Harness) -> None:
        out = harness.run_pass()
        assert "OOM_KILL runner=" not in out, out
        assert harness.oom_posts() == 0
        row = harness.last_row()
        assert row["oom_kill_total"] == 6
        assert row["oom_kill_delta"] == 0

    def test_a_rise_logs_oom_kill_with_delta_and_total(self, harness: _Harness) -> None:
        harness.run_pass()
        harness.set_oom_kill(7)
        out = harness.run_pass()
        assert f"OOM_KILL runner={RUNNER} delta=1 total=7" in out, out

    def test_a_rise_posts_to_slack_in_the_same_pass_bypassing_the_dwell(
        self, harness: _Harness
    ) -> None:
        harness.run_pass()
        harness.set_oom_kill(7)
        harness.run_pass()
        assert harness.oom_posts() == 1, (
            "one kill must reach Slack in the pass that saw it, not after the "
            "three-pass announcement dwell"
        )
        sink = harness.slack_log.read_text(encoding="utf-8")
        assert RUNNER in sink
        assert "delta=1" in sink

    def test_the_fleet_observation_carries_total_and_delta(
        self, harness: _Harness
    ) -> None:
        harness.run_pass()
        harness.set_oom_kill(7)
        harness.run_pass()
        row = harness.last_row()
        assert row["oom_kill_total"] == 7
        assert row["oom_kill_delta"] == 1

    def test_a_multi_kill_rise_reports_the_whole_delta(self, harness: _Harness) -> None:
        harness.run_pass()
        harness.set_oom_kill(9)
        out = harness.run_pass()
        assert f"OOM_KILL runner={RUNNER} delta=3 total=9" in out, out


class TestFlatCounterIsQuiet:
    """AC2 -- the same total is never announced twice."""

    def test_the_pass_after_a_kill_reports_delta_zero(self, harness: _Harness) -> None:
        harness.run_pass()
        harness.set_oom_kill(7)
        harness.run_pass()
        out = harness.run_pass()
        assert "OOM_KILL runner=" not in out, out
        assert harness.oom_posts() == 1
        assert harness.last_row()["oom_kill_delta"] == 0

    def test_a_recreated_container_is_a_new_baseline_not_a_replay(
        self, harness: _Harness
    ) -> None:
        harness.run_pass()
        harness.recreate(RECREATED_ID)
        harness.set_oom_kill(3)
        out = harness.run_pass()
        assert "OOM_KILL runner=" not in out, out
        assert harness.oom_posts() == 0
        row = harness.last_row()
        assert row["oom_kill_total"] == 3
        assert row["oom_kill_delta"] == 0


class TestUnreadableCounterIsLoud:
    """AC3 -- an unreadable counter is an error, never a zero."""

    def test_a_missing_memory_events_file_is_reported_for_that_runner(
        self, harness: _Harness
    ) -> None:
        harness.run_pass()
        harness.remove_counter()
        out = harness.run_pass()
        assert f"OOM_COUNTER_UNREADABLE runner={RUNNER}" in out, out
        assert "OOM_KILL runner=" not in out

    def test_an_unreadable_counter_is_an_actionable_finding(
        self, harness: _Harness
    ) -> None:
        harness.run_pass()
        baseline = int(harness.state()["alert_count"])  # type: ignore[arg-type]
        harness.remove_counter()
        harness.run_pass()
        assert int(harness.state()["alert_count"]) == baseline + 1  # type: ignore[arg-type]

    def test_an_unreadable_counter_is_published_as_null_not_zero(
        self, harness: _Harness
    ) -> None:
        harness.run_pass()
        harness.remove_counter()
        harness.run_pass()
        row = harness.last_row()
        assert row["oom_kill_total"] is None
        assert row["oom_kill_delta"] is None

    def test_a_counter_that_returns_after_an_outage_alerts_the_missed_kills(
        self, harness: _Harness
    ) -> None:
        harness.run_pass()
        harness.remove_counter()
        harness.run_pass()
        harness.set_oom_kill(8)
        out = harness.run_pass()
        assert f"OOM_KILL runner={RUNNER} delta=2 total=8" in out, out


class TestDisabledScanIsSaidAloud:
    """The off switch exists for harnesses without a cgroup tree; it is never silent."""

    def test_a_disabled_scan_logs_that_kills_are_not_watched(
        self, harness: _Harness
    ) -> None:
        harness.extra_env = {"RUNNER_MONITOR_OOM_KILL_SCAN": "false"}
        out = harness.run_pass()
        assert "OOM-kill counter scan DISABLED" in out, out
        assert "NOT watched" in out
        assert harness.last_row()["oom_kill_total"] is None
