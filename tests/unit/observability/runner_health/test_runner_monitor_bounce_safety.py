# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Behavioral tests for runner-monitor.sh bounce-safety hardening (OMN-16947).

Root cause of the 2026-08-29 self-inflicted fleet outage: ``*/10`` auto-bounce
fired against a freshly-recovered, 100%-healthy 88-runner fleet and
force-recreated ALL 88 at once. Recovering the resulting orphaned/``Dead``
container state needed two ``systemctl restart docker`` cycles.

The chain, reconstructed from the live monitor state file
(``healthy: 88, unhealthy_count: 1, wedge_count: 1, alert_count: 88,
remediation_target_count: 88``) and the GitHub Actions queue:

  1. ``oldest_queued_job_age_seconds()`` had no upper bound. Abandoned queued
     workflow runs (``OmniNode-ai/omnibase_infra`` run 32215978710, queued
     2026-08-19; ``omniclaude`` run 29019863632, queued 2026-07-09) are never
     reaped by GitHub, so it returned ~895_303s (10.4 days) on EVERY tick,
     forever. With the fleet idle (``busy_count == 0``) the SILENT-WEDGE
     predicate was therefore permanently true.
  2. ``collect_remediation_targets()`` expanded a wedge finding to
     ``seq 1 $EXPECTED_RUNNERS`` -- every runner in the fleet, unconditionally,
     including runners that were present, registered, and ``Listening for
     Jobs``.
  3. ``current_alert_count`` is derived from the target count, so a single
     false-positive wedge read out as "88 actionable" and ``auto_bounce()``
     force-recreated the whole fleet in one compose call.

These tests drive the REAL shell script end-to-end against PATH-injected mock
binaries and pin the four invariants that break the chain:

  * a runner that is present, registered, and listening is NEVER a bounce
    target -- a wedge is an alert-only, fleet-level finding that contributes
    exactly one actionable count and zero recreate targets,
  * a queued job older than ``WEDGE_QUEUE_AGE_MAX_SECONDS`` is a zombie, not
    wedge evidence,
  * no single tick can ever recreate more than
    ``AUTO_BOUNCE_MAX_TARGETS_PER_TICK`` containers, enforced inside
    ``auto_bounce()`` itself so no upstream collection bug can mass-recreate,
  * the bounce path is sequential and verified: each target must reach
    ``running`` before the next is touched, and a target that fails to recover
    halts the batch instead of cascading.

Plus the two silent-failure classes that preceded the outage:

  * compose env interpolation failure (the 6-day
    ``DEPLOY_RUNNER_OMNI_HOME is missing a value`` no-op) must fail the tick
    LOUDLY and block the bounce, not log "AUTO-BOUNCE dispatched" and do
    nothing,
  * ``expected_count`` drifting from the compose-declared runner service count
    (the 72-vs-88 gap that hid 16 runners from monitor scope) must alert.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import textwrap
import time
from collections.abc import Iterable
from pathlib import Path

import pytest

from tests.unit.observability.runner_health._resolve_modern_bash import (
    resolve_modern_bash,
)

REPO_ROOT = Path(__file__).parents[4]
MONITOR_SCRIPT = REPO_ROOT / "docker" / "runners" / "runner-monitor.sh"

PREFIX = "omninode-runner"
NOW = 1_750_000_000

pytestmark = pytest.mark.unit


def _require_tools() -> None:
    # `flock` is deliberately NOT required: runner-monitor.sh carries a `mkdir`
    # lock fallback for exactly this case, and these tests assert bounce-target
    # *selection*, not lock contention. Requiring flock here would skip the
    # whole module on macOS and leave the OMN-16947 invariants unproven on
    # every developer machine.
    resolve_modern_bash()
    for tool in ("bash", "jq"):
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} not available; shell behavior test requires it")


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


class Scenario:
    """Filesystem-backed control surface for the mocked docker/gh binaries.

    The mocks are dumb shell that read these files, so a test declares the
    world (which containers exist, what compose does) in Python and the real
    script observes it exactly as it would observe a live host.
    """

    def __init__(self, root: Path, *, fleet_count: int) -> None:
        self.root = root
        self.fleet_count = fleet_count
        root.mkdir(parents=True, exist_ok=True)
        (root / "status").mkdir(exist_ok=True)
        self.calls = root / "calls.log"
        self.calls.write_text("", encoding="utf-8")
        # Defaults: whole fleet present, Up (healthy), inspect says running.
        self.set_present(range(1, fleet_count + 1))
        self.set_compose_services(range(1, fleet_count + 1))
        (root / "compose_config_rc").write_text("0", encoding="utf-8")
        (root / "compose_config_stderr").write_text("", encoding="utf-8")
        # `docker compose up` heals its named targets by default.
        (root / "recreate_heals").write_text("1", encoding="utf-8")
        self.set_queued_jobs([])

    def set_present(
        self, indices: Iterable[int], status: str = "Up 3 hours (healthy)"
    ) -> None:
        lines = []
        for i in indices:
            lines.append(f"{PREFIX}-{i}\t{status}")
            (self.root / "status" / f"{PREFIX}-{i}").write_text(
                "running", encoding="utf-8"
            )
        (self.root / "ps.tsv").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    def set_missing(self, indices: Iterable[int]) -> None:
        """Remove containers entirely (the MISSING (no container) class)."""
        present = [
            line
            for line in (self.root / "ps.tsv").read_text(encoding="utf-8").splitlines()
            if line and line.split("\t")[0] not in {f"{PREFIX}-{i}" for i in indices}
        ]
        (self.root / "ps.tsv").write_text(
            "\n".join(present) + ("\n" if present else ""), encoding="utf-8"
        )
        for i in indices:
            (self.root / "status" / f"{PREFIX}-{i}").write_text(
                "missing", encoding="utf-8"
            )

    def set_compose_services(self, indices: Iterable[int]) -> None:
        names = ["omninode-deploy-runner"] + [f"{PREFIX}-{i}" for i in indices]
        (self.root / "compose_services.txt").write_text(
            "\n".join(names) + "\n", encoding="utf-8"
        )

    def break_compose_interpolation(self, message: str) -> None:
        (self.root / "compose_config_rc").write_text("1", encoding="utf-8")
        (self.root / "compose_config_stderr").write_text(message, encoding="utf-8")

    def set_recreate_heals(self, heals: bool) -> None:
        (self.root / "recreate_heals").write_text(
            "1" if heals else "0", encoding="utf-8"
        )

    def never_heals(self, indices: Iterable[int]) -> None:
        """These containers stay non-running no matter how often they're started."""
        (self.root / "never_heal.txt").write_text(
            "\n".join(f"{PREFIX}-{i}" for i in indices) + "\n", encoding="utf-8"
        )

    def set_queued_jobs(self, ages_seconds: list[int]) -> None:
        """Declare self-hosted jobs queued for N seconds as of NOW."""
        runs = {
            "total_count": len(ages_seconds),
            "workflow_runs": [{"id": 1000 + idx} for idx, _ in enumerate(ages_seconds)],
        }
        (self.root / "queued_runs.json").write_text(json.dumps(runs), encoding="utf-8")
        for idx, age in enumerate(ages_seconds):
            jobs = {
                "jobs": [
                    {
                        "status": "queued",
                        "labels": ["self-hosted", "omnibase-ci"],
                        # The date mock resolves `date -d <created_at>` by
                        # reading the epoch straight out of this string.
                        "created_at": f"EPOCH:{NOW - age}",
                    }
                ]
            }
            (self.root / f"queued_jobs_{1000 + idx}.json").write_text(
                json.dumps(jobs), encoding="utf-8"
            )

    def runners_json(self, *, status: str = "online", busy: bool = False) -> str:
        return json.dumps(
            {
                "total_count": self.fleet_count,
                "runners": [
                    {
                        "name": f"{PREFIX}-{i}",
                        "status": status,
                        "busy": busy,
                        "labels": [
                            {"name": "self-hosted"},
                            {"name": "omnibase-ci"},
                        ],
                    }
                    for i in range(1, self.fleet_count + 1)
                ],
            }
        )

    def compose_recreate_targets(self) -> list[list[str]]:
        """Each `docker compose ... up` call's target service list, in order."""
        batches = []
        for line in self.calls.read_text(encoding="utf-8").splitlines():
            if not line.startswith("compose ") or " up " not in line:
                continue
            tokens = line.split()
            targets = [t for t in tokens if t.startswith(f"{PREFIX}-")]
            if targets:
                batches.append(targets)
        return batches

    def all_recreated(self) -> list[str]:
        return [svc for batch in self.compose_recreate_targets() for svc in batch]


def _make_mock_bin(bindir: Path, scen: Scenario) -> None:
    bindir.mkdir(parents=True, exist_ok=True)
    root = scen.root

    _write_exec(
        bindir / "docker",
        f"""\
        set -uo pipefail
        SCEN="{root}"
        cmd="${{1:-}}"
        case "${{cmd}}" in
          ps)
            cat "${{SCEN}}/ps.tsv" 2>/dev/null || true
            ;;
          inspect)
            fmt="$*"
            name="${{fmt##* }}"
            if [[ "${{fmt}}" == *OOMKilled* ]]; then
              echo "false"
            elif [[ "${{fmt}}" == *RestartCount* ]]; then
              echo "0"
            elif [[ "${{fmt}}" == *State.Status* ]]; then
              cat "${{SCEN}}/status/${{name}}" 2>/dev/null || echo "missing"
            else
              echo "false"
            fi
            ;;
          logs)
            echo "Listening for Jobs"
            ;;
          exec)
            echo "27.0.0"
            ;;
          start)
            name="${{2}}"
            echo "start $*" >> "${{SCEN}}/calls.log"
            if ! grep -qx "${{name}}" "${{SCEN}}/never_heal.txt" 2>/dev/null; then
              echo "running" > "${{SCEN}}/status/${{name}}"
            fi
            ;;
          compose)
            shift
            sub=""
            for a in "$@"; do
              case "${{a}}" in
                config|up|down|ps) sub="${{a}}"; break ;;
              esac
            done
            if [[ "${{sub}}" == "config" ]]; then
              if [[ "$*" == *"--services"* ]]; then
                cat "${{SCEN}}/compose_services.txt"
                exit 0
              fi
              rc="$(cat "${{SCEN}}/compose_config_rc")"
              if [[ "${{rc}}" != "0" ]]; then
                cat "${{SCEN}}/compose_config_stderr" >&2
              fi
              echo "compose config $*" >> "${{SCEN}}/calls.log"
              exit "${{rc}}"
            fi
            echo "compose $*" >> "${{SCEN}}/calls.log"
            if [[ "${{sub}}" == "up" ]] && [[ "$(cat "${{SCEN}}/recreate_heals")" == "1" ]]; then
              for a in "$@"; do
                case "${{a}}" in
                  {PREFIX}-*)
                    if ! grep -qx "${{a}}" "${{SCEN}}/never_heal.txt" 2>/dev/null; then
                      echo "running" > "${{SCEN}}/status/${{a}}"
                    fi
                    ;;
                esac
              done
            fi
            ;;
          *)
            : ;;
        esac
        exit 0
        """,
    )

    _write_exec(
        bindir / "gh",
        f"""\
        set -uo pipefail
        SCEN="{root}"
        path=""
        for a in "$@"; do
          if [[ "${{a}}" == /* ]]; then path="${{a}}"; fi
        done
        if [[ "$*" == *"registration-token"* ]]; then
          echo "mock-registration-token"
        elif [[ "${{path}}" == *"/actions/runners?"* ]]; then
          cat "${{SCEN}}/runners.json"
        elif [[ "${{path}}" == *"/actions/runs?status=queued"* ]]; then
          cat "${{SCEN}}/queued_runs.json"
        elif [[ "${{path}}" == *"/actions/runs/"*"/jobs"* ]]; then
          rid="${{path#*/actions/runs/}}"
          rid="${{rid%%/jobs*}}"
          cat "${{SCEN}}/queued_jobs_${{rid}}.json" 2>/dev/null || echo '{{"jobs":[]}}'
        else
          echo '{{}}'
        fi
        exit 0
        """,
    )

    _write_exec(
        bindir / "curl",
        """\
        set -uo pipefail
        echo '{"ok":true}'
        exit 0
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

    # `date -u -d "EPOCH:<n>" +%s` -> <n>. Everything else is frozen at NOW so
    # queued-job ages are exactly what the scenario declared.
    _write_exec(
        bindir / "date",
        f"""\
        set -uo pipefail
        args="$*"
        if [[ "${{args}}" == *"EPOCH:"* ]]; then
          rest="${{args#*EPOCH:}}"
          echo "${{rest%% *}}"
        elif [[ "${{args}}" == *"+%s"* ]]; then
          echo "{NOW}"
        elif [[ "${{args}}" == *"-Iseconds"* ]]; then
          echo "2026-08-29T00:00:00+00:00"
        else
          echo "00:00:00"
        fi
        exit 0
        """,
    )


def _write_fleet_config(path: Path, *, expected: int, burst: int | None = None) -> None:
    path.write_text(
        textwrap.dedent(
            f"""\
            version: "1.0"
            github_org: OmniNode-ai
            runner_host: 192.168.86.201
            runner_group: omnibase-ci
            runner_name_prefix: {PREFIX}
            expected_count: {expected}
            burst_count: {burst if burst is not None else expected}
            """
        ),
        encoding="utf-8",
    )


def _run_monitor(
    tmp_path: Path,
    scen: Scenario,
    *,
    expected_count: int | None = None,
    runner_status: str = "online",
    runner_busy: bool = False,
    extra_env: dict[str, str] | None = None,
) -> tuple[dict[str, object], str]:
    """Run the real monitor script; return (parsed state file, stdout+stderr)."""
    _require_tools()
    bindir = tmp_path / "bin"
    (scen.root / "runners.json").write_text(
        scen.runners_json(status=runner_status, busy=runner_busy), encoding="utf-8"
    )
    _make_mock_bin(bindir, scen)

    fleet_config = tmp_path / "runner_fleet.yaml"
    _write_fleet_config(fleet_config, expected=expected_count or scen.fleet_count)
    state_file = tmp_path / "runner-monitor-state.json"
    _write_required_compose_overrides(tmp_path)

    env = {
        "PATH": f"{bindir}:{os.environ.get('PATH', '')}",
        "HOME": str(tmp_path),
        "RUNNER_FLEET_CONFIG_PATH": str(fleet_config),
        "RUNNER_MONITOR_STATE_FILE": str(state_file),
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_CHANNEL_ID": "C-test",
        "RUNNER_GITHUB_TOKEN": "ghp-test",
        "WEDGE_WATCH_REPOS": "OmniNode-ai/omnibase_infra",
        "MONITOR_AUTO_BOUNCE": "1",
        "AUTO_BOUNCE_VERIFY_RETRY_COUNT": "2",
        "AUTO_BOUNCE_VERIFY_RETRY_SLEEP_SECONDS": "0",
        "AUTO_BOUNCE_PER_CONTAINER_BUDGET_SECONDS": "5",
        "AUTO_BOUNCE_HARD_LIMIT_SECONDS": "5",
        "AUTO_BOUNCE_LOCKFILE": str(tmp_path / "bounce.lock"),
        "AUTO_BOUNCE_BOUNCE_LOG": str(tmp_path / "bounce.log"),
    }
    if extra_env:
        env.update(extra_env)

    modern_bash = resolve_modern_bash()
    proc = subprocess.run(
        [modern_bash, str(MONITOR_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = proc.stdout + proc.stderr
    # auto_bounce dispatches a background subshell; wait for it to settle.
    _wait_for_bounce_quiescence(tmp_path)
    state: dict[str, object] = {}
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
    bounce_log = tmp_path / "bounce.log"
    if bounce_log.exists():
        output += "\n" + bounce_log.read_text(encoding="utf-8")
    return state, output


def _wait_for_bounce_quiescence(tmp_path: Path, timeout: float = 120.0) -> None:
    """Block until the detached bounce subshell releases its lock.

    ``auto_bounce()`` acquires the lock in the foreground *before* forking, so
    "lock still held" is a sound proxy for "bounce still running" under both
    the flock and the mkdir fallback path.
    """
    lock = tmp_path / "bounce.lock"
    lockdir = tmp_path / "bounce.lock.d"
    have_flock = shutil.which("flock") is not None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if lockdir.exists():
            time.sleep(0.05)
            continue
        if have_flock and lock.exists():
            probe = subprocess.run(
                ["flock", "-n", str(lock), "true"],
                capture_output=True,
                check=False,
            )
            if probe.returncode != 0:
                time.sleep(0.05)
                continue
        # Released. Give the subshell a beat to flush its final log write.
        time.sleep(0.2)
        return
    raise AssertionError("auto_bounce subshell did not finish within timeout")


# ---------------------------------------------------------------------------
# 1. A healthy, registered, listening runner is NEVER a bounce target
# ---------------------------------------------------------------------------


def test_wedge_on_healthy_fleet_dispatches_zero_recreates(tmp_path: Path) -> None:
    """The exact 2026-08-29 shape: 88/88 healthy + one wedge finding.

    Before OMN-16947 this produced ``remediation_target_count == 88`` and a
    single compose call force-recreating the entire fleet. A wedge is a
    fleet-level *alert*, not a licence to recreate runners that are present,
    registered, and listening.
    """
    scen = Scenario(tmp_path / "scen", fleet_count=8)
    # A job queued 20 minutes -- past WEDGE_QUEUE_AGE_SECONDS (600) but well
    # inside the zombie ceiling, so this is a genuine live wedge signal.
    scen.set_queued_jobs([1_200])

    state, output = _run_monitor(tmp_path, scen)

    assert state["wedge_count"] == 1, f"expected a wedge finding; got {state}"
    assert state["healthy"] == 8
    assert state["remediation_target_count"] == 0, (
        "a wedge must contribute ZERO auto-bounce targets"
    )
    assert state["alert_count"] == 1, (
        "a wedge is exactly one actionable finding, not one-per-runner"
    )
    assert scen.all_recreated() == [], (
        "no container may be force-recreated when every runner is present, "
        f"registered and listening; got {scen.all_recreated()}"
    )
    assert "WEDGE" in output.upper()


def test_healthy_runner_is_never_in_the_bounce_set(tmp_path: Path) -> None:
    """Only runners with hard per-runner failure evidence get recreated."""
    scen = Scenario(tmp_path / "scen", fleet_count=6)
    scen.set_missing([3])
    scen.set_queued_jobs([1_200])  # wedge also firing, to stack the two paths

    _, _ = _run_monitor(tmp_path, scen)

    recreated = scen.all_recreated()
    assert recreated == [f"{PREFIX}-3"], (
        "only the missing container is actionable; the 5 healthy runners must "
        f"not be touched. got {recreated}"
    )


# ---------------------------------------------------------------------------
# 2. Zombie queued jobs are not wedge evidence
# ---------------------------------------------------------------------------


def test_queued_job_older_than_ceiling_is_a_zombie_not_a_wedge(
    tmp_path: Path,
) -> None:
    """A 10-day-old queued run is abandoned, not proof the fleet is wedged.

    Live evidence: ``omnibase_infra`` run 32215978710 has sat ``queued`` since
    2026-08-19, so the unbounded age check reported a wedge on every tick for
    ten days straight.
    """
    scen = Scenario(tmp_path / "scen", fleet_count=4)
    scen.set_queued_jobs([895_303])  # 10.4 days, the observed value

    state, output = _run_monitor(tmp_path, scen)

    assert state["wedge_count"] == 0, (
        "a queued job past WEDGE_QUEUE_AGE_MAX_SECONDS must not pin the wedge "
        f"signal on; got {state}"
    )
    assert state["zombie_queued_count"] == 1
    assert scen.all_recreated() == []
    assert "ZOMBIE" in output.upper()


def test_fresh_and_zombie_queued_jobs_together_still_detect_the_wedge(
    tmp_path: Path,
) -> None:
    """A zombie must not mask a real wedge signal from a fresh queued job."""
    scen = Scenario(tmp_path / "scen", fleet_count=4)
    scen.set_queued_jobs([895_303, 1_800])

    state, _ = _run_monitor(tmp_path, scen)

    assert state["wedge_count"] == 1
    assert state["zombie_queued_count"] == 1


# ---------------------------------------------------------------------------
# 3. Hard per-tick cap on the bounce set
# ---------------------------------------------------------------------------


def test_bounce_set_is_capped_per_tick(tmp_path: Path) -> None:
    """Even with 10 genuinely-dead runners, one tick recreates at most N."""
    scen = Scenario(tmp_path / "scen", fleet_count=12)
    scen.set_missing(range(1, 11))

    state, output = _run_monitor(
        tmp_path, scen, extra_env={"AUTO_BOUNCE_MAX_TARGETS_PER_TICK": "3"}
    )

    recreated = scen.all_recreated()
    assert len(recreated) == 3, (
        f"cap is 3 targets per tick; {len(recreated)} were recreated: {recreated}"
    )
    # The alert must still report the TRUE scope, not the capped scope --
    # under-reporting the outage is how the 72-vs-88 gap stayed invisible.
    assert state["remediation_target_count"] == 10
    assert "CAP" in output.upper() or "capped" in output


def test_cap_is_enforced_inside_auto_bounce_not_only_at_collection(
    tmp_path: Path,
) -> None:
    """The cap is the last line of defence, below any collection logic.

    The 2026-08-29 outage came from a *collection* bug. A cap that only lives
    in ``collect_remediation_targets()`` would not have stopped it, so the
    truncation must happen inside ``auto_bounce()`` against whatever list it
    is handed.
    """
    script = MONITOR_SCRIPT.read_text(encoding="utf-8")
    start = script.index("auto_bounce() {")
    end = script.index("\n}\n", start)
    body = script[start:end]
    assert "AUTO_BOUNCE_MAX_TARGETS_PER_TICK" in body, (
        "auto_bounce() must itself truncate its target list; a cap applied "
        "only by the caller cannot contain a caller bug"
    )


# ---------------------------------------------------------------------------
# 4. Sequential, verified bounce
# ---------------------------------------------------------------------------


def test_targets_are_recreated_one_at_a_time(tmp_path: Path) -> None:
    """Each target is its own compose call, so one failure cannot cascade."""
    scen = Scenario(tmp_path / "scen", fleet_count=6)
    scen.set_missing([2, 4, 5])

    _run_monitor(tmp_path, scen, extra_env={"AUTO_BOUNCE_MAX_TARGETS_PER_TICK": "5"})

    batches = scen.compose_recreate_targets()
    assert len(batches) == 3, f"expected 3 single-target calls, got {batches}"
    assert all(len(batch) == 1 for batch in batches), (
        f"each compose call must name exactly one service; got {batches}"
    )
    assert [b[0] for b in batches] == [
        f"{PREFIX}-2",
        f"{PREFIX}-4",
        f"{PREFIX}-5",
    ]


def test_bounce_halts_when_a_target_fails_to_recover(tmp_path: Path) -> None:
    """A canary that will not come back stops the batch; it does not cascade."""
    scen = Scenario(tmp_path / "scen", fleet_count=6)
    scen.set_missing([2, 4, 5])
    scen.never_heals([2])

    _, output = _run_monitor(
        tmp_path, scen, extra_env={"AUTO_BOUNCE_MAX_TARGETS_PER_TICK": "5"}
    )

    recreated = scen.all_recreated()
    assert f"{PREFIX}-2" in recreated
    assert f"{PREFIX}-4" not in recreated, (
        "the first target never reached running, so the batch must halt "
        f"before touching the next; got {recreated}"
    )
    assert f"{PREFIX}-5" not in recreated
    assert "HALT" in output.upper() or "aborting" in output.lower()


# ---------------------------------------------------------------------------
# 5. Compose interpolation preflight (the 6-day silent no-op)
# ---------------------------------------------------------------------------


def test_broken_compose_interpolation_fails_the_tick_loudly(tmp_path: Path) -> None:
    """The exact 6-day silent-failure class, now a loud, actionable finding.

    ``.monitor-env`` was missing ``DEPLOY_RUNNER_OMNI_HOME``, so every
    cron-driven ``docker compose up`` aborted during interpolation while the
    monitor logged "AUTO-BOUNCE dispatched" -- 864 error lines, zero successful
    recreates, zero alerts, for six days.
    """
    scen = Scenario(tmp_path / "scen", fleet_count=4)
    scen.set_missing([2])
    scen.break_compose_interpolation(
        "error while interpolating services.omninode-deploy-runner.environment: "
        "required variable DEPLOY_RUNNER_OMNI_HOME is missing a value"
    )

    state, output = _run_monitor(tmp_path, scen)

    assert state["compose_interpolation_ok"] is False
    assert int(str(state["alert_count"])) >= 1, (
        "a broken monitor env must be an actionable alert, not a silent no-op"
    )
    assert "DEPLOY_RUNNER_OMNI_HOME" in output
    assert "INTERPOLATION" in output.upper() or "PREFLIGHT" in output.upper()


def test_broken_compose_interpolation_blocks_the_bounce(tmp_path: Path) -> None:
    """Do not claim to dispatch a bounce that cannot possibly succeed."""
    scen = Scenario(tmp_path / "scen", fleet_count=4)
    scen.set_missing([2])
    scen.break_compose_interpolation("required variable FOO is missing a value")

    _, output = _run_monitor(tmp_path, scen)

    assert scen.all_recreated() == [], (
        "compose cannot interpolate, so no recreate may be attempted; "
        f"got {scen.all_recreated()}"
    )
    assert "AUTO-BOUNCE dispatched" not in output, (
        "reporting a dispatch that was impossible is the exact 6-day "
        "silent-failure signature"
    )


def test_healthy_compose_interpolation_records_ok(tmp_path: Path) -> None:
    scen = Scenario(tmp_path / "scen", fleet_count=4)
    state, _ = _run_monitor(tmp_path, scen)
    assert state["compose_interpolation_ok"] is True


# ---------------------------------------------------------------------------
# 6. expected_count vs compose-declared service count drift
# ---------------------------------------------------------------------------


def test_expected_count_below_compose_service_count_alerts(tmp_path: Path) -> None:
    """The 72-vs-88 gap: 16 runners silently outside monitor scope.

    ``runner-monitor.sh`` iterates ``seq 1 $EXPECTED_RUNNERS``, so any runner
    declared in compose above that index is invisible to detection, alerting,
    and auto-bounce alike.
    """
    scen = Scenario(tmp_path / "scen", fleet_count=10)
    scen.set_compose_services(range(1, 11))

    state, output = _run_monitor(tmp_path, scen, expected_count=6)

    assert state["fleet_count_drift"] is True
    assert state["compose_runner_service_count"] == 10
    assert state["expected_runner_count"] == 6
    assert int(str(state["alert_count"])) >= 1
    assert "DRIFT" in output.upper()
    assert "10" in output and "6" in output


def test_expected_count_matching_compose_reports_no_drift(tmp_path: Path) -> None:
    scen = Scenario(tmp_path / "scen", fleet_count=8)
    state, _ = _run_monitor(tmp_path, scen, expected_count=8)
    assert state["fleet_count_drift"] is False
    assert state["compose_runner_service_count"] == 8


def test_expected_count_above_compose_service_count_alerts(tmp_path: Path) -> None:
    """Drift in the other direction is equally invisible and equally wrong."""
    scen = Scenario(tmp_path / "scen", fleet_count=5)
    scen.set_compose_services(range(1, 6))

    state, _ = _run_monitor(tmp_path, scen, expected_count=9)

    assert state["fleet_count_drift"] is True
    assert state["compose_runner_service_count"] == 5
    assert state["expected_runner_count"] == 9
