# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the OMN-13915 runner listener-liveness root-cause fix.

Incident: 37/48 self-hosted runner listeners were dead-in-container since
~2026-06-29 while every Docker healthcheck stayed green. The fix has three
enforced layers, all verified here without touching the live fleet:

1. ``docker/runners/healthcheck.sh`` — asserts the ``Runner.Listener`` process
   for THIS runner home is alive AND its ``_diag`` heartbeat is fresh. Proven
   functionally with a synthetic listener process that is killed mid-test
   (the DoD "synthetic kill" without a live container).
2. ``docker/runners/entrypoint.sh`` — watchdog that recycles the runner when
   the listener dies under a still-alive wrapper tree (content assertions).
3. ``.github/workflows/runner-fleet-canary.yml`` + ``scripts/ci/runner_fleet_canary.sh``
   — scheduled GitHub-hosted canary comparing the org runner registry against
   ``config/runner_fleet.yaml`` expected_count. Proven functionally against a
   local HTTP stub of the org runners API (synthetic offline fleet).

Ticket: OMN-13915
"""

from __future__ import annotations

import http.server
import json
import os
import shutil
import stat
import subprocess
import threading
import time
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
HEALTHCHECK = REPO_ROOT / "docker" / "runners" / "healthcheck.sh"
ENTRYPOINT = REPO_ROOT / "docker" / "runners" / "entrypoint.sh"
COMPOSE = REPO_ROOT / "docker" / "docker-compose.runners.yml"
CANARY_SCRIPT = REPO_ROOT / "scripts" / "ci" / "runner_fleet_canary.sh"
CANARY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "runner-fleet-canary.yml"
FLEET_CONFIG = REPO_ROOT / "config" / "runner_fleet.yaml"
RUNBOOK = REPO_ROOT / "docs" / "runbooks" / "runner-fleet-listener-liveness.md"


def _run_healthcheck(runner_home: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["RUNNER_HOME"] = str(runner_home)
    env["RUNNER_HEALTH_EGRESS_CHECK"] = "0"  # offline determinism
    env["RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS"] = "900"
    return subprocess.run(
        ["bash", str(HEALTHCHECK)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


@pytest.fixture
def synthetic_runner_home(tmp_path: Path) -> Path:
    """A fake RUNNER_HOME with a bin/Runner.Listener sleeper and fresh _diag."""
    home = tmp_path / "actions-runner"
    (home / "bin").mkdir(parents=True)
    (home / "_diag").mkdir()
    listener = home / "bin" / "Runner.Listener"
    listener.write_text("#!/bin/sh\nsleep 120\n", encoding="utf-8")
    listener.chmod(listener.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    (home / "_diag" / "Runner_20260703-000000-utc.log").write_text(
        "heartbeat\n", encoding="utf-8"
    )
    return home


class TestHealthcheckSyntheticKill:
    """DoD: a dead Runner.Listener flips the healthcheck — proven synthetically."""

    def test_healthcheck_lifecycle_alive_stale_dead(
        self, synthetic_runner_home: Path
    ) -> None:
        listener = synthetic_runner_home / "bin" / "Runner.Listener"
        proc = subprocess.Popen([str(listener)])
        try:
            # Give pgrep a moment to see the process.
            time.sleep(0.5)

            # 1. Listener alive + fresh heartbeat => healthy (exit 0).
            result = _run_healthcheck(synthetic_runner_home)
            assert result.returncode == 0, (
                f"expected healthy with live listener + fresh _diag; "
                f"got rc={result.returncode} out={result.stdout} err={result.stderr}"
            )

            # 2. Listener alive but heartbeat STALE => unhealthy. This is the
            #    exact OMN-13915 zombie mode a point-in-time pgrep cannot catch.
            diag_log = (
                synthetic_runner_home / "_diag" / "Runner_20260703-000000-utc.log"
            )
            stale = time.time() - 3600
            os.utime(diag_log, (stale, stale))
            result = _run_healthcheck(synthetic_runner_home)
            assert result.returncode == 1, (
                f"expected unhealthy with stale _diag heartbeat; "
                f"got rc={result.returncode} out={result.stdout}"
            )
            assert "heartbeat" in result.stdout

            # 3. SYNTHETIC KILL: listener process dies => unhealthy.
            proc.terminate()
            proc.wait(timeout=10)
            time.sleep(0.5)
            result = _run_healthcheck(synthetic_runner_home)
            assert result.returncode == 1, (
                f"expected unhealthy after listener kill; "
                f"got rc={result.returncode} out={result.stdout}"
            )
            assert "not running" in result.stdout
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_healthcheck_missing_diag_dir_fails_closed(
        self, synthetic_runner_home: Path
    ) -> None:
        """A 'live' listener with no _diag directory is the same divergence."""
        listener = synthetic_runner_home / "bin" / "Runner.Listener"
        proc = subprocess.Popen([str(listener)])
        try:
            time.sleep(0.5)
            shutil.rmtree(synthetic_runner_home / "_diag")
            result = _run_healthcheck(synthetic_runner_home)
            assert result.returncode == 1
            assert "_diag" in result.stdout
        finally:
            proc.kill()

    def test_healthcheck_pattern_is_runner_home_anchored(self) -> None:
        """The pgrep pattern must anchor to ${RUNNER_HOME}/bin/Runner.Listener.

        A loose 'Runner.Listener' substring match can be satisfied by wrapper
        processes or (in CI, which itself runs inside a runner container) the
        host runner's own listener — masking a dead listener.
        """
        content = HEALTHCHECK.read_text(encoding="utf-8")
        assert "bin/Runner\\.Listener" in content
        assert "${RUNNER_HOME//./" in content, (
            "healthcheck pgrep pattern must be derived from RUNNER_HOME"
        )


class TestEntrypointWatchdog:
    """The entrypoint must supervise the listener, not just the wrapper tree."""

    def test_entrypoint_has_watchdog(self) -> None:
        content = ENTRYPOINT.read_text(encoding="utf-8")
        assert "WATCHDOG" in content, "entrypoint must log watchdog transitions"
        assert "LISTENER_SUPERVISE_INTERVAL" in content
        assert "LISTENER_SUPERVISE_MISSES" in content
        assert "LISTENER_PGREP_PATTERN" in content
        assert "OMN-13915" in content

    def test_entrypoint_watchdog_restart_is_bounded(self) -> None:
        """Unbounded silent restarts would hide a crash-looping listener."""
        content = ENTRYPOINT.read_text(encoding="utf-8")
        assert "LISTENER_RESTART_MAX" in content
        assert "listener_restarts" in content

    def test_entrypoint_pattern_is_runner_home_anchored(self) -> None:
        content = ENTRYPOINT.read_text(encoding="utf-8")
        assert "${RUNNER_HOME//./" in content, (
            "watchdog pgrep pattern must be derived from RUNNER_HOME"
        )

    def test_entrypoint_bash_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(ENTRYPOINT)], check=False, capture_output=True, text=True
        )
        assert result.returncode == 0, f"bash -n failed: {result.stderr}"

    def test_healthcheck_bash_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(HEALTHCHECK)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"bash -n failed: {result.stderr}"

    def test_canary_script_bash_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(CANARY_SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"bash -n failed: {result.stderr}"


class TestComposeHealthcheckWiring:
    """The compose healthcheck stanza must invoke the mounted script."""

    def test_compose_healthcheck_invokes_script(self) -> None:
        content = COMPOSE.read_text(encoding="utf-8")
        assert "/usr/local/bin/healthcheck.sh" in content
        assert "./runners/healthcheck.sh:/usr/local/bin/healthcheck.sh:ro" in content

    def test_compose_documents_recreate_requirement(self) -> None:
        """Root cause: healthcheck DEFINITION is frozen at container creation.

        The compose file must carry the rollout warning so nobody assumes a
        file sync alone updates live containers (OMN-13915).
        """
        content = COMPOSE.read_text(encoding="utf-8")
        assert "OMN-13915" in content
        assert "container creation" in content


def _canary_runners_payload(online: int, offline: int) -> dict[str, object]:
    runners = []
    for i in range(1, online + offline + 1):
        runners.append(
            {
                "name": f"omninode-runner-{i}",
                "status": "online" if i <= online else "offline",
                "busy": False,
                "labels": [
                    {"name": "self-hosted"},
                    {"name": "omnibase-ci"},
                    {"name": "linux"},
                ],
            }
        )
    return {"total_count": len(runners), "runners": runners}


class _StubHandler(http.server.BaseHTTPRequestHandler):
    payload: bytes = b"{}"

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *args: object) -> None:  # silence request logging
        pass


def _run_canary(
    payload: dict[str, object], max_offline: str = "5"
) -> subprocess.CompletedProcess[str]:
    handler = type(
        "Handler", (_StubHandler,), {"payload": json.dumps(payload).encode()}
    )
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = dict(os.environ)
        env.update(
            {
                "GITHUB_API_URL": f"http://127.0.0.1:{server.server_port}",
                "RUNNER_FLEET_STATUS_TOKEN": "test-token",
                "RUNNER_CANARY_MAX_OFFLINE": max_offline,
                "RUNNER_FLEET_CONFIG_PATH": str(FLEET_CONFIG),
            }
        )
        env.pop("GITHUB_STEP_SUMMARY", None)
        env.pop("SLACK_BOT_TOKEN", None)
        env.pop("SLACK_CHANNEL_ID", None)
        return subprocess.run(
            ["bash", str(CANARY_SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            cwd=REPO_ROOT,
            timeout=60,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)


class TestFleetCanaryFunctional:
    """DoD: canary alerts on a synthetic offline fleet without any Docker change."""

    def test_canary_passes_on_healthy_fleet(self) -> None:
        expected = yaml.safe_load(FLEET_CONFIG.read_text(encoding="utf-8"))[
            "expected_count"
        ]
        result = _run_canary(_canary_runners_payload(online=expected, offline=0))
        assert result.returncode == 0, (
            f"healthy fleet must pass: rc={result.returncode} "
            f"out={result.stdout} err={result.stderr}"
        )

    def test_canary_fails_on_incident_shape_fleet(self) -> None:
        """The exact 2026-07-03 incident shape: 11 online / 37 offline."""
        result = _run_canary(_canary_runners_payload(online=11, offline=37))
        assert result.returncode != 0, (
            f"37 offline must breach the threshold: out={result.stdout}"
        )
        assert "offline" in (result.stdout + result.stderr)

    def test_canary_fails_on_missing_registrations(self) -> None:
        """Runners that dropped registration entirely count as offline."""
        result = _run_canary(_canary_runners_payload(online=11, offline=0))
        assert result.returncode != 0, (
            f"37 missing registrations must breach: out={result.stdout}"
        )

    def test_canary_fails_closed_without_token(self) -> None:
        env = dict(os.environ)
        env.pop("RUNNER_FLEET_STATUS_TOKEN", None)
        env.pop("CROSS_REPO_PAT", None)
        env["RUNNER_FLEET_CONFIG_PATH"] = str(FLEET_CONFIG)
        result = subprocess.run(
            ["bash", str(CANARY_SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            cwd=REPO_ROOT,
            timeout=30,
        )
        assert result.returncode != 0
        assert "token" in (result.stdout + result.stderr).lower()


class TestFleetCanaryWorkflow:
    """The canary must be an ENFORCED scheduled surface, not an opt-in script."""

    @staticmethod
    def _load_workflow() -> dict:  # type: ignore[type-arg]
        loaded = yaml.safe_load(CANARY_WORKFLOW.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict)
        return loaded

    def test_workflow_exists(self) -> None:
        assert CANARY_WORKFLOW.exists(), (
            "runner-fleet-canary.yml missing — the OMN-13915 canary must be a "
            "scheduled workflow, not an opt-in script (enforcement-not-detection)."
        )

    def test_workflow_is_scheduled_at_most_every_15_minutes(self) -> None:
        workflow = self._load_workflow()
        triggers = workflow.get("on") or workflow.get(True)
        assert triggers is not None
        schedule = triggers.get("schedule")
        assert schedule, "canary must run on a schedule"
        crons = [entry["cron"] for entry in schedule]
        assert any("*/15" in c or "*/10" in c or "*/5 " in c for c in crons), (
            f"canary cadence must be <= 15 minutes, got {crons}"
        )
        assert "workflow_dispatch" in triggers, "manual trigger required"

    def test_workflow_runs_on_github_hosted_only(self) -> None:
        """The canary must not share fate with the fleet it watches."""
        workflow = self._load_workflow()
        for job_name, job in workflow["jobs"].items():
            runs_on = job.get("runs-on")
            assert runs_on == "ubuntu-latest", (
                f"job {job_name} must run on ubuntu-latest (GitHub-hosted), "
                f"got {runs_on!r} — a self-hosted canary dies with the fleet."
            )
            assert "self-hosted" not in str(runs_on)

    def test_workflow_invokes_canary_script(self) -> None:
        content = CANARY_WORKFLOW.read_text(encoding="utf-8")
        assert "scripts/ci/runner_fleet_canary.sh" in content


class TestRunbook:
    """DoD: runbook states Docker-healthy is NOT sufficient evidence."""

    def test_runbook_exists_with_authoritative_signal_note(self) -> None:
        assert RUNBOOK.exists()
        content = RUNBOOK.read_text(encoding="utf-8")
        assert "NOT sufficient" in content
        assert "canary" in content.lower()
        assert "OMN-13915" in content
