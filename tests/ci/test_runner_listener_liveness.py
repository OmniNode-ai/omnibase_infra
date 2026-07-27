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
import re
import shutil
import signal
import stat
import subprocess
import threading
import time
from pathlib import Path

import pytest
import yaml

# tests/ci/ is the project-recognized home for CI/CD parity tests (OMN-4307).
# These tests exercise local subprocesses and a localhost HTTP stub but no
# external infrastructure, matching the registered ``ci`` marker semantics.
pytestmark = pytest.mark.ci

REPO_ROOT = Path(__file__).resolve().parents[2]
HEALTHCHECK = REPO_ROOT / "docker" / "runners" / "healthcheck.sh"
ENTRYPOINT = REPO_ROOT / "docker" / "runners" / "entrypoint.sh"
COMPOSE = REPO_ROOT / "docker" / "docker-compose.runners.yml"
CANARY_SCRIPT = REPO_ROOT / "scripts" / "ci" / "runner_fleet_canary.sh"
CANARY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "runner-fleet-canary.yml"
FLEET_CONFIG = REPO_ROOT / "config" / "runner_fleet.yaml"
RUNBOOK = REPO_ROOT / "docs" / "runbooks" / "runner-fleet-listener-liveness.md"


def _run_healthcheck(
    runner_home: Path, max_diag_age: str | None = "900"
) -> subprocess.CompletedProcess[str]:
    """Run healthcheck.sh against ``runner_home``.

    ``max_diag_age=None`` leaves ``RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS`` UNSET so
    the script's own default is what gets exercised (OMN-15233 — the default is
    the thing that was miscalibrated; a test that always pins the threshold can
    never catch that).
    """
    env = dict(os.environ)
    env["RUNNER_HOME"] = str(runner_home)
    env["RUNNER_HEALTH_EGRESS_CHECK"] = "0"  # offline determinism
    if max_diag_age is None:
        env.pop("RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS", None)
    else:
        env["RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS"] = max_diag_age
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
    if os.getpid() == 1:
        pytest.skip(
            "pytest is PID 1 in this harness, so every synthetic listener would "
            "inherit PPID 1 and be indistinguishable from an OMN-15233 orphan"
        )
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


def _spawn_orphan(command: Path) -> int:
    """Spawn ``command`` detached so this test process is NOT its parent.

    The intermediate shell exits immediately, so the child is reparented (to
    PID 1 on macOS/Linux without a subreaper) — the exact orphan shape from the
    OMN-15233 incident. Returns the orphan's pid.
    """
    spawn = subprocess.run(
        ["bash", "-c", f'"{command}" >/dev/null 2>&1 & echo $!'],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return int(spawn.stdout.strip())


def _ppid_of(pid: int) -> int | None:
    result = subprocess.run(
        ["ps", "-o", "ppid=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    raw = result.stdout.strip()
    return int(raw) if raw.isdigit() else None


def _kill_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class TestHealthcheckIdleThresholdRecalibration:
    """OMN-15233 (a): the default threshold must clear the IDLE _diag cadence.

    When a runner is idle the ONLY thing writing ``_diag`` is the OAuth/AAD
    token refresh, on a ~50-minute cadence; the minutes-scale cadence holds
    only while jobs run. The shipped 900s default therefore read unhealthy for
    ~35 of every 50 idle minutes with nothing degraded — the 2026-07-27
    "13 -> 37 -> 59 unhealthy growth" while the GitHub registry reported 64/64
    online throughout (59 -> 4 resolved with 8 restarts; the untouched control
    group self-healed).
    """

    def test_idle_runner_past_old_900s_window_reads_healthy(
        self, synthetic_runner_home: Path
    ) -> None:
        """45 min of idle _diag silence is HEALTHY under the script default."""
        listener = synthetic_runner_home / "bin" / "Runner.Listener"
        proc = subprocess.Popen([str(listener)])
        try:
            time.sleep(0.5)
            diag_log = (
                synthetic_runner_home / "_diag" / "Runner_20260703-000000-utc.log"
            )
            idle = time.time() - 2700  # 45 min — inside the idle OAuth cadence
            os.utime(diag_log, (idle, idle))

            result = _run_healthcheck(synthetic_runner_home, max_diag_age=None)
            assert result.returncode == 0, (
                "an idle runner 45 min past its last _diag write must read "
                "HEALTHY on the script default (the 900s default flagged it by "
                f"arithmetic): rc={result.returncode} out={result.stdout}"
            )
        finally:
            proc.kill()

    def test_recalibration_did_not_disable_the_check(
        self, synthetic_runner_home: Path
    ) -> None:
        """Past the recalibrated window, silence is still UNHEALTHY."""
        listener = synthetic_runner_home / "bin" / "Runner.Listener"
        proc = subprocess.Popen([str(listener)])
        try:
            time.sleep(0.5)
            diag_log = (
                synthetic_runner_home / "_diag" / "Runner_20260703-000000-utc.log"
            )
            dead = time.time() - 9000  # 2.5 h — well past any benign cadence
            os.utime(diag_log, (dead, dead))

            result = _run_healthcheck(synthetic_runner_home, max_diag_age=None)
            assert result.returncode == 1, (
                f"2.5 h of silence must still fail: out={result.stdout}"
            )
            assert "heartbeat" in result.stdout
        finally:
            proc.kill()

    def test_default_threshold_clears_idle_cadence_and_is_justified(self) -> None:
        content = HEALTHCHECK.read_text(encoding="utf-8")
        match = re.search(r"RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS:-(\d+)", content)
        assert match, "RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS default missing"
        assert int(match.group(1)) >= 3600, (
            "alert threshold must exceed the ~50-min idle OAuth-refresh cadence "
            f"(>= 3600s), got {match.group(1)}s"
        )
        assert "idle" in content.lower(), (
            "the threshold must carry an inline justification against the idle "
            "_diag cadence — an unexplained magic number invites the next "
            "well-meaning tightening back to 900s"
        )


class TestHealthcheckOrphanInversion:
    """OMN-15233 (b): the zombie shape #2194 exists to catch scored HEALTHY.

    An orphaned ``Runner.Listener`` reparented to PPID 1 keeps holding the
    GitHub session; the watchdog's replacement crash-loops every ~5 min on
    ``TaskAgentSessionConflictException``; every crash mints a fresh
    ``Runner_*.log``, which keeps the ``_diag`` mtime heartbeat fresh — so the
    mtime-only check read HEALTHY forever. Runners 1/43/55/57 sat in this state
    with 88-234 log files (vs 3-7 normal) and were found only by process scan.
    """

    def test_duplicate_listeners_read_unhealthy(
        self, synthetic_runner_home: Path
    ) -> None:
        """Two listeners = a contested broker session, whatever _diag says."""
        listener = synthetic_runner_home / "bin" / "Runner.Listener"
        first = subprocess.Popen([str(listener)])
        second = subprocess.Popen([str(listener)])
        try:
            time.sleep(0.5)
            result = _run_healthcheck(synthetic_runner_home, max_diag_age=None)
            assert result.returncode == 1, (
                "duplicate Runner.Listener processes with a fresh _diag must "
                f"read UNHEALTHY: rc={result.returncode} out={result.stdout}"
            )
            assert "duplicate" in result.stdout.lower()
        finally:
            first.kill()
            second.kill()

    def test_orphaned_ppid1_listener_reads_unhealthy(
        self, synthetic_runner_home: Path
    ) -> None:
        """A single listener reparented to PPID 1 is an orphan, not health."""
        listener = synthetic_runner_home / "bin" / "Runner.Listener"
        orphan_pid = _spawn_orphan(listener)
        try:
            ppid = None
            for _ in range(50):
                ppid = _ppid_of(orphan_pid)
                if ppid == 1:
                    break
                time.sleep(0.1)
            if ppid != 1:
                pytest.skip(
                    f"host did not reparent the orphan to PID 1 (ppid={ppid}); "
                    "a subreaper is active, so the incident precondition cannot "
                    "be reproduced here"
                )
            result = _run_healthcheck(synthetic_runner_home, max_diag_age=None)
            assert result.returncode == 1, (
                "a PPID-1 orphan with a fresh _diag must read UNHEALTHY — this "
                "is the exact state that scored HEALTHY on runners 1/43/55/57: "
                f"rc={result.returncode} out={result.stdout}"
            )
            assert "PPID 1" in result.stdout
        finally:
            _kill_pid(orphan_pid)


class TestHealthcheckCrashLoopRate:
    """OMN-15233 (b): crash-loop detection must be RATE-based, not cumulative."""

    def test_crash_loop_rate_flags_but_history_does_not(
        self, synthetic_runner_home: Path
    ) -> None:
        listener = synthetic_runner_home / "bin" / "Runner.Listener"
        proc = subprocess.Popen([str(listener)])
        diag = synthetic_runner_home / "_diag"
        try:
            time.sleep(0.5)

            # (1) A long-lived clean runner: 234 historical Runner_*.log files
            #     (the observed zombie-runner count) all aged 10 days, plus the
            #     one active log. A CUMULATIVE count would flag this forever.
            old = time.time() - 10 * 86400
            for i in range(234):
                path = diag / f"Runner_2026070{i % 10}-{i:06d}-utc.log"
                path.write_text("historical\n", encoding="utf-8")
                os.utime(path, (old, old))
            result = _run_healthcheck(synthetic_runner_home, max_diag_age=None)
            assert result.returncode == 0, (
                "234 historical logs with one active log must read HEALTHY — a "
                "cumulative count would red-line every long-lived container: "
                f"rc={result.returncode} out={result.stdout}"
            )

            # (2) The crash-loop shape: a replacement listener restarting every
            #     ~5 min mints ~12 fresh Runner_*.log files per hour.
            for i in range(12):
                path = diag / f"Runner_20260727-{i:06d}-utc.log"
                path.write_text("crash\n", encoding="utf-8")
            result = _run_healthcheck(synthetic_runner_home, max_diag_age=None)
            assert result.returncode == 1, (
                "12 listener starts inside the rate window must read UNHEALTHY: "
                f"rc={result.returncode} out={result.stdout}"
            )
            assert "crash-looping" in result.stdout
        finally:
            proc.kill()

    def test_crash_loop_signal_is_documented_as_rate_not_cumulative(self) -> None:
        content = HEALTHCHECK.read_text(encoding="utf-8")
        assert "RUNNER_HEALTH_MAX_LOG_STARTS_PER_HOUR" in content
        assert "RUNNER_HEALTH_LOG_RATE_WINDOW_MINUTES" in content
        assert "NOT CUMULATIVE" in content, (
            "the rate-vs-cumulative choice is the whole point of the layer and "
            "must be stated where the next editor will read it"
        )
        assert "-mmin" in content


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


def _make_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


@pytest.fixture
def synthetic_entrypoint_home(tmp_path: Path) -> Path:
    """A fake RUNNER_HOME the real entrypoint.sh can supervise end-to-end.

    - ``run.sh`` launches ``bin/Runner.Listener`` by absolute path (so the
      watchdog's RUNNER_HOME-anchored pgrep pattern matches) and waits on it.
    - ``bin/Runner.Listener`` is a sleeper: ALIVE but writing nothing — the
      exact OMN-14564 hung-listener shape.
    - In-place ``.runner`` + ``.credentials`` skip registration entirely.
    - The only ``_diag`` log is pre-aged one hour → heartbeat STALE.
    """
    home = tmp_path / "actions-runner"
    (home / "bin").mkdir(parents=True)
    (home / "_diag").mkdir()
    listener = home / "bin" / "Runner.Listener"
    _make_executable(listener, "#!/bin/bash\nsleep 300\n")
    worker = home / "bin" / "Runner.Worker"
    _make_executable(worker, "#!/bin/bash\nsleep 300\n")
    _make_executable(home / "run.sh", f'#!/bin/bash\n"{listener}" &\nwait $!\n')
    (home / ".runner").write_text("{}\n", encoding="utf-8")
    (home / ".credentials").write_text("{}\n", encoding="utf-8")
    diag_log = home / "_diag" / "Runner_20260716-000000-utc.log"
    diag_log.write_text("heartbeat\n", encoding="utf-8")
    stale = time.time() - 3600
    os.utime(diag_log, (stale, stale))
    return home


def _entrypoint_env(home: Path, log_file: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "RUNNER_NAME": "synthetic-runner",
            "RUNNER_LABELS": "synthetic",
            "GITHUB_ORG_URL": "https://github.com/OmniNode-ai",
            "RUNNER_HOME": str(home),
            "LOG_FILE": str(log_file),
            "LISTENER_SUPERVISE_INTERVAL": "1",
            "LISTENER_HEARTBEAT_MISSES": "2",
            "LISTENER_RESTART_MAX": "0",
            "LISTENER_HEARTBEAT_MAX_AGE_SECONDS": "60",
        }
    )
    return env


class TestEntrypointHungListenerWatchdog:
    """OMN-14564: a listener can hang ALIVE (AAD/OAuth token-refresh deadlock).

    Incident 2026-07-16..23: 11/64 runners GitHub-offline for ~6 days with
    Runner.Listener still in the process table — pgrep-existence supervision
    (OMN-13915) passes forever, the hung listener never exits so run.sh never
    respawns it, and only the healthcheck's _diag staleness layer flagged it.
    The entrypoint watchdog must turn that same detection into remediation.
    """

    def test_entrypoint_has_heartbeat_watchdog(self) -> None:
        content = ENTRYPOINT.read_text(encoding="utf-8")
        assert "OMN-14564" in content
        assert "LISTENER_HEARTBEAT_MAX_AGE_SECONDS" in content
        assert "LISTENER_HEARTBEAT_MISSES" in content
        assert "_listener_heartbeat_stale" in content

    def test_kill_threshold_clears_the_benign_broker_quiet_ceiling(self) -> None:
        """The watchdog KILL threshold must clear the observed benign
        broker-quiet ceiling (~50 min). Live 2026-07-23T05:25-06:02Z readback:
        a fleet-wide broker-quiet window silenced _diag on 53/64 listeners for
        35-50 min while GitHub kept them online and docker-"unhealthy" runners
        were actively executing jobs — killing at the then-current 900s alert
        threshold would have mass-recycled ~50 healthy-but-quiet listeners
        mid-window.

        OMN-15233 note: the alert threshold is now 4500s, i.e. ABOVE this kill
        threshold. The ordering flipped on purpose — this watchdog additionally
        requires LISTENER_HEARTBEAT_MISSES consecutive ticks and never fires
        while a Runner.Worker runs, so it is the narrower signal. What must
        hold is the >= 3600s floor, not an ordering between the two."""
        content = ENTRYPOINT.read_text(encoding="utf-8")
        match = re.search(r"LISTENER_HEARTBEAT_MAX_AGE_SECONDS:-(\d+)", content)
        assert match, "LISTENER_HEARTBEAT_MAX_AGE_SECONDS default missing"
        assert int(match.group(1)) >= 3600, (
            "kill threshold must clear the observed ~50 min benign "
            "broker-quiet ceiling (>= 3600s)"
        )

    def test_entrypoint_has_worker_job_guard(self) -> None:
        """The heartbeat recycle must never kill an executing job."""
        content = ENTRYPOINT.read_text(encoding="utf-8")
        assert "WORKER_PGREP_PATTERN" in content
        assert "bin/Runner\\.Worker" in content

    def test_recycle_kills_listener_binary_explicitly(self) -> None:
        """A hung listener ignores the wrapper-tree TERM and never exits on
        its own; leaving it alive collides with the respawned listener's
        broker session. The recycle path must pkill the listener pattern."""
        content = ENTRYPOINT.read_text(encoding="utf-8")
        assert "_recycle_runner_tree" in content
        assert 'pkill -KILL -f "${LISTENER_PGREP_PATTERN}"' in content

    @pytest.mark.skipif(
        os.getuid() == 0, reason="entrypoint takes root-only paths (gosu/groupmod)"
    )
    def test_entrypoint_recycles_hung_listener(
        self, synthetic_entrypoint_home: Path, tmp_path: Path
    ) -> None:
        """Functional: alive-but-silent listener → watchdog kills + recycles.

        LISTENER_RESTART_MAX=0 makes the first supervised recycle a terminal
        container exit (rc=1), giving the test a deterministic endpoint.
        """
        home = synthetic_entrypoint_home
        stdout_path = tmp_path / "entrypoint-stdout.log"
        with stdout_path.open("wb") as stdout_file:
            proc = subprocess.Popen(
                ["bash", str(ENTRYPOINT)],
                stdout=stdout_file,
                stderr=subprocess.STDOUT,
                env=_entrypoint_env(home, tmp_path / "listener.log"),
                start_new_session=True,
            )
            try:
                rc = proc.wait(timeout=90)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                # Belt-and-braces: never leak synthetic sleepers.
                subprocess.run(["pkill", "-KILL", "-f", str(home)], check=False)
        output = stdout_path.read_text(encoding="utf-8")
        assert rc == 1, f"expected terminal exit 1 (RESTART_MAX=0); got {rc}: {output}"
        assert "OMN-14564 hung-listener mode" in output
        assert "killing listener" in output
        # The hung listener process itself must be gone, not just the wrapper.
        pgrep = subprocess.run(
            ["pgrep", "-f", f"{home}/bin/Runner.Listener"],
            check=False,
            capture_output=True,
        )
        assert pgrep.returncode != 0, "hung listener survived the recycle"

    @pytest.mark.skipif(
        os.getuid() == 0, reason="entrypoint takes root-only paths (gosu/groupmod)"
    )
    def test_entrypoint_never_recycles_while_worker_running(
        self, synthetic_entrypoint_home: Path, tmp_path: Path
    ) -> None:
        """Functional: stale heartbeat + active Runner.Worker → NO recycle."""
        home = synthetic_entrypoint_home
        worker = subprocess.Popen([str(home / "bin" / "Runner.Worker")])
        stdout_path = tmp_path / "entrypoint-stdout.log"
        proc = None
        try:
            with stdout_path.open("wb") as stdout_file:
                proc = subprocess.Popen(
                    ["bash", str(ENTRYPOINT)],
                    stdout=stdout_file,
                    stderr=subprocess.STDOUT,
                    env=_entrypoint_env(home, tmp_path / "listener.log"),
                    start_new_session=True,
                )
                # Long enough for >2 supervise ticks (interval=1, misses=2):
                # without the worker guard the hung-listener miss line appears
                # within ~2s.
                time.sleep(6)
                assert proc.poll() is None, "entrypoint exited despite active job"
            output = stdout_path.read_text(encoding="utf-8")
            assert "OMN-14564 hung-listener mode" not in output, (
                f"watchdog counted heartbeat misses while a Runner.Worker "
                f"job was executing: {output}"
            )
        finally:
            worker.kill()
            worker.wait(timeout=10)
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            subprocess.run(["pkill", "-KILL", "-f", str(home)], check=False)


@pytest.fixture
def synthetic_reap_home(tmp_path: Path) -> Path:
    """A RUNNER_HOME whose ``run.sh`` reproduces the session-conflict failure.

    The real ``run.sh`` cannot register while a surviving listener still owns
    this runner's GitHub broker session — it dies on
    ``TaskAgentSessionConflictException``. This stub asserts the same
    precondition: if a listener for this home is already running when run.sh
    starts, it prints that exception and exits non-zero. So the ONLY way the
    entrypoint gets a clean start is by reaping first.
    """
    home = tmp_path / "actions-runner"
    (home / "bin").mkdir(parents=True)
    (home / "_diag").mkdir()
    listener = home / "bin" / "Runner.Listener"
    _make_executable(listener, "#!/bin/bash\nsleep 300\n")
    _make_executable(
        home / "run.sh",
        "#!/bin/bash\n"
        f'if pgrep -f "{listener}" >/dev/null 2>&1; then\n'
        '  echo "TaskAgentSessionConflictException: session held by another listener"\n'
        "  exit 1\n"
        "fi\n"
        f'"{listener}" &\n'
        "wait $!\n",
    )
    (home / ".runner").write_text("{}\n", encoding="utf-8")
    (home / ".credentials").write_text("{}\n", encoding="utf-8")
    (home / "_diag" / "Runner_20260727-000000-utc.log").write_text(
        "heartbeat\n", encoding="utf-8"
    )
    return home


class TestEntrypointOrphanReap:
    """OMN-15233: reap the orphan BEFORE spawning a replacement.

    Spawn-without-reap is what manufactures ``TaskAgentSessionConflictException``
    in the first place: the orphaned listener still owns the session, so every
    replacement dies within ~5 min and mints a fresh ``Runner_*.log`` that keeps
    the ``_diag`` heartbeat looking healthy.
    """

    def test_reap_precedes_the_run_sh_spawn(self) -> None:
        """Static: the reap call must sit ABOVE the spawn in the loop body."""
        content = ENTRYPOINT.read_text(encoding="utf-8")
        assert "_reap_orphaned_listeners" in content
        assert "OMN-15233" in content
        reap_call = content.index("if ! _reap_orphaned_listeners; then")
        spawn = content.index('_as_runner "${RUNNER_HOME}/run.sh"')
        assert reap_call < spawn, (
            "the reap must run before run.sh is spawned; reaping afterwards "
            "cannot prevent the session conflict"
        )

    def test_reap_escalates_term_to_kill(self) -> None:
        """A listener deadlocked in token refresh ignores TERM (OMN-14564)."""
        content = ENTRYPOINT.read_text(encoding="utf-8")
        reap = content[content.index("_reap_orphaned_listeners() {") :]
        reap = reap[: reap.index("\n}\n")]
        assert 'pkill -TERM -f "${LISTENER_PGREP_PATTERN}"' in reap
        assert 'pkill -KILL -f "${LISTENER_PGREP_PATTERN}"' in reap
        assert "LISTENER_REAP_TIMEOUT_SECONDS" in reap

    @pytest.mark.skipif(
        os.getuid() == 0, reason="entrypoint takes root-only paths (gosu/groupmod)"
    )
    def test_entrypoint_reaps_orphan_and_replacement_sees_no_session_conflict(
        self, synthetic_reap_home: Path, tmp_path: Path
    ) -> None:
        """Functional: orphan present at start → reaped → clean replacement.

        DoD: "kill the parent and confirm no TaskAgentSessionConflictException
        in the replacement's logs."
        """
        home = synthetic_reap_home
        orphan_pid = _spawn_orphan(home / "bin" / "Runner.Listener")
        time.sleep(0.5)
        assert _ppid_of(orphan_pid) is not None, "orphan failed to start"

        env = dict(os.environ)
        env.update(
            {
                "RUNNER_NAME": "synthetic-runner",
                "RUNNER_LABELS": "synthetic",
                "GITHUB_ORG_URL": "https://github.com/OmniNode-ai",
                "RUNNER_HOME": str(home),
                "LOG_FILE": str(tmp_path / "listener.log"),
                "LISTENER_SUPERVISE_INTERVAL": "1",
                "LISTENER_REAP_TIMEOUT_SECONDS": "5",
            }
        )
        stdout_path = tmp_path / "entrypoint-stdout.log"
        proc = None
        try:
            with stdout_path.open("wb") as stdout_file:
                proc = subprocess.Popen(
                    ["bash", str(ENTRYPOINT)],
                    stdout=stdout_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                deadline = time.time() + 60
                output = ""
                while time.time() < deadline:
                    output = stdout_path.read_text(encoding="utf-8")
                    if "REAP: no Runner.Listener remains" in output:
                        break
                    if proc.poll() is not None:
                        break
                    time.sleep(0.5)
            output = stdout_path.read_text(encoding="utf-8")
            assert "REAP: no Runner.Listener remains" in output, (
                f"entrypoint never reaped the orphan before spawning: {output}"
            )
            # Give run.sh time to clear its session-conflict guard and spawn.
            # Assert against LOG_FILE (the tee'd run.sh output), NOT the
            # entrypoint stdout — the entrypoint's own REAP banner names the
            # exception, so asserting on stdout would be vacuous.
            time.sleep(2)
            replacement_log = (tmp_path / "listener.log").read_text(encoding="utf-8")
            assert "TaskAgentSessionConflictException" not in replacement_log, (
                f"replacement spawned into a contested session: {replacement_log}"
            )
            output = stdout_path.read_text(encoding="utf-8")
            assert proc.poll() is None, (
                f"entrypoint exited instead of running the replacement: {output}"
            )
            assert _ppid_of(orphan_pid) is None, "the orphan survived the reap"
        finally:
            _kill_pid(orphan_pid)
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            subprocess.run(["pkill", "-KILL", "-f", str(home)], check=False)


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

    def test_runbook_records_registry_crosscheck_before_restart_sweep(self) -> None:
        """OMN-15233 interim rule: the Docker-unhealthy count is not a
        degradation metric on this fleet. An operator who restart-sweeps off it
        alone repeats the 2026-07-27 false-positive sweep."""
        content = RUNBOOK.read_text(encoding="utf-8")
        assert "OMN-15233" in content
        assert "before ANY restart sweep" in content
        assert "actions/runners" in content, (
            "the runbook must name the registry probe the operator runs, not "
            "just tell them to 'cross-check'"
        )
