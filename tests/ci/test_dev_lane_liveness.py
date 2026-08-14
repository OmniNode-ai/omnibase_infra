# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the OMN-15190 lab/dev lane liveness detector.

Incident class: the `.201` lab/dev lane (compose project ``omnibase-infra``)
was GC/idle-reclaimed to zero containers at least five times (2026-07-13,
07-14, 07-24, 07-26 x2). Every occurrence was found REACTIVELY, by whichever
PR happened to hit the resulting org-wide ``occ-autobind`` /
``occ-companion-effect`` connection-refused cascade. No monitor existed:
``scripts/system_health_check.sh`` held most of the logic and was wired into
nothing, and its severity contract actively *passed* a down lane
(``runtime_containers: yellow`` -> exit 0).

Operator ruling 2026-07-29 (WS-4) reverses the posture: the lab lane is
KEEP-ALIVE, so lane-down is a defect. These tests drive the REAL bash
artifact (never a reimplementation of its logic) with a stubbed ``docker``
and real localhost sockets, and pin both directions of every verdict:

* the failure modes that actually strand CI are RED,
* the states that do NOT strand CI are NOT red (a permanently-red check is a
  disabled check — the same reasoning ``healthcheck.sh`` layer 4 documents).

Ticket: OMN-15190
"""

from __future__ import annotations

import http.server
import os
import socket
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

# tests/ci/ is the project-recognized home for CI/CD parity tests (OMN-4307).
# Local subprocesses + localhost sockets only; no external infrastructure.
pytestmark = pytest.mark.ci

REPO_ROOT = Path(__file__).resolve().parents[2]
HEALTH_GATE = REPO_ROOT / "scripts" / "system_health_check.sh"
LANE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "dev-lane-liveness.yml"
CANARY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "runner-fleet-canary.yml"

# A container census in the exact `docker ps --format '{{.Names}}|{{.State}}|
# {{.Status}}'` shape, transcribed from the live `.201` dev lane on
# 2026-07-29T00:5xZ (compose project omnibase-infra). Used as the "lane is
# fine" baseline so the healthy-path assertions run against the real input
# distribution rather than a hand-shrunk approximation.
LIVE_LANE_ROWS = "\n".join(
    [
        "omninode-runtime|running|Up 9 hours (healthy)",
        "omninode-runtime-effects|running|Up 2 hours (healthy)",
        "omnibase-intelligence-api|running|Up 9 hours (healthy)",
        "omnibase-infra-postgres|running|Up 10 hours (healthy)",
        "omnibase-infra-redpanda|running|Up 10 hours (healthy)",
        "omnibase-infra-valkey|running|Up 10 hours (healthy)",
        "omnibase-infra-intelligence-migration|exited|Exited (0) 2 hours ago",
    ]
)

RUNTIME_CONTAINER_NAMES = "\n".join(
    [
        "omninode-runtime",
        "omninode-runtime-effects",
        "omnibase-intelligence-api",
        "omnibase-infra-redpanda",
    ]
)


class _HealthHandler(http.server.BaseHTTPRequestHandler):
    """Minimal /health responder; status code is set per-test on the class."""

    status_code = 200

    def do_GET(self) -> None:
        self.send_response(self.status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture
def health_server() -> Iterator[tuple[int, type[_HealthHandler]]]:
    """A localhost HTTP server standing in for the lane's runtime /health."""
    handler = type("_ScopedHealthHandler", (_HealthHandler,), {"status_code": 200})
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], handler
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def open_broker_port() -> Iterator[int]:
    """A listening TCP socket standing in for the lane's published broker port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


def _closed_port() -> int:
    """A port number with nothing listening on it."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port: int = sock.getsockname()[1]
    sock.close()
    return port


def _install_docker_stub(
    tmp_path: Path,
    lane_rows: str,
    running_names: str = RUNTIME_CONTAINER_NAMES,
    daemon_ok: bool = True,
) -> Path:
    """Write a ``docker`` stub whose census output the test controls.

    The stub answers the three invocations the ``--lane`` subset makes:
    the compose-project-label census, the plain running-name list, and the
    ``rpk cluster health`` exec. Nothing else is emulated.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    rows_file = tmp_path / "lane_rows.txt"
    rows_file.write_text(lane_rows)
    names_file = tmp_path / "running_names.txt"
    names_file.write_text(running_names)

    fail_block = (
        ""
        if daemon_ok
        else (
            'echo "Cannot connect to the Docker daemon at unix:///var/run/'
            'docker.sock. Is the docker daemon running?" >&2\nexit 1\n'
        )
    )

    stub = f"""#!/usr/bin/env bash
{fail_block}case "$*" in
  *"label=com.docker.compose.project"*)
    cat {rows_file!s}
    ;;
  "ps --format {{{{.Names}}}}")
    cat {names_file!s}
    ;;
  *"rpk cluster health"*)
    echo "Healthy: true"
    ;;
  *)
    exit 0
    ;;
esac
"""
    docker = bin_dir / "docker"
    docker.write_text(stub)
    docker.chmod(0o755)
    return bin_dir


def _run_lane_gate(
    tmp_path: Path,
    lane_rows: str,
    broker_port: int,
    main_port: int,
    keepalive: str = "1",
    running_names: str = RUNTIME_CONTAINER_NAMES,
    daemon_ok: bool = True,
) -> subprocess.CompletedProcess[str]:
    bin_dir = _install_docker_stub(
        tmp_path, lane_rows, running_names=running_names, daemon_ok=daemon_ok
    )
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["LANE_PROBE_HOST"] = "127.0.0.1"
    env["DEV_LANE_BROKER_PORT"] = str(broker_port)
    env["DEV_LANE_MAIN_PORT"] = str(main_port)
    env["ONEX_LANE_KEEPALIVE"] = keepalive
    return subprocess.run(
        ["bash", str(HEALTH_GATE), "--lane", "--ci"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )


# ---------------------------------------------------------------------------
# The failure mode the ticket exists for: the lane is GONE.
# ---------------------------------------------------------------------------


def test_lane_absent_is_red_under_keepalive(
    tmp_path: Path,
    open_broker_port: int,
    health_server: tuple[int, type[_HealthHandler]],
) -> None:
    """Zero containers in the compose project is the OMN-15190 signature.

    Note the probe targets are deliberately REACHABLE here: the verdict must
    come from lane membership, not from an incidentally-dead port.
    """
    main_port, _ = health_server
    result = _run_lane_gate(tmp_path, "", open_broker_port, main_port)

    assert result.returncode == 1, result.stdout
    assert '"overall": "red"' in result.stdout
    assert "ZERO containers" in result.stdout
    assert "OMN-15190" in result.stdout


def test_lane_absent_is_advisory_when_keepalive_disabled(
    tmp_path: Path,
    open_broker_port: int,
    health_server: tuple[int, type[_HealthHandler]],
) -> None:
    """Negative control for the ruling itself.

    Identical input, one env flip. ``ONEX_LANE_KEEPALIVE=0`` reproduces the
    pre-ruling posture — the lane is torn down, the gate says yellow, and
    yellow exits 0. That exit-0 IS the defect the WS-4 ruling closes: the
    canonical health gate reported success while the lane stranded every
    repo's receipt path. Without this control the RED above could be an
    unconditional failure rather than a ruling-driven verdict.
    """
    main_port, _ = health_server
    result = _run_lane_gate(tmp_path, "", open_broker_port, main_port, keepalive="0")

    assert result.returncode == 0, result.stdout
    assert '"overall": "red"' not in result.stdout
    assert "ZERO containers" in result.stdout


# ---------------------------------------------------------------------------
# Reachability on the exact path CI publishers use.
# ---------------------------------------------------------------------------


def test_healthy_lane_is_green(
    tmp_path: Path,
    open_broker_port: int,
    health_server: tuple[int, type[_HealthHandler]],
) -> None:
    """Non-vacuity: the live lane census must not manufacture a red."""
    main_port, _ = health_server
    result = _run_lane_gate(tmp_path, LIVE_LANE_ROWS, open_broker_port, main_port)

    assert result.returncode == 0, result.stdout
    assert '"overall": "green"' in result.stdout
    assert "up and reachable" in result.stdout


def test_broker_port_refused_is_red(
    tmp_path: Path, health_server: tuple[int, type[_HealthHandler]]
) -> None:
    """Containers up but the publish port dead still strands the whole org."""
    main_port, _ = health_server
    result = _run_lane_gate(tmp_path, LIVE_LANE_ROWS, _closed_port(), main_port)

    assert result.returncode == 1, result.stdout
    assert '"overall": "red"' in result.stdout
    assert "refused connection" in result.stdout
    assert "occ-autobind" in result.stdout


def test_health_endpoint_non_200_is_red(
    tmp_path: Path,
    open_broker_port: int,
    health_server: tuple[int, type[_HealthHandler]],
) -> None:
    main_port, handler = health_server
    handler.status_code = 503
    result = _run_lane_gate(tmp_path, LIVE_LANE_ROWS, open_broker_port, main_port)

    assert result.returncode == 1, result.stdout
    assert "/health" in result.stdout
    assert "503" in result.stdout


# ---------------------------------------------------------------------------
# Degradation that every "is it up?" probe misses.
# ---------------------------------------------------------------------------


def test_nonzero_exit_oneshot_is_red(
    tmp_path: Path,
    open_broker_port: int,
    health_server: tuple[int, type[_HealthHandler]],
) -> None:
    """The OMN-15312 class: lane serving, schema silently unapplied.

    This exact shape was live on `.201` while this test was written —
    ``omnibase-infra-forward-migration`` Exited (3) with every runtime
    container healthy and :8085/health returning 200.
    """
    main_port, _ = health_server
    rows = (
        LIVE_LANE_ROWS
        + "\nomnibase-infra-forward-migration|exited|Exited (3) 15 minutes ago"
    )
    result = _run_lane_gate(tmp_path, rows, open_broker_port, main_port)

    assert result.returncode == 1, result.stdout
    assert "nonzero-exit" in result.stdout
    assert "omnibase-infra-forward-migration(exit 3)" in result.stdout


def test_zero_exit_oneshot_is_not_red(
    tmp_path: Path,
    open_broker_port: int,
    health_server: tuple[int, type[_HealthHandler]],
) -> None:
    """Discriminator for the test above: exited != failed.

    Migration/provisioner one-shots are SUPPOSED to exit. A check that
    flagged every exited container would be red on every healthy lane.
    """
    main_port, _ = health_server
    rows = (
        LIVE_LANE_ROWS
        + "\nomnibase-infra-redpanda-partition-cap|exited|Exited (0) 15 minutes ago"
    )
    result = _run_lane_gate(tmp_path, rows, open_broker_port, main_port)

    assert result.returncode == 0, result.stdout
    assert '"overall": "green"' in result.stdout


def test_restarting_container_is_red(
    tmp_path: Path,
    open_broker_port: int,
    health_server: tuple[int, type[_HealthHandler]],
) -> None:
    """A crash-looping lane service is a defect, not a transient."""
    main_port, _ = health_server
    rows = LIVE_LANE_ROWS + "\nomnibase-infra-migration-gate|restarting|Restarting (1)"
    result = _run_lane_gate(tmp_path, rows, open_broker_port, main_port)

    assert result.returncode == 1, result.stdout
    assert "not-running" in result.stdout
    assert "omnibase-infra-migration-gate(restarting)" in result.stdout


def test_docker_unhealthy_is_yellow_not_red(
    tmp_path: Path,
    open_broker_port: int,
    health_server: tuple[int, type[_HealthHandler]],
) -> None:
    """Docker health is a secondary signal, never the verdict.

    A running-but-unhealthy sidecar does not strand the CI publish path, and
    the fleet has twice proven Docker health alone inverts (OMN-13915:
    37/48 "Up (healthy)" with dead listeners; OMN-15233: 59/64 unhealthy
    while the registry read 64/64 online).
    """
    main_port, _ = health_server
    rows = LIVE_LANE_ROWS.replace(
        "omninode-runtime-effects|running|Up 2 hours (healthy)",
        "omninode-runtime-effects|running|Up 2 hours (unhealthy)",
    )
    result = _run_lane_gate(tmp_path, rows, open_broker_port, main_port)

    assert result.returncode == 0, result.stdout
    assert '"overall": "yellow"' in result.stdout
    assert "docker-unhealthy" in result.stdout


def test_docker_daemon_unreachable_is_red_fail_closed(
    tmp_path: Path,
    open_broker_port: int,
    health_server: tuple[int, type[_HealthHandler]],
) -> None:
    """Indeterminate is not health — the inversion OMN-13915 shipped with."""
    main_port, _ = health_server
    result = _run_lane_gate(
        tmp_path, LIVE_LANE_ROWS, open_broker_port, main_port, daemon_ok=False
    )

    assert result.returncode == 1, result.stdout
    assert "fail-closed" in result.stdout


def test_missing_runtime_containers_are_red_under_keepalive(
    tmp_path: Path,
    open_broker_port: int,
    health_server: tuple[int, type[_HealthHandler]],
) -> None:
    """The severity inversion this ticket fixes, on the second surface.

    ``runtime_containers`` scored yellow ("runtime profile not active") when
    the lane had no runtime services at all — and yellow exits 0.
    """
    main_port, _ = health_server
    result = _run_lane_gate(
        tmp_path,
        LIVE_LANE_ROWS,
        open_broker_port,
        main_port,
        running_names="omnibase-infra-redpanda",
    )

    assert result.returncode == 1, result.stdout
    assert "KEEP-ALIVE" in result.stdout


def test_missing_runtime_containers_are_advisory_without_keepalive(
    tmp_path: Path,
    open_broker_port: int,
    health_server: tuple[int, type[_HealthHandler]],
) -> None:
    """Negative control for the severity flip above."""
    main_port, _ = health_server
    result = _run_lane_gate(
        tmp_path,
        LIVE_LANE_ROWS,
        open_broker_port,
        main_port,
        keepalive="0",
        running_names="omnibase-infra-redpanda",
    )

    assert result.returncode == 0, result.stdout
    assert "runtime profile not active" in result.stdout


# ---------------------------------------------------------------------------
# Enforcement wiring: detection that does not fire is not a control.
# ---------------------------------------------------------------------------


def test_lane_probe_is_wired_to_a_firing_surface() -> None:
    workflow = yaml.safe_load(LANE_WORKFLOW.read_text())
    jobs = workflow["jobs"]

    assert "dev-lane-liveness" in jobs, (
        "the lane probe must be wired to a firing surface — an unwired script "
        "is exactly the OMN-15190 pre-state"
    )
    lane_job = jobs["dev-lane-liveness"]
    probe_step = next(
        step
        for step in lane_job["steps"]
        if "system_health_check.sh" in step.get("run", "")
    )

    assert "--lane" in probe_step["run"]
    assert probe_step["env"]["LANE_PROBE_HOST"] == "host.docker.internal", (
        "inside the deploy-runner container localhost is the container itself; "
        "an unset probe host manufactures a false RED (OMN-14958)"
    )
    assert str(probe_step["env"]["ONEX_LANE_KEEPALIVE"]) == "1"

    # `on` parses as the boolean True in YAML 1.1 unless quoted.
    triggers = workflow.get("on", workflow.get(True))
    assert "schedule" in triggers, "the probe must run on a schedule, not on demand"


def test_lane_probe_runs_where_the_lane_is_reachable() -> None:
    """The lane job is self-hosted BY DESIGN, and only on the deploy runner.

    `omnibase-deploy` is the one runner carrying both docker.sock and the
    host-gateway alias, so it is the only place either half of the probe
    (compose-project membership, lane host-port reachability) is observable.
    The lane's broker host-port is on the tailnet — GitHub-hosted compute
    could only ever assert "I cannot see it."
    """
    workflow = yaml.safe_load(LANE_WORKFLOW.read_text())
    assert workflow["jobs"]["dev-lane-liveness"]["runs-on"] == [
        "self-hosted",
        "omnibase-deploy",
    ]


def test_fleet_canary_is_not_borrowed_for_the_lane_probe() -> None:
    """The OMN-13915 fate boundary stays intact and unshared.

    The lane probe deliberately does NOT live in runner-fleet-canary.yml: that
    workflow asserts every one of its jobs is GitHub-hosted, because a canary
    sharing fate with the fleet it watches proves nothing. Folding a
    self-hosted job in would have required narrowing that guard. This pins the
    separation so a later consolidation cannot silently weaken it.
    """
    canary = yaml.safe_load(CANARY_WORKFLOW.read_text())
    assert list(canary["jobs"]) == ["fleet-status"], (
        "runner-fleet-canary.yml must stay single-job and GitHub-hosted; put "
        "lane/host-scoped probes in dev-lane-liveness.yml instead"
    )
    assert canary["jobs"]["fleet-status"]["runs-on"] == "ubuntu-latest"
