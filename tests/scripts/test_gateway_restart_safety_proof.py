# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""scripts/gateway_restart_safety_proof.sh -- OMN-15521 AC5 remediation.

A prior remediation-round finding (OMN-15521) was that AC5 -- "the OMN-12912
restart/redelivery proof (source-offset-ack / dedupe receipt) is re-run
against the newly deployed forwarder" -- had NO implementing work at all: the
PR body and runbook converted the ticket's own "that receipt lands on
OMN-12912, not this ticket" filing instruction into a blanket "out of scope,"
skipping the proof even though the container was force-recreated during the
same session (the one moment it was cheap to capture).

This script is the implementing work: a real, executable restart-durability
smoke proof driven via subprocess against the REAL script, exactly the
convention tests/scripts/test_deploy_gateway.py already uses, with
`docker`/`sudo` replaced by inspectable fakes.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROOF_SCRIPT = REPO_ROOT / "scripts" / "gateway_restart_safety_proof.sh"

_FAKE_DOCKER = """#!/usr/bin/env bash
set -eu
case "$*" in
  "exec omninode-gateway-forwarder python3 -c "*)
    state="${GW_TEST_SNAPSHOT_STATE:?}"
    if [ -f "${state}" ]; then
      cat "${state}"
    else
      printf '0\\t0\\n'
    fi
    exit 0
    ;;
  "inspect omninode-gateway-forwarder --format {{.State.Health.Status}}")
    printf '%s\\n' "${GW_TEST_HEALTH_STATUS:-healthy}"
    exit 0
    ;;
  *)
    printf 'fake docker: unexpected invocation: %s\\n' "$*" >&2
    exit 1
    ;;
esac
"""

_FAKE_SUDO = """#!/usr/bin/env bash
set -eu
if [ "${1:-}" = "systemctl" ] && [ "${2:-}" = "reload" ]; then
  printf '%s\\n' "$*" >> "${GW_TEST_SYSTEMCTL_LOG:?}"
  # Simulate the restart producing GW_TEST_AFTER_SNAPSHOT (if set) by
  # overwriting the snapshot state file the fake docker exec reads.
  if [ -n "${GW_TEST_AFTER_SNAPSHOT:-}" ] && [ -n "${GW_TEST_SNAPSHOT_STATE:-}" ]; then
    printf '%s' "${GW_TEST_AFTER_SNAPSHOT}" > "${GW_TEST_SNAPSHOT_STATE}"
  fi
  exit 0
fi
exec "$@"
"""


def _write_fake_bin(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    docker = bin_dir / "docker"
    docker.write_text(_FAKE_DOCKER, encoding="utf-8")
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)

    sudo = bin_dir / "sudo"
    sudo.write_text(_FAKE_SUDO, encoding="utf-8")
    sudo.chmod(sudo.stat().st_mode | stat.S_IEXEC)


class _Harness:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.bin_dir = tmp_path / "bin"
        self.systemctl_log = tmp_path / "systemctl.log"
        self.snapshot_state = tmp_path / "snapshot.state"
        _write_fake_bin(self.bin_dir)
        self.snapshot_state.write_text("292\t1785611578.75", encoding="utf-8")

    def env(self, **overrides: str) -> dict[str, str]:
        e = os.environ.copy()
        e["PATH"] = f"{self.bin_dir}{os.pathsep}{e['PATH']}"
        e["GW_TEST_SYSTEMCTL_LOG"] = str(self.systemctl_log)
        e["GW_TEST_SNAPSHOT_STATE"] = str(self.snapshot_state)
        e["GATEWAY_RESTART_PROOF_HEALTHY_TIMEOUT_SECONDS"] = "3"
        e.update(overrides)
        return e

    def run(self, *args: str, **env_overrides: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(PROOF_SCRIPT), *args],
            cwd=REPO_ROOT,
            env=self.env(**env_overrides),
            capture_output=True,
            text=True,
            check=False,
        )


@pytest.fixture
def harness(tmp_path: Path) -> _Harness:
    return _Harness(tmp_path)


@pytest.mark.unit
def test_script_exists_and_is_executable() -> None:
    assert PROOF_SCRIPT.is_file(), (
        "scripts/gateway_restart_safety_proof.sh must exist (AC5)"
    )
    mode = PROOF_SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, (
        "scripts/gateway_restart_safety_proof.sh must be executable"
    )


@pytest.mark.unit
def test_help_documents_ac5_scope() -> None:
    result = subprocess.run(
        ["bash", str(PROOF_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "OMN-15521" in result.stdout
    assert "OMN-12912" in result.stdout


@pytest.mark.unit
def test_proof_green_when_records_survive_restart(harness: _Harness) -> None:
    """GREEN: rows before (292) <= rows after (293, i.e. traffic kept
    flowing and nothing was lost) and the container reports healthy -- the
    exact restart-durability probe this script exists to run.
    """
    result = harness.run(
        GW_TEST_HEALTH_STATUS="healthy",
        GW_TEST_AFTER_SNAPSHOT="293\t1785611600.0",
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "AC5-PROOF OK" in result.stdout
    assert "rows 292 -> 293" in result.stdout
    assert (
        "systemctl reload onex-gateway-forwarder" in harness.systemctl_log.read_text()
    )


@pytest.mark.unit
def test_proof_red_when_records_are_lost_across_restart(harness: _Harness) -> None:
    """RED (exists-but-wrong): the durable marker store must not lose rows
    across a restart. Simulates exactly the failure OMN-12912's WAL +
    synchronous=FULL design is meant to prevent -- the store came back with
    FEWER rows than before the restart.
    """
    result = harness.run(
        GW_TEST_HEALTH_STATUS="healthy",
        GW_TEST_AFTER_SNAPSHOT="0\t0",
    )
    assert result.returncode != 0
    assert "AC5-PROOF FAILED" in result.stdout + result.stderr
    assert "did NOT survive" in result.stdout + result.stderr


@pytest.mark.unit
def test_proof_red_when_container_never_reports_healthy(harness: _Harness) -> None:
    """RED: a restart that never returns the container to Docker-healthy
    must fail loudly, not report success -- this is the "false-green
    gateway" OMN-12912's own PR description names as the failure mode its
    readiness gating exists to prevent.
    """
    result = harness.run(
        GW_TEST_HEALTH_STATUS="unhealthy",
        GW_TEST_AFTER_SNAPSHOT="293\t1785611600.0",
    )
    assert result.returncode != 0
    assert "did not report Docker-healthy" in result.stdout + result.stderr


@pytest.mark.unit
def test_proof_warns_on_empty_store_before_restart(harness: _Harness) -> None:
    """An empty store before the restart makes the proof trivially true (an
    empty set survives anything) -- must warn, not silently claim strong
    evidence.
    """
    harness.snapshot_state.write_text("0\t0", encoding="utf-8")
    result = harness.run(
        GW_TEST_HEALTH_STATUS="healthy",
        GW_TEST_AFTER_SNAPSHOT="0\t0",
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "trivially true" in result.stdout + result.stderr
