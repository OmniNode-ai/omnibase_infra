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
  "exec omninode-gateway-forwarder true")
    # Reachability probe (require_reachable()). GW_TEST_UNREACHABLE_BEFORE
    # simulates a container that was never reachable in the first place;
    # GW_TEST_UNREACHABLE_AFTER simulates one that stopped responding
    # sometime during the restart (discriminated by whether the reload has
    # already been logged -- the same ordering the real script uses).
    if [ "${GW_TEST_UNREACHABLE_BEFORE:-0}" = "1" ]; then
      exit 1
    fi
    if [ "${GW_TEST_UNREACHABLE_AFTER:-0}" = "1" ] && [ -s "${GW_TEST_SYSTEMCTL_LOG:-/dev/null}" ]; then
      exit 1
    fi
    exit 0
    ;;
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
  "inspect omninode-gateway-forwarder --format {{.Id}}|{{.State.StartedAt}}")
    cat "${GW_TEST_IDENTITY_STATE:?}"
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
  # Simulate the reload actually RECREATING the container (default: yes --
  # matches real `systemctl reload` per deploy-gateway.sh's own doc). Set
  # GW_TEST_RELOAD_RECREATES=0 to simulate a reload that exits 0 and reports
  # healthy without recreating anything -- the false-green this script's
  # container_identity() check exists to catch.
  if [ "${GW_TEST_RELOAD_RECREATES:-1}" = "1" ] && [ -n "${GW_TEST_IDENTITY_STATE:-}" ]; then
    printf 'id-after-reload|2026-08-01T00:05:00.000000000Z' > "${GW_TEST_IDENTITY_STATE}"
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
        self.identity_state = tmp_path / "identity.state"
        _write_fake_bin(self.bin_dir)
        self.snapshot_state.write_text("292\t1785611578.75", encoding="utf-8")
        self.identity_state.write_text(
            "id-before-reload|2026-08-01T00:00:00.000000000Z", encoding="utf-8"
        )

    def env(self, **overrides: str) -> dict[str, str]:
        e = os.environ.copy()
        e["PATH"] = f"{self.bin_dir}{os.pathsep}{e['PATH']}"
        e["GW_TEST_SYSTEMCTL_LOG"] = str(self.systemctl_log)
        e["GW_TEST_SNAPSHOT_STATE"] = str(self.snapshot_state)
        e["GW_TEST_IDENTITY_STATE"] = str(self.identity_state)
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
    assert "Container identity changed across restart" in result.stdout
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


# ---------------------------------------------------------------------------
# Remediation round 3 (OMN-15521): a fake `sudo` that exits 0 and does
# nothing must not read as a successful restart-safety proof. The prior
# version only checked Docker-healthy + row-count-not-decreasing, both of
# which a stale, never-recreated container trivially satisfies.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_proof_red_when_reload_does_not_actually_recreate_container(
    harness: _Harness,
) -> None:
    """RED (exists-but-wrong, not merely absent): a `sudo systemctl reload`
    that exits 0, reports the container Docker-healthy (because the STALE
    container was already healthy), and leaves row counts unchanged (no
    GW_TEST_AFTER_SNAPSHOT override, so before == after) must NOT be reported
    as AC5-PROOF OK. Proved by execution with a fake `sudo` that never
    recreates the container (GW_TEST_RELOAD_RECREATES=0): before this fix,
    this exact fixture printed 'AC5-PROOF OK ... rows 292 -> 292' with
    returncode 0. The container's own identity (Id + StartedAt) is the only
    signal that distinguishes this from a genuine restart.
    """
    result = harness.run(
        GW_TEST_HEALTH_STATUS="healthy",
        GW_TEST_RELOAD_RECREATES="0",
    )
    assert result.returncode != 0
    assert "AC5-PROOF FAILED" in result.stdout + result.stderr
    assert "was NOT actually recreated" in result.stdout + result.stderr
    assert "AC5-PROOF OK" not in result.stdout


@pytest.mark.unit
def test_proof_red_when_container_unreachable_after_restart(
    harness: _Harness,
) -> None:
    """RED (OMN-15521 remediation round 3): if the container stops
    responding to `docker exec` sometime during the restart (reachable
    before, not after), the proof must fail loudly instead of silently
    reading the idempotency store as empty. Before this fix, snapshot()'s own
    `except Exception: print('0\\t0')` (plus the `|| printf '0\\t0\\n'`
    fallback) collapsed an unreachable container to the same "0\\t0" reading
    as a genuinely empty store, so after_count(0) < before_count(292) was the
    only signal -- and if before_count also happened to read 0 (e.g. a
    concurrently-unreachable pre-check), the comparison went vacuously green.
    """
    result = harness.run(
        GW_TEST_HEALTH_STATUS="healthy",
        GW_TEST_UNREACHABLE_AFTER="1",
    )
    assert result.returncode != 0
    assert "AC5-PROOF FAILED" in result.stdout + result.stderr
    assert "not reachable via 'docker exec'" in result.stdout + result.stderr
    assert "after restart" in result.stdout + result.stderr
    assert "AC5-PROOF OK" not in result.stdout


@pytest.mark.unit
def test_proof_red_when_container_unreachable_before_restart(
    harness: _Harness,
) -> None:
    """Fail-closed precondition: the proof must refuse to even start against
    a container it cannot reach before mutating anything (reloading a
    container it never proved was reachable would make the "before" snapshot
    meaningless).
    """
    result = harness.run(GW_TEST_UNREACHABLE_BEFORE="1")
    assert result.returncode != 0
    assert "AC5-PROOF FAILED" in result.stdout + result.stderr
    assert "not reachable via 'docker exec'" in result.stdout + result.stderr
    assert "before restart" in result.stdout + result.stderr
    assert not harness.systemctl_log.exists(), (
        "must not attempt the restart at all if the pre-restart reachability "
        "check fails"
    )
