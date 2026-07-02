# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Systemd unit must execute deploy-agent code from the canonical repo copy."""

from pathlib import Path

_DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deploy"


def test_service_uses_canonical_repo_source() -> None:
    service = _DEPLOY_DIR / "deploy-agent.service"
    text = service.read_text()

    assert "WorkingDirectory=/data/omninode/omnibase_infra/scripts/deploy-agent" in text
    assert (
        "Environment=DEPLOY_AGENT_DIR=/data/omninode/omnibase_infra/scripts/deploy-agent"
        in text
    )
    assert "WorkingDirectory=/data/omninode/deploy-agent" not in text

    # ExecStart must invoke the interpreter from a venv rooted under the
    # canonical repo copy, not the legacy standalone `/data/omninode/deploy-agent`
    # install. A prior drift shipped WorkingDirectory pointed at the canonical
    # path while ExecStart still hardcoded the legacy venv's python -- systemd
    # ignores WorkingDirectory for resolving the ExecStart binary path, so that
    # combination silently ran the stale legacy venv/interpreter.
    exec_start_lines = [
        line for line in text.splitlines() if line.startswith("ExecStart=")
    ]
    assert exec_start_lines, "service file must declare ExecStart"
    assert all(
        "/data/omninode/omnibase_infra/scripts/deploy-agent/.venv/bin/python" in line
        for line in exec_start_lines
    ), exec_start_lines
    assert all(
        "/data/omninode/deploy-agent/venv" not in line for line in exec_start_lines
    )


def test_service_declares_no_watchdog() -> None:
    """OMN-13760: the base unit must not arm WatchdogSec.

    The agent runs minutes-long synchronous rebuilds that block the event loop,
    so a systemd liveness watchdog SIGABRTs it mid-rebuild. Restart=on-failure
    covers genuine crashes instead.
    """
    text = (_DEPLOY_DIR / "deploy-agent.service").read_text()
    unit_lines = [ln.strip() for ln in text.splitlines()]
    assert not any(ln.startswith("WatchdogSec=") for ln in unit_lines), (
        "deploy-agent.service must not declare WatchdogSec (see OMN-13760)"
    )


def test_override_drop_in_disables_watchdog() -> None:
    """The tracked .201 drop-in must explicitly disable the watchdog.

    Belt-and-suspenders in case an older base unit carrying WatchdogSec=30 is
    still installed on the host.
    """
    override = _DEPLOY_DIR / "deploy-agent.service.d" / "override.conf"
    text = override.read_text()
    assert "WatchdogSec=0" in text
