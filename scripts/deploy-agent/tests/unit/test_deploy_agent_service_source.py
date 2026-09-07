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


def test_override_drop_in_is_wired_to_prod_broker() -> None:
    """OMN-15181 Finding 3: the single live deploy-agent must consume prod.

    Live /proc/<pid>/environ readback (2026-07-26) showed the only running
    deploy-agent process wired to KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:39092 /
    KAFKA_ENVIRONMENT=stability-test -- zero consumer presence on the prod
    broker, so a real gated prod redeploy command would sit unconsumed even
    after the network (Finding 1) and resolver (Finding 2) gaps are fixed.
    This locks the repoint to the prod broker's host-mapped external
    listener (docker/docker-compose.prod.yml,
    ${PROD_REDPANDA_EXTERNAL_PORT:-49092}:19092, advertised externally as
    192.168.86.201:49092 -- reachable from omninode-pc itself) so a future
    edit can't silently drift the only live instance back to stability-test
    (or leave it there) without failing this test.
    """
    override = _DEPLOY_DIR / "deploy-agent.service.d" / "override.conf"
    text = override.read_text()

    exec_start_lines = [
        line for line in text.splitlines() if line.startswith("ExecStart=/usr/bin/env")
    ]
    assert exec_start_lines, "override.conf must declare an ExecStart override"
    exec_start = exec_start_lines[0]

    assert "KAFKA_BOOTSTRAP_SERVERS=192.168.86.201:49092" in exec_start
    assert "KAFKA_ENVIRONMENT=prod" in exec_start
    assert "127.0.0.1:39092" not in exec_start
    assert "stability-test" not in exec_start

    env_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("Environment=KAFKA_ENVIRONMENT=")
    ]
    assert env_lines == ["Environment=KAFKA_ENVIRONMENT=prod"]


def test_dev_lane_unit_exists_and_is_fenced_to_dev() -> None:
    """OMN-16939: the dev-lane instance must be a tracked unit, not a hand-edit.

    The prod drop-in pins deploy-agent.service to 192.168.86.201:49092, so
    before this unit existed the DEV broker's rebuild-requested topic had zero
    consumer groups and four signed commands sat on it unconsumed.
    """
    text = (_DEPLOY_DIR / "deploy-agent-dev.service").read_text()

    assert "Environment=DEPLOY_AGENT_ALLOWED_LANES=dev" in text
    # The assertion that keeps the dev instance off the prod bus: the literal
    # dev broker address is the point of the test, not a fallback.
    dev_broker_line = (
        "Environment=KAFKA_BOOTSTRAP_SERVERS=192.168.86.201:19092"  # kafka-fallback-ok
    )
    assert dev_broker_line in text
    # must not collide with the prod instance's port or state dir
    assert "Environment=DEPLOY_AGENT_PORT=8098" in text
    assert (
        "Environment=DEPLOY_AGENT_STATE_DIR=/data/omninode/deploy-agent/state/jobs-dev"
        in text
    )
    # canonical repo copy for both cwd and interpreter (OMN-13760)
    assert "WorkingDirectory=/data/omninode/omnibase_infra/scripts/deploy-agent" in text
    exec_start_lines = [
        line for line in text.splitlines() if line.startswith("ExecStart=")
    ]
    assert exec_start_lines
    assert all(
        "/data/omninode/omnibase_infra/scripts/deploy-agent/.venv/bin/python" in line
        for line in exec_start_lines
    ), exec_start_lines
    assert not any(ln.strip().startswith("WatchdogSec=") for ln in text.splitlines())


def test_dev_unit_does_not_touch_the_prod_bus() -> None:
    """Negative control for the test above: the dev unit must never name the
    prod broker port, and the prod drop-in must never name the dev one."""
    dev = (_DEPLOY_DIR / "deploy-agent-dev.service").read_text()
    prod_override = (
        _DEPLOY_DIR / "deploy-agent.service.d" / "override.conf"
    ).read_text()

    dev_directives = [
        ln for ln in dev.splitlines() if ln and not ln.lstrip().startswith("#")
    ]
    assert not any("49092" in ln for ln in dev_directives), dev_directives
    assert "192.168.86.201:49092" in prod_override
    assert "Environment=DEPLOY_AGENT_ALLOWED_LANES=prod" in prod_override
