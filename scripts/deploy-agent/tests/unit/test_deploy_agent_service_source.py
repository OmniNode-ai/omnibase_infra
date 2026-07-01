# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Systemd unit must execute deploy-agent code from the canonical repo copy."""

from pathlib import Path


def test_service_uses_canonical_repo_source() -> None:
    service = Path(__file__).resolve().parents[2] / "deploy" / "deploy-agent.service"
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
