# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The stability follow retires the folded effects container that holds :18086 (OMN-24562).

Main publishes the effects port once the effects role is folded, so a runtime-effects
container left by an older compose file collides with main on the host port. The
stability lane retires exactly the profile-mapped services, and nothing else.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from deploy_agent.events import EnumRuntimeLane, Scope
from deploy_agent.executor import (
    DeployExecutor,
    _requested_services_for_up,
    lane_config_for,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
STABILITY_PROJECT = "label=com.docker.compose.project=" + (
    lane_config_for(EnumRuntimeLane.STABILITY_TEST).compose_project
)


@pytest.fixture(autouse=True)
def _committed_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read this checkout's runtime policy, not the deploy host's clone path."""
    monkeypatch.setattr(
        "deploy_agent.executor.RUNTIME_POLICY_ENV_FILE",
        REPO_ROOT / "docker/runtime-policy.env",
    )


def test_stability_disables_only_the_folded_effects_service() -> None:
    config = lane_config_for(EnumRuntimeLane.STABILITY_TEST)
    assert config.disabled_services == frozenset({"runtime-effects"})
    requested = _requested_services_for_up(
        Scope.RUNTIME, [], lane=EnumRuntimeLane.STABILITY_TEST
    )
    assert "runtime-effects" not in requested
    # Positive control: the services the lane still runs by name stay requested.
    assert {"omninode-runtime", "runtime-worker"} <= set(requested)


def test_stability_retires_the_old_effects_container_and_keeps_volumes() -> None:
    calls: list[list[str]] = []

    def run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd, 0, stdout="935bf70c5d47\n" if cmd[1] == "ps" else "", stderr=""
        )

    with patch("deploy_agent.executor._run", side_effect=run):
        DeployExecutor()._retire_folded_dev_services(EnumRuntimeLane.STABILITY_TEST)

    assert calls == [
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            STABILITY_PROJECT,
            "--filter",
            "label=com.docker.compose.service=runtime-effects",
        ],
        ["docker", "stop", "--time", "30", "935bf70c5d47"],
        ["docker", "rm", "935bf70c5d47"],
    ]
    assert all("-v" not in c and "--volumes" not in c for c in calls)


def test_stability_with_no_old_container_removes_nothing() -> None:
    calls: list[list[str]] = []

    def run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("deploy_agent.executor._run", side_effect=run):
        DeployExecutor()._retire_folded_dev_services(EnumRuntimeLane.STABILITY_TEST)

    assert [c[1] for c in calls] == ["ps"]
