# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23525: a roll retires the infra routing decisions consumer's container.

omnibase_infra#5803 (OMN-23072) dropped ``infra-routing-decisions-consumer`` from
the dev compose files, and the roll that started its container ran ten seconds
before that merge, so the container of the previous revision kept running on
h201. It holds consumer group ``infra-routing-decisions-postgres``, which
``projection_infra_routing_decisions`` takes over in main, and the runtime's
attach fails with ``ConsumerGroupFenceError`` (main DEGRADED). Compose only warns
about an orphan; the deploy agent retires only the services the runtime policy
names in ``DEV_RUNTIME_MAIN_RETIRED_SERVICES``, so the name belongs there.

These tests read the shipped policy (no patched fold profiles or policy values).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.executor import DeployExecutor, lane_config_for

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
CONSUMER = "infra-routing-decisions-consumer"


@pytest.fixture(autouse=True)
def _committed_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read this checkout's policy, not the deploy host's clone path."""
    monkeypatch.setattr(
        "deploy_agent.executor.RUNTIME_POLICY_ENV_FILE",
        REPO_ROOT / "docker/runtime-policy.env",
    )


def test_the_policy_retires_the_routing_decisions_consumer() -> None:
    contract = yaml.safe_load(
        (REPO_ROOT / "contracts/services/runtime_policy.contract.yaml").read_text(
            "utf-8"
        )
    )
    main = contract["profiles"]["dev"]["services"]["main"]
    assert CONSUMER in main["retired_services"]
    line = next(
        item
        for item in (REPO_ROOT / "docker/runtime-policy.env")
        .read_text("utf-8")
        .splitlines()
        if item.startswith("DEV_RUNTIME_MAIN_RETIRED_SERVICES=")
    )
    assert CONSUMER in line.split("=", 1)[1].split(",")


def test_the_agent_retires_its_container_with_the_shipped_policy() -> None:
    calls: list[list[str]] = []
    target = f"label=com.docker.compose.service={CONSUMER}"

    def run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        found = cmd[1] == "ps" and target in cmd
        return subprocess.CompletedProcess(
            cmd, 0, stdout="routing-a\n" if found else "", stderr=""
        )

    assert CONSUMER in lane_config_for(EnumRuntimeLane.DEV).disabled_services
    with patch("deploy_agent.executor._run", side_effect=run):
        DeployExecutor()._retire_folded_dev_services(EnumRuntimeLane.DEV)

    project = (
        "label=com.docker.compose.project="
        f"{lane_config_for(EnumRuntimeLane.DEV).compose_project}"
    )
    census = [c for c in calls if c[1] == "ps" and target in c]
    assert len(census) == 1
    assert project in census[0]
    assert [c for c in calls if c[1] == "stop"] == [
        ["docker", "stop", "--time", "30", "routing-a"]
    ]
    assert [c for c in calls if c[1] == "rm"] == [["docker", "rm", "routing-a"]]
    assert all("-v" not in c and "--volumes" not in c for c in calls)
