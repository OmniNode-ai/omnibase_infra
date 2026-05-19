# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for HandlerRunnerHealthCollector snapshot reconciliation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.observability.runner_health.enum_runner_health_state import (
    EnumRunnerHealthState,
)
from omnibase_infra.observability.runner_health.handler_runner_health_collector import (
    HandlerRunnerHealthCollector,
)

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_runner_health_handler_reconciles_github_and_docker_sources() -> None:
    handler = HandlerRunnerHealthCollector(
        github_org="OmniNode-ai",
        runner_host="ci-host",
        runner_count=3,
        runner_prefix="omninode-runner",
    )
    github_data: list[dict[str, object]] = [
        {"name": "omninode-runner-1", "status": "online", "busy": False},
        {"name": "omninode-runner-2", "status": "offline", "busy": False},
    ]
    docker_data = {
        "omninode-runner-1": {"status": "healthy", "uptime": "Up 2h (healthy)"},
        "omninode-runner-2": {
            "status": "restarting",
            "uptime": "Restarting (1) 5s ago",
        },
        "omninode-runner-3": {"status": "running", "uptime": "Up 4h"},
    }

    with (
        patch.object(
            handler._collector,
            "_fetch_github_runners",
            new_callable=AsyncMock,
            return_value=(github_data, None),
        ),
        patch.object(
            handler._collector,
            "_fetch_docker_status",
            new_callable=AsyncMock,
            return_value=(docker_data, None),
        ),
        patch.object(
            handler._collector,
            "_fetch_host_disk",
            new_callable=AsyncMock,
            return_value=31.0,
        ),
    ):
        snapshot = await handler.collect(correlation_id=uuid4())

    states = {runner.name: runner.state for runner in snapshot.runners}
    assert states == {
        "omninode-runner-1": EnumRunnerHealthState.HEALTHY,
        "omninode-runner-2": EnumRunnerHealthState.CRASH_LOOPING,
        "omninode-runner-3": EnumRunnerHealthState.STALE_REGISTRATION,
    }
    assert snapshot.expected_runners == 3
    assert snapshot.observed_runners == 3
    assert snapshot.healthy_count == 1
    assert snapshot.degraded_count == 2
    assert snapshot.github_source_ok
    assert snapshot.docker_source_ok
    assert snapshot.host == "ci-host"
