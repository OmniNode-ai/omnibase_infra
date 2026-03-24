# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for runner health collector."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.observability.runner_health.collector_runner_health import (
    CollectorRunnerHealth,
)
from omnibase_infra.observability.runner_health.enum_runner_health_state import (
    EnumRunnerHealthState,
)


@pytest.mark.unit
class TestCollectorRunnerHealth:
    @pytest.fixture
    def handler(self) -> CollectorRunnerHealth:
        return CollectorRunnerHealth(
            github_org="OmniNode-ai",
            runner_host="192.168.86.201",
            runner_count=10,
            runner_prefix="omninode-runner",
        )

    def test_classify_healthy(self, handler: CollectorRunnerHealth) -> None:
        state = handler._classify_runner(
            github_status="online",
            github_busy=False,
            docker_status="healthy",
            docker_uptime="Up 2 hours (healthy)",
        )
        assert state == EnumRunnerHealthState.HEALTHY

    def test_classify_github_offline_docker_healthy(
        self, handler: CollectorRunnerHealth
    ) -> None:
        state = handler._classify_runner(
            github_status="offline",
            github_busy=False,
            docker_status="healthy",
            docker_uptime="Up 1 hour (healthy)",
        )
        assert state == EnumRunnerHealthState.GITHUB_OFFLINE

    def test_classify_crash_loop(self, handler: CollectorRunnerHealth) -> None:
        state = handler._classify_runner(
            github_status="offline",
            github_busy=False,
            docker_status="restarting",
            docker_uptime="Restarting (1) 5 seconds ago",
        )
        assert state == EnumRunnerHealthState.CRASH_LOOPING

    def test_classify_missing(self, handler: CollectorRunnerHealth) -> None:
        state = handler._classify_runner(
            github_status="offline",
            github_busy=False,
            docker_status="not_found",
            docker_uptime="",
        )
        assert state == EnumRunnerHealthState.MISSING

    @pytest.mark.asyncio
    async def test_collect_with_mocked_sources(
        self, handler: CollectorRunnerHealth
    ) -> None:
        github_data: list[dict[str, object]] = [
            {"name": "omninode-runner-1", "status": "online", "busy": False},
            {"name": "omninode-runner-2", "status": "offline", "busy": False},
        ]
        docker_data: dict[str, dict[str, str]] = {
            "omninode-runner-1": {"status": "healthy", "uptime": "Up 2h (healthy)"},
            "omninode-runner-2": {
                "status": "restarting",
                "uptime": "Restarting (1) 5s ago",
            },
        }

        with (
            patch.object(
                handler,
                "_fetch_github_runners",
                new_callable=AsyncMock,
                return_value=(github_data, None),
            ),
            patch.object(
                handler,
                "_fetch_docker_status",
                new_callable=AsyncMock,
                return_value=(docker_data, None),
            ),
            patch.object(
                handler,
                "_fetch_host_disk",
                new_callable=AsyncMock,
                return_value=38.0,
            ),
        ):
            snapshot = await handler.collect(correlation_id=uuid4())

        assert snapshot.observed_runners == 2
        assert snapshot.healthy_count == 1
        assert snapshot.degraded_count == 1
        assert snapshot.runners[1].state == EnumRunnerHealthState.CRASH_LOOPING
        assert snapshot.github_source_ok
        assert snapshot.docker_source_ok

    @pytest.mark.asyncio
    async def test_collect_github_failure(self, handler: CollectorRunnerHealth) -> None:
        with (
            patch.object(
                handler,
                "_fetch_github_runners",
                new_callable=AsyncMock,
                return_value=([], "GitHub API exit code 1: Not Found"),
            ),
            patch.object(
                handler,
                "_fetch_docker_status",
                new_callable=AsyncMock,
                return_value=({}, None),
            ),
            patch.object(
                handler,
                "_fetch_host_disk",
                new_callable=AsyncMock,
                return_value=25.0,
            ),
        ):
            snapshot = await handler.collect(correlation_id=uuid4())

        assert not snapshot.github_source_ok
        assert snapshot.docker_source_ok
        assert len(snapshot.source_errors) == 1
        assert "GitHub" in snapshot.source_errors[0]
        assert snapshot.observed_runners == 0

    @pytest.mark.asyncio
    async def test_collect_orphaned_docker_container(
        self, handler: CollectorRunnerHealth
    ) -> None:
        github_data: list[dict[str, object]] = [
            {"name": "omninode-runner-1", "status": "online", "busy": False},
        ]
        docker_data: dict[str, dict[str, str]] = {
            "omninode-runner-1": {"status": "healthy", "uptime": "Up 2h (healthy)"},
            "omninode-runner-99": {"status": "running", "uptime": "Up 5d"},
        }

        with (
            patch.object(
                handler,
                "_fetch_github_runners",
                new_callable=AsyncMock,
                return_value=(github_data, None),
            ),
            patch.object(
                handler,
                "_fetch_docker_status",
                new_callable=AsyncMock,
                return_value=(docker_data, None),
            ),
            patch.object(
                handler,
                "_fetch_host_disk",
                new_callable=AsyncMock,
                return_value=25.0,
            ),
        ):
            snapshot = await handler.collect(correlation_id=uuid4())

        assert snapshot.observed_runners == 2
        orphan = [r for r in snapshot.runners if r.name == "omninode-runner-99"]
        assert len(orphan) == 1
        assert orphan[0].state == EnumRunnerHealthState.STALE_REGISTRATION
