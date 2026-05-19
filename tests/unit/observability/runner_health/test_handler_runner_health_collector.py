# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerRunnerHealthCollector."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.observability.runner_health.enum_runner_health_state import (
    EnumRunnerHealthState,
)
from omnibase_infra.observability.runner_health.handler_runner_health_collector import (
    HandlerRunnerHealthCollector,
)


@pytest.mark.unit
class TestHandlerRunnerHealthCollector:
    @pytest.fixture
    def handler(self) -> HandlerRunnerHealthCollector:
        return HandlerRunnerHealthCollector(
            github_org="OmniNode-ai",
            runner_host="192.168.86.201",
            runner_count=10,
            runner_prefix="omninode-runner",
        )

    def test_handler_id(self, handler: HandlerRunnerHealthCollector) -> None:
        assert handler.handler_id == "handler-runner-health-collector"

    def test_handler_type(self, handler: HandlerRunnerHealthCollector) -> None:
        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER

    def test_handler_category(self, handler: HandlerRunnerHealthCollector) -> None:
        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT

    def test_classify_missing(self, handler: HandlerRunnerHealthCollector) -> None:
        state = handler._classify_runner(
            github_status="offline",
            github_busy=False,
            docker_status="not_found",
            docker_uptime="",
        )
        assert state == EnumRunnerHealthState.MISSING

    def test_classify_crash_looping(
        self, handler: HandlerRunnerHealthCollector
    ) -> None:
        state = handler._classify_runner(
            github_status="offline",
            github_busy=False,
            docker_status="restarting",
            docker_uptime="Restarting (1) 5 seconds ago",
        )
        assert state == EnumRunnerHealthState.CRASH_LOOPING

    def test_classify_docker_unhealthy(
        self, handler: HandlerRunnerHealthCollector
    ) -> None:
        state = handler._classify_runner(
            github_status="online",
            github_busy=False,
            docker_status="exited",
            docker_uptime="Exited (1) 3 minutes ago",
        )
        assert state == EnumRunnerHealthState.DOCKER_UNHEALTHY

    def test_classify_github_offline(
        self, handler: HandlerRunnerHealthCollector
    ) -> None:
        state = handler._classify_runner(
            github_status="offline",
            github_busy=False,
            docker_status="healthy",
            docker_uptime="Up 1 hour (healthy)",
        )
        assert state == EnumRunnerHealthState.GITHUB_OFFLINE

    def test_classify_healthy(self, handler: HandlerRunnerHealthCollector) -> None:
        state = handler._classify_runner(
            github_status="online",
            github_busy=False,
            docker_status="healthy",
            docker_uptime="Up 2 hours (healthy)",
        )
        assert state == EnumRunnerHealthState.HEALTHY

    @pytest.mark.asyncio
    async def test_collect_returns_snapshot(
        self, handler: HandlerRunnerHealthCollector
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
                return_value=42.0,
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
    async def test_collect_github_source_failure(
        self, handler: HandlerRunnerHealthCollector
    ) -> None:
        with (
            patch.object(
                handler._collector,
                "_fetch_github_runners",
                new_callable=AsyncMock,
                return_value=([], "GitHub API exit code 1: Not Found"),
            ),
            patch.object(
                handler._collector,
                "_fetch_docker_status",
                new_callable=AsyncMock,
                return_value=({}, None),
            ),
            patch.object(
                handler._collector,
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
    async def test_collect_docker_source_failure(
        self, handler: HandlerRunnerHealthCollector
    ) -> None:
        github_data: list[dict[str, object]] = [
            {"name": "omninode-runner-1", "status": "online", "busy": False},
        ]
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
                return_value=({}, "SSH/Docker exit code 255: Connection refused"),
            ),
            patch.object(
                handler._collector,
                "_fetch_host_disk",
                new_callable=AsyncMock,
                return_value=0.0,
            ),
        ):
            snapshot = await handler.collect(correlation_id=uuid4())

        assert snapshot.github_source_ok
        assert not snapshot.docker_source_ok
        assert len(snapshot.source_errors) == 1
        assert "SSH" in snapshot.source_errors[0]

    def test_collect_uses_asyncio_subprocess(
        self, handler: HandlerRunnerHealthCollector
    ) -> None:
        """Verify underlying collector uses asyncio.create_subprocess_exec (non-blocking)."""
        import inspect

        assert "create_subprocess_exec" in inspect.getsource(
            handler._collector._fetch_github_runners
        )
        assert "create_subprocess_exec" in inspect.getsource(
            handler._collector._fetch_docker_status
        )


__all__: list[str] = []
