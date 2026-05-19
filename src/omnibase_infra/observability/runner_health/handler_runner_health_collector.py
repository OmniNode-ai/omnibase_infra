# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler wrapper around CollectorRunnerHealth — exposes handler classification metadata."""

from __future__ import annotations

from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.observability.runner_health.collector_runner_health import (
    CollectorRunnerHealth,
)
from omnibase_infra.observability.runner_health.enum_runner_health_state import (
    EnumRunnerHealthState,
)
from omnibase_infra.observability.runner_health.model_runner_health_snapshot import (
    ModelRunnerHealthSnapshot,
)


class HandlerRunnerHealthCollector:
    """ONEX handler that collects runner health from GitHub API and SSH Docker inspection.

    Wraps ``CollectorRunnerHealth`` with handler classification properties
    (``handler_id``, ``handler_type``, ``handler_category``) so it can be
    registered and routed within the ONEX handler system.
    """

    def __init__(
        self,
        github_org: str,
        runner_host: str,
        runner_count: int,
        runner_prefix: str = "omninode-runner",
    ) -> None:
        self._collector = CollectorRunnerHealth(
            github_org=github_org,
            runner_host=runner_host,
            runner_count=runner_count,
            runner_prefix=runner_prefix,
        )

    @property
    def handler_id(self) -> str:
        return "handler-runner-health-collector"

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def _classify_runner(
        self,
        github_status: str,
        github_busy: bool,
        docker_status: str,
        docker_uptime: str,
    ) -> EnumRunnerHealthState:
        """Delegate classification to the underlying collector."""
        return self._collector._classify_runner(
            github_status=github_status,
            github_busy=github_busy,
            docker_status=docker_status,
            docker_uptime=docker_uptime,
        )

    async def _fetch_github_runners(self) -> tuple[list[dict[str, object]], str | None]:
        """Delegate to underlying collector."""
        return await self._collector._fetch_github_runners()

    async def _fetch_docker_status(
        self,
    ) -> tuple[dict[str, dict[str, str]], str | None]:
        """Delegate to underlying collector."""
        return await self._collector._fetch_docker_status()

    async def _fetch_host_disk(self) -> float:
        """Delegate to underlying collector."""
        return await self._collector._fetch_host_disk()

    async def collect(self, correlation_id: UUID) -> ModelRunnerHealthSnapshot:
        """Collect a point-in-time health snapshot for all monitored runners."""
        return await self._collector.collect(correlation_id=correlation_id)


__all__ = ["HandlerRunnerHealthCollector"]
