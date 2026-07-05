# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that starts a runner-fleet-maintain tick by dispatching a snapshot gather (OMN-13942)."""

from __future__ import annotations

import logging
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_runner_fleet_snapshot_gather_command import (
    ModelRunnerFleetSnapshotGatherCommand,
)

logger = logging.getLogger(__name__)


class HandlerRunnerFleetMaintainStart:
    """Receives the maintain-start command and emits a snapshot-gather command.

    No I/O here -- this ORCHESTRATOR handler only routes; the EFFECT node
    does the actual gh-api/SSH gathering.
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self, correlation_id: UUID
    ) -> ModelRunnerFleetSnapshotGatherCommand:
        """Start a runner-fleet-maintain tick.

        Args:
            correlation_id: Workflow correlation ID for this tick.

        Returns:
            ModelRunnerFleetSnapshotGatherCommand for the snapshot EFFECT node.
        """
        logger.info(
            "Starting runner-fleet-maintain tick (correlation_id=%s)", correlation_id
        )
        return ModelRunnerFleetSnapshotGatherCommand(correlation_id=correlation_id)


__all__ = ["HandlerRunnerFleetMaintainStart"]
