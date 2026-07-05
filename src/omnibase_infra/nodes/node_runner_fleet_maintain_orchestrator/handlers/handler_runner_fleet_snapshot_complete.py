# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that receives a gathered snapshot and dispatches health classification (OMN-13942)."""

from __future__ import annotations

import logging

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.model_runner_fleet_health_evaluate_command import (
    ModelRunnerFleetHealthEvaluateCommand,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_runner_fleet_snapshot import (
    ModelRunnerFleetSnapshot,
)

logger = logging.getLogger(__name__)


class HandlerRunnerFleetSnapshotComplete:
    """Receives the gathered snapshot and emits a health-evaluate command.

    No I/O here -- this ORCHESTRATOR handler only routes; classification
    happens in the COMPUTE node.
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self, snapshot: ModelRunnerFleetSnapshot
    ) -> ModelRunnerFleetHealthEvaluateCommand:
        """Transform a gathered snapshot into a health-evaluate command.

        Args:
            snapshot: Facts-only snapshot from node_runner_health_snapshot_effect.

        Returns:
            ModelRunnerFleetHealthEvaluateCommand for the health COMPUTE node.
        """
        logger.info(
            "Snapshot gathered: %d runners observed (correlation_id=%s)",
            len(snapshot.runners),
            snapshot.correlation_id,
        )
        return ModelRunnerFleetHealthEvaluateCommand(
            correlation_id=snapshot.correlation_id,
            snapshot=snapshot,
        )


__all__ = ["HandlerRunnerFleetSnapshotComplete"]
