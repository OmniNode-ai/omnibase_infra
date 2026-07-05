# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that receives a health verdict and emits the terminal maintain-completed event (OMN-13942).

Increment 1 terminates here: the verdict (health report + recorded
recommended actions) is published. NOTHING in this handler -- or anywhere
reachable from this workflow -- restarts a runner, cancels a run, or
dequeues/re-enqueues a merge-queue entry. Increment 2 (design-only, not
built) would insert a grant-gated recovery EFFECT + FSM reducer between this
handler and the terminal event.
"""

from __future__ import annotations

import logging

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.model_runner_fleet_health_verdict import (
    ModelRunnerFleetHealthVerdict,
)
from omnibase_infra.nodes.node_runner_fleet_maintain_orchestrator.models.model_runner_fleet_maintain_completed_event import (
    ModelRunnerFleetMaintainCompletedEvent,
)

logger = logging.getLogger(__name__)


class HandlerRunnerFleetHealthVerdictComplete:
    """Receives the classified health verdict and emits the terminal completed event."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self, verdict: ModelRunnerFleetHealthVerdict
    ) -> ModelRunnerFleetMaintainCompletedEvent:
        """Build the terminal runner-fleet-maintain-completed event.

        Args:
            verdict: Classified health verdict from the COMPUTE node.

        Returns:
            ModelRunnerFleetMaintainCompletedEvent (health report, no mutation).
        """
        logger.info(
            "Runner-fleet-maintain tick complete: %d/%d online, %d recommended actions "
            "(recorded only, correlation_id=%s)",
            verdict.online_count,
            verdict.expected_count,
            len(verdict.recommended_actions),
            verdict.correlation_id,
        )
        return ModelRunnerFleetMaintainCompletedEvent(
            correlation_id=verdict.correlation_id,
            verdict=verdict,
        )


__all__ = ["HandlerRunnerFleetHealthVerdictComplete"]
