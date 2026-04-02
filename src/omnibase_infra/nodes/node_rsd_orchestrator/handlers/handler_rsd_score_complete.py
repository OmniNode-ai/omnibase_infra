# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that processes completed scoring and emits final result."""

from __future__ import annotations

import logging
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_rsd_orchestrator.models.model_rsd_result import (
    ModelRsdResult,
)
from omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_ticket_score import (
    ModelRsdTicketScore,
)

logger = logging.getLogger(__name__)


class HandlerRsdScoreComplete:
    """Receives computed scores and emits final RSD result."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self,
        correlation_id: UUID,
        ticket_scores: tuple[ModelRsdTicketScore, ...],
        ranked_ticket_ids: tuple[str, ...],
    ) -> ModelRsdResult:
        """Transform score result into final RSD result.

        Args:
            correlation_id: Workflow correlation ID.
            ticket_scores: Computed ticket scores.
            ranked_ticket_ids: Tickets in ranked order.

        Returns:
            ModelRsdResult as the final workflow output.
        """
        logger.info(
            "RSD scoring complete: %d tickets ranked (correlation_id=%s)",
            len(ticket_scores),
            correlation_id,
        )

        return ModelRsdResult(
            correlation_id=correlation_id,
            ticket_scores=ticket_scores,
            ranked_ticket_ids=ranked_ticket_ids,
            success=True,
        )
