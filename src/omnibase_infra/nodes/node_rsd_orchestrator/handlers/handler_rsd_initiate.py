# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that initiates the RSD scoring workflow."""

from __future__ import annotations

import logging
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_rsd_data_fetch_effect.models.model_rsd_data_fetch_request import (
    ModelRsdDataFetchRequest,
)
from omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_factor_weights import (
    ModelRsdFactorWeights,
)

logger = logging.getLogger(__name__)


class HandlerRsdInitiate:
    """Receives RSD scoring command and dispatches data fetch."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self,
        correlation_id: UUID,
        ticket_ids: tuple[str, ...],
        weights: ModelRsdFactorWeights,
        include_dependencies: bool = True,
        include_agent_requests: bool = True,
        include_plan_overrides: bool = True,
    ) -> ModelRsdDataFetchRequest:
        """Transform RSD command into a data fetch request.

        Args:
            correlation_id: Workflow correlation ID.
            ticket_ids: Tickets to score.
            weights: Factor weights.
            include_dependencies: Whether to fetch dependency data.
            include_agent_requests: Whether to fetch agent requests.
            include_plan_overrides: Whether to fetch plan overrides.

        Returns:
            ModelRsdDataFetchRequest to dispatch to the effect node.
        """
        logger.info(
            "RSD scoring initiated for %d tickets (correlation_id=%s)",
            len(ticket_ids),
            correlation_id,
        )

        return ModelRsdDataFetchRequest(
            correlation_id=correlation_id,
            ticket_ids=ticket_ids,
            include_dependencies=include_dependencies,
            include_agent_requests=include_agent_requests,
            include_plan_overrides=include_plan_overrides,
        )
