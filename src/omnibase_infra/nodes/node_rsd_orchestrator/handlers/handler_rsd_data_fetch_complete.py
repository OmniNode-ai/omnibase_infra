# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that processes completed data fetch and dispatches scoring."""

from __future__ import annotations

import logging
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_rsd_data_fetch_effect.models.model_agent_request_data import (
    ModelAgentRequestData,
)
from omnibase_infra.nodes.node_rsd_data_fetch_effect.models.model_dependency_edge import (
    ModelDependencyEdge,
)
from omnibase_infra.nodes.node_rsd_data_fetch_effect.models.model_plan_override_data import (
    ModelPlanOverrideData,
)
from omnibase_infra.nodes.node_rsd_data_fetch_effect.models.model_ticket_data import (
    ModelTicketData,
)
from omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_factor_weights import (
    ModelRsdFactorWeights,
)
from omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_score_input import (
    ModelRsdScoreInput,
)

logger = logging.getLogger(__name__)


class HandlerRsdDataFetchComplete:
    """Receives fetched data and dispatches to score compute node."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self,
        correlation_id: UUID,
        tickets: tuple[ModelTicketData, ...],
        dependency_edges: tuple[ModelDependencyEdge, ...],
        agent_requests: tuple[ModelAgentRequestData, ...],
        plan_overrides: tuple[ModelPlanOverrideData, ...],
        success: bool,
        fetch_errors: tuple[str, ...] = (),
    ) -> ModelRsdScoreInput:
        """Transform fetched data into score compute input.

        Args:
            correlation_id: Workflow correlation ID.
            tickets: Fetched ticket data.
            dependency_edges: Dependency graph edges.
            agent_requests: Agent requests.
            plan_overrides: Plan overrides.
            success: Whether fetch succeeded.
            fetch_errors: Errors from fetch.

        Returns:
            ModelRsdScoreInput to dispatch to the compute node.
        """
        if not success:
            logger.warning(
                "Data fetch had errors (correlation_id=%s): %s",
                correlation_id,
                fetch_errors,
            )

        logger.info(
            "Data fetch complete: %d tickets, %d edges (correlation_id=%s)",
            len(tickets),
            len(dependency_edges),
            correlation_id,
        )

        return ModelRsdScoreInput(
            correlation_id=correlation_id,
            tickets=tickets,
            dependency_edges=dependency_edges,
            agent_requests=agent_requests,
            plan_overrides=plan_overrides,
            weights=ModelRsdFactorWeights(),
        )
