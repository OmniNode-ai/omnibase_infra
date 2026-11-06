# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that fetches ticket and dependency data for RSD scoring.

This is an EFFECT handler - performs I/O against Linear API or database.
"""

from __future__ import annotations

import logging

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_rsd_data_fetch_effect.models.model_rsd_data_fetch_request import (
    ModelRsdDataFetchRequest,
)
from omnibase_infra.nodes.node_rsd_data_fetch_effect.models.model_rsd_data_fetch_result import (
    ModelRsdDataFetchResult,
)

logger = logging.getLogger(__name__)


class HandlerRsdDataFetch:
    """Fetches ticket data, dependency edges, agent requests, and plan overrides."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        request: ModelRsdDataFetchRequest,
    ) -> ModelRsdDataFetchResult:
        """Fetch all RSD-relevant data for the given tickets.

        Returns empty results as a baseline. Wire to Linear API or database
        adapter via the container registry for production data.

        Args:
            request: Workflow correlation ID, tickets and data inclusion flags.

        Returns:
            ModelRsdDataFetchResult with fetched data.
        """
        correlation_id = request.correlation_id
        ticket_ids = request.ticket_ids
        include_dependencies = request.include_dependencies
        include_agent_requests = request.include_agent_requests
        include_plan_overrides = request.include_plan_overrides
        logger.info(
            "Fetching RSD data for %d tickets (correlation_id=%s)",
            len(ticket_ids),
            correlation_id,
        )

        # stub-ok: effect handler baseline — wire adapter via container registry
        return ModelRsdDataFetchResult(
            correlation_id=correlation_id,
            tickets=(),
            dependency_edges=(),
            agent_requests=(),
            plan_overrides=(),
            fetch_errors=(),
            success=True,
        )
