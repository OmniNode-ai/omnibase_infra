# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler for registry API operation: list_nodes.

Ticket: OMN-4481
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_registry_api_effect.models import (
    ModelRegistryApiRequest,
    ModelRegistryApiResponse,
)

if TYPE_CHECKING:
    from omnibase_infra.services.registry_api.registry_discovery import (
        ServiceRegistryDiscovery,
    )

logger = logging.getLogger(__name__)

__all__ = ["HandlerRegistryApiListNodes"]


class HandlerRegistryApiListNodes:
    """Handler for operation: list_nodes.

    Lists registered nodes with optional state and node_type filters,
    delegating to ServiceRegistryDiscovery.list_nodes().

    Attributes:
        _service: Registry discovery service instance.
    """

    def __init__(self, service: ServiceRegistryDiscovery) -> None:
        """Initialise the handler with a registry discovery service.

        Args:
            service: Registry discovery service for node listing.
        """
        self._service = service

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role: INFRA_HANDLER."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification: EFFECT (external I/O)."""
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelRegistryApiRequest
    ) -> ModelRegistryApiResponse:
        """Handle list_nodes operation.

        Calls ServiceRegistryDiscovery.list_nodes(),
        and returns a ModelRegistryApiResponse containing paginated node data.

        Args:
            request: Typed registry API request with a correlation ID.

        Returns:
            ModelRegistryApiResponse with nodes list, pagination info, and
            any warnings from the service layer.
        """
        correlation_id = request.correlation_id
        nodes, pagination, warnings = await self._service.list_nodes(
            limit=request.limit,
            offset=request.offset,
            state=request.state,
            node_type=request.node_type,
            correlation_id=correlation_id,
        )
        logger.debug(
            "list_nodes returned %d nodes",
            len(nodes),
            extra={"correlation_id": str(correlation_id)},
        )
        return ModelRegistryApiResponse(
            operation=request.operation,
            correlation_id=correlation_id,
            success=True,
            data={
                "results": [n.model_dump(mode="json") for n in nodes],
                "pagination": pagination.model_dump(mode="json"),
            },
            warnings=[w.message for w in warnings],
        )
