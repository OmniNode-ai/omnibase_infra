# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler for registry API operation: list_nodes.

Ticket: OMN-4481
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_registry_api_effect.models import (
    ModelRegistryApiRequest,
    ModelRegistryApiResponse,
)

if TYPE_CHECKING:
    from omnibase_infra.services.registry_api.service import ServiceRegistryDiscovery

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

    async def handle(self, request: object, correlation_id: UUID) -> object:
        """Handle list_nodes operation.

        Deserialises the request, calls ServiceRegistryDiscovery.list_nodes(),
        and returns a ModelRegistryApiResponse containing paginated node data.

        Args:
            request: ModelRegistryApiRequest (or mapping-compatible object).
            correlation_id: Distributed tracing identifier.

        Returns:
            ModelRegistryApiResponse with nodes list, pagination info, and
            any warnings from the service layer.
        """
        req = ModelRegistryApiRequest.model_validate(request)
        nodes, pagination, warnings = await self._service.list_nodes(
            limit=req.limit,
            offset=req.offset,
            state=req.state,
            node_type=req.node_type,
            correlation_id=correlation_id,
        )
        logger.debug(
            "list_nodes returned %d nodes",
            len(nodes),
            extra={"correlation_id": str(correlation_id)},
        )
        return ModelRegistryApiResponse(
            operation=req.operation,
            correlation_id=correlation_id,
            success=True,
            data={
                "results": [n.model_dump(mode="json") for n in nodes],
                "pagination": pagination.model_dump(mode="json"),
            },
            warnings=[w.message for w in warnings],
        )
