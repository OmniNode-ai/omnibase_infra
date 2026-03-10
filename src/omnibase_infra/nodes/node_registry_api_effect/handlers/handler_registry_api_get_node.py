# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler for registry API operation: get_node.

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

__all__ = ["HandlerRegistryApiGetNode"]


class HandlerRegistryApiGetNode:
    """Handler for operation: get_node.

    Fetches a single registered node by UUID, delegating to
    ServiceRegistryDiscovery.get_node().

    Attributes:
        _service: Registry discovery service instance.
    """

    def __init__(self, service: ServiceRegistryDiscovery) -> None:
        """Initialise the handler with a registry discovery service.

        Args:
            service: Registry discovery service for single node lookup.
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
        """Handle get_node operation.

        Deserialises the request, calls ServiceRegistryDiscovery.get_node(),
        and returns a ModelRegistryApiResponse with the node data or
        success=False if not found.

        Args:
            request: ModelRegistryApiRequest (or mapping-compatible object).
                Must include node_id.
            correlation_id: Distributed tracing identifier.

        Returns:
            ModelRegistryApiResponse with node data or success=False if not found.

        Raises:
            ValueError: If node_id is not provided in the request.
        """
        req = ModelRegistryApiRequest.model_validate(request)
        if req.node_id is None:
            raise ValueError("node_id is required for get_node operation")
        node, warnings = await self._service.get_node(
            node_id=req.node_id,
            correlation_id=correlation_id,
        )
        logger.debug(
            "get_node for %s found=%s",
            req.node_id,
            node is not None,
            extra={"correlation_id": str(correlation_id)},
        )
        return ModelRegistryApiResponse(
            operation=req.operation,
            correlation_id=correlation_id,
            success=node is not None,
            data={"result": node.model_dump() if node is not None else None},
            warnings=[w.message for w in warnings],
            error="Node not found" if node is None else None,
        )
