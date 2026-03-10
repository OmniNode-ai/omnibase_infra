# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler for registry API operation: list_topics.

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

__all__ = ["HandlerRegistryApiListTopics"]


class HandlerRegistryApiListTopics:
    """Handler for operation: list_topics.

    Lists topics with pagination, delegating to
    ServiceRegistryDiscovery.list_topics().

    Attributes:
        _service: Registry discovery service instance.
    """

    def __init__(self, service: ServiceRegistryDiscovery) -> None:
        """Initialise the handler with a registry discovery service.

        Args:
            service: Registry discovery service for topic listing.
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
        """Handle list_topics operation.

        Deserialises the request, calls ServiceRegistryDiscovery.list_topics(),
        and returns a ModelRegistryApiResponse with paginated topic data.

        Args:
            request: ModelRegistryApiRequest (or mapping-compatible object).
            correlation_id: Distributed tracing identifier.

        Returns:
            ModelRegistryApiResponse with topics list, pagination info, and
            any warnings from the service layer.
        """
        req = ModelRegistryApiRequest.model_validate(request)
        topics, pagination, warnings = await self._service.list_topics(
            limit=req.limit,
            offset=req.offset,
            correlation_id=correlation_id,
        )
        logger.debug(
            "list_topics returned %d topics",
            len(topics),
            extra={"correlation_id": str(correlation_id)},
        )
        return ModelRegistryApiResponse(
            operation=req.operation,
            correlation_id=correlation_id,
            success=True,
            data={
                "results": [t.model_dump() for t in topics],
                "pagination": pagination.model_dump(),
            },
            warnings=[w.message for w in warnings],
        )
