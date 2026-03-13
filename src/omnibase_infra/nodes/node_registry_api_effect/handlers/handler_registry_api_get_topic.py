# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler for registry API operation: get_topic.

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

__all__ = ["HandlerRegistryApiGetTopic"]


class HandlerRegistryApiGetTopic:
    """Handler for operation: get_topic.

    Fetches a single topic by suffix, delegating to
    ServiceRegistryDiscovery.get_topic_detail().

    Attributes:
        _service: Registry discovery service instance.
    """

    def __init__(self, service: ServiceRegistryDiscovery) -> None:
        """Initialise the handler with a registry discovery service.

        Args:
            service: Registry discovery service for topic detail lookup.
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
        """Handle get_topic operation.

        Deserialises the request, calls ServiceRegistryDiscovery.get_topic_detail(),
        and returns a ModelRegistryApiResponse with the topic data or
        success=False if not found.

        Args:
            request: ModelRegistryApiRequest (or mapping-compatible object).
                Must include topic_suffix.
            correlation_id: Distributed tracing identifier.

        Returns:
            ModelRegistryApiResponse with topic data or success=False if not found.

        Raises:
            ValueError: If topic_suffix is not provided in the request.
        """
        req = ModelRegistryApiRequest.model_validate(request)
        if req.topic_suffix is None:
            raise ValueError("topic_suffix is required for get_topic operation")
        topic, warnings = await self._service.get_topic_detail(
            topic_suffix=req.topic_suffix,
            correlation_id=correlation_id,
        )
        logger.debug(
            "get_topic for %r found=%s",
            req.topic_suffix,
            topic is not None,
            extra={"correlation_id": str(correlation_id)},
        )
        return ModelRegistryApiResponse(
            operation=req.operation,
            correlation_id=correlation_id,
            success=topic is not None,
            data={
                "result": topic.model_dump(mode="json") if topic is not None else None
            },
            warnings=[w.message for w in warnings],
            error="Topic not found" if topic is None else None,
        )
