# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Protocol for topic catalog service implementations.

Defines ``ProtocolTopicCatalogService`` — the structural interface shared by
``ServiceTopicCatalog`` (Consul KV) and ``ServiceTopicCatalogPostgres``
(PostgreSQL). Any object that implements ``build_catalog`` with the correct
signature satisfies this protocol.

Related Tickets:
    - OMN-2746: Replace ServiceTopicCatalog Consul KV backend with PostgreSQL

.. versionadded:: 0.10.0
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from omnibase_infra.models.catalog.model_topic_catalog_response import (
    ModelTopicCatalogResponse,
)


@runtime_checkable
class ProtocolTopicCatalogService(Protocol):
    """Structural interface for topic catalog service implementations.

    Any class that exposes ``build_catalog`` with the correct signature
    satisfies this protocol — no explicit inheritance required.

    Implementations:
        - ``ServiceTopicCatalog``: Consul KV backend (legacy)
        - ``ServiceTopicCatalogPostgres``: PostgreSQL backend (OMN-2746)

    Example:
        >>> def make_handler(svc: ProtocolTopicCatalogService) -> HandlerTopicCatalogQuery:
        ...     return HandlerTopicCatalogQuery(catalog_service=svc)
    """

    async def build_catalog(
        self,
        correlation_id: UUID,
        include_inactive: bool = False,
        topic_pattern: str | None = None,
    ) -> ModelTopicCatalogResponse:
        """Build (or return cached) topic catalog snapshot.

        Args:
            correlation_id: Correlation ID for tracing.
            include_inactive: Include topics with no publishers/subscribers.
            topic_pattern: Optional fnmatch glob to filter topic suffixes.

        Returns:
            ModelTopicCatalogResponse with topics and any partial-failure warnings.
        """
        ...


__all__: list[str] = ["ProtocolTopicCatalogService"]
