# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler for registry API operation: list_instances.

Consul service discovery was decommissioned (OMN-4857). This handler always
returns an empty instance list. Full implementation via the PostgreSQL registry
is pending (OMN-2909).

Ticket: OMN-4857
"""

from __future__ import annotations

from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_registry_api_effect.models import (
    ModelRegistryApiResponse,
)

__all__ = ["HandlerRegistryApiListInstances"]


class HandlerRegistryApiListInstances:
    """Handler for operation: list_instances.

    Consul service discovery was decommissioned (OMN-4857). Returns an empty
    instance list. A PostgreSQL-backed implementation is planned under OMN-2909.
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role: INFRA_HANDLER."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification: EFFECT (external I/O)."""
        return EnumHandlerTypeCategory.EFFECT

    async def handle(self, request: object, correlation_id: UUID) -> object:
        """Handle list_instances operation.

        Consul was decommissioned (OMN-4857). Always returns an empty instance
        list. A PostgreSQL-backed implementation is tracked under OMN-2909.

        Args:
            request: Ignored.
            correlation_id: Distributed tracing identifier.

        Returns:
            ModelRegistryApiResponse with an empty instance list in ``data``.
        """
        return ModelRegistryApiResponse(
            operation="list_instances",
            correlation_id=correlation_id,
            success=True,
            data={"instances": [], "total": 0},
        )
