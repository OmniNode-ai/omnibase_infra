# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Stub handler for registry API operation: get_node.

Full implementation pending. See node_registry_api_effect contract.yaml
and services/registry_api/service.py for business logic.

Ticket: OMN-2909
"""

from __future__ import annotations

from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory


class HandlerRegistryApiGetNode:
    """Stub handler for operation: get_node.

    Fetches a single registered node by UUID.
    Full implementation is delegated to the FastAPI service layer.

    Raises:
        NotImplementedError: Always — full implementation pending.
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
        """Handle get_node operation.

        Raises:
            NotImplementedError: Full implementation pending.
        """
        raise NotImplementedError(
            "HandlerRegistryApiGetNode is not yet implemented. "
            "See node_registry_api_effect contract for full spec. "
            f"Correlation ID: {correlation_id}"
        )
