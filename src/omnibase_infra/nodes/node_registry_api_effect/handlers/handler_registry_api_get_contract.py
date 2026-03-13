# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler for registry API operation: get_contract.

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

__all__ = ["HandlerRegistryApiGetContract"]


class HandlerRegistryApiGetContract:
    """Handler for operation: get_contract.

    Fetches a single contract by ID, delegating to
    ServiceRegistryDiscovery.get_contract().

    Attributes:
        _service: Registry discovery service instance.
    """

    def __init__(self, service: ServiceRegistryDiscovery) -> None:
        """Initialise the handler with a registry discovery service.

        Args:
            service: Registry discovery service for contract lookup.
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
        """Handle get_contract operation.

        Deserialises the request, calls ServiceRegistryDiscovery.get_contract(),
        and returns a ModelRegistryApiResponse with the contract data or
        success=False if not found.

        Args:
            request: ModelRegistryApiRequest (or mapping-compatible object).
                Must include contract_id.
            correlation_id: Distributed tracing identifier.

        Returns:
            ModelRegistryApiResponse with contract data or success=False if not found.

        Raises:
            ValueError: If contract_id is not provided in the request.
        """
        req = ModelRegistryApiRequest.model_validate(request)
        if req.contract_id is None:
            raise ValueError("contract_id is required for get_contract operation")
        contract, warnings = await self._service.get_contract(
            contract_id=str(req.contract_id),
            correlation_id=correlation_id,
        )
        logger.debug(
            "get_contract for %s found=%s",
            req.contract_id,
            contract is not None,
            extra={"correlation_id": str(correlation_id)},
        )
        return ModelRegistryApiResponse(
            operation=req.operation,
            correlation_id=correlation_id,
            success=contract is not None,
            data={
                "result": contract.model_dump(mode="json")
                if contract is not None
                else None
            },
            warnings=[w.message for w in warnings],
            error="Contract not found" if contract is None else None,
        )
