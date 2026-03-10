# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler for registry API operation: get_discovery.

Returns full dashboard payload by composing list_nodes, list_instances,
and get_widget_mapping handlers. Does not duplicate their logic.

Ticket: OMN-4482
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
    from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_get_widget_mapping import (
        HandlerRegistryApiGetWidgetMapping,
    )
    from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_list_instances import (
        HandlerRegistryApiListInstances,
    )
    from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_list_nodes import (
        HandlerRegistryApiListNodes,
    )

logger = logging.getLogger(__name__)

__all__ = ["HandlerRegistryApiGetDiscovery"]


class HandlerRegistryApiGetDiscovery:
    """Handler for operation: get_discovery.

    Aggregates the full dashboard payload by composing three handlers:
    - ``HandlerRegistryApiListNodes`` — registered nodes
    - ``HandlerRegistryApiListInstances`` — live Consul instances
    - ``HandlerRegistryApiGetWidgetMapping`` — capability-to-widget config

    Does not duplicate logic from the composed handlers.

    Attributes:
        _list_nodes: Handler for listing registered nodes.
        _list_instances: Handler for listing Consul instances.
        _get_widget_mapping: Handler for loading widget mapping YAML.
    """

    def __init__(
        self,
        list_nodes: HandlerRegistryApiListNodes,
        list_instances: HandlerRegistryApiListInstances,
        get_widget_mapping: HandlerRegistryApiGetWidgetMapping,
    ) -> None:
        """Initialise the handler with composed sub-handlers.

        Args:
            list_nodes: Handler that lists registered nodes.
            list_instances: Handler that lists Consul service instances.
            get_widget_mapping: Handler that loads widget mapping configuration.
        """
        self._list_nodes = list_nodes
        self._list_instances = list_instances
        self._get_widget_mapping = get_widget_mapping

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role: INFRA_HANDLER."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification: EFFECT (external I/O)."""
        return EnumHandlerTypeCategory.EFFECT

    async def handle(self, request: object, correlation_id: UUID) -> object:
        """Handle get_discovery operation.

        Calls list_nodes, list_instances, and get_widget_mapping in sequence
        and merges their data payloads into a single discovery response.

        Args:
            request: ModelRegistryApiRequest (or compatible mapping).
            correlation_id: Distributed tracing identifier.

        Returns:
            ModelRegistryApiResponse with aggregated discovery payload.
        """
        nodes_request = ModelRegistryApiRequest(
            operation="list_nodes",
            correlation_id=correlation_id,
        )

        nodes_resp = await self._list_nodes.handle(nodes_request, correlation_id)
        instances_resp = await self._list_instances.handle(object(), correlation_id)
        mapping_resp = await self._get_widget_mapping.handle(object(), correlation_id)

        warnings: list[str] = []
        nodes_data: dict = {}
        instances_data: dict = {}
        mapping_data: dict = {}

        if isinstance(nodes_resp, ModelRegistryApiResponse):
            nodes_data = nodes_resp.data
            warnings.extend(nodes_resp.warnings)
        if isinstance(instances_resp, ModelRegistryApiResponse):
            instances_data = instances_resp.data
            warnings.extend(instances_resp.warnings)
        if isinstance(mapping_resp, ModelRegistryApiResponse):
            mapping_data = mapping_resp.data
            warnings.extend(mapping_resp.warnings)

        discovery_payload = {
            "nodes": nodes_data.get("nodes", []),
            "nodes_total": nodes_data.get("total", 0),
            "instances": instances_data.get("instances", []),
            "instances_total": instances_data.get("total", 0),
            "widget_mapping": mapping_data.get("widget_mapping", {}),
        }

        logger.debug(
            "HandlerRegistryApiGetDiscovery: aggregated %d nodes, %d instances",
            discovery_payload["nodes_total"],
            discovery_payload["instances_total"],
        )

        return ModelRegistryApiResponse(
            operation="get_discovery",
            correlation_id=correlation_id,
            success=True,
            data=discovery_payload,
            warnings=warnings,
        )
