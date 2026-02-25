# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""MCP Tool Discovery Service - Discovers MCP-enabled orchestrators from the event bus registry.

This service queries the ONEX node registration projection for nodes that have
registered with the ``mcp-enabled`` capability tag. It replaces the previous
Consul-backed discovery approach.

Discovery Flow:
    1. Query ProjectionReaderRegistration for ACTIVE nodes with tag "mcp-enabled"
    2. For each projection, extract MCP config from capabilities.mcp
    3. Filter: only ORCHESTRATOR nodes with mcp.expose=True are eligible
    4. Build ModelMCPToolDefinition from projection + mcp contract config
    5. Warn if registry returns 0 eligible nodes

Hot-reload path:
    ServiceMCPToolSync handles incremental updates via Kafka node-registration
    events (registered / updated / deregistered / expired). No Consul calls
    are needed there either; the event payload carries the full MCP metadata.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import uuid4

from omnibase_infra.enums import EnumInfraTransportType, EnumRegistrationState
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.models.mcp.model_mcp_tool_definition import ModelMCPToolDefinition

if TYPE_CHECKING:
    from omnibase_infra.projectors import ProjectionReaderRegistration

logger = logging.getLogger(__name__)


class ServiceMCPToolDiscovery:
    """Discovers MCP-enabled orchestrators from the event bus registry.

    Queries the ONEX registration projection (PostgreSQL-backed) for nodes
    that have registered with the ``mcp-enabled`` capability tag and
    ``mcp.expose: true`` in their contract configuration.

    Only nodes in the ACTIVE registration state are returned (cold-start
    discovery should not surface nodes that are pending or expired).

    Attributes:
        _reader: Projection reader for querying registered nodes.
        _query_limit: Maximum nodes to fetch from the registry per query.

    Example:
        >>> from omnibase_infra.projectors import ProjectionReaderRegistration
        >>> reader = ProjectionReaderRegistration(pool)
        >>> discovery = ServiceMCPToolDiscovery(reader)
        >>> tools = await discovery.discover_all()
        >>> for tool in tools:
        ...     print(f"{tool.name}: {tool.description}")
    """

    # Capability tag used to identify MCP-eligible nodes in the registry.
    # Nodes register this tag when they have mcp.expose=True in their contract.
    CAPABILITY_TAG_MCP_ENABLED = "mcp-enabled"

    # Node type prefix required for MCP exposure.
    # Only ORCHESTRATOR nodes may be exposed as MCP tools.
    ORCHESTRATOR_NODE_TYPE_PREFIX = "orchestrator"

    def __init__(
        self,
        reader: ProjectionReaderRegistration,
        *,
        query_limit: int = 100,
    ) -> None:
        """Initialize the discovery service.

        Args:
            reader: Projection reader for querying the ONEX node registry.
            query_limit: Maximum nodes to fetch from the registry (default: 100).
        """
        self._reader = reader
        self._query_limit = query_limit

        logger.debug(
            "ServiceMCPToolDiscovery initialized",
            extra={
                "query_limit": self._query_limit,
                "source": "event_bus_registry",
            },
        )

    async def discover_all(self) -> list[ModelMCPToolDefinition]:
        """Cold start: query the event bus registry for all MCP-enabled orchestrators.

        Queries the registration projection for all ACTIVE nodes with the
        ``mcp-enabled`` capability tag, then filters to orchestrators that
        have ``mcp.expose: true`` in their contract configuration.

        Returns:
            List of discovered tool definitions (may be empty).

        Raises:
            InfraConnectionError: If the database connection fails.
            InfraTimeoutError: If the query times out.
            RuntimeHostError: For other registry query errors.
        """
        correlation_id = uuid4()

        logger.info(
            "Starting MCP tool discovery from event bus registry",
            extra={
                "capability_tag": self.CAPABILITY_TAG_MCP_ENABLED,
                "state_filter": EnumRegistrationState.ACTIVE.value,
                "query_limit": self._query_limit,
                "correlation_id": str(correlation_id),
            },
        )

        try:
            projections = await self._reader.get_by_capability_tag(
                tag=self.CAPABILITY_TAG_MCP_ENABLED,
                state=EnumRegistrationState.ACTIVE,
                limit=self._query_limit,
                correlation_id=correlation_id,
            )
        except (InfraConnectionError, InfraTimeoutError, RuntimeHostError):
            # Already typed infra errors — re-raise with correlation intact.
            raise
        except Exception as exc:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="get_by_capability_tag",
                target_name="ProjectionReaderRegistration",
            )
            raise InfraConnectionError(
                "Unexpected error querying MCP-enabled nodes from event bus registry",
                context=context,
            ) from exc

        tools: list[ModelMCPToolDefinition] = []

        for proj in projections:
            tool = self._projection_to_tool(proj, correlation_id)
            if tool is not None:
                tools.append(tool)

        if not tools:
            logger.warning(
                "MCP tool discovery returned 0 eligible nodes",
                extra={
                    "capability_tag": self.CAPABILITY_TAG_MCP_ENABLED,
                    "state_filter": EnumRegistrationState.ACTIVE.value,
                    "query_limit": self._query_limit,
                    "registry_hits": len(projections),
                    "correlation_id": str(correlation_id),
                },
            )

        logger.info(
            "MCP tool discovery complete",
            extra={
                "tool_count": len(tools),
                "correlation_id": str(correlation_id),
            },
        )

        return tools

    def _projection_to_tool(
        self,
        proj: object,
        correlation_id: object,
    ) -> ModelMCPToolDefinition | None:
        """Convert a registration projection to an MCP tool definition.

        Validates that the projection represents an ORCHESTRATOR node with
        ``mcp.expose: true`` in its capabilities.

        Args:
            proj: A ModelRegistrationProjection instance from the registry.
            correlation_id: Correlation ID for tracing.

        Returns:
            ModelMCPToolDefinition if the projection is eligible, else None.
        """
        from omnibase_infra.models.projection import ModelRegistrationProjection

        if not isinstance(proj, ModelRegistrationProjection):
            return None

        # Only orchestrators may be exposed as MCP tools.
        node_type_str = (
            proj.node_type.value
            if hasattr(proj.node_type, "value")
            else str(proj.node_type)
        ).lower()
        if not node_type_str.startswith(self.ORCHESTRATOR_NODE_TYPE_PREFIX):
            logger.debug(
                "Skipping non-orchestrator node with mcp-enabled tag",
                extra={
                    "entity_id": str(proj.entity_id),
                    "node_type": node_type_str,
                    "correlation_id": str(correlation_id),
                },
            )
            return None

        # Require mcp.expose=True in capabilities.
        mcp_config = proj.capabilities.mcp
        if mcp_config is None or not mcp_config.expose:
            logger.debug(
                "Skipping node without mcp.expose=True in capabilities",
                extra={
                    "entity_id": str(proj.entity_id),
                    "node_type": node_type_str,
                    "has_mcp_config": mcp_config is not None,
                    "correlation_id": str(correlation_id),
                },
            )
            return None

        # Derive tool name: prefer mcp.tool_name, fall back to entity_id string.
        tool_name = mcp_config.tool_name or str(proj.entity_id)

        # Derive description from mcp config or generate one.
        description = mcp_config.description or f"ONEX orchestrator: {tool_name}"

        # Derive version from projection.
        version = str(proj.node_version) if proj.node_version else "1.0.0"

        tool = ModelMCPToolDefinition(
            name=tool_name,
            description=description,
            version=version,
            parameters=[],
            input_schema={"type": "object", "properties": {}},
            orchestrator_node_id=str(proj.entity_id),
            orchestrator_service_id=None,  # Not applicable; Consul removed
            endpoint=None,  # Endpoint resolved at invocation time via registry
            timeout_seconds=mcp_config.timeout_seconds,
            metadata={
                "entity_id": str(proj.entity_id),
                "node_type": node_type_str,
                "source": "event_bus_registry",
            },
        )

        logger.info(
            "Discovered MCP tool from event bus registry",
            extra={
                "tool_name": tool_name,
                "entity_id": str(proj.entity_id),
                "node_type": node_type_str,
                "correlation_id": str(correlation_id),
            },
        )

        return tool

    def describe(self) -> dict[str, object]:
        """Return service metadata for observability."""
        return {
            "service_name": "ServiceMCPToolDiscovery",
            "source": "event_bus_registry",
            "capability_tag": self.CAPABILITY_TAG_MCP_ENABLED,
            "query_limit": self._query_limit,
        }


__all__ = ["ServiceMCPToolDiscovery"]
