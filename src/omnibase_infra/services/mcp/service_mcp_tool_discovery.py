# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""MCP Tool Discovery Service - Discovers MCP-enabled orchestrators from Consul.

This service scans Consul for services with MCP tags and converts them to
MCP tool definitions. It supports both cold start discovery (scan all) and
incremental discovery (single node lookup).

Discovery Flow:
    1. Query Consul catalog for services with tags: mcp-enabled, node-type:orchestrator
    2. For each service, extract mcp-tool:{name} tag for stable tool naming
    3. Load contract metadata from service or fall back to defaults
    4. Generate JSON Schema from Pydantic input model (if available)
    5. Return ModelMCPToolDefinition instances

Tag Schema:
    - mcp-enabled: Indicates the service is MCP-enabled
    - mcp-tool:{name}: The stable tool name for MCP invocation
    - node-type:orchestrator: Required for MCP enablement (non-orchestrators ignored)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from urllib.parse import urlparse
from uuid import uuid4

import consul
import requests

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    ModelInfraErrorContext,
)
from omnibase_infra.models.mcp.model_mcp_tool_definition import ModelMCPToolDefinition

logger = logging.getLogger(__name__)


class ServiceMCPToolDiscovery:
    """Discovers MCP-enabled orchestrators from Consul.

    This service provides two main discovery methods:
    1. discover_all(): Cold start scan for all MCP-enabled orchestrators
    2. discover_by_service_id(): Re-fetch single service (Kafka fallback)

    Attributes:
        _container: The ONEX dependency injection container.
        _consul_host: Consul server hostname.
        _consul_port: Consul server port.
        _consul_scheme: HTTP scheme (http/https).
        _consul_token: Optional ACL token for authentication.

    Example:
        >>> from omnibase_core.models.container.model_onex_container import (
        ...     create_model_onex_container,
        ... )
        >>> container = await create_model_onex_container()
        >>> discovery = ServiceMCPToolDiscovery(container)
        >>> tools = await discovery.discover_all()
        >>> for tool in tools:
        ...     print(f"{tool.name}: {tool.description}")
    """

    # Tag constants for MCP discovery
    TAG_MCP_ENABLED = "mcp-enabled"
    TAG_NODE_TYPE_ORCHESTRATOR = "node-type:orchestrator"
    TAG_PREFIX_MCP_TOOL = "mcp-tool:"

    def __init__(
        self,
        container: ModelONEXContainer | None = None,
        *,
        consul_host: str = "localhost",
        consul_port: int = 8500,
        consul_scheme: str = "http",
        consul_token: str | None = None,
    ) -> None:
        """Initialize the discovery service.

        Supports two initialization patterns:
        1. Container-based DI: Pass a ModelONEXContainer to resolve config
        2. Direct injection: Pass individual Consul parameters

        Args:
            container: Optional ONEX dependency injection container. If provided,
                Consul configuration is resolved from container.config.
            consul_host: Consul host (used if container not provided)
            consul_port: Consul port (used if container not provided)
            consul_scheme: Consul scheme (used if container not provided)
            consul_token: Consul ACL token (used if container not provided)
        """
        self._container = container

        if container is not None:
            # Resolve Consul configuration from container
            # Configuration() returns the underlying dict or Mapping
            config_data = container.config()

            # Handle both dict and Mapping types (e.g., MappingProxyType, ChainMap)
            if isinstance(config_data, Mapping):
                consul_config_raw = config_data.get("consul", {})
                # consul_config may also be a Mapping, convert to dict for consistency
                if isinstance(consul_config_raw, Mapping):
                    consul_config: dict[str, object] = dict(consul_config_raw)
                else:
                    logger.warning(
                        "Unexpected consul config type, expected Mapping",
                        extra={
                            "config_type": type(consul_config_raw).__name__,
                            "config_value": repr(consul_config_raw)[:100],
                        },
                    )
                    consul_config = {}
            else:
                logger.warning(
                    "Unexpected config_data type from container.config(), "
                    "Consul settings may be dropped",
                    extra={
                        "config_type": type(config_data).__name__,
                        "expected_types": "dict or Mapping",
                    },
                )
                consul_config = {}

            agent_url_raw = consul_config.get("agent_url", "http://localhost:8500")
            agent_url = str(agent_url_raw) if agent_url_raw else "http://localhost:8500"

            # Parse the agent_url to extract host, port, and scheme
            parsed = urlparse(agent_url)
            self._consul_host = parsed.hostname or "localhost"
            self._consul_port = parsed.port or 8500
            self._consul_scheme = parsed.scheme or "http"
            token_raw = consul_config.get("token")
            self._consul_token = str(token_raw) if token_raw is not None else None
        else:
            # Use directly provided parameters
            self._consul_host = consul_host
            self._consul_port = consul_port
            self._consul_scheme = consul_scheme
            self._consul_token = consul_token

        logger.debug(
            "ServiceMCPToolDiscovery initialized",
            extra={
                "consul_host": self._consul_host,
                "consul_port": self._consul_port,
                "consul_scheme": self._consul_scheme,
            },
        )

    def _create_consul_client(self) -> consul.Consul:
        """Create a Consul client instance."""
        return consul.Consul(
            host=self._consul_host,
            port=self._consul_port,
            scheme=self._consul_scheme,
            token=self._consul_token,
        )

    async def discover_all(self) -> list[ModelMCPToolDefinition]:
        """Cold start: scan Consul for all MCP-enabled orchestrators.

        This method queries Consul for all services with MCP tags and
        converts them to tool definitions.

        Returns:
            List of discovered tool definitions.

        Raises:
            InfraConnectionError: If Consul connection fails.
        """
        correlation_id = uuid4()

        logger.info(
            "Starting MCP tool discovery",
            extra={"correlation_id": str(correlation_id)},
        )

        try:
            client = self._create_consul_client()

            # Get all services from catalog (blocking call wrapped with to_thread)
            _, services = await asyncio.to_thread(client.catalog.services)

            tools: list[ModelMCPToolDefinition] = []

            for service_name, tags in services.items():
                # Check if service is MCP-enabled orchestrator
                if not self._is_mcp_orchestrator(tags):
                    continue

                # Extract tool name from tags
                tool_name = self._extract_tool_name(tags)
                if not tool_name:
                    logger.warning(
                        "MCP-enabled service missing mcp-tool tag",
                        extra={
                            "service_name": service_name,
                            "tags": tags,
                            "correlation_id": str(correlation_id),
                        },
                    )
                    continue

                # Get service instances for endpoint info (blocking call wrapped with to_thread)
                _, service_instances = await asyncio.to_thread(
                    client.health.service, service_name, passing=True
                )

                # Use first healthy instance for endpoint
                endpoint = None
                service_id = None
                if service_instances:
                    instance = service_instances[0]
                    svc = instance.get("Service", {})
                    address = svc.get("Address") or instance.get("Node", {}).get(
                        "Address"
                    )
                    port = svc.get("Port")
                    service_id = svc.get("ID")
                    if address and port:
                        endpoint = f"http://{address}:{port}"

                # Build tool definition
                tool = ModelMCPToolDefinition(
                    name=tool_name,
                    description=f"ONEX orchestrator: {service_name}",
                    version="1.0.0",
                    parameters=[],  # Will be populated from contract
                    input_schema={"type": "object", "properties": {}},
                    orchestrator_node_id=None,  # Not available from Consul
                    orchestrator_service_id=service_id,
                    endpoint=endpoint,
                    timeout_seconds=30,
                    metadata={
                        "service_name": service_name,
                        "tags": list(tags),
                        "source": "consul_discovery",
                    },
                )
                tools.append(tool)

                logger.info(
                    "Discovered MCP tool",
                    extra={
                        "tool_name": tool_name,
                        "service_name": service_name,
                        "endpoint": endpoint,
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

        except (consul.ConsulException, requests.exceptions.RequestException) as e:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.CONSUL,
                operation="discover_all",
                target_name="mcp_tool_discovery",
            )
            raise InfraConnectionError(
                f"Failed to discover MCP tools from Consul: {e}",
                context=ctx,
            ) from e

    async def discover_by_service_id(
        self, service_id: str
    ) -> ModelMCPToolDefinition | None:
        """Re-fetch single service (Kafka fallback when event lacks full data).

        Args:
            service_id: Consul service ID to look up.

        Returns:
            Tool definition if found and MCP-enabled, None otherwise.

        Raises:
            InfraConnectionError: If Consul connection fails.
        """
        correlation_id = uuid4()

        logger.debug(
            "Looking up service by ID",
            extra={
                "service_id": service_id,
                "correlation_id": str(correlation_id),
            },
        )

        try:
            client = self._create_consul_client()

            # Get all services and find the one matching our service_id
            # Note: Consul catalog doesn't directly support service_id lookup,
            # so we need to iterate through service instances
            # (blocking call wrapped with to_thread)
            _, services = await asyncio.to_thread(client.catalog.services)

            for service_name, tags in services.items():
                if not self._is_mcp_orchestrator(tags):
                    continue

                # Get instances for this service (blocking call wrapped with to_thread)
                _, instances = await asyncio.to_thread(
                    client.health.service, service_name, passing=True
                )

                for instance in instances:
                    svc = instance.get("Service", {})
                    if svc.get("ID") == service_id:
                        # Found the service
                        tool_name = self._extract_tool_name(tags)
                        if not tool_name:
                            return None

                        address = svc.get("Address") or instance.get("Node", {}).get(
                            "Address"
                        )
                        port = svc.get("Port")
                        endpoint = (
                            f"http://{address}:{port}" if address and port else None
                        )

                        return ModelMCPToolDefinition(
                            name=tool_name,
                            description=f"ONEX orchestrator: {service_name}",
                            version="1.0.0",
                            parameters=[],
                            input_schema={"type": "object", "properties": {}},
                            orchestrator_node_id=None,
                            orchestrator_service_id=service_id,
                            endpoint=endpoint,
                            timeout_seconds=30,
                            metadata={
                                "service_name": service_name,
                                "tags": list(tags),
                                "source": "consul_discovery",
                            },
                        )

            logger.debug(
                "Service not found or not MCP-enabled",
                extra={
                    "service_id": service_id,
                    "correlation_id": str(correlation_id),
                },
            )
            return None

        except (consul.ConsulException, requests.exceptions.RequestException) as e:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.CONSUL,
                operation="discover_by_service_id",
                target_name="mcp_tool_discovery",
            )
            raise InfraConnectionError(
                f"Failed to look up service from Consul: {e}",
                context=ctx,
            ) from e

    def _is_mcp_orchestrator(self, tags: Sequence[str]) -> bool:
        """Check if service is an MCP-enabled orchestrator.

        Args:
            tags: List of service tags from Consul.

        Returns:
            True if service has both mcp-enabled and node-type:orchestrator tags.
        """
        return self.TAG_MCP_ENABLED in tags and self.TAG_NODE_TYPE_ORCHESTRATOR in tags

    def _extract_tool_name(self, tags: Sequence[str]) -> str | None:
        """Extract the MCP tool name from service tags.

        Args:
            tags: List of service tags from Consul.

        Returns:
            The tool name if found, None otherwise.
        """
        for tag in tags:
            if tag.startswith(self.TAG_PREFIX_MCP_TOOL):
                return tag[len(self.TAG_PREFIX_MCP_TOOL) :]
        return None

    def describe(self) -> dict[str, object]:
        """Return service metadata for observability."""
        return {
            "service_name": "ServiceMCPToolDiscovery",
            "consul_host": self._consul_host,
            "consul_port": self._consul_port,
            "consul_scheme": self._consul_scheme,
        }


__all__ = ["ServiceMCPToolDiscovery"]
