# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""MCP Server Lifecycle - Orchestrates startup and shutdown of MCP services.

This module provides the MCPServerLifecycle class that manages the complete
lifecycle of the MCP server, including:
- Cold start: Discover tools from Consul and populate registry
- Hot reload: Start Kafka subscription for real-time updates
- Handler initialization: Set up HandlerMCP with registry and executor
- Graceful shutdown: Clean up all resources

Architecture:
    MCPServerLifecycle acts as the composition root for MCP services, wiring
    together the discovery, registry, sync, and execution components.

Usage:
    ```python
    lifecycle = MCPServerLifecycle(container=container, config=config)
    await lifecycle.start()
    # ... server is running ...
    await lifecycle.shutdown()
    ```
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.adapters.adapter_onex_tool_execution import (
    AdapterONEXToolExecution,
)
from omnibase_infra.models.mcp.model_mcp_server_config import ModelMCPServerConfig
from omnibase_infra.services.mcp.service_mcp_tool_discovery import (
    ServiceMCPToolDiscovery,
)
from omnibase_infra.services.mcp.service_mcp_tool_registry import (
    ServiceMCPToolRegistry,
)
from omnibase_infra.services.mcp.service_mcp_tool_sync import ServiceMCPToolSync

if TYPE_CHECKING:
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.handlers.handler_mcp import HandlerMCP
    from omnibase_infra.models.mcp.model_mcp_tool_definition import (
        ModelMCPToolDefinition,
    )

logger = logging.getLogger(__name__)


class MCPServerLifecycle:
    """Orchestrates startup and shutdown of MCP server components.

    This class manages the lifecycle of all MCP-related services:
    - ServiceMCPToolRegistry: In-memory cache of tool definitions
    - ServiceMCPToolDiscovery: Consul scanner for MCP-enabled orchestrators
    - ServiceMCPToolSync: Kafka listener for hot reload
    - AdapterONEXToolExecution: Dispatcher bridge for tool execution

    Lifecycle Phases:
        1. start(): Initialize services and populate registry
           - Create registry, discovery, executor
           - Cold start: scan Consul for MCP-enabled orchestrators
           - Start Kafka subscription (if enabled)
        2. get_handler(): Create configured HandlerMCP
        3. shutdown(): Clean up all resources

    Attributes:
        _container: ONEX container for dependency injection.
        _config: Server configuration.
        _registry: Tool registry instance.
        _discovery: Consul discovery service.
        _sync: Kafka sync service (if enabled).
        _executor: Tool execution adapter.
        _started: Whether the lifecycle has been started.

    Example:
        >>> config = ModelMCPServerConfig(
        ...     consul_host="consul.local",
        ...     http_port=8090,
        ... )
        >>> lifecycle = MCPServerLifecycle(container=container, config=config)
        >>> await lifecycle.start()
        >>> handler = lifecycle.get_handler()
        >>> # Use handler with uvicorn/transport
        >>> await lifecycle.shutdown()
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        config: ModelMCPServerConfig,
        bus: EventBusKafka | None = None,
    ) -> None:
        """Initialize the lifecycle manager.

        Args:
            container: ONEX container for dependency injection.
            config: Server configuration.
            bus: Optional Kafka event bus for hot reload. If not provided,
                Kafka subscription is skipped even if kafka_enabled=True.
        """
        self._container = container
        self._config = config
        self._bus = bus

        # Services (initialized during start())
        self._registry: ServiceMCPToolRegistry | None = None
        self._discovery: ServiceMCPToolDiscovery | None = None
        self._sync: ServiceMCPToolSync | None = None
        self._executor: AdapterONEXToolExecution | None = None

        # State
        self._started = False

        logger.debug(
            "MCPServerLifecycle initialized",
            extra={
                "consul_host": config.consul_host,
                "consul_port": config.consul_port,
                "kafka_enabled": config.kafka_enabled,
                "http_port": config.http_port,
            },
        )

    @property
    def is_running(self) -> bool:
        """Return True if the lifecycle has been started."""
        return self._started

    @property
    def registry(self) -> ServiceMCPToolRegistry | None:
        """Return the tool registry (available after start())."""
        return self._registry

    @property
    def executor(self) -> AdapterONEXToolExecution | None:
        """Return the execution adapter (available after start())."""
        return self._executor

    async def start(self) -> None:
        """Start all MCP server components.

        This method performs the following steps:
        1. Create registry, discovery, and executor instances
        2. Cold start: discover all MCP-enabled tools from Consul
        3. Populate the registry with discovered tools
        4. Start Kafka subscription for hot reload (if enabled)

        Raises:
            RuntimeError: If already started.
        """
        if self._started:
            logger.debug("MCPServerLifecycle already started")
            return

        correlation_id = uuid4()

        logger.info(
            "Starting MCP server lifecycle",
            extra={"correlation_id": str(correlation_id)},
        )

        # Create services
        self._registry = ServiceMCPToolRegistry()
        self._executor = AdapterONEXToolExecution(
            container=self._container,
            default_timeout=self._config.default_timeout,
        )

        # Dev mode: scan local contracts instead of Consul
        if self._config.dev_mode:
            logger.info(
                "Dev mode: discovering tools from local contracts",
                extra={
                    "contracts_dir": self._config.contracts_dir,
                    "correlation_id": str(correlation_id),
                },
            )
            tools = await self._discover_from_contracts(correlation_id)
            for tool in tools:
                await self._registry.upsert_tool(tool, "dev-mode")
            logger.info(
                "Dev mode discovery complete",
                extra={
                    "tool_count": len(tools),
                    "correlation_id": str(correlation_id),
                },
            )
        else:
            # Production mode: discover from Consul
            self._discovery = ServiceMCPToolDiscovery(
                consul_host=self._config.consul_host,
                consul_port=self._config.consul_port,
                consul_scheme=self._config.consul_scheme,
                consul_token=self._config.consul_token,
            )

            logger.info(
                "Cold start: discovering tools from Consul",
                extra={"correlation_id": str(correlation_id)},
            )

            try:
                tools = await self._discovery.discover_all()

                # Populate registry
                for tool in tools:
                    # Use "0" as event_id to ensure any future Kafka updates take precedence
                    # (numeric offsets like "1", "2" sort after "0" alphabetically)
                    await self._registry.upsert_tool(tool, "0")

                logger.info(
                    "Cold start complete",
                    extra={
                        "tool_count": len(tools),
                        "correlation_id": str(correlation_id),
                    },
                )

            except Exception as e:
                logger.warning(
                    "Cold start discovery failed - continuing with empty registry",
                    extra={
                        "error": str(e),
                        "correlation_id": str(correlation_id),
                    },
                )

        # Start Kafka sync (if enabled, bus provided, and not in dev mode)
        # Dev mode doesn't use Consul discovery, so Kafka sync is not applicable
        if (
            self._config.kafka_enabled
            and self._bus is not None
            and self._discovery is not None
        ):
            logger.info(
                "Starting Kafka subscription for hot reload",
                extra={"correlation_id": str(correlation_id)},
            )

            self._sync = ServiceMCPToolSync(
                registry=self._registry,
                discovery=self._discovery,
                bus=self._bus,
            )
            await self._sync.start()

        self._started = True

        logger.info(
            "MCP server lifecycle started",
            extra={
                "tool_count": self._registry.tool_count,
                "kafka_enabled": self._config.kafka_enabled and self._bus is not None,
                "correlation_id": str(correlation_id),
            },
        )

    async def _discover_from_contracts(
        self, correlation_id: UUID
    ) -> list[ModelMCPToolDefinition]:
        """Scan local contracts for MCP-enabled orchestrators (dev mode).

        Args:
            correlation_id: Correlation ID for logging.

        Returns:
            List of discovered tool definitions from local contracts.
        """
        from pathlib import Path

        import yaml

        from omnibase_infra.models.mcp.model_mcp_tool_definition import (
            ModelMCPToolDefinition,
        )

        tools: list[ModelMCPToolDefinition] = []
        contracts_dir = self._config.contracts_dir

        if not contracts_dir:
            logger.warning(
                "Dev mode enabled but no contracts_dir specified",
                extra={"correlation_id": str(correlation_id)},
            )
            return tools

        contracts_path = Path(contracts_dir)
        if not contracts_path.exists():
            logger.warning(
                "Contracts directory does not exist",
                extra={
                    "contracts_dir": contracts_dir,
                    "correlation_id": str(correlation_id),
                },
            )
            return tools

        # Scan for contract.yaml files
        for contract_file in contracts_path.rglob("contract.yaml"):
            try:
                with contract_file.open("r") as f:
                    contract = yaml.safe_load(f)

                if not contract:
                    continue

                # Check for MCP configuration
                mcp_config = contract.get("mcp", {})
                if not mcp_config.get("expose", False):
                    continue

                # Only orchestrators can be exposed
                node_type = contract.get("node_type", "")
                if "ORCHESTRATOR" not in node_type:
                    logger.debug(
                        "Skipping non-orchestrator with mcp.expose",
                        extra={
                            "contract": str(contract_file),
                            "node_type": node_type,
                            "correlation_id": str(correlation_id),
                        },
                    )
                    continue

                # Build tool definition
                name = contract.get("name", contract_file.parent.name)
                tool_name = mcp_config.get("tool_name", name)
                description = mcp_config.get(
                    "description", contract.get("description", f"ONEX: {name}")
                )
                timeout = mcp_config.get("timeout_seconds", 30)
                version = contract.get("node_version", "1.0.0")

                tool = ModelMCPToolDefinition(
                    name=tool_name,
                    description=description,
                    version=version,
                    parameters=[],  # Will be populated from input_model
                    input_schema={"type": "object", "properties": {}},
                    orchestrator_node_id=name,
                    orchestrator_service_id=None,
                    endpoint=None,  # Local dev mode - no endpoint
                    timeout_seconds=timeout,
                    metadata={
                        "contract_path": str(contract_file),
                        "node_type": node_type,
                        "source": "local_contract",
                    },
                )
                tools.append(tool)

                logger.info(
                    "Discovered MCP tool from contract",
                    extra={
                        "tool_name": tool_name,
                        "contract": str(contract_file),
                        "correlation_id": str(correlation_id),
                    },
                )

            except yaml.YAMLError as e:
                logger.warning(
                    "Failed to parse contract YAML",
                    extra={
                        "contract": str(contract_file),
                        "error": str(e),
                        "correlation_id": str(correlation_id),
                    },
                )
            except Exception as e:
                logger.warning(
                    "Error processing contract",
                    extra={
                        "contract": str(contract_file),
                        "error": str(e),
                        "correlation_id": str(correlation_id),
                    },
                )

        return tools

    async def shutdown(self) -> None:
        """Shutdown all MCP server components.

        This method performs graceful cleanup:
        1. Stop Kafka subscription
        2. Clear registry
        3. Close executor HTTP client

        Safe to call multiple times.
        """
        if not self._started:
            logger.debug("MCPServerLifecycle already stopped")
            return

        correlation_id = uuid4()

        logger.info(
            "Shutting down MCP server lifecycle",
            extra={"correlation_id": str(correlation_id)},
        )

        # Stop Kafka sync
        if self._sync is not None:
            await self._sync.stop()
            self._sync = None

        # Clear registry
        if self._registry is not None:
            await self._registry.clear()
            self._registry = None

        # Close executor
        if self._executor is not None:
            await self._executor.close()
            self._executor = None

        # Clear discovery (stateless, no cleanup needed)
        self._discovery = None

        self._started = False

        logger.info(
            "MCP server lifecycle shutdown complete",
            extra={"correlation_id": str(correlation_id)},
        )

    def describe(self) -> dict[str, object]:
        """Return lifecycle metadata for observability."""
        return {
            "service_name": "MCPServerLifecycle",
            "started": self._started,
            "config": {
                "consul_host": self._config.consul_host,
                "consul_port": self._config.consul_port,
                "kafka_enabled": self._config.kafka_enabled,
                "http_port": self._config.http_port,
            },
            "registry_tool_count": (self._registry.tool_count if self._registry else 0),
            "sync_running": self._sync.is_running if self._sync else False,
        }


__all__ = ["MCPServerLifecycle", "ModelMCPServerConfig"]
