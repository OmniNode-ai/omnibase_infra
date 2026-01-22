# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""MCP Handler - Model Context Protocol integration for ONEX nodes.

Exposes ONEX nodes as MCP tools for AI agent integration via streamable HTTP transport.
This handler enables AI agents (Claude, etc.) to discover and invoke ONEX nodes as tools.

The handler implements the MCP protocol specification using the official MCP Python SDK,
providing a bridge between the ONEX node ecosystem and AI agent tool interfaces.

Key Features:
    - Streamable HTTP transport for production scalability
    - Dynamic tool discovery from ONEX node registry
    - Contract-to-MCP schema generation
    - Request/response correlation for observability
    - Internal uvicorn server lifecycle management (OMN-1282)

Note:
    This handler requires the `mcp` package (anthropic-ai/mcp-python-sdk).
    Install via: poetry add mcp
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import uvicorn
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import (
    InfraUnavailableError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.handlers.models.mcp import (
    EnumMcpOperationType,
    ModelMcpHandlerConfig,
    ModelMcpToolResult,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker, MixinEnvelopeExtraction
from omnibase_infra.services.mcp import MCPServerLifecycle, ModelMCPServerConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Sequence

    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.adapters.adapter_onex_tool_execution import (
        AdapterONEXToolExecution,
    )
    from omnibase_infra.services.mcp.service_mcp_tool_registry import (
        ServiceMCPToolRegistry,
    )
    from omnibase_spi.protocols.types.protocol_mcp_tool_types import (
        ProtocolMCPToolDefinition,
    )

logger = logging.getLogger(__name__)

# Handler ID for ModelHandlerOutput
HANDLER_ID_MCP: str = "mcp-handler"

# Shutdown timeout constants (can be overridden via class attributes)
_DEFAULT_SHUTDOWN_TIMEOUT: float = 5.0
_DEFAULT_CANCEL_TIMEOUT: float = 1.0
_DEFAULT_STARTUP_TIMEOUT: float = 2.0

# Error message truncation limit for health check responses
_ERROR_MESSAGE_MAX_LENGTH: int = 200


def _require_config_value[T](
    config: dict[str, object],
    key: str,
    expected_type: type[T],
    correlation_id: UUID,
) -> T:
    """Extract required config value or raise ProtocolConfigurationError.

    Per CLAUDE.md configuration rules, the `.env` file is the SINGLE SOURCE OF TRUTH.
    There should be ZERO hardcoded fallbacks - all configuration must be explicitly
    provided. If missing, this function raises an error rather than using defaults.

    Args:
        config: Configuration dictionary to extract value from.
        key: Configuration key to look up.
        expected_type: Expected Python type for the value.
        correlation_id: Correlation ID for error context.

    Returns:
        The validated configuration value.

    Raises:
        ProtocolConfigurationError: If value is missing or has wrong type.
    """
    value = config.get(key)
    if value is None:
        raise ProtocolConfigurationError(
            f"Missing required config: '{key}'. Must be set in .env or runtime config.",
            context=ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.MCP,
                operation="initialize",
                target_name="handler_mcp",
            ),
        )
    if not isinstance(value, expected_type):
        raise ProtocolConfigurationError(
            f"Invalid config type for '{key}': expected {expected_type.__name__}, "
            f"got {type(value).__name__}",
            context=ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.MCP,
                operation="initialize",
                target_name="handler_mcp",
            ),
        )
    return value


# Supported operations
_SUPPORTED_OPERATIONS: frozenset[str] = frozenset(
    {op.value for op in EnumMcpOperationType}
)


class HandlerMCP(MixinEnvelopeExtraction, MixinAsyncCircuitBreaker):
    """MCP protocol handler for exposing ONEX nodes as AI agent tools.

    This handler creates an MCP server using streamable HTTP transport,
    enabling AI agents to discover and invoke ONEX nodes as tools.

    The handler integrates with the ONEX registry to dynamically expose
    registered nodes as MCP tools, translating ONEX contracts into
    MCP tool definitions.

    Architecture:
        - Uses official MCP Python SDK for protocol compliance
        - Streamable HTTP transport for production deployments
        - Stateless mode for horizontal scaling
        - JSON response mode for compatibility

    Security Features:
        - Tool execution timeout enforcement (via config.timeout_seconds)
        - Request size limits inherited from ONEX nodes
        - Correlation ID propagation for tracing
        - Circuit breaker protection against cascading failures

    Authentication:
        Authentication is NOT yet implemented in this MVP version. The MCP
        endpoint is currently open/unauthenticated. Authentication will be
        added in a future release via:
        - Bearer token validation in the transport layer
        - Integration with ONEX identity service for token verification
        - Optional API key support for service-to-service communication
        See: TODO(OMN-1288) for authentication implementation tracking

        For production deployments before authentication is implemented,
        deploy behind an API gateway with authentication or restrict
        network access to trusted clients.

    Dispatcher Integration:
        This MVP version uses placeholder tool execution. Full ONEX dispatcher
        integration is planned to enable:
        - Routing tool calls to the appropriate ONEX node
        - Timeout enforcement via asyncio.wait_for()
        - Full observability through the ONEX runtime
        See: TODO(OMN-1288) for dispatcher integration tracking

    Class Attributes:
        shutdown_timeout: Timeout for graceful server shutdown (default: 5.0s).
        cancel_timeout: Timeout for forced cancellation after graceful fails (default: 1.0s).
        startup_timeout: Timeout for server readiness check during startup (default: 2.0s).
    """

    # Configurable timeout attributes (can be overridden on subclasses or instances)
    shutdown_timeout: float = _DEFAULT_SHUTDOWN_TIMEOUT
    cancel_timeout: float = _DEFAULT_CANCEL_TIMEOUT
    startup_timeout: float = _DEFAULT_STARTUP_TIMEOUT

    def __init__(
        self,
        container: ModelONEXContainer | None = None,
        registry: ServiceMCPToolRegistry | None = None,
        executor: AdapterONEXToolExecution | None = None,
    ) -> None:
        """Initialize HandlerMCP with optional ONEX container for dependency injection.

        Args:
            container: Optional ONEX container providing dependency injection for
                services, configuration, and runtime context. When None, the handler
                operates in standalone mode without container-based DI.
            registry: Optional MCP tool registry for dynamic tool discovery.
                If provided, tools are looked up from this registry. If not
                provided, the handler uses its local _tool_registry dict.
            executor: Optional tool execution adapter for dispatching to
                ONEX orchestrators. If provided, tool calls are routed through
                this adapter. If not provided, placeholder execution is used.

        Note:
            The container parameter is optional to support two instantiation paths:
            1. Registry-based: RuntimeHostProcess creates handlers via registry lookup
               with no-argument constructor calls. Container is None in this case.
            2. DI-based: Explicit container injection for full ONEX integration.

            When container is provided, it enables future DI-based service resolution
            (e.g., dispatcher routing, metrics integration).

        MCP Integration (OMN-1281):
            When registry and executor are provided, the handler operates in
            "integrated mode" with full MCP tool discovery and execution:
            - Tools are discovered from Consul via ServiceMCPToolDiscovery
            - Tool list is cached in ServiceMCPToolRegistry
            - Tool execution routes through AdapterONEXToolExecution
            - Hot reload updates are received via ServiceMCPToolSync

        Server Lifecycle (OMN-1282):
            The handler owns its uvicorn server lifecycle. When initialize() is
            called, the handler starts a uvicorn server in a background task.
            When shutdown() is called, the server is gracefully stopped.
        """
        self._container = container
        self._config: ModelMcpHandlerConfig | None = None
        self._initialized: bool = False
        self._tool_registry: dict[str, ProtocolMCPToolDefinition] = {}

        # MCP integration components (OMN-1281)
        self._mcp_registry: ServiceMCPToolRegistry | None = registry
        self._mcp_executor: AdapterONEXToolExecution | None = executor

        # Server lifecycle components (OMN-1282)
        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task[None] | None = None
        self._lifecycle: MCPServerLifecycle | None = None
        self._skip_server: bool = False  # Track if server was intentionally skipped
        self._server_started_at: float | None = None  # Timestamp for uptime tracking

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role of this handler.

        Returns:
            EnumHandlerType.INFRA_HANDLER - This handler is an infrastructure
            protocol/transport handler that exposes ONEX nodes via MCP.
        """
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification of this handler.

        Returns:
            EnumHandlerTypeCategory.EFFECT - This handler performs side-effecting
            I/O operations (tool execution via MCP protocol).
        """
        return EnumHandlerTypeCategory.EFFECT

    @property
    def transport_type(self) -> EnumInfraTransportType:
        """Return the transport protocol identifier.

        Returns:
            EnumInfraTransportType.MCP - Model Context Protocol transport.
        """
        return EnumInfraTransportType.MCP

    def _create_json_endpoint(
        self,
        response_factory: Callable[[], Coroutine[object, object, dict[str, object]]],
    ) -> Callable[[Request], Coroutine[object, object, JSONResponse]]:
        """Create a JSON endpoint that wraps an async response factory.

        This method creates a Starlette-compatible async route handler that:
        1. Calls the provided response_factory to generate response data
        2. Wraps the data in a JSONResponse

        Args:
            response_factory: Async callable that returns the response data dict.
                The factory is called on each request to generate fresh data.

        Returns:
            Async function suitable for Starlette Route.
        """

        async def endpoint(_request: Request) -> JSONResponse:
            data = await response_factory()
            return JSONResponse(data)

        return endpoint

    def _create_health_endpoint(
        self,
    ) -> Callable[[Request], Coroutine[object, object, JSONResponse]]:
        """Create health endpoint with explicit handler binding.

        Returns a coroutine function that closes over `self` explicitly,
        avoiding fragile closure patterns with intermediate variables.

        Returns:
            Async function suitable for Starlette Route.
        """
        # Capture reference explicitly in closure scope
        handler = self

        async def get_health_data() -> dict[str, object]:
            """Return health status data for the MCP server."""
            tool_count = 0
            if handler._lifecycle and handler._lifecycle.registry:
                tool_count = handler._lifecycle.registry.tool_count
            return {
                "status": "healthy",
                "tool_count": tool_count,
                "initialized": handler._initialized,
            }

        return self._create_json_endpoint(get_health_data)

    def _create_tools_list_endpoint(
        self,
    ) -> Callable[[Request], Coroutine[object, object, JSONResponse]]:
        """Create tools list endpoint with explicit handler binding.

        Returns a coroutine function that closes over `self` explicitly,
        avoiding fragile closure patterns with intermediate variables.

        Returns:
            Async function suitable for Starlette Route.
        """
        # Capture reference explicitly in closure scope
        handler = self

        async def get_tools_data() -> dict[str, object]:
            """Return list of available MCP tools."""
            if handler._lifecycle and handler._lifecycle.registry:
                tools = await handler._lifecycle.registry.list_tools()
                return {
                    "tools": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "endpoint": t.endpoint,
                        }
                        for t in tools
                    ]
                }
            return {"tools": []}

        return self._create_json_endpoint(get_tools_data)

    async def _wait_for_server_ready(
        self,
        host: str,
        port: int,
        timeout: float = 2.0,
        poll_interval: float = 0.05,
    ) -> None:
        """Wait for server to be ready by polling TCP connect.

        Args:
            host: Server host
            port: Server port
            timeout: Maximum time to wait
            poll_interval: Time between connection attempts

        Raises:
            ProtocolConfigurationError: If server doesn't start within timeout

        Note:
            Circuit Breaker Failures Are NOT Recorded Here

            This method is for startup verification, not runtime health checking.
            TCP connect failures during startup are expected and transient - the
            server is still spinning up and will become available shortly.

            Circuit breaker tracking is intentionally omitted because:

            1. Startup retries are bounded and transient - the method either succeeds
               within the timeout or raises ProtocolConfigurationError, ending startup.

            2. Recording startup failures would pollute circuit breaker metrics with
               expected transient failures, potentially triggering an open circuit
               before the server even starts.

            3. Circuit breakers are designed for runtime fault tolerance - detecting
               when a previously-healthy service becomes unhealthy. Startup behavior
               is fundamentally different: we expect failures until success.

            4. If the server fails to start within timeout, we fail fast with
               ProtocolConfigurationError rather than entering a degraded state.

            Circuit breaker tracking should occur during runtime operations (e.g.,
            tool execution, health checks) where failures indicate actual service
            degradation rather than expected startup latency.
        """
        import socket

        start_time = time.perf_counter()
        last_error: Exception | None = None

        while time.perf_counter() - start_time < timeout:
            # Check if server task has failed
            if self._server_task is not None and self._server_task.done():
                exc = self._server_task.exception()
                if exc:
                    ctx = ModelInfraErrorContext.with_correlation(
                        transport_type=EnumInfraTransportType.MCP,
                        operation="server_startup",
                        target_name="mcp_handler",
                    )
                    raise ProtocolConfigurationError(
                        f"Server failed to start: {exc}",
                        context=ctx,
                    ) from exc

            # Try TCP connect
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(poll_interval)
                # TCP PROTOCOL REQUIREMENT (NOT a config fallback):
                # When a server binds to 0.0.0.0 (INADDR_ANY), it listens on all
                # interfaces but you cannot connect() to 0.0.0.0 - it's not a
                # routable address. TCP requires connecting to a specific interface.
                # Using 127.0.0.1 (loopback) is the correct way to reach a local
                # server that bound to 0.0.0.0. This is standard TCP/IP behavior,
                # not an environment configuration fallback.
                effective_host = "127.0.0.1" if host == "0.0.0.0" else host  # noqa: S104
                result = sock.connect_ex((effective_host, port))
                sock.close()
                if result == 0:
                    return  # Server is ready
            except Exception as e:
                last_error = e

            await asyncio.sleep(poll_interval)

        # Timeout reached
        ctx = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.MCP,
            operation="server_startup",
            target_name="mcp_handler",
        )
        raise ProtocolConfigurationError(
            f"Server failed to start within {timeout}s. Last error: {last_error}",
            context=ctx,
        )

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize MCP handler with configuration and optionally start uvicorn server.

        This method performs the following steps:
        1. Parse and validate handler configuration
        2. Initialize MCPServerLifecycle for tool discovery (unless skip_server=True)
        3. Create Starlette app with /health and /mcp/tools endpoints
        4. Start uvicorn server in a background task (unless skip_server=True)

        Args:
            config: Configuration dict containing:
                - host: Host to bind MCP server to (default: 0.0.0.0)
                - port: Port for MCP endpoint (default: 8090)
                - path: URL path for MCP endpoint (default: /mcp)
                - stateless: Enable stateless mode (default: True)
                - json_response: Return JSON responses (default: True)
                - timeout_seconds: Tool execution timeout (default: 30.0)
                - max_tools: Maximum tools to expose (default: 100)
                - consul_host: Consul server hostname (REQUIRED - no default)
                - consul_port: Consul server port (REQUIRED - no default)
                - kafka_enabled: Whether to enable Kafka hot reload (REQUIRED - no default)
                - dev_mode: Whether to run in development mode (REQUIRED - no default)
                - contracts_dir: Directory for contract scanning in dev mode (optional)
                - skip_server: Skip starting uvicorn server (default: False).
                    Use for unit testing to avoid port binding.

        Raises:
            ProtocolConfigurationError: If configuration is invalid or required
                config values (consul_host, consul_port, kafka_enabled, dev_mode)
                are missing. Per CLAUDE.md, .env is the single source of truth -
                no hardcoded fallbacks are used.
        """
        init_correlation_id = uuid4()

        logger.info(
            "Initializing %s",
            self.__class__.__name__,
            extra={
                "handler": self.__class__.__name__,
                "correlation_id": str(init_correlation_id),
            },
        )

        try:
            # Use Pydantic validation for type-safe configuration parsing.
            # Pydantic handles type coercion (e.g., str "8090" -> int 8090) automatically.
            # ValidationError will be raised for truly invalid types that cannot be coerced.
            self._config = ModelMcpHandlerConfig(**config)

            # Initialize tool registry (empty until tools are registered)
            self._tool_registry = {}

            # Initialize circuit breaker for tool execution resilience
            # Configuration from contract.yaml: threshold=5, reset_timeout=60.0
            self._init_circuit_breaker(
                threshold=5,
                reset_timeout=60.0,
                service_name="mcp-handler",
                transport_type=EnumInfraTransportType.MCP,
            )

            # Check if server startup should be skipped (for unit testing)
            skip_server_val = config.get("skip_server")
            skip_server: bool = (
                skip_server_val if isinstance(skip_server_val, bool) else False
            )
            self._skip_server = skip_server

            if not skip_server:
                # Build MCPServerConfig from handler config (OMN-1282)
                # Map handler config fields to lifecycle config fields
                #
                # Per CLAUDE.md: .env is the SINGLE SOURCE OF TRUTH.
                # No hardcoded fallbacks - all required config must be explicit.
                # The _require_config_value helper validates type, cast() is for mypy.
                consul_host = _require_config_value(
                    config, "consul_host", str, init_correlation_id
                )
                consul_port = _require_config_value(
                    config, "consul_port", int, init_correlation_id
                )
                kafka_enabled = _require_config_value(
                    config, "kafka_enabled", bool, init_correlation_id
                )
                dev_mode = _require_config_value(
                    config, "dev_mode", bool, init_correlation_id
                )
                # contracts_dir is optional - only used when dev_mode=True
                contracts_dir_val = config.get("contracts_dir")
                contracts_dir: str | None = (
                    contracts_dir_val if isinstance(contracts_dir_val, str) else None
                )

                server_config = ModelMCPServerConfig(
                    consul_host=consul_host,
                    consul_port=consul_port,
                    kafka_enabled=kafka_enabled,
                    http_host=self._config.host,
                    http_port=self._config.port,
                    default_timeout=self._config.timeout_seconds,
                    dev_mode=dev_mode,
                    contracts_dir=contracts_dir,
                )

                # Wrap entire server startup in try/except to ensure cleanup
                # if ANY step fails after lifecycle starts. This prevents:
                # - Orphan lifecycle resources (registry, executor, sync)
                # - Orphan server tasks
                # - Resource leaks from partial initialization
                try:
                    # Create and start MCPServerLifecycle for tool discovery
                    # Container is required for lifecycle initialization
                    if self._container is None:
                        raise ValueError(
                            "Container required for MCPServerLifecycle initialization"
                        )
                    self._lifecycle = MCPServerLifecycle(
                        container=self._container,
                        config=server_config,
                        bus=None,
                    )
                    await self._lifecycle.start()

                    # Update MCP registry and executor references from lifecycle
                    if self._lifecycle.registry is not None:
                        self._mcp_registry = self._lifecycle.registry
                    if self._lifecycle.executor is not None:
                        self._mcp_executor = self._lifecycle.executor

                    # Create Starlette app with HTTP endpoints (OMN-1282)
                    # Use factory methods for explicit handler reference binding
                    health_endpoint = self._create_health_endpoint()
                    tools_list_endpoint = self._create_tools_list_endpoint()

                    app = Starlette(
                        routes=[
                            Route("/health", health_endpoint, methods=["GET"]),
                            Route("/mcp/tools", tools_list_endpoint, methods=["GET"]),
                        ],
                    )

                    # Create uvicorn server config and server
                    uvicorn_config = uvicorn.Config(
                        app=app,
                        host=self._config.host,
                        port=self._config.port,
                        log_level="info",
                    )
                    self._server = uvicorn.Server(uvicorn_config)

                    # Start server in background task
                    self._server_task = asyncio.create_task(self._server.serve())

                    # Wait for server to be ready before marking as initialized
                    await self._wait_for_server_ready(
                        self._config.host,
                        self._config.port,
                        timeout=self.startup_timeout,
                    )
                    self._server_started_at = time.time()

                except Exception as startup_error:
                    # Any failure during server startup - clean up all resources
                    # This handles failures in:
                    # - lifecycle.start() (Consul/contract discovery)
                    # - Starlette app creation
                    # - uvicorn config/server creation
                    # - server task creation
                    # - server readiness check
                    logger.exception(
                        "MCP server startup failed, cleaning up resources",
                        extra={
                            "host": self._config.host,
                            "port": self._config.port,
                            "lifecycle_created": self._lifecycle is not None,
                            "server_created": self._server is not None,
                            "server_task_created": self._server_task is not None,
                            "correlation_id": str(init_correlation_id),
                        },
                    )
                    # shutdown() safely handles partially initialized state:
                    # - Checks each component before cleanup
                    # - Safe to call even if components weren't created
                    await self.shutdown()
                    ctx = ModelInfraErrorContext(
                        transport_type=EnumInfraTransportType.MCP,
                        operation="initialize",
                        target_name="mcp_handler",
                        correlation_id=init_correlation_id,
                    )
                    raise ProtocolConfigurationError(
                        f"MCP server startup failed: {startup_error}",
                        context=ctx,
                    ) from startup_error

            self._initialized = True

            tool_count = 0
            if self._lifecycle and self._lifecycle.registry:
                tool_count = self._lifecycle.registry.tool_count

            if skip_server:
                logger.info(
                    "%s initialized successfully (server skipped)",
                    self.__class__.__name__,
                    extra={
                        "handler": self.__class__.__name__,
                        "host": self._config.host,
                        "port": self._config.port,
                        "path": self._config.path,
                        "stateless": self._config.stateless,
                        "skip_server": True,
                        "correlation_id": str(init_correlation_id),
                    },
                )
            else:
                logger.info(
                    "%s initialized successfully - uvicorn server running",
                    self.__class__.__name__,
                    extra={
                        "handler": self.__class__.__name__,
                        "host": self._config.host,
                        "port": self._config.port,
                        "path": self._config.path,
                        "stateless": self._config.stateless,
                        "tool_count": tool_count,
                        "url": f"http://{self._config.host}:{self._config.port}",
                        "correlation_id": str(init_correlation_id),
                    },
                )

        except ValidationError as e:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.MCP,
                operation="initialize",
                target_name="mcp_handler",
                correlation_id=init_correlation_id,
            )
            raise ProtocolConfigurationError(
                f"Invalid MCP handler configuration: {e}", context=ctx
            ) from e
        except (TypeError, ValueError) as e:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.MCP,
                operation="initialize",
                target_name="mcp_handler",
                correlation_id=init_correlation_id,
            )
            raise ProtocolConfigurationError(
                f"Invalid MCP handler configuration: {e}", context=ctx
            ) from e

    async def shutdown(self) -> None:
        """Shutdown MCP handler with timeout protection.

        This method performs graceful shutdown with timeout protection:
        1. Signal uvicorn server to stop
        2. Wait for server task with timeout (max 5s graceful, 1s forced)
        3. Shutdown MCPServerLifecycle (registry, discovery, sync)
        4. Clear tool registry and reset state

        Safe to call multiple times. Never hangs indefinitely (max ~6s with defaults).

        Note:
            Timeouts are configurable via class attributes:
            - shutdown_timeout: Graceful shutdown timeout (default: 5.0s)
            - cancel_timeout: Forced cancellation timeout (default: 1.0s)
        """
        shutdown_correlation_id = uuid4()

        logger.info(
            "Shutting down %s",
            self.__class__.__name__,
            extra={
                "handler": self.__class__.__name__,
                "correlation_id": str(shutdown_correlation_id),
            },
        )

        # Stop uvicorn server with timeout protection (OMN-1282)
        if (
            self._server is not None
            and self._server_task is not None
            and not self._skip_server
        ):
            # Signal server to stop
            self._server.should_exit = True

            try:
                # Wait for graceful shutdown with timeout
                logger.debug(
                    "Waiting for server task to complete",
                    extra={
                        "timeout_seconds": self.shutdown_timeout,
                        "correlation_id": str(shutdown_correlation_id),
                    },
                )
                await asyncio.wait_for(self._server_task, timeout=self.shutdown_timeout)
            except TimeoutError:
                logger.warning(
                    "Server shutdown timed out, forcing cancellation",
                    extra={
                        "timeout_seconds": self.shutdown_timeout,
                        "correlation_id": str(shutdown_correlation_id),
                    },
                )
                self._server_task.cancel()
                try:
                    await asyncio.wait_for(
                        self._server_task, timeout=self.cancel_timeout
                    )
                except (TimeoutError, asyncio.CancelledError):
                    pass  # Best effort
            except asyncio.CancelledError:
                logger.debug(
                    "Server task was cancelled",
                    extra={"correlation_id": str(shutdown_correlation_id)},
                )

        # Shutdown lifecycle (registry, discovery, sync)
        if self._lifecycle is not None:
            logger.debug(
                "Shutting down MCPServerLifecycle",
                extra={"correlation_id": str(shutdown_correlation_id)},
            )
            await self._lifecycle.shutdown()
            self._lifecycle = None

        # Clear registry and executor references
        self._mcp_registry = None
        self._mcp_executor = None

        # Clear all state
        self._tool_registry.clear()
        self._config = None
        self._initialized = False
        self._server = None
        self._server_task = None
        self._skip_server = False
        self._server_started_at = None

        logger.info(
            "%s shutdown complete",
            self.__class__.__name__,
            extra={
                "handler": self.__class__.__name__,
                "correlation_id": str(shutdown_correlation_id),
            },
        )

    async def execute(
        self, envelope: dict[str, object]
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Execute MCP operation from envelope.

        Supported operations:
            - mcp.list_tools: List all available MCP tools
            - mcp.call_tool: Invoke a specific tool
            - mcp.describe: Return handler metadata

        Args:
            envelope: Request envelope containing:
                - operation: One of the supported MCP operations
                - payload: Operation-specific payload
                - correlation_id: Optional correlation ID
                - envelope_id: Optional envelope ID

        Returns:
            ModelHandlerOutput containing operation result.

        Raises:
            RuntimeHostError: If handler not initialized.
            ProtocolConfigurationError: If operation invalid.
        """
        correlation_id = self._extract_correlation_id(envelope)
        input_envelope_id = self._extract_envelope_id(envelope)

        if not self._initialized:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.MCP,
                operation="execute",
                target_name="mcp_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "HandlerMCP not initialized. Call initialize() first.", context=ctx
            )

        operation = envelope.get("operation")
        if not isinstance(operation, str):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.MCP,
                operation="execute",
                target_name="mcp_handler",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "Missing or invalid 'operation' in envelope", context=ctx
            )

        if operation not in _SUPPORTED_OPERATIONS:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.MCP,
                operation=operation,
                target_name="mcp_handler",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                f"Operation '{operation}' not supported. "
                f"Available: {', '.join(sorted(_SUPPORTED_OPERATIONS))}",
                context=ctx,
            )

        payload = envelope.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        # Route to operation handler
        if operation == EnumMcpOperationType.LIST_TOOLS.value:
            return await self._handle_list_tools(
                payload, correlation_id, input_envelope_id
            )
        elif operation == EnumMcpOperationType.CALL_TOOL.value:
            return await self._handle_call_tool(
                payload, correlation_id, input_envelope_id
            )
        else:  # mcp.describe
            return await self._handle_describe(correlation_id, input_envelope_id)

    async def _handle_list_tools(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Handle mcp.list_tools operation.

        Returns a list of all registered MCP tools with their schemas.
        """
        tools = self._get_tool_definitions()

        # Convert to MCP-compatible format
        tool_list = [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": self._build_input_schema(tool),
            }
            for tool in tools
        ]

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_MCP,
            result={
                "status": "success",
                "payload": {"tools": tool_list},
                "correlation_id": str(correlation_id),
            },
        )

    async def _handle_call_tool(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Handle mcp.call_tool operation.

        Invokes the specified tool with provided arguments.
        """
        # Parse tool call request
        tool_name = payload.get("tool_name")
        if not isinstance(tool_name, str):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.MCP,
                operation="mcp.call_tool",
                target_name="mcp_handler",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "Missing or invalid 'tool_name' in payload", context=ctx
            )

        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}

        # Check if tool exists
        if tool_name not in self._tool_registry:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.MCP,
                operation="mcp.call_tool",
                target_name=tool_name,
                correlation_id=correlation_id,
            )
            raise InfraUnavailableError(
                f"Tool '{tool_name}' not found in registry", context=ctx
            )

        # Execute tool (placeholder - actual execution delegates to ONEX node)
        start_time = time.perf_counter()

        try:
            result = await self._execute_tool(tool_name, arguments, correlation_id)
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            tool_result = ModelMcpToolResult(
                success=True,
                content=result,
                is_error=False,
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
            )

        except InfraUnavailableError as e:
            # Circuit breaker open or tool unavailable
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "Tool execution failed: infrastructure unavailable",
                extra={
                    "tool_name": tool_name,
                    "error": str(e),
                    "error_type": "InfraUnavailableError",
                    "correlation_id": str(correlation_id),
                    "execution_time_ms": execution_time_ms,
                },
            )
            tool_result = ModelMcpToolResult(
                success=False,
                content=str(e),
                is_error=True,
                error_message=str(e),
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
            )

        except (RuntimeHostError, ProtocolConfigurationError) as e:
            # Handler or configuration errors
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "Tool execution failed: runtime or configuration error",
                extra={
                    "tool_name": tool_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "correlation_id": str(correlation_id),
                    "execution_time_ms": execution_time_ms,
                },
            )
            tool_result = ModelMcpToolResult(
                success=False,
                content=str(e),
                is_error=True,
                error_message=str(e),
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
            )

        except (TimeoutError, OSError) as e:
            # Network/IO errors during tool execution
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(
                "Tool execution failed: network or timeout error",
                extra={
                    "tool_name": tool_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "correlation_id": str(correlation_id),
                    "execution_time_ms": execution_time_ms,
                },
            )
            tool_result = ModelMcpToolResult(
                success=False,
                content=str(e),
                is_error=True,
                error_message=str(e),
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
            )

        except (ValueError, TypeError, KeyError) as e:
            # Data validation or type errors
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "Tool execution failed: data validation error",
                extra={
                    "tool_name": tool_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "correlation_id": str(correlation_id),
                    "execution_time_ms": execution_time_ms,
                },
            )
            tool_result = ModelMcpToolResult(
                success=False,
                content=str(e),
                is_error=True,
                error_message=str(e),
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
            )

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_MCP,
            result={
                "status": "success" if tool_result.success else "error",
                "payload": tool_result.model_dump(),
                "correlation_id": str(correlation_id),
            },
        )

    async def _handle_describe(
        self,
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Handle mcp.describe operation."""
        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_MCP,
            result={
                "status": "success",
                "payload": self.describe(),
                "correlation_id": str(correlation_id),
            },
        )

    def _get_tool_definitions(self) -> Sequence[ProtocolMCPToolDefinition]:
        """Get all registered tool definitions."""
        return list(self._tool_registry.values())

    def _build_input_schema(self, tool: ProtocolMCPToolDefinition) -> dict[str, object]:
        """Build JSON Schema for tool input from MCP tool definition."""
        properties: dict[str, object] = {}
        required: list[str] = []

        for param in tool.parameters:
            param_schema: dict[str, object] = {
                "type": param.parameter_type,
                "description": param.description,
            }
            if param.schema:
                param_schema.update(param.schema)

            properties[param.name] = param_schema

            if param.required:
                required.append(param.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    async def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, object],
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Execute a registered tool.

        This method delegates to the ONEX orchestrator that provides this tool.
        When operating in integrated mode (with registry and executor), the tool
        is looked up from the MCP registry and executed via the execution adapter.

        Circuit breaker protection is applied to prevent cascading failures
        when tool execution repeatedly fails.

        Integration Mode (OMN-1281):
            When _mcp_registry and _mcp_executor are configured:
            1. Look up the tool definition from the MCP registry
            2. Delegate execution to AdapterONEXToolExecution
            3. The adapter dispatches to the orchestrator endpoint
            4. Timeout is enforced by the adapter using the tool's timeout_seconds

        Legacy Mode:
            When registry/executor are not configured, returns placeholder response
            for backward compatibility.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            correlation_id: Correlation ID for tracing.

        Returns:
            Tool execution result.

        Raises:
            InfraUnavailableError: If tool not found or execution fails.
        """
        # Check circuit breaker before tool execution
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("execute_tool", correlation_id)

        try:
            # Integrated mode: use MCP registry and executor (OMN-1281)
            if self._mcp_registry is not None and self._mcp_executor is not None:
                # Look up tool from registry
                tool = await self._mcp_registry.get_tool(tool_name)
                if tool is None:
                    ctx = ModelInfraErrorContext.with_correlation(
                        correlation_id=correlation_id,
                        transport_type=self.transport_type,
                        operation="execute_tool",
                        target_name=tool_name,
                    )
                    raise InfraUnavailableError(
                        f"Tool not found: {tool_name}",
                        context=ctx,
                    )

                logger.info(
                    "Executing MCP tool via adapter",
                    extra={
                        "tool_name": tool_name,
                        "argument_count": len(arguments),
                        "correlation_id": str(correlation_id),
                    },
                )

                # Execute via adapter
                result = await self._mcp_executor.execute(
                    tool=tool,
                    arguments=arguments,
                    correlation_id=correlation_id,
                )

                # Reset circuit breaker on success
                async with self._circuit_breaker_lock:
                    await self._reset_circuit_breaker()

                return result

            # Legacy mode: placeholder response for backward compatibility
            logger.info(
                "Tool execution requested (placeholder mode)",
                extra={
                    "tool_name": tool_name,
                    "argument_count": len(arguments),
                    "correlation_id": str(correlation_id),
                },
            )

            placeholder_result: dict[str, object] = {
                "message": f"Tool '{tool_name}' executed successfully",
                "arguments_received": list(arguments.keys()),
            }

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return placeholder_result

        except InfraUnavailableError:
            # Record failure in circuit breaker and re-raise
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("execute_tool", correlation_id)
            raise

        except Exception:
            # Record failure in circuit breaker
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("execute_tool", correlation_id)
            raise

    def register_tool(self, tool: ProtocolMCPToolDefinition) -> bool:
        """Register an MCP tool definition.

        Args:
            tool: Tool definition to register.

        Returns:
            True if tool was registered successfully, False if max tool limit exceeded.

        Note:
            Callers MUST check the return value. If False, the tool was NOT registered
            due to the max_tools limit being reached. Silently ignoring a False return
            will lead to tools being unavailable without any error being raised.

            The tool registry is a simple dict and is NOT thread-safe. If concurrent
            registration is required, external synchronization must be provided by
            the caller.

        Example:
            if not handler.register_tool(my_tool):
                ctx = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.MCP,
                    operation="register_tool",
                    target_name=my_tool.name,
                    correlation_id=uuid4(),
                )
                raise ProtocolConfigurationError(
                    f"Failed to register tool: {my_tool.name}",
                    context=ctx,
                )
        """
        if self._config and len(self._tool_registry) >= self._config.max_tools:
            logger.warning(
                "Maximum tool limit reached, tool not registered",
                extra={"tool_name": tool.name, "max_tools": self._config.max_tools},
            )
            return False

        self._tool_registry[tool.name] = tool
        logger.info(
            "Tool registered",
            extra={
                "tool_name": tool.name,
                "tool_type": tool.tool_type,
                "parameter_count": len(tool.parameters),
            },
        )
        return True

    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister an MCP tool.

        Args:
            tool_name: Name of the tool to unregister.

        Returns:
            True if tool was unregistered, False if not found.
        """
        if tool_name in self._tool_registry:
            del self._tool_registry[tool_name]
            logger.info("Tool unregistered", extra={"tool_name": tool_name})
            return True
        return False

    def describe(self) -> dict[str, object]:
        """Return handler metadata and capabilities.

        Returns:
            dict containing handler type, category, transport type,
            supported operations, configuration, tool count, and server state.
        """
        config_dict: dict[str, object] = {}
        if self._config:
            config_dict = {
                "host": self._config.host,
                "port": self._config.port,
                "path": self._config.path,
                "stateless": self._config.stateless,
                "json_response": self._config.json_response,
                "timeout_seconds": self._config.timeout_seconds,
                "max_tools": self._config.max_tools,
            }

        # Include lifecycle tool count if available (OMN-1282)
        tool_count = len(self._tool_registry)
        if self._lifecycle and self._lifecycle.registry:
            tool_count = self._lifecycle.registry.tool_count

        return {
            "handler_type": self.handler_type.value,
            "handler_category": self.handler_category.value,
            "transport_type": self.transport_type.value,
            "supported_operations": sorted(_SUPPORTED_OPERATIONS),
            "tool_count": tool_count,
            "config": config_dict,
            "initialized": self._initialized,
            "server_running": self._server is not None
            and self._server_task is not None,
            "lifecycle_running": self._lifecycle is not None
            and self._lifecycle.is_running,
            "version": "0.1.0-mvp",
        }

    async def health_check(self) -> dict[str, object]:
        """Check handler health and server status.

        Returns unhealthy if:
        - Not initialized
        - Server task has crashed/completed unexpectedly
        - Server task was cancelled

        Note:
            When skip_server=True was used during initialization, the handler is
            considered healthy if initialized, even without a running server.
            This enables unit testing without actual port binding.
        """
        if not self._initialized:
            return {
                "healthy": False,
                "reason": "not_initialized",
                "transport_type": self.transport_type.value,
            }

        if self._skip_server:
            return {
                "healthy": True,
                "skip_server": True,
                "transport_type": self.transport_type.value,
                "initialized": True,
            }

        # Capture server task reference once to avoid TOCTOU race conditions.
        # If _server_task is reassigned (e.g., by concurrent shutdown()),
        # we work with the captured reference consistently.
        server_task = self._server_task

        # Check server task state
        if server_task is None:
            return {
                "healthy": False,
                "reason": "server_task_missing",
                "transport_type": self.transport_type.value,
                "initialized": True,
            }

        if server_task.done():
            # Task completed - check why
            if server_task.cancelled():
                return {
                    "healthy": False,
                    "reason": "server_cancelled",
                    "transport_type": self.transport_type.value,
                    "initialized": True,
                }

            exc = server_task.exception()
            if exc is not None:
                return {
                    "healthy": False,
                    "reason": "server_crashed",
                    "error": str(exc)[:_ERROR_MESSAGE_MAX_LENGTH],
                    "transport_type": self.transport_type.value,
                    "initialized": True,
                }

            # Exited cleanly but unexpectedly
            return {
                "healthy": False,
                "reason": "server_exited",
                "transport_type": self.transport_type.value,
                "initialized": True,
            }

        # Task is still running - healthy
        # Include lifecycle tool count if available (OMN-1282)
        tool_count = len(self._tool_registry)
        if self._lifecycle and self._lifecycle.registry:
            tool_count = self._lifecycle.registry.tool_count

        lifecycle_running = self._lifecycle is not None and self._lifecycle.is_running

        return {
            "healthy": True,
            "initialized": True,
            "server_running": True,
            "tool_count": tool_count,
            "transport_type": self.transport_type.value,
            "lifecycle_running": lifecycle_running,
            "uptime_seconds": (
                time.time() - self._server_started_at
                if self._server_started_at is not None
                else None
            ),
        }


__all__: list[str] = ["HandlerMCP", "HANDLER_ID_MCP"]
