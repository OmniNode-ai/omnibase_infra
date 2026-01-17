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

Note:
    This handler requires the `mcp` package (anthropic-ai/mcp-python-sdk).
    Install via: poetry add mcp
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from pydantic import ValidationError

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

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_spi.protocols.types.protocol_mcp_tool_types import (
        ProtocolMCPToolDefinition,
    )

logger = logging.getLogger(__name__)

# Handler ID for ModelHandlerOutput
HANDLER_ID_MCP: str = "mcp-handler"

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
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize HandlerMCP with ONEX container for dependency injection.

        Args:
            container: ONEX container providing dependency injection for
                services, configuration, and runtime context.
        """
        self._container = container
        self._config: ModelMcpHandlerConfig | None = None
        self._initialized: bool = False
        self._tool_registry: dict[str, ProtocolMCPToolDefinition] = {}

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

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize MCP handler with configuration.

        Args:
            config: Configuration dict containing:
                - host: Host to bind MCP server to (default: 0.0.0.0)
                - port: Port for MCP endpoint (default: 8090)
                - path: URL path for MCP endpoint (default: /mcp)
                - stateless: Enable stateless mode (default: True)
                - json_response: Return JSON responses (default: True)
                - timeout_seconds: Tool execution timeout (default: 30.0)
                - max_tools: Maximum tools to expose (default: 100)

        Raises:
            ProtocolConfigurationError: If configuration is invalid.
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

            # Note: The MCP server is created lazily when start_server() is called
            # This allows the handler to be initialized before tools are registered
            self._initialized = True

            logger.info(
                "%s initialized successfully",
                self.__class__.__name__,
                extra={
                    "handler": self.__class__.__name__,
                    "host": self._config.host,
                    "port": self._config.port,
                    "path": self._config.path,
                    "stateless": self._config.stateless,
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
        """Shutdown MCP handler and release resources."""
        self._tool_registry.clear()
        self._config = None
        self._initialized = False
        logger.info("HandlerMCP shutdown complete")

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

        This method delegates to the ONEX node that provides this tool.
        The actual implementation will route through the ONEX dispatcher.

        Circuit breaker protection is applied to prevent cascading failures
        when tool execution repeatedly fails.

        Timeout Enforcement:
            The tool execution timeout (config.timeout_seconds, default: 30.0s)
            will be enforced when dispatcher integration is complete. The timeout
            will be applied using asyncio.wait_for() around the dispatcher call.

            Currently, timeout enforcement is handled at the protocol level by:
            - uvicorn request timeout settings
            - MCP SDK's internal timeout handling
            - HTTP client timeouts on the caller side

            See: TODO(OMN-1288) for dispatcher timeout integration

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            correlation_id: Correlation ID for tracing.

        Returns:
            Tool execution result.

        Raises:
            InfraUnavailableError: If tool execution fails or circuit is open.
        """
        # Check circuit breaker before tool execution
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("execute_tool", correlation_id)

        try:
            # TODO(OMN-1288): Implement actual tool execution via ONEX dispatcher
            # Integration plan:
            # 1. Look up the ONEX node that provides this tool from container registry
            # 2. Build a ModelEventEnvelope for the node with proper correlation ID
            # 3. Dispatch to the node via the ONEX runtime dispatcher
            # 4. Apply timeout enforcement via asyncio.wait_for(dispatch(), timeout)
            #    using self._config.timeout_seconds (default: 30.0s)
            # 5. Transform the node response to MCP-compatible format
            # 6. Handle dispatcher errors (timeout, node not found, execution failure)
            #
            # For now, return a placeholder response
            logger.info(
                "Tool execution requested",
                extra={
                    "tool_name": tool_name,
                    "argument_count": len(arguments),
                    "correlation_id": str(correlation_id),
                },
            )

            result: dict[str, object] = {
                "message": f"Tool '{tool_name}' executed successfully",
                "arguments_received": list(arguments.keys()),
            }

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return result

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
                raise RuntimeError(f"Failed to register tool: {my_tool.name}")
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
            supported operations, configuration, and tool count.
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

        return {
            "handler_type": self.handler_type.value,
            "handler_category": self.handler_category.value,
            "transport_type": self.transport_type.value,
            "supported_operations": sorted(_SUPPORTED_OPERATIONS),
            "tool_count": len(self._tool_registry),
            "config": config_dict,
            "initialized": self._initialized,
            "version": "0.1.0-mvp",
        }

    async def health_check(self) -> dict[str, object]:
        """Check handler health and connectivity.

        Returns:
            Health status including initialization state and tool count.
        """
        return {
            "healthy": self._initialized,
            "initialized": self._initialized,
            "tool_count": len(self._tool_registry),
            "transport_type": self.transport_type.value,
        }


__all__: list[str] = ["HandlerMCP", "HANDLER_ID_MCP"]
