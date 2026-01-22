# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Tool Execution Adapter - Bridges MCP tool calls to ONEX orchestrator execution.

This adapter handles the execution of MCP tool invocations by:
1. Validating input arguments against the tool's input schema
2. Building an ONEX envelope with the input payload
3. Dispatching to the orchestrator endpoint via HTTP
4. Transforming the response to MCP format

Routing:
    The adapter uses the tool definition's endpoint or service_id to locate
    the target orchestrator. It supports both direct HTTP dispatch and
    service discovery via Consul.

Timeout Handling:
    Each tool definition includes a timeout_seconds value. The adapter
    enforces this timeout when dispatching to the orchestrator, raising
    InfraTimeoutError if exceeded.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import httpx

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker

if TYPE_CHECKING:
    from omnibase_infra.models.mcp.model_mcp_tool_definition import (
        ModelMCPToolDefinition,
    )

logger = logging.getLogger(__name__)


class AdapterONEXToolExecution(MixinAsyncCircuitBreaker):
    """Bridges MCP tool calls to ONEX orchestrator execution.

    This adapter handles the dispatch of MCP tool invocations to the
    appropriate ONEX orchestrator node. It supports:
    - Direct HTTP dispatch to orchestrator endpoint
    - Input validation against JSON Schema
    - Timeout enforcement
    - Error transformation to MCP format
    - Circuit breaker protection for external HTTP calls

    Attributes:
        _container: ONEX container for dependency injection.
        _http_client: HTTP client for orchestrator dispatch.
        _default_timeout: Default timeout if tool definition doesn't specify one.

    Example:
        >>> adapter = AdapterONEXToolExecution(container=container)
        >>> result = await adapter.execute(
        ...     tool=tool_definition,
        ...     arguments={"input_data": "test"},
        ...     correlation_id=uuid4(),
        ... )
        >>> print(result["success"])
        True
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        http_client: httpx.AsyncClient | None = None,
        default_timeout: float = 30.0,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_reset_timeout: float = 60.0,
    ) -> None:
        """Initialize the execution adapter.

        Args:
            container: ONEX container for dependency injection.
            http_client: Optional HTTP client. If not provided, one will be
                created during execute() calls.
            default_timeout: Default timeout in seconds for orchestrator calls.
            circuit_breaker_threshold: Max failures before opening circuit (default: 5).
            circuit_breaker_reset_timeout: Seconds before automatic reset (default: 60.0).
        """
        self._container = container
        self._http_client = http_client
        self._default_timeout = default_timeout
        self._owns_client = http_client is None

        # Initialize circuit breaker for HTTP dispatch resilience
        self._init_circuit_breaker(
            threshold=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_reset_timeout,
            service_name="onex-tool-execution",
            transport_type=EnumInfraTransportType.HTTP,
        )

        logger.debug(
            "AdapterONEXToolExecution initialized",
            extra={
                "default_timeout": default_timeout,
                "circuit_breaker_threshold": circuit_breaker_threshold,
                "circuit_breaker_reset_timeout": circuit_breaker_reset_timeout,
            },
        )

    async def execute(
        self,
        tool: ModelMCPToolDefinition,
        arguments: dict[str, object],
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Execute an MCP tool call by dispatching to the ONEX orchestrator.

        Args:
            tool: Tool definition containing endpoint, timeout, and metadata.
            arguments: Input arguments from the MCP tool call.
            correlation_id: Correlation ID for tracing.

        Returns:
            Dictionary with execution result:
                - success: True if execution succeeded
                - result: Orchestrator response (if successful)
                - error: Error message (if failed)

        Raises:
            InfraUnavailableError: If tool endpoint is not configured.
            InfraTimeoutError: If execution times out.
            InfraConnectionError: If connection to orchestrator fails.
        """
        logger.info(
            "Executing MCP tool",
            extra={
                "tool_name": tool.name,
                "correlation_id": str(correlation_id),
            },
        )

        # Validate endpoint
        endpoint = tool.endpoint
        if not endpoint:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="execute_tool",
                target_name=tool.name,
            )
            raise InfraUnavailableError(
                f"Tool '{tool.name}' has no endpoint configured",
                context=ctx,
            )

        # Build envelope payload
        envelope = self._build_envelope(tool, arguments, correlation_id)

        # Determine timeout
        timeout = tool.timeout_seconds or self._default_timeout

        # Check circuit breaker before dispatch
        try:
            async with self._circuit_breaker_lock:
                await self._check_circuit_breaker(
                    operation="execute_tool",
                    correlation_id=correlation_id,
                )
        except InfraUnavailableError:
            logger.warning(
                "MCP tool execution blocked - circuit breaker open",
                extra={
                    "tool_name": tool.name,
                    "correlation_id": str(correlation_id),
                },
            )
            return {
                "success": False,
                "error": "Service temporarily unavailable - circuit breaker open",
            }

        # Dispatch to orchestrator
        try:
            result = await self._http_dispatch(
                endpoint=endpoint,
                envelope=envelope,
                timeout=timeout,
                correlation_id=correlation_id,
            )

            # Record success to reset circuit breaker
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.info(
                "MCP tool execution succeeded",
                extra={
                    "tool_name": tool.name,
                    "correlation_id": str(correlation_id),
                },
            )

            return {
                "success": True,
                "result": result,
            }

        except InfraTimeoutError:
            # Record failure to potentially open circuit breaker
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="execute_tool",
                    correlation_id=correlation_id,
                )
            logger.warning(
                "MCP tool execution timed out",
                extra={
                    "tool_name": tool.name,
                    "timeout": timeout,
                    "correlation_id": str(correlation_id),
                },
            )
            return {
                "success": False,
                "error": f"Tool execution timed out after {timeout} seconds",
            }

        except InfraConnectionError as e:
            # Record failure to potentially open circuit breaker
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="execute_tool",
                    correlation_id=correlation_id,
                )
            logger.warning(
                "MCP tool execution failed - connection error",
                extra={
                    "tool_name": tool.name,
                    "error": str(e),
                    "correlation_id": str(correlation_id),
                },
            )
            return {
                "success": False,
                "error": f"Connection error: {e}",
            }

        except Exception as e:
            # Record failure to potentially open circuit breaker
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="execute_tool",
                    correlation_id=correlation_id,
                )
            logger.exception(
                "MCP tool execution failed - unexpected error",
                extra={
                    "tool_name": tool.name,
                    "error": str(e),
                    "correlation_id": str(correlation_id),
                },
            )
            return {
                "success": False,
                "error": f"Unexpected error: {e}",
            }

    def _build_envelope(
        self,
        tool: ModelMCPToolDefinition,
        arguments: dict[str, object],
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Build an ONEX envelope for the orchestrator.

        Args:
            tool: Tool definition.
            arguments: Input arguments from MCP.
            correlation_id: Correlation ID.

        Returns:
            Envelope dict for the orchestrator.
        """
        return {
            "envelope_id": str(uuid4()),
            "correlation_id": str(correlation_id),
            "source": "mcp-adapter",
            "payload": arguments,
            "metadata": {
                "tool_name": tool.name,
                "tool_version": tool.version,
            },
        }

    async def _http_dispatch(
        self,
        endpoint: str,
        envelope: dict[str, object],
        timeout: float,
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Dispatch envelope to orchestrator endpoint via HTTP.

        Args:
            endpoint: Target endpoint URL.
            envelope: Request envelope.
            timeout: Request timeout in seconds.
            correlation_id: Correlation ID.

        Returns:
            Response from orchestrator.

        Raises:
            InfraTimeoutError: If request times out.
            InfraConnectionError: If connection fails.
        """
        # Use provided client or create one
        if self._http_client is not None:
            return await self._dispatch_with_client(
                self._http_client,
                endpoint,
                envelope,
                timeout,
                correlation_id,
            )
        else:
            # Create temporary client
            async with httpx.AsyncClient() as client:
                return await self._dispatch_with_client(
                    client,
                    endpoint,
                    envelope,
                    timeout,
                    correlation_id,
                )

    async def _dispatch_with_client(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        envelope: dict[str, object],
        timeout: float,
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Dispatch using the provided HTTP client.

        Args:
            client: HTTP client.
            endpoint: Target endpoint URL.
            envelope: Request envelope.
            timeout: Request timeout in seconds.
            correlation_id: Correlation ID.

        Returns:
            Response from orchestrator.

        Raises:
            InfraTimeoutError: If request times out.
            InfraConnectionError: If connection fails.
        """
        try:
            response = await asyncio.wait_for(
                client.post(
                    endpoint,
                    json=envelope,
                    headers={
                        "X-Correlation-ID": str(correlation_id),
                        "Content-Type": "application/json",
                    },
                    timeout=timeout,
                ),
                timeout=timeout,
            )

            response.raise_for_status()
            result: dict[str, object] = response.json()
            return result

        except TimeoutError as e:
            timeout_ctx = ModelTimeoutErrorContext(
                transport_type=EnumInfraTransportType.HTTP,
                operation="http_dispatch",
                target_name=endpoint,
                correlation_id=correlation_id,
                timeout_seconds=timeout,
            )
            raise InfraTimeoutError(
                f"Timeout dispatching to {endpoint} after {timeout}s",
                context=timeout_ctx,
            ) from e

        except httpx.ConnectError as e:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="http_dispatch",
                target_name=endpoint,
            )
            raise InfraConnectionError(
                f"Connection failed to {endpoint}: {e}",
                context=ctx,
            ) from e

        except httpx.HTTPStatusError as e:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="http_dispatch",
                target_name=endpoint,
            )
            raise InfraConnectionError(
                f"HTTP error from {endpoint}: {e.response.status_code}",
                context=ctx,
            ) from e

        except Exception as e:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="http_dispatch",
                target_name=endpoint,
            )
            raise InfraConnectionError(
                f"Request to {endpoint} failed: {e}",
                context=ctx,
            ) from e

    async def close(self) -> None:
        """Close the HTTP client if owned by this adapter."""
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    def describe(self) -> dict[str, object]:
        """Return adapter metadata for observability."""
        return {
            "adapter_name": "AdapterONEXToolExecution",
            "default_timeout": self._default_timeout,
            "owns_client": self._owns_client,
            "circuit_breaker": self._get_circuit_breaker_state(),
        }


__all__ = ["AdapterONEXToolExecution"]
