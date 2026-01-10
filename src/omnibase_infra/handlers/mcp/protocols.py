# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocols for MCP handler tool execution.

This module defines the ProtocolToolExecutor protocol for tool execution backends.
It enables duck typing and dependency injection for different tool execution
strategies (local ONEX nodes, remote services, etc.).

Protocol Responsibilities:
    ProtocolToolExecutor: Execute tools and return results with proper error handling

Concurrency Safety:
    All protocol implementations MUST be safe for concurrent async coroutine calls.
    Multiple coroutines may invoke execute_tool() simultaneously.

Error Handling and Sanitization:
    All implementations MUST follow ONEX error sanitization guidelines.

    NEVER include in error messages:
    - Passwords, API keys, tokens, secrets
    - Full connection strings with credentials
    - PII (names, emails, SSNs, phone numbers)
    - Raw tool argument content (may contain secrets)

    SAFE to include in error messages:
    - Tool names
    - Operation names (e.g., "execute")
    - Correlation IDs (always include for tracing)
    - Error codes
    - Argument keys (not values)

Related:
    - OMN-1288: MCP Handler transport for ONEX integration
    - HandlerMCP: The handler that uses this protocol
    - MixinAsyncCircuitBreaker: Circuit breaker for resilience

.. versionadded:: 0.1.0
"""

from __future__ import annotations

__all__ = ["ProtocolToolExecutor"]

from typing import Protocol, runtime_checkable
from uuid import UUID


@runtime_checkable
class ProtocolToolExecutor(Protocol):
    """Protocol for tool execution backends.

    Tool executors are responsible for invoking tools and returning results.
    They bridge the MCP handler to actual tool implementations (ONEX nodes,
    external services, etc.).

    Protocol Verification:
        Per ONEX conventions, protocol compliance is verified via duck typing
        rather than isinstance checks. Verify required methods exist:

        **Validation Approach** (duck typing check):
        Use when you need to verify an object implements the executor interface
        before passing it to components that expect an executor.

        .. code-block:: python

            # Verify required methods exist
            if hasattr(executor, 'execute_tool') and callable(executor.execute_tool):
                handler.set_executor(executor)
            else:
                raise TypeError("Object does not implement ProtocolToolExecutor")

    Example:
        .. code-block:: python

            from uuid import UUID, uuid4

            class LocalNodeExecutor:
                '''Executor that routes to local ONEX nodes.'''

                def __init__(self, node_registry):
                    self._registry = node_registry

                async def execute_tool(
                    self,
                    tool_name: str,
                    arguments: dict[str, object],
                    correlation_id: UUID,
                ) -> dict[str, object]:
                    node = self._registry.get(tool_name)
                    if not node:
                        raise ValueError(f"Tool not found: {tool_name}")
                    return await node.execute(arguments, correlation_id)

            # Verify protocol compliance via duck typing
            executor = LocalNodeExecutor(registry)
            assert hasattr(executor, 'execute_tool')
            assert callable(executor.execute_tool)

    Concurrency Safety:
        Implementations MUST be safe for concurrent async coroutine calls.
        Multiple coroutines may invoke execute_tool() simultaneously with
        different tools and arguments.

        Design Requirements:
            - Stateless Executors (Recommended): Keep executors stateless by
              not caching tool-specific state. This requires no synchronization.
            - Stateful Executors: If state is required (e.g., connection pools),
              use appropriate synchronization primitives (asyncio.Lock for async state).

    Note:
        Method bodies in this Protocol use ``...`` (Ellipsis) rather than
        ``raise NotImplementedError()``. This is the standard Python convention
        for ``typing.Protocol`` classes per PEP 544.

    .. versionadded:: 0.1.0
    """

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, object],
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Execute a tool and return results.

        This is the primary execution method. The executor receives a tool name
        and arguments, invokes the appropriate backend, and returns the results.

        Concurrency Safety:
            This method MUST be safe to call concurrently from multiple
            coroutines. Implementations should not rely on instance state
            that could be modified by concurrent calls without proper
            synchronization.

        Error Handling:
            Implementations SHOULD raise appropriate exceptions on failure:
            - InfraUnavailableError: Tool not found or backend unavailable
            - InfraTimeoutError: Execution timed out
            - ProtocolConfigurationError: Invalid arguments or configuration

            All error messages MUST be sanitized:
            - Include tool_name and correlation_id
            - Never include raw argument values (may contain secrets)
            - Use generic descriptions for internal errors

        Args:
            tool_name: Name of the tool to execute. This should match a
                registered tool in the handler's registry.
            arguments: Tool arguments as key-value pairs. Values may be any
                JSON-serializable type. Implementations MUST NOT log or
                include argument values in error messages.
            correlation_id: Correlation ID for distributed tracing. MUST be
                propagated to all downstream operations and included in logs.

        Returns:
            dict[str, object]: Tool execution result. The structure depends
                on the specific tool, but typically includes:
                - Status or success indicator
                - Result data
                - Any metadata (execution time, etc.)

        Raises:
            InfraUnavailableError: If the tool is not found or the backend
                is unavailable.
            InfraTimeoutError: If execution exceeds the configured timeout.
            ProtocolConfigurationError: If arguments are invalid.

        Example:
            .. code-block:: python

                result = await executor.execute_tool(
                    tool_name="node_compute_hash",
                    arguments={"input": "data", "algorithm": "sha256"},
                    correlation_id=uuid4(),
                )
                # result: {"hash": "abc123...", "algorithm": "sha256"}
        """
        ...
