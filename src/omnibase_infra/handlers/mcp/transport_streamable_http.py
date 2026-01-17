# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""MCP Streamable HTTP Transport for ONEX.

Provides streamable HTTP transport integration for exposing ONEX nodes
as MCP tools. This transport is recommended for production deployments.

The transport uses the official MCP Python SDK's streamable HTTP implementation,
configured for stateless operation and JSON responses for scalability.

Security:
    Authentication is handled at the handler level (HandlerMCP), not at the
    transport level. The transport exposes the raw HTTP endpoint without any
    authentication middleware.

    For production deployments before HandlerMCP authentication is implemented:
    - Deploy behind an API gateway with authentication (recommended)
    - Use network-level access controls (VPC, firewall rules)
    - Restrict access to trusted clients only

    The transport layer is intentionally kept simple to allow flexibility in
    authentication strategies (API gateway, service mesh, direct auth).

    See: HandlerMCP class docstring for authentication implementation status
    See: TODO(OMN-1288) for authentication implementation tracking

Usage:
    from omnibase_infra.handlers.models.mcp import ModelMcpHandlerConfig

    config = ModelMcpHandlerConfig(host="0.0.0.0", port=8090, path="/mcp")
    transport = TransportMCPStreamableHttp(config)
    await transport.start(tool_registry)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.handlers.models.mcp import ModelMcpHandlerConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import uvicorn
    from starlette.applications import Starlette

    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_spi.protocols.types.protocol_mcp_tool_types import (
        ProtocolMCPToolDefinition,
    )

logger = logging.getLogger(__name__)


class TransportMCPStreamableHttp:
    """Streamable HTTP transport for MCP server.

    This class provides a wrapper around the MCP SDK's streamable HTTP
    transport, integrating it with ONEX's tool registry.

    The transport creates an ASGI application that can be:
    1. Run standalone via uvicorn
    2. Mounted into an existing FastAPI/Starlette application

    Attributes:
        config: MCP handler configuration containing host, port, path, etc.
        _container: Optional ONEX container for dependency injection.
    """

    def __init__(
        self,
        config: ModelMcpHandlerConfig | None = None,
        container: ModelONEXContainer | None = None,
    ) -> None:
        """Initialize the streamable HTTP transport.

        Args:
            config: MCP handler configuration. If None, uses defaults.
            container: Optional ONEX container for dependency injection.
                      Provides access to shared services and configuration
                      when integrating with the ONEX runtime.
        """
        self._config = config or ModelMcpHandlerConfig()
        self._container = container
        self._app: Starlette | None = None
        self._server: uvicorn.Server | None = None
        self._running = False
        self._tool_handlers: dict[str, Callable[..., object]] = {}

    @property
    def is_running(self) -> bool:
        """Check if the transport is currently running."""
        return self._running

    @property
    def app(self) -> Starlette | None:
        """Get the ASGI application (available after create_app is called)."""
        return self._app

    def create_app(
        self,
        tools: Sequence[ProtocolMCPToolDefinition],
        tool_executor: Callable[[str, dict[str, object]], object],
    ) -> Starlette:
        """Create the ASGI application for the MCP server.

        This method creates a Starlette application with the MCP server
        mounted at the configured path.

        Args:
            tools: Sequence of tool definitions to expose.
            tool_executor: Callback function to execute tool calls.
                          Signature: (tool_name, arguments) -> result

        Returns:
            Starlette ASGI application.

        Note:
            The MCP SDK is imported lazily to allow the module to be
            imported even if the MCP SDK is not installed.
        """
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError as e:
            raise ImportError(
                "MCP SDK not installed. Install via: poetry add mcp"
            ) from e

        from starlette.applications import Starlette
        from starlette.routing import Mount

        # Create FastMCP server with streamable HTTP configuration
        mcp = FastMCP(
            "ONEX MCP Server",
            stateless_http=self._config.stateless,
            json_response=self._config.json_response,
        )

        # Register tools from the provided definitions
        for tool_def in tools:
            self._register_tool(mcp, tool_def, tool_executor)

        # Create Starlette app with MCP server mounted
        self._app = Starlette(
            routes=[
                Mount(self._config.path, app=mcp.streamable_http_app()),
            ],
        )

        logger.info(
            "MCP streamable HTTP transport app created",
            extra={
                "path": self._config.path,
                "tool_count": len(tools),
                "stateless": self._config.stateless,
                "json_response": self._config.json_response,
            },
        )

        return self._app

    def _register_tool(
        self,
        mcp: object,  # FastMCP type, but using object to avoid import issues
        tool_def: ProtocolMCPToolDefinition,
        tool_executor: Callable[[str, dict[str, object]], object],
    ) -> None:
        """Register a tool with the MCP server.

        Creates a wrapper function with a unique name that calls the tool_executor
        with the tool name and arguments.

        Note:
            Each tool handler gets a unique function name (onex_tool_{name}) to avoid
            potential conflicts with FastMCP's internal function registry. While FastMCP
            uses the explicit `name` parameter for tool identification, having unique
            function names ensures robustness across different MCP SDK versions.

        Args:
            mcp: FastMCP server instance.
            tool_def: Tool definition.
            tool_executor: Callback to execute the tool.
        """
        from mcp.server.fastmcp import FastMCP

        if not isinstance(mcp, FastMCP):
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.MCP,
                operation="register_tool",
            )
            raise ProtocolConfigurationError(
                f"Expected FastMCP instance, got {type(mcp).__name__}",
                context=context,
            )

        tool_name = tool_def.name

        # Create a handler factory that produces uniquely-named functions per tool.
        # This avoids potential issues where FastMCP might use __name__ internally.
        def _make_tool_handler(name: str) -> Callable[..., object]:
            def handler(**kwargs: object) -> object:
                """Wrapper that routes to the ONEX tool executor."""
                return tool_executor(name, kwargs)

            # Set unique function name for this tool - ensures no naming collisions
            handler.__name__ = f"onex_tool_{name}"
            handler.__qualname__ = f"TransportMCPStreamableHttp.onex_tool_{name}"
            return handler

        handler = _make_tool_handler(tool_name)
        mcp.tool(name=tool_name, description=tool_def.description)(handler)

        # Store the handler for reference
        self._tool_handlers[tool_name] = handler

        logger.debug(
            "Tool registered with MCP server",
            extra={
                "tool_name": tool_name,
                "parameter_count": len(tool_def.parameters),
            },
        )

    async def start(
        self,
        tools: Sequence[ProtocolMCPToolDefinition],
        tool_executor: Callable[[str, dict[str, object]], object],
    ) -> None:
        """Start the MCP server.

        This method creates the ASGI app and starts it using uvicorn.

        Args:
            tools: Sequence of tool definitions to expose.
            tool_executor: Callback function to execute tool calls.

        Raises:
            Exception: If the server fails to start (e.g., port already in use).
                      State is reset to not-running on failure.

        Note:
            Port binding occurs during uvicorn server startup, not during
            configuration. If the configured port is unavailable at bind time,
            the server will fail to start and raise an exception.

            For testing scenarios where port availability needs to be checked,
            note that there is an inherent TOCTOU (time-of-check-time-of-use)
            race between checking port availability and actually binding.
            Production deployments should handle startup failures gracefully.
        """
        import uvicorn

        if self._running:
            logger.warning("MCP transport already running")
            return

        app = self.create_app(tools, tool_executor)

        logger.info(
            "Starting MCP streamable HTTP transport",
            extra={
                "host": self._config.host,
                "port": self._config.port,
                "path": self._config.path,
            },
        )

        # Run uvicorn server - only set _running after successful server creation
        try:
            config = uvicorn.Config(
                app,
                host=self._config.host,
                port=self._config.port,
                log_level="info",
            )
            self._server = uvicorn.Server(config)
            self._running = True  # Only set after successful server creation
            await self._server.serve()
        except Exception:
            self._running = False
            self._server = None
            raise

    async def stop(self) -> None:
        """Stop the MCP server.

        Signals uvicorn to exit gracefully. This method sets the shutdown flag
        and clears local state, but does NOT block waiting for shutdown completion.

        Shutdown Behavior:
            1. Sets ``should_exit = True`` on the uvicorn server, which signals
               the server's main loop to stop accepting new connections.
            2. Clears local state (``_running``, ``_app``, ``_server``).
            3. Returns immediately - actual shutdown completes asynchronously.

        Important:
            The actual server shutdown happens when the ``serve()`` coroutine
            (started by ``start()``) detects ``should_exit`` and returns.
            Callers that need to wait for full shutdown should await the
            ``start()`` coroutine completion, not just call ``stop()``.

        Usage Pattern:
            .. code-block:: python

                # Start in background task
                server_task = asyncio.create_task(transport.start(tools, executor))

                # ... do work ...

                # Signal shutdown
                await transport.stop()

                # Wait for full shutdown
                await server_task

        Note:
            This design follows uvicorn's cooperative shutdown model where
            setting ``should_exit`` signals intent, and the server gracefully
            finishes in-flight requests before the ``serve()`` coroutine returns.
        """
        if not self._running:
            return

        # Signal the uvicorn server to exit gracefully.
        # Per uvicorn's design, setting should_exit = True causes serve() to:
        # 1. Stop accepting new connections
        # 2. Wait for in-flight requests to complete (with configurable timeout)
        # 3. Return from the serve() coroutine
        if self._server is not None:
            self._server.should_exit = True
            logger.info(
                "Signaled MCP transport shutdown",
                extra={
                    "host": self._config.host,
                    "port": self._config.port,
                },
            )

        # Clear local state immediately. The caller of start() should await
        # that coroutine to ensure the server has fully stopped.
        self._running = False
        self._app = None
        self._server = None
        self._tool_handlers.clear()

        logger.info("MCP streamable HTTP transport stopped")


__all__ = ["TransportMCPStreamableHttp"]
