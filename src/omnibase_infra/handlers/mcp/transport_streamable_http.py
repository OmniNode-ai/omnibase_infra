# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""MCP Streamable HTTP Transport for ONEX.

Provides streamable HTTP transport integration for exposing ONEX nodes
as MCP tools. This transport is recommended for production deployments.

The transport uses the official MCP Python SDK's streamable HTTP implementation,
configured for stateless operation and JSON responses for scalability.

Usage:
    from omnibase_infra.handlers.models.mcp import ModelMcpHandlerConfig

    config = ModelMcpHandlerConfig(host="0.0.0.0", port=8090, path="/mcp")
    transport = TransportMCPStreamableHttp(config)
    await transport.start(tool_registry)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_infra.handlers.models.mcp import ModelMcpHandlerConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import uvicorn
    from omnibase_spi.protocols.types.protocol_mcp_tool_types import (
        ProtocolMCPToolDefinition,
    )
    from starlette.applications import Starlette

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
    """

    def __init__(self, config: ModelMcpHandlerConfig | None = None) -> None:
        """Initialize the streamable HTTP transport.

        Args:
            config: MCP handler configuration. If None, uses defaults.
        """
        self._config = config or ModelMcpHandlerConfig()
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

        Creates a wrapper function that calls the tool_executor with
        the tool name and arguments.

        Args:
            mcp: FastMCP server instance.
            tool_def: Tool definition.
            tool_executor: Callback to execute the tool.
        """
        from mcp.server.fastmcp import FastMCP

        assert isinstance(mcp, FastMCP)

        # Create a closure that captures the tool name
        tool_name = tool_def.name

        @mcp.tool(name=tool_name, description=tool_def.description)
        def tool_wrapper(**kwargs: object) -> object:
            """Wrapper that routes to the ONEX tool executor."""
            return tool_executor(tool_name, kwargs)

        # Store the handler for reference
        self._tool_handlers[tool_name] = tool_wrapper

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
        """
        import uvicorn

        if self._running:
            logger.warning("MCP transport already running")
            return

        app = self.create_app(tools, tool_executor)
        self._running = True

        logger.info(
            "Starting MCP streamable HTTP transport",
            extra={
                "host": self._config.host,
                "port": self._config.port,
                "path": self._config.path,
            },
        )

        # Run uvicorn server
        config = uvicorn.Config(
            app,
            host=self._config.host,
            port=self._config.port,
            log_level="info",
        )
        self._server = uvicorn.Server(config)
        await self._server.serve()

    async def stop(self) -> None:
        """Stop the MCP server."""
        if not self._running:
            return

        # Signal the uvicorn server to exit gracefully
        if self._server is not None:
            self._server.should_exit = True

        self._running = False
        self._app = None
        self._server = None
        self._tool_handlers.clear()

        logger.info("MCP streamable HTTP transport stopped")


__all__ = ["TransportMCPStreamableHttp"]
