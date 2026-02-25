# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""MCP Streamable HTTP Transport for ONEX.

Provides streamable HTTP transport integration for exposing ONEX nodes
as MCP tools. This transport is recommended for production deployments.

The transport uses the official MCP Python SDK's streamable HTTP implementation,
configured for stateless operation and JSON responses for scalability.

Security:
    Authentication is enforced via MCPAuthMiddleware, an ASGI middleware that
    validates bearer token / API key on every request to the MCP endpoint.

    The ``/health`` endpoint is explicitly exempted from authentication so that
    infrastructure health checks can reach it without credentials.

    Configure via ModelMcpHandlerConfig:
    - ``auth_enabled=True`` (default): auth middleware is active
    - ``auth_enabled=False``: middleware is bypassed; WARNING logged at startup
    - ``api_key``: the secret token value (from Infisical/env)

    Supported auth schemes (either accepted):
    - ``Authorization: Bearer <token>``
    - ``X-API-Key: <token>``

    Unauthenticated requests receive HTTP 401 with a JSON error body.
    Auth failures are logged with: timestamp, remote IP, rejection reason.
    Successful tool invocations are logged with: timestamp, masked token
    (last 4 chars), tool name (path), correlation ID.

Usage:
    from omnibase_infra.handlers.models.mcp import ModelMcpHandlerConfig

    config = ModelMcpHandlerConfig(
        host="0.0.0.0", port=8090, path="/mcp",
        auth_enabled=True, api_key="secret-token",
    )
    transport = TransportMCPStreamableHttp(config)
    await transport.start(tool_registry)
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.handlers.models.mcp import ModelMcpHandlerConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import uvicorn
    from starlette.applications import Starlette
    from starlette.types import ASGIApp, Receive, Scope, Send

    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_spi.protocols.types.protocol_mcp_tool_types import (
        ProtocolMCPToolDefinition,
    )

logger = logging.getLogger(__name__)

# Endpoints explicitly exempted from auth (case-insensitive prefix match)
_AUTH_EXEMPT_PATHS: frozenset[str] = frozenset({"/health"})


class MCPAuthMiddleware:
    """ASGI middleware that enforces bearer token / API-key authentication.

    Exempts ``/health`` (and any path in ``_AUTH_EXEMPT_PATHS``) from auth.
    All other paths require a valid ``Authorization: Bearer <token>`` or
    ``X-API-Key: <token>`` header.

    Audit logging:
        - Auth failures: WARNING with timestamp, remote IP, rejection reason.
        - Successful authenticated requests: INFO with timestamp, masked token
          (last 4 chars), path, and correlation ID from ``X-Correlation-ID``.

    Args:
        app: The inner ASGI application to wrap.
        api_key: The expected token value. If empty or None, every request is
            rejected with 401 (misconfiguration guard).
    """

    def __init__(self, app: ASGIApp, api_key: str) -> None:
        self._app = app
        self._api_key = api_key

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self._app(scope, receive, send)
            return

        path: str = scope.get("path", "")

        # Exempt health and other explicitly excluded paths
        if path in _AUTH_EXEMPT_PATHS:
            await self._app(scope, receive, send)
            return

        # Extract remote IP for audit logging
        client = scope.get("client")
        remote_ip: str = client[0] if client else "unknown"

        # Extract headers (ASGI headers are list of (name_bytes, value_bytes))
        headers: dict[str, str] = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }

        token: str | None = None

        # Accept Authorization: Bearer <token>
        auth_header = headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[len("bearer ") :]

        # Accept X-API-Key: <token>
        if token is None:
            token = headers.get("x-api-key")

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if not token or not self._api_key or token != self._api_key:
            reason = (
                "missing token"
                if not token
                else (
                    "server misconfiguration" if not self._api_key else "invalid token"
                )
            )
            logger.warning(
                "MCP auth rejected",
                extra={
                    "timestamp": timestamp,
                    "remote_ip": remote_ip,
                    "path": path,
                    "reason": reason,
                },
            )
            await self._send_401(send)
            return

        # Auth passed — log masked token (last 4 chars) for audit trail
        masked = f"****{token[-4:]}" if len(token) >= 4 else "****"
        correlation_id = headers.get("x-correlation-id", "")
        logger.info(
            "MCP auth accepted",
            extra={
                "timestamp": timestamp,
                "masked_token": masked,
                "path": path,
                "correlation_id": correlation_id,
            },
        )

        await self._app(scope, receive, send)

    @staticmethod
    async def _send_401(send: Send) -> None:
        """Send an HTTP 401 response with a JSON error body."""
        body = json.dumps(
            {
                "error": "Unauthorized",
                "detail": "Valid bearer token or X-API-Key required",
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class TransportMCPStreamableHttp:
    """Streamable HTTP transport for MCP server.

    This class provides a wrapper around the MCP SDK's streamable HTTP
    transport, integrating it with ONEX's tool registry.

    The transport creates an ASGI application that can be:
    1. Run standalone via uvicorn
    2. Mounted into an existing FastAPI/Starlette application

    Authentication:
        When ``config.auth_enabled`` is True (default), every request to the
        MCP endpoint is validated by ``MCPAuthMiddleware`` before reaching the
        inner application. The ``/health`` endpoint is exempt.

        When ``config.auth_enabled`` is False, a startup WARNING is logged and
        no auth is enforced — intended for local development only.

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
        mounted at the configured path, wrapped by ``MCPAuthMiddleware``
        when ``config.auth_enabled`` is True.

        Args:
            tools: Sequence of tool definitions to expose.
            tool_executor: Callback function to execute tool calls.
                          Signature: (tool_name, arguments) -> result

        Returns:
            Starlette ASGI application (may be wrapped in auth middleware).

        Note:
            The MCP SDK is imported lazily to allow the module to be
            imported even if the MCP SDK is not installed.

            When ``auth_enabled=False``, a WARNING is logged at startup
            and no authentication is enforced. Do not use in production.
        """
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError as e:
            raise ImportError("MCP SDK not installed. Install via: uv add mcp") from e

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

        # Apply authentication middleware (R1, R3 — OMN-2701)
        if self._config.auth_enabled:
            api_key = self._config.api_key or ""
            self._app = MCPAuthMiddleware(self._app, api_key=api_key)  # type: ignore[assignment]
        else:
            logger.warning(
                "MCP auth disabled — do not use in production",
                extra={"path": self._config.path},
            )

        logger.info(
            "MCP streamable HTTP transport app created",
            extra={
                "path": self._config.path,
                "tool_count": len(tools),
                "stateless": self._config.stateless,
                "json_response": self._config.json_response,
                "auth_enabled": self._config.auth_enabled,
            },
        )

        return self._app  # type: ignore[return-value]

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


__all__ = ["MCPAuthMiddleware", "TransportMCPStreamableHttp"]
