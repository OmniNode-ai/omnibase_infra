# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Mock-based MCP protocol test fixtures.

IMPORTANT: These fixtures provide MOCK MCP endpoints, NOT real MCP SDK integration.

What these fixtures provide:
- Mock JSON-RPC handlers implementing MCP protocol methods (initialize, tools/list, tools/call)
- Mock tool registry with deterministic behavior
- Call history tracking for assertion verification
- Infrastructure detection for conditional Consul integration

What these fixtures do NOT provide:
- Real MCP SDK server (mcp.server.fastmcp)
- Real MCP client connections
- Actual streamable HTTP transport

Why mocks are necessary:
The MCP SDK's streamable_http_app() requires proper task group initialization
via run() before handling requests. This lifecycle management is incompatible
with direct ASGI testing via httpx.ASGITransport because:
1. The SDK expects to own its event loop and task groups
2. Direct ASGI testing bypasses the SDK's connection lifecycle
3. The SDK's internal state machines require proper initialization sequences

The mock approach allows us to test:
- JSON-RPC protocol compliance (request/response format)
- Error handling (unknown tools, malformed JSON, missing fields)
- Tool discovery and invocation flow
- Argument passing and result structure

For real MCP SDK integration tests, see:
    tests/integration/services/mcp/e2e/test_mcp_real_e2e.py

Fixture dependency graph:
    infra_availability (session) - Detects Consul/PostgreSQL at fixture time
    mcp_app_dev_mode (function) - Mock MCP app for basic protocol tests
    mcp_http_client (function) - httpx client for mcp_app_dev_mode
    mcp_app_full_infra (function) - Mock MCP app with real Consul discovery

Related Ticket: OMN-1408
"""

from __future__ import annotations

import json as json_module  # Aliased to avoid shadowing in handler scopes
import os
import socket
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, TypedDict

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse

from omnibase_core.models.errors import ModelOnexError
from omnibase_infra.models.types import JsonDict

if TYPE_CHECKING:
    from omnibase_infra.services.mcp import MCPServerLifecycle


# ============================================================================
# TYPE DEFINITIONS FOR FIXTURES
# ============================================================================


class MCPDevModeFixture(TypedDict):
    """Type definition for mcp_app_dev_mode fixture result."""

    app: Starlette
    call_history: list[JsonDict]
    path: str


class MCPFullInfraFixture(TypedDict):
    """Type definition for mcp_app_full_infra fixture result."""

    app: Starlette
    path: str
    lifecycle: MCPServerLifecycle


pytestmark = [
    pytest.mark.mcp_protocol,
    pytest.mark.asyncio,
]


# ============================================================================
# ASSERTION HELPERS
# ============================================================================


def assert_mcp_content_valid(result: JsonDict) -> None:
    """Validate MCP tool call result content structure per MCP spec.

    This helper validates the mandatory content array structure required
    by the MCP protocol for successful tool call responses.

    Args:
        result: The 'result' field from a successful MCP JSON-RPC response.
            Expected to be the value of data["result"] where data is the
            full JSON-RPC response.

    Raises:
        AssertionError: If content structure is invalid with descriptive message.

    MCP Spec Requirements:
        - result MUST contain "content" key
        - content MUST be a list (array)
        - content MUST be non-empty (prevents false positive from empty array)
        - Each content item MUST be a dict with a "type" field
    """
    # MANDATORY: MCP tool call result MUST contain content array per spec
    assert "content" in result, f"MCP tool call result missing content array: {result}"
    assert isinstance(result["content"], list), (
        f"MCP result content must be array, got: {type(result['content'])}"
    )
    # Success response MUST have non-empty content (prevents false positive from empty array)
    assert len(result["content"]) > 0, (
        f"MCP tool call result content must be non-empty: {result}"
    )
    # Each content item MUST have a 'type' field per MCP spec
    for i, item in enumerate(result["content"]):
        assert isinstance(item, dict), (
            f"Content item {i} must be dict, got: {type(item)}"
        )
        assert "type" in item, f"Content item {i} missing required 'type' field: {item}"


# ============================================================================
# INFRASTRUCTURE DETECTION (in fixture, not at import)
# ============================================================================


@pytest.fixture(scope="session")
def infra_availability() -> dict[str, bool]:
    """Compute infrastructure availability at fixture time, not import time.

    This avoids network calls during test collection, which can cause
    timeouts and failures in CI environments without infrastructure.

    Returns:
        Dictionary with availability flags:
            - consul: True if Consul is reachable
            - postgres: True if PostgreSQL credentials are configured
            - full_infra: True if both Consul and PostgreSQL are available
    """
    consul_host = os.getenv("CONSUL_HOST")
    postgres_host = os.getenv("POSTGRES_HOST")
    postgres_password = os.getenv("POSTGRES_PASSWORD")

    consul_available = False
    if consul_host:
        try:
            consul_port_str = os.getenv("CONSUL_PORT", "28500")
            try:
                consul_port = int(consul_port_str)
            except ValueError as e:
                raise ModelOnexError(
                    f"Invalid CONSUL_PORT value: {consul_port_str!r} - must be an integer"
                ) from e
            with socket.socket() as s:
                s.settimeout(2.0)
                consul_available = s.connect_ex((consul_host, consul_port)) == 0
        except (OSError, TimeoutError):
            pass

    postgres_available = bool(postgres_host and postgres_password)

    return {
        "consul": consul_available,
        "postgres": postgres_available,
        "full_infra": consul_available and postgres_available,
    }


# ============================================================================
# MOCK TOOL DEFINITIONS
# ============================================================================


class MockToolDefinition:
    """Mock tool definition for testing.

    Conforms to ProtocolMCPToolDefinition protocol for use with
    TransportMCPStreamableHttp.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: list[object] | None = None,
    ) -> None:
        """Initialize mock tool definition.

        Args:
            name: Tool name (unique identifier).
            description: Human-readable description.
            parameters: Parameter definitions (optional).
        """
        self.name = name
        self.description = description
        self.parameters = parameters or []


# ============================================================================
# JSON-RPC HANDLER FACTORY
# ============================================================================


"""Type alias for sync or async executor callables."""
SyncExecutor = Callable[[str, JsonDict], JsonDict]
AsyncExecutor = Callable[[str, JsonDict], Awaitable[JsonDict]]


def create_json_rpc_handler(
    available_tools: list[JsonDict],
    executor: SyncExecutor | AsyncExecutor,
    server_name: str,
    call_history: list[JsonDict] | None = None,
) -> Callable[[Request], Awaitable[JSONResponse]]:
    """Factory for JSON-RPC endpoint handlers.

    Creates an async handler function that implements the MCP JSON-RPC protocol
    for testing purposes. This avoids the MCP SDK lifecycle complexity while
    providing deterministic behavior for E2E tests.

    Args:
        available_tools: List of tool definitions with name, description, inputSchema.
        executor: Callback to execute tool calls, receives (tool_name, arguments).
            Can be either sync or async - async executors will be awaited.
        server_name: Server name for MCP initialize response.
        call_history: Optional list to track tool calls for testing.

    Returns:
        Async handler function compatible with Starlette Route.
    """
    import asyncio
    import inspect

    async def mcp_endpoint(request: Request) -> JSONResponse:
        """Handle MCP JSON-RPC requests."""
        try:
            body = await request.json()
        except json_module.JSONDecodeError:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": "Parse error"},
                    "id": None,
                },
                status_code=400,
            )

        # Validate JSON-RPC format
        if "jsonrpc" not in body:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request: missing jsonrpc field",
                    },
                    "id": body.get("id"),
                },
                status_code=200,
            )

        method = body.get("method", "")
        params = body.get("params", {})
        request_id = body.get("id", 1)

        # Handle different MCP methods
        if method == "initialize":
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": server_name, "version": "1.0.0"},
                    },
                    "id": request_id,
                }
            )

        if method == "tools/list":
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "result": {"tools": available_tools},
                    "id": request_id,
                }
            )

        if method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})

            # Check if tool exists
            tool_names = {str(t["name"]) for t in available_tools}
            if tool_name not in tool_names:
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32602,
                            "message": f"Unknown tool: {tool_name}",
                        },
                        "id": request_id,
                    }
                )

            # Execute tool via callback (supports both sync and async executors)
            result = executor(tool_name, arguments)
            if asyncio.iscoroutine(result) or inspect.isawaitable(result):
                result = await result

            # Track call if history provided
            if call_history is not None:
                # Extract correlation_id from result if available (nested in result.result)
                correlation_id = None
                if isinstance(result, dict):
                    inner_result = result.get("result")
                    if isinstance(inner_result, dict):
                        correlation_id = inner_result.get("correlation_id")

                call_history.append(
                    {
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "correlation_id": correlation_id,
                    }
                )

            # Return MCP tool result format
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json_module.dumps(result),
                            }
                        ],
                        "isError": False,
                    },
                    "id": request_id,
                }
            )

        # Unknown method
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Method not found: {method}"},
                "id": request_id,
            }
        )

    return mcp_endpoint


# ============================================================================
# MCP APP FIXTURE (ASGI-based testing without real HTTP server)
# ============================================================================


@pytest.fixture
async def mcp_app_dev_mode() -> MCPDevModeFixture:
    """Create mock MCP app for dev mode testing.

    This fixture creates a simple mock Starlette app that handles MCP
    JSON-RPC requests directly without the complex MCP SDK lifecycle.

    The mock approach is used because the MCP SDK's streamable_http_app()
    requires proper task group initialization via run() before handling
    requests, which is incompatible with direct ASGI testing.

    Yields:
        MCPDevModeFixture containing:
            - app: The Starlette ASGI application
            - call_history: List of recorded tool calls
            - path: The MCP endpoint path
    """
    from uuid import uuid4

    from starlette.routing import Route

    # Track tool calls for assertions
    call_history: list[JsonDict] = []

    # Define available tools
    available_tools: list[JsonDict] = [
        {
            "name": "mock_compute",
            "description": "Mock compute tool - echoes input for deterministic testing",
            "inputSchema": {"type": "object", "properties": {}},
        }
    ]

    def mock_executor(tool_name: str, arguments: JsonDict) -> JsonDict:
        """Mock executor that returns deterministic results."""
        correlation_id = str(uuid4())
        return {
            "success": True,
            "result": {
                "status": "success",
                "echo": arguments.get("input_value"),
                "tool_name": tool_name,
                "correlation_id": correlation_id,
            },
        }

    # Create handler using shared factory
    mcp_endpoint = create_json_rpc_handler(
        available_tools=available_tools,
        executor=mock_executor,
        server_name="mock-mcp-server",
        call_history=call_history,
    )

    # Create Starlette app with the MCP endpoint
    app = Starlette(
        routes=[
            Route("/mcp/", mcp_endpoint, methods=["POST"]),
            Route("/mcp", mcp_endpoint, methods=["POST"]),
        ]
    )

    return MCPDevModeFixture(
        app=app,
        call_history=call_history,
        path="/mcp",
    )


# ============================================================================
# HTTP CLIENT FIXTURE (for ASGI testing)
# ============================================================================


@pytest.fixture
async def mcp_http_client(
    mcp_app_dev_mode: MCPDevModeFixture,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for testing MCP app via ASGI transport.

    Uses httpx.AsyncClient with ASGITransport to test the MCP app
    without starting a real HTTP server.

    Args:
        mcp_app_dev_mode: The MCP app fixture with typed structure.

    Yields:
        Configured httpx.AsyncClient.
    """
    app = mcp_app_dev_mode["app"]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
        follow_redirects=True,
    ) as client:
        yield client


# ============================================================================
# FULL INFRA FIXTURES (when available)
# ============================================================================


@pytest.fixture
async def mcp_app_full_infra(
    infra_availability: dict[str, bool],
) -> AsyncGenerator[MCPFullInfraFixture, None]:
    """Create MCP app with real Consul discovery using mock HTTP layer.

    This fixture requires full infrastructure (Consul + PostgreSQL)
    and will skip if not available. Uses real Consul for tool discovery
    but mocks the HTTP/JSON-RPC layer (same approach as mcp_app_dev_mode)
    because the MCP SDK's streamable_http_app() requires task group
    initialization that is incompatible with direct ASGI testing.

    Args:
        infra_availability: Infrastructure availability flags.

    Yields:
        MCPFullInfraFixture containing:
            - app: The Starlette ASGI application (mock JSON-RPC)
            - path: The MCP endpoint path
            - lifecycle: MCPServerLifecycle for cleanup
    """
    if not infra_availability["full_infra"]:
        pytest.skip(
            f"Full infrastructure required. "
            f"Consul: {infra_availability['consul']}, "
            f"PostgreSQL: {infra_availability['postgres']}"
        )

    from starlette.routing import Route

    from omnibase_infra.services.mcp import MCPServerLifecycle, ModelMCPServerConfig

    # Use lifecycle to discover real tools from Consul
    consul_port_str = os.getenv("CONSUL_PORT", "28500")
    try:
        consul_port = int(consul_port_str)
    except ValueError as e:
        raise ModelOnexError(
            f"Invalid CONSUL_PORT value: {consul_port_str!r} - must be an integer"
        ) from e

    lifecycle_config = ModelMCPServerConfig(
        dev_mode=False,
        consul_host=os.getenv("CONSUL_HOST", "localhost"),
        consul_port=consul_port,
        http_port=8090,  # Not used for ASGI testing
        http_host="127.0.0.1",
        kafka_enabled=False,
    )

    lifecycle = MCPServerLifecycle(lifecycle_config)
    await lifecycle.start()

    # Get discovered tools from registry and convert to dict format
    available_tools: list[JsonDict] = []
    if lifecycle.registry:
        registry_tools = await lifecycle.registry.list_tools()
        for tool in registry_tools:
            available_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {"type": "object", "properties": {}},
                }
            )

    # Real executor that routes to ONEX nodes via the MCP infrastructure
    async def real_executor(tool_name: str, arguments: JsonDict) -> JsonDict:
        """Execute MCP tool by routing to ONEX nodes via lifecycle infrastructure.

        This executor validates tools against the registry (discovered from Consul)
        and dispatches to ONEX orchestrators when endpoints are available.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments from the MCP call.

        Returns:
            Execution result dictionary with:
                - success: Whether execution succeeded
                - result: Execution result from ONEX (if dispatched)
                - tool_name: Name of the executed tool
                - source: Indicates execution source ("onex_dispatch" or "integration_test")
                - validation: Tool validation details (for integration test mode)

        Raises:
            ModelOnexError: If tool not found in registry.
        """
        import logging
        from uuid import uuid4

        logger = logging.getLogger(__name__)
        correlation_id = uuid4()

        # Validate tool exists in registry (discovered from Consul)
        if lifecycle.registry is None:
            return {
                "success": False,
                "error": "Registry not available",
                "tool_name": tool_name,
            }

        tool = await lifecycle.registry.get_tool(tool_name)
        if tool is None:
            raise ModelOnexError(f"Tool not found in registry: {tool_name}")

        # If tool has an endpoint, dispatch to ONEX orchestrator via HTTP
        if tool.endpoint and lifecycle.executor is not None:
            logger.info(
                "Dispatching tool to ONEX orchestrator",
                extra={
                    "tool_name": tool_name,
                    "endpoint": tool.endpoint,
                    "correlation_id": str(correlation_id),
                },
            )

            result = await lifecycle.executor.execute(
                tool=tool,
                arguments=arguments,
                correlation_id=correlation_id,
            )

            return {
                "success": result.get("success", False),
                "result": result.get("result"),
                "error": result.get("error"),
                "tool_name": tool_name,
                "source": "onex_dispatch",
                "correlation_id": str(correlation_id),
            }

        # No endpoint (local dev mode or Consul-discovered without endpoint)
        # Return structured response indicating integration test mode with validation
        logger.info(
            "Tool validated but no endpoint - integration test mode",
            extra={
                "tool_name": tool_name,
                "orchestrator_node_id": tool.orchestrator_node_id,
                "correlation_id": str(correlation_id),
            },
        )

        return {
            "success": True,
            "tool_name": tool_name,
            "source": "integration_test",
            "validation": {
                "tool_exists": True,
                "tool_version": tool.version,
                "orchestrator_node_id": tool.orchestrator_node_id,
                "orchestrator_service_id": tool.orchestrator_service_id,
                "has_endpoint": False,
                "timeout_seconds": tool.timeout_seconds,
            },
            "arguments_received": arguments,
            "correlation_id": str(correlation_id),
        }

    # Create handler using shared factory
    mcp_endpoint = create_json_rpc_handler(
        available_tools=available_tools,
        executor=real_executor,
        server_name="onex-mcp-server",
    )

    # Create mock Starlette app with real tool discovery
    app = Starlette(
        routes=[
            Route("/mcp/", mcp_endpoint, methods=["POST"]),
            Route("/mcp", mcp_endpoint, methods=["POST"]),
        ]
    )

    yield MCPFullInfraFixture(
        app=app,
        path="/mcp",
        lifecycle=lifecycle,
    )

    await lifecycle.shutdown()
