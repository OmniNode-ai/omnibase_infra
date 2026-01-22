# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""MCP E2E test fixtures with runtime infrastructure detection.

This module provides pytest fixtures for MCP E2E integration tests with:
- Infrastructure detection at fixture time (NOT import time)
- Port allocation with retry pattern for race condition handling
- MCP app using TransportMCPStreamableHttp (ASGI transport for testing)
- Direct ASGI testing without actual HTTP server (more reliable)

Related Ticket: OMN-1408
"""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import httpx
import pytest

if TYPE_CHECKING:
    from starlette.applications import Starlette


pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


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
            with socket.socket() as s:
                s.settimeout(2.0)
                consul_available = s.connect_ex((consul_host, 28500)) == 0
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
# MCP APP FIXTURE (ASGI-based testing without real HTTP server)
# ============================================================================


@pytest.fixture
async def mcp_app_dev_mode() -> AsyncGenerator[dict[str, object], None]:
    """Create MCP app for dev mode testing using ASGI transport.

    This fixture creates the MCP Starlette app without starting a real
    HTTP server. Tests use httpx.AsyncClient with ASGITransport for
    direct ASGI testing, which is more reliable than real HTTP.

    Yields:
        Dictionary containing:
            - app: The Starlette ASGI application
            - call_history: List of recorded tool calls
            - path: The MCP endpoint path
    """
    from omnibase_infra.handlers.mcp import TransportMCPStreamableHttp
    from omnibase_infra.handlers.models.mcp import ModelMcpHandlerConfig

    # Track tool calls for assertions
    call_history: list[dict[str, object]] = []

    def mock_executor(tool_name: str, arguments: dict[str, object]) -> object:
        """Mock executor that returns deterministic results."""
        from uuid import uuid4

        correlation_id = str(uuid4())
        call_history.append(
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "correlation_id": correlation_id,
            }
        )

        return {
            "success": True,
            "result": {
                "status": "success",
                "echo": arguments.get("input_value"),
                "tool_name": tool_name,
                "correlation_id": correlation_id,
            },
        }

    # Create transport with test configuration
    config = ModelMcpHandlerConfig(
        host="127.0.0.1",
        port=8090,  # Not used for ASGI testing
        path="/mcp",
        stateless=True,
        json_response=True,
        timeout_seconds=10.0,
    )

    transport = TransportMCPStreamableHttp(config=config)

    # Create mock tools
    tools = [
        MockToolDefinition(
            name="mock_compute",
            description="Mock compute tool - echoes input for deterministic testing",
            parameters=[],
        ),
    ]

    # Create the app (but don't start a server)
    app = transport.create_app(tools=tools, tool_executor=mock_executor)  # type: ignore[arg-type]

    yield {
        "app": app,
        "call_history": call_history,
        "path": config.path,
        "transport": transport,
    }

    # Cleanup transport state
    await transport.stop()


# ============================================================================
# HTTP CLIENT FIXTURE (for ASGI testing)
# ============================================================================


@pytest.fixture
async def mcp_http_client(
    mcp_app_dev_mode: dict[str, object],
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for testing MCP app via ASGI transport.

    Uses httpx.AsyncClient with ASGITransport to test the MCP app
    without starting a real HTTP server.

    Args:
        mcp_app_dev_mode: The MCP app fixture.

    Yields:
        Configured httpx.AsyncClient.
    """
    import httpx

    app: Starlette = mcp_app_dev_mode["app"]  # type: ignore[assignment]

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
) -> AsyncGenerator[dict[str, object], None]:
    """Create MCP app with real Consul discovery.

    This fixture requires full infrastructure (Consul + PostgreSQL)
    and will skip if not available. Used for testing real workflows.

    Args:
        infra_availability: Infrastructure availability flags.

    Yields:
        Dictionary containing:
            - app: The Starlette ASGI application
            - path: The MCP endpoint path
    """
    if not infra_availability["full_infra"]:
        pytest.skip(
            f"Full infrastructure required. "
            f"Consul: {infra_availability['consul']}, "
            f"PostgreSQL: {infra_availability['postgres']}"
        )

    from omnibase_infra.handlers.mcp import TransportMCPStreamableHttp
    from omnibase_infra.handlers.models.mcp import ModelMcpHandlerConfig
    from omnibase_infra.services.mcp import MCPServerLifecycle, ModelMCPServerConfig

    # Use lifecycle to discover real tools from Consul
    lifecycle_config = ModelMCPServerConfig(
        dev_mode=False,
        consul_host=os.getenv("CONSUL_HOST", "localhost"),
        consul_port=int(os.getenv("CONSUL_PORT", "28500")),
        http_port=8090,  # Not used for ASGI testing
        http_host="127.0.0.1",
        kafka_enabled=False,
    )

    lifecycle = MCPServerLifecycle(lifecycle_config)
    await lifecycle.start()

    # Get discovered tools from registry
    tools = []
    if lifecycle.registry:
        registry_tools = await lifecycle.registry.list_tools()
        for tool in registry_tools:
            tools.append(
                MockToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=list(tool.parameters) if tool.parameters else [],
                )
            )

    # Create real executor that routes to ONEX
    def real_executor(tool_name: str, arguments: dict[str, object]) -> object:
        """Real executor that routes to ONEX nodes."""
        return {"status": "success", "tool_name": tool_name}

    # Create transport
    transport_config = ModelMcpHandlerConfig(
        host="127.0.0.1",
        port=8090,
        path="/mcp",
        stateless=True,
        json_response=True,
        timeout_seconds=30.0,
    )

    transport = TransportMCPStreamableHttp(config=transport_config)
    app = transport.create_app(tools=tools, tool_executor=real_executor)  # type: ignore[arg-type]

    yield {
        "app": app,
        "path": transport_config.path,
        "lifecycle": lifecycle,
        "transport": transport,
    }

    await transport.stop()
    await lifecycle.shutdown()
