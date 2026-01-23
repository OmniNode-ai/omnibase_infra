# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Mock-based MCP protocol tests for tool discovery.

IMPORTANT: These are NOT true integration tests. They use mock JSON-RPC handlers
to test the MCP protocol handling logic WITHOUT the real MCP SDK.

What these tests verify:
- JSON-RPC protocol compliance for tools/list method
- Response structure validation
- Mock tool registry behavior

What these tests do NOT verify:
- Real MCP SDK server lifecycle (startup, shutdown, task groups)
- Actual MCP client library behavior
- Real network transport behavior

Why mocks are used:
The MCP SDK's streamable_http_app() requires proper task group initialization
via run() before handling requests, which is incompatible with direct ASGI
testing via httpx. These mock tests provide fast, deterministic protocol
validation without the SDK complexity.

For real MCP SDK integration tests, see:
    tests/integration/services/mcp/e2e/test_mcp_real_e2e.py

Related Ticket: OMN-1408
"""

from __future__ import annotations

import httpx
import pytest

from .conftest import MCPDevModeFixture, MCPFullInfraFixture

pytestmark = [
    pytest.mark.mcp_protocol,
    pytest.mark.asyncio,
    pytest.mark.timeout(10),
]


class TestMockMCPToolDiscovery:
    """Mock-based MCP protocol tests for tool discovery.

    Uses mock JSON-RPC handlers (not real MCP SDK) to verify protocol compliance.
    """

    async def test_list_tools_returns_discovered_nodes(
        self,
        mcp_http_client: httpx.AsyncClient,
        mcp_app_dev_mode: MCPDevModeFixture,
    ) -> None:
        """MCP tools/list returns ONEX nodes from registry.

        Verifies:
        - tools/list request returns HTTP 200 success
        - Response contains result with tools array (no error)
        - Tools array includes mock_compute tool
        """
        client = mcp_http_client
        path = mcp_app_dev_mode["path"]

        # Send MCP JSON-RPC request to list tools
        response = await client.post(
            f"{path}/",
            json={
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": 1,
            },
            headers={"Content-Type": "application/json"},
        )

        # Tool listing should succeed
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        # Parse and verify response structure
        data = response.json()
        assert "error" not in data, f"Unexpected error in response: {data.get('error')}"
        assert "result" in data, f"Expected 'result' in response, got: {data}"
        result = data["result"]
        assert "tools" in result, f"Expected 'tools' in result, got: {result}"
        tools = result["tools"]
        tool_names = {t.get("name") for t in tools}
        assert "mock_compute" in tool_names, (
            f"Expected 'mock_compute' in tools, got: {tool_names}"
        )

    async def test_initialize_returns_mcp_protocol_fields(
        self,
        mcp_http_client: httpx.AsyncClient,
        mcp_app_dev_mode: MCPDevModeFixture,
    ) -> None:
        """MCP initialize returns required protocol fields.

        Verifies:
        - Initialize request succeeds with HTTP 200 status
        - Response contains result with protocolVersion, capabilities, serverInfo
        - serverInfo contains name and version fields
        """
        client = mcp_http_client
        path = mcp_app_dev_mode["path"]

        # Send a basic request
        response = await client.post(
            f"{path}/",
            json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0.0"},
                },
                "id": 1,
            },
            headers={"Content-Type": "application/json"},
        )

        # Initialize must succeed in dev mode
        assert response.status_code == 200, (
            f"Initialize must succeed in dev mode, got {response.status_code}"
        )

        # Verify response structure - must have result, not error
        data = response.json()
        assert "error" not in data, f"Unexpected error in response: {data.get('error')}"
        assert "result" in data, f"Expected 'result' in response, got: {data}"

        # Verify initialize response contains expected MCP protocol fields
        result = data["result"]
        assert "protocolVersion" in result, (
            f"Initialize response must contain 'protocolVersion', got: {result}"
        )
        assert "capabilities" in result, (
            f"Initialize response must contain 'capabilities', got: {result}"
        )
        assert "serverInfo" in result, (
            f"Initialize response must contain 'serverInfo', got: {result}"
        )

        # Verify serverInfo has required fields
        server_info = result["serverInfo"]
        assert "name" in server_info, (
            f"serverInfo must contain 'name', got: {server_info}"
        )
        assert "version" in server_info, (
            f"serverInfo must contain 'version', got: {server_info}"
        )


class TestMCPToolDiscoveryWithInfra:
    """MCP tool discovery with real infrastructure (Consul).

    NOTE: Still uses mock JSON-RPC layer, but discovers tools from real Consul.
    """

    async def test_tool_discovery_with_real_consul(
        self,
        mcp_app_full_infra: MCPFullInfraFixture,
    ) -> None:
        """Discover tools from real Consul registry when infrastructure available.

        This test requires Consul and PostgreSQL to be running.
        It verifies that real ONEX nodes registered in Consul are
        discoverable via MCP with HTTP 200 responses.

        Note: Infrastructure availability check is handled by mcp_app_full_infra
        fixture which depends on infra_availability and skips if unavailable.
        """
        app = mcp_app_full_infra["app"]
        path = mcp_app_full_infra["path"]

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
            base_url="http://testserver",
            follow_redirects=True,
        ) as client:
            response = await client.post(
                f"{path}/",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "id": 1,
                },
                headers={"Content-Type": "application/json"},
            )

            # Tool listing must succeed when infrastructure is available
            assert response.status_code == 200, (
                f"Expected 200, got {response.status_code}"
            )

            # Verify response structure - must have result, not error
            data = response.json()
            assert "error" not in data, (
                f"Unexpected error in response: {data.get('error')}"
            )
            assert "result" in data, f"Expected 'result' in response, got: {data}"
            result = data["result"]
            assert "tools" in result, f"Expected 'tools' in result, got: {result}"

            # Verify tools is a list (may be empty if nothing registered in Consul)
            tools = result["tools"]
            assert isinstance(tools, list), (
                f"Expected 'tools' to be a list, got: {type(tools).__name__}"
            )

            # Verify each discovered tool has required MCP tool fields
            # (tools list may be empty if Consul has no registered services)
            for tool in tools:
                assert "name" in tool, f"Tool must have 'name', got: {tool}"
                assert isinstance(tool["name"], str), (
                    f"Tool 'name' must be a string, got: {type(tool['name']).__name__}"
                )
                assert "description" in tool, (
                    f"Tool must have 'description', got: {tool}"
                )
                assert "inputSchema" in tool, (
                    f"Tool must have 'inputSchema', got: {tool}"
                )
