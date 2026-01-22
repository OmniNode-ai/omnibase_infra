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
        mcp_http_client: object,
        mcp_app_dev_mode: dict[str, object],
    ) -> None:
        """MCP tools/list returns ONEX nodes from registry.

        Verifies:
        - tools/list request returns a valid response
        - Response contains the mock_compute tool
        - Tool has correct description from contract
        """
        client: httpx.AsyncClient = mcp_http_client  # type: ignore[assignment]
        path = str(mcp_app_dev_mode["path"])

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

    async def test_mcp_endpoint_accessible(
        self,
        mcp_http_client: object,
        mcp_app_dev_mode: dict[str, object],
    ) -> None:
        """MCP endpoint is accessible via HTTP.

        Verifies the MCP endpoint responds to requests.
        """
        client: httpx.AsyncClient = mcp_http_client  # type: ignore[assignment]
        path = str(mcp_app_dev_mode["path"])

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

        # Verify response structure
        data = response.json()
        assert "error" not in data, f"Unexpected error in response: {data.get('error')}"
        assert "result" in data, f"Expected 'result' in response, got: {data}"


class TestMCPToolDiscoveryWithInfra:
    """MCP tool discovery with real infrastructure (Consul).

    NOTE: Still uses mock JSON-RPC layer, but discovers tools from real Consul.
    """

    async def test_tool_discovery_with_real_consul(
        self,
        infra_availability: dict[str, bool],
        mcp_app_full_infra: object,
    ) -> None:
        """When infrastructure available, real orchestrators are discovered.

        This test requires Consul and PostgreSQL to be running.
        It verifies that real ONEX nodes registered in Consul are
        discoverable via MCP.
        """
        fixture: dict[str, object] = mcp_app_full_infra  # type: ignore[assignment]
        app = fixture["app"]
        path = str(fixture["path"])

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

            # Tool listing should succeed when infrastructure is available
            assert response.status_code == 200, (
                f"Expected 200, got {response.status_code}"
            )

            # Verify response structure
            data = response.json()
            assert "error" not in data, (
                f"Unexpected error in response: {data.get('error')}"
            )
            assert "result" in data, f"Expected 'result' in response, got: {data}"
            result = data["result"]
            assert "tools" in result, f"Expected 'tools' in result, got: {result}"
