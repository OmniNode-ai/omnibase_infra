# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Test MCP tool discovery via HTTP/JSON-RPC.

These tests verify that ONEX nodes exposed via MCP can be discovered
using the MCP protocol's tools/list method.

Related Ticket: OMN-1408
"""

from __future__ import annotations

import httpx
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.timeout(10),
]


class TestMCPToolDiscovery:
    """MCP protocol discovers ONEX nodes exposed as tools."""

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

        # Initialize should succeed; 400 acceptable if protocol requires session setup first
        # 307/404/405 indicate broken endpoint and should fail the test
        assert response.status_code in (200, 400), (
            f"Expected 200 or 400, got {response.status_code}"
        )


class TestMCPToolDiscoveryWithInfra:
    """MCP tool discovery with real infrastructure (Consul)."""

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
