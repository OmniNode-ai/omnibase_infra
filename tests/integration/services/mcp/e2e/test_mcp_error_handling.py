# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Test MCP error handling via HTTP/JSON-RPC.

These tests verify that errors at both the MCP protocol level
and ONEX execution level are properly handled and returned
to clients in a structured format.

Related Tickets:
    - OMN-1408: MCP E2E Integration Tests
"""

from __future__ import annotations

import httpx
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.timeout(10),
]


class TestMCPRoutingErrors:
    """MCP-level errors (tool not found, invalid routing)."""

    async def test_nonexistent_tool_returns_error(
        self,
        mcp_http_client: object,
        mcp_app_dev_mode: dict[str, object],
    ) -> None:
        """Calling nonexistent tool returns error response.

        The MCP protocol should return an error for unknown tools.
        """
        client: httpx.AsyncClient = mcp_http_client  # type: ignore[assignment]
        path = str(mcp_app_dev_mode["path"])

        response = await client.post(
            f"{path}/",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "nonexistent_tool_xyz_12345",
                    "arguments": {},
                },
                "id": 1,
            },
            headers={"Content-Type": "application/json"},
        )

        # Response should be valid (might be error or 404)
        assert response.status_code in (200, 307, 400, 404)

        # If 200, check for error in JSON-RPC response
        if response.status_code == 200:
            data = response.json()
            # For nonexistent tools, MCP MUST return an error
            assert "error" in data, f"Expected error for nonexistent tool, got: {data}"
            # JSON-RPC error response should have message or code
            assert "message" in data["error"] or "code" in data["error"]


class TestMCPBasicFunctionality:
    """Basic MCP functionality tests."""

    async def test_tool_call_with_empty_arguments(
        self,
        mcp_http_client: object,
        mcp_app_dev_mode: dict[str, object],
    ) -> None:
        """Tool call with empty arguments succeeds.

        Verifies the MCP layer handles empty argument dicts correctly.
        """
        client: httpx.AsyncClient = mcp_http_client  # type: ignore[assignment]
        path = str(mcp_app_dev_mode["path"])

        response = await client.post(
            f"{path}/",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "mock_compute",
                    "arguments": {},
                },
                "id": 1,
            },
            headers={"Content-Type": "application/json"},
        )

        # Empty arguments should succeed
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    async def test_tool_call_records_to_history(
        self,
        mcp_http_client: object,
        mcp_app_dev_mode: dict[str, object],
    ) -> None:
        """Tool calls are recorded in call history.

        Verifies the executor receives and records calls correctly.
        """
        client: httpx.AsyncClient = mcp_http_client  # type: ignore[assignment]
        path = str(mcp_app_dev_mode["path"])
        call_history: list[dict[str, object]] = mcp_app_dev_mode["call_history"]  # type: ignore[assignment]

        initial_count = len(call_history)

        response = await client.post(
            f"{path}/",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "mock_compute",
                    "arguments": {"test_key": "test_value"},
                },
                "id": 1,
            },
            headers={"Content-Type": "application/json"},
        )

        # Tool call should succeed
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        # Call should be recorded in history
        assert len(call_history) > initial_count, (
            "Expected call to be recorded in history"
        )
        latest_call = call_history[-1]
        assert latest_call["tool_name"] == "mock_compute"
        arguments = latest_call["arguments"]
        assert isinstance(arguments, dict), f"Expected dict, got {type(arguments)}"
        assert arguments["test_key"] == "test_value"


class TestMCPTimeoutHandling:
    """MCP timeout handling tests."""

    async def test_normal_execution_completes_within_timeout(
        self,
        mcp_http_client: object,
        mcp_app_dev_mode: dict[str, object],
    ) -> None:
        """Normal tool execution completes within timeout.

        This test verifies async execution works correctly within
        the configured timeout period.
        """
        client: httpx.AsyncClient = mcp_http_client  # type: ignore[assignment]
        path = str(mcp_app_dev_mode["path"])

        # Make a normal call - should complete quickly
        response = await client.post(
            f"{path}/",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "mock_compute",
                    "arguments": {"input_value": "timeout_test"},
                },
                "id": 1,
            },
            headers={"Content-Type": "application/json"},
        )

        # Normal execution should succeed
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"


class TestMCPProtocolCompliance:
    """MCP protocol compliance tests."""

    async def test_json_rpc_format_required(
        self,
        mcp_http_client: object,
        mcp_app_dev_mode: dict[str, object],
    ) -> None:
        """MCP endpoint requires JSON-RPC format.

        Sending invalid JSON-RPC should return an error.
        """
        client: httpx.AsyncClient = mcp_http_client  # type: ignore[assignment]
        path = str(mcp_app_dev_mode["path"])

        # Send invalid request (missing jsonrpc field)
        response = await client.post(
            f"{path}/",
            json={"method": "tools/list", "id": 1},
            headers={"Content-Type": "application/json"},
        )

        # Invalid JSON-RPC should return error (400) or be handled gracefully (200 with error)
        assert response.status_code in (200, 400), (
            f"Expected 200 or 400, got {response.status_code}"
        )

        # If 200, the response should contain an error for invalid JSON-RPC
        if response.status_code == 200:
            data = response.json()
            assert "error" in data, f"Expected error for invalid JSON-RPC, got: {data}"

    async def test_initialize_method_supported(
        self,
        mcp_http_client: object,
        mcp_app_dev_mode: dict[str, object],
    ) -> None:
        """MCP initialize method is supported.

        The initialize method is required by MCP protocol.
        """
        client: httpx.AsyncClient = mcp_http_client  # type: ignore[assignment]
        path = str(mcp_app_dev_mode["path"])

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

        # Initialize is a required MCP method, should succeed
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
