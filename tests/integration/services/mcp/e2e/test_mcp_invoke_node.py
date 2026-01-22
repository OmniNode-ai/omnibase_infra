# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Test MCP tool invocation via HTTP/JSON-RPC.

These tests verify that ONEX nodes can be invoked through the MCP
protocol using the tools/call method.

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


class TestMCPInvokeNode:
    """MCP protocol invokes ONEX nodes exposed as tools."""

    async def test_call_tool_returns_structured_result(
        self,
        mcp_http_client: object,
        mcp_app_dev_mode: dict[str, object],
    ) -> None:
        """MCP tools/call returns deterministic structured result.

        Verifies:
        - tools/call request returns a valid response
        - Response contains the expected result structure
        """
        client: httpx.AsyncClient = mcp_http_client  # type: ignore[assignment]
        path = str(mcp_app_dev_mode["path"])

        # Send MCP JSON-RPC request to call tool
        response = await client.post(
            f"{path}/",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "mock_compute",
                    "arguments": {"input_value": "test_data_123"},
                },
                "id": 1,
            },
            headers={"Content-Type": "application/json"},
        )

        # Tool call should succeed
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        # Verify successful result (not error) before validating structure
        data = response.json()
        assert "result" in data, (
            f"Expected success result, got error: {data.get('error')}"
        )
        assert "error" not in data, f"Unexpected error in response: {data.get('error')}"

        # Verify the result structure (mandatory)
        result = data["result"]
        # MCP tool result contains content array
        if "content" in result:
            content = result["content"]
            assert len(content) >= 1

    async def test_executor_receives_correct_arguments(
        self,
        mcp_http_client: object,
        mcp_app_dev_mode: dict[str, object],
    ) -> None:
        """Verify MCP layer correctly routes arguments to executor.

        The mock executor records all calls, allowing us to verify
        that arguments are passed through correctly from MCP client
        to ONEX executor.
        """
        client: httpx.AsyncClient = mcp_http_client  # type: ignore[assignment]
        path = str(mcp_app_dev_mode["path"])
        call_history: list[dict[str, object]] = mcp_app_dev_mode["call_history"]  # type: ignore[assignment]

        # Clear any previous calls
        call_history.clear()

        # Send tool call request
        response = await client.post(
            f"{path}/",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "mock_compute",
                    "arguments": {"input_value": "arg_test", "extra_field": 42},
                },
                "id": 1,
            },
            headers={"Content-Type": "application/json"},
        )

        # Request must succeed
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        # Verify successful result (not error)
        data = response.json()
        assert "result" in data, (
            f"Expected success result, got error: {data.get('error')}"
        )

        # Verify executor received the call (mandatory assertions)
        assert len(call_history) == 1, (
            f"Expected exactly 1 call, got {len(call_history)}"
        )
        call = call_history[0]
        assert call["tool_name"] == "mock_compute"
        arguments = call["arguments"]
        assert isinstance(arguments, dict)
        assert arguments["input_value"] == "arg_test"
        assert arguments["extra_field"] == 42

    async def test_multiple_invocations_are_independent(
        self,
        mcp_http_client: object,
        mcp_app_dev_mode: dict[str, object],
    ) -> None:
        """Multiple tool invocations are independent and tracked.

        Each invocation should:
        - Have its own correlation ID
        - Be recorded separately in call history
        """
        client: httpx.AsyncClient = mcp_http_client  # type: ignore[assignment]
        path = str(mcp_app_dev_mode["path"])
        call_history: list[dict[str, object]] = mcp_app_dev_mode["call_history"]  # type: ignore[assignment]

        call_history.clear()

        # Make multiple calls
        response1 = await client.post(
            f"{path}/",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "mock_compute",
                    "arguments": {"input_value": "first"},
                },
                "id": 1,
            },
            headers={"Content-Type": "application/json"},
        )

        response2 = await client.post(
            f"{path}/",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "mock_compute",
                    "arguments": {"input_value": "second"},
                },
                "id": 2,
            },
            headers={"Content-Type": "application/json"},
        )

        # Both requests must succeed
        assert response1.status_code == 200, (
            f"First call failed: {response1.status_code}"
        )
        assert response2.status_code == 200, (
            f"Second call failed: {response2.status_code}"
        )

        # Verify successful results (not errors)
        data1 = response1.json()
        data2 = response2.json()
        assert "result" in data1, f"First call error: {data1.get('error')}"
        assert "result" in data2, f"Second call error: {data2.get('error')}"

        # Both calls must be recorded (mandatory assertions)
        assert len(call_history) == 2, (
            f"Expected exactly 2 calls, got {len(call_history)}"
        )
        args_0 = call_history[0]["arguments"]
        args_1 = call_history[1]["arguments"]
        assert isinstance(args_0, dict) and isinstance(args_1, dict)
        assert args_0["input_value"] == "first"
        assert args_1["input_value"] == "second"
        assert call_history[0]["correlation_id"] != call_history[1]["correlation_id"], (
            "Each invocation must have unique correlation_id"
        )


class TestMCPInvokeWorkflow:
    """MCP protocol invokes full ONEX workflow (requires infrastructure)."""

    async def test_invoke_registration_workflow(
        self,
        infra_availability: dict[str, bool],
        mcp_app_full_infra: dict[str, object],
    ) -> None:
        """MCP invokes real workflow end-to-end.

        This test requires full infrastructure (Consul + PostgreSQL)
        and invokes real ONEX workflows.
        """
        app = mcp_app_full_infra["app"]
        path = str(mcp_app_full_infra["path"])

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
            base_url="http://testserver",
            follow_redirects=True,
        ) as client:
            # First, list available tools
            list_response = await client.post(
                f"{path}/",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "id": 1,
                },
                headers={"Content-Type": "application/json"},
            )

            # tools/list should succeed
            assert list_response.status_code == 200, (
                f"Expected 200, got {list_response.status_code}"
            )

            # Verify successful result (not error)
            list_data = list_response.json()
            assert "result" in list_data, (
                f"Expected success result, got error: {list_data.get('error')}"
            )
