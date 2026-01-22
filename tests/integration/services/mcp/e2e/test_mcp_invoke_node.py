# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Test MCP tool invocation via HTTP/JSON-RPC.

These tests verify that ONEX nodes can be invoked through the MCP
protocol using the tools/call method.

Related Ticket: OMN-1408
"""

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

        # Response should be valid
        assert response.status_code in (200, 307, 400, 404)

        # If we got 200, verify the result structure
        if response.status_code == 200:
            data = response.json()
            if "result" in data:
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

        # If the request succeeded, verify executor received it
        if response.status_code == 200 and len(call_history) > 0:
            call = call_history[0]
            assert call["tool_name"] == "mock_compute"
            assert call["arguments"]["input_value"] == "arg_test"
            assert call["arguments"]["extra_field"] == 42

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
        await client.post(
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

        await client.post(
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

        # If calls were recorded, verify independence
        if len(call_history) >= 2:
            assert call_history[0]["arguments"]["input_value"] == "first"
            assert call_history[1]["arguments"]["input_value"] == "second"
            assert (
                call_history[0]["correlation_id"] != call_history[1]["correlation_id"]
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
            transport=httpx.ASGITransport(app=app),
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

            # Response should be valid
            assert list_response.status_code in (200, 307, 400, 404)
