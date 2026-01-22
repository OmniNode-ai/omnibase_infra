# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Mock-based MCP protocol tests for tool invocation.

IMPORTANT: These are NOT true integration tests. They use mock JSON-RPC handlers
to test the MCP protocol handling logic WITHOUT the real MCP SDK.

What these tests verify:
- JSON-RPC protocol compliance for tools/call method
- Argument passing through the mock executor
- Call history tracking and correlation IDs
- Concurrent request handling

What these tests do NOT verify:
- Real MCP SDK server lifecycle (startup, shutdown, task groups)
- Actual MCP client library behavior
- Real ONEX node execution

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


class TestMockMCPInvokeNode:
    """Mock-based MCP protocol tests for tool invocation.

    Uses mock JSON-RPC handlers (not real MCP SDK) to verify protocol compliance.
    """

    async def test_call_tool_returns_structured_result(
        self,
        mcp_http_client: object,
        mcp_app_dev_mode: dict[str, object],
    ) -> None:
        """MCP tools/call returns deterministic structured result.

        Verifies:
        - tools/call request returns a valid response
        - Response contains the expected result structure
        - Call was recorded in executor history
        """
        client: httpx.AsyncClient = mcp_http_client  # type: ignore[assignment]
        path = str(mcp_app_dev_mode["path"])
        call_history: list[dict[str, object]] = mcp_app_dev_mode["call_history"]  # type: ignore[assignment]

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

        # Tool call should succeed (HTTP level)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        # Verify successful result (not error) before validating structure
        data = response.json()
        assert "result" in data, (
            f"Expected success result, got error: {data.get('error')}"
        )
        assert "error" not in data, f"Unexpected error in response: {data.get('error')}"

        # Verify the result structure (mandatory assertions)
        result = data["result"]
        # MCP tool result MUST contain content array
        assert "content" in result, (
            f"MCP result must contain 'content' array, got keys: {list(result.keys())}"
        )
        content = result["content"]
        assert len(content) >= 1, "MCP content array must not be empty"

        # Verify call was recorded in executor history (prevents silent failures)
        assert len(call_history) >= 1, (
            f"Expected at least 1 call recorded in history, got {len(call_history)}. "
            "Tool call may have silently failed to reach executor."
        )
        # Verify the recorded call matches what we sent
        last_call = call_history[-1]
        assert last_call["tool_name"] == "mock_compute", (
            f"Expected tool_name 'mock_compute', got '{last_call['tool_name']}'"
        )

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

    async def test_concurrent_invocations_are_independent(
        self,
        mcp_http_client: object,
        mcp_app_dev_mode: dict[str, object],
    ) -> None:
        """Concurrent tool invocations are independent and tracked.

        Unlike sequential tests, this verifies the MCP server correctly
        handles multiple simultaneous requests without race conditions.
        """
        import asyncio

        client: httpx.AsyncClient = mcp_http_client  # type: ignore[assignment]
        path = str(mcp_app_dev_mode["path"])
        call_history: list[dict[str, object]] = mcp_app_dev_mode["call_history"]  # type: ignore[assignment]

        async def make_call(value: str) -> httpx.Response:
            return await client.post(
                f"{path}/",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "mock_compute",
                        "arguments": {"input_value": value},
                    },
                    "id": 1,
                },
                headers={"Content-Type": "application/json"},
            )

        # Make 5 concurrent calls
        responses = await asyncio.gather(
            make_call("concurrent_1"),
            make_call("concurrent_2"),
            make_call("concurrent_3"),
            make_call("concurrent_4"),
            make_call("concurrent_5"),
        )

        # All requests must succeed
        for i, response in enumerate(responses):
            assert response.status_code == 200, (
                f"Call {i + 1} failed: {response.status_code}"
            )
            data = response.json()
            assert "result" in data, f"Call {i + 1} error: {data.get('error')}"

        # All calls must be recorded
        assert len(call_history) == 5, f"Expected 5 calls, got {len(call_history)}"

        # All correlation_ids must be unique
        correlation_ids = [call["correlation_id"] for call in call_history]
        assert len(set(correlation_ids)) == 5, (
            "Each concurrent call must have unique correlation_id"
        )


class TestMCPInvokeWorkflow:
    """MCP tool invocation with real infrastructure (Consul).

    NOTE: Still uses mock JSON-RPC layer, but executes against real Consul services.
    """

    async def test_invoke_registration_workflow(
        self,
        mcp_app_full_infra: dict[str, object],
    ) -> None:
        """MCP invokes real workflow end-to-end.

        This test requires full infrastructure (Consul + PostgreSQL)
        and invokes real ONEX workflows through the registry.

        Note: Infrastructure availability check is handled by mcp_app_full_infra
        fixture which depends on infra_availability and skips if unavailable.

        Test flow:
        1. List available tools from Consul-discovered registry
        2. Select a discovered tool
        3. Invoke the tool via tools/call
        4. Verify the real executor returns structured response
        """
        import json as json_module

        app = mcp_app_full_infra["app"]
        path = str(mcp_app_full_infra["path"])

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
            base_url="http://testserver",
            follow_redirects=True,
        ) as client:
            # Step 1: List available tools from Consul-discovered registry
            list_response = await client.post(
                f"{path}/",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "id": 1,
                },
                headers={"Content-Type": "application/json"},
            )

            # tools/list must succeed
            assert list_response.status_code == 200, (
                f"Expected 200, got {list_response.status_code}"
            )

            list_data = list_response.json()
            assert "result" in list_data, (
                f"Expected success result, got error: {list_data.get('error')}"
            )

            tools = list_data["result"].get("tools", [])

            # Step 2: Skip if no tools discovered (Consul may have no services)
            if not tools:
                pytest.skip(
                    "No tools discovered from Consul. "
                    "Ensure ONEX services are registered in Consul."
                )

            # Step 3: Select first discovered tool and invoke it
            tool_name = str(tools[0]["name"])

            invoke_response = await client.post(
                f"{path}/",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": {"test_input": "workflow_test"},
                    },
                    "id": 2,
                },
                headers={"Content-Type": "application/json"},
            )

            # tools/call must succeed
            assert invoke_response.status_code == 200, (
                f"Expected 200, got {invoke_response.status_code}"
            )

            invoke_data = invoke_response.json()
            assert "result" in invoke_data, (
                f"Expected success result, got error: {invoke_data.get('error')}"
            )

            # Step 4: Parse MCP content and verify executor response structure
            result = invoke_data["result"]
            assert "content" in result, "MCP result must contain content array"
            assert len(result["content"]) >= 1, "Content array must not be empty"

            # MCP wraps executor result as JSON string in content[0].text
            content_item = result["content"][0]
            assert content_item.get("type") == "text", (
                f"Expected text content, got {content_item.get('type')}"
            )

            # Parse the JSON-stringified executor response
            executor_result = json_module.loads(content_item["text"])

            # Verify executor response structure (from real_executor in conftest)
            assert "success" in executor_result, (
                f"Executor result missing 'success': {executor_result}"
            )
            assert "tool_name" in executor_result, (
                f"Executor result missing 'tool_name': {executor_result}"
            )
            assert executor_result["tool_name"] == tool_name, (
                f"Tool name mismatch: expected {tool_name}, "
                f"got {executor_result['tool_name']}"
            )

            # Verify source indicates where execution came from
            assert "source" in executor_result, (
                f"Executor result missing 'source': {executor_result}"
            )
            # Source is either "onex_dispatch" (real endpoint) or "integration_test"
            assert executor_result["source"] in ("onex_dispatch", "integration_test"), (
                f"Unexpected source: {executor_result['source']}"
            )

            # If integration_test mode, verify validation structure
            if executor_result["source"] == "integration_test":
                assert "validation" in executor_result, (
                    "Integration test mode must include validation details"
                )
                validation = executor_result["validation"]
                assert validation.get("tool_exists") is True, (
                    "Tool must be validated as existing"
                )
                # Verify additional validation fields exist
                assert "tool_version" in validation
                assert "orchestrator_node_id" in validation

            # Verify correlation_id is present (required for tracing)
            assert "correlation_id" in executor_result, (
                "Executor result must include correlation_id for tracing"
            )
