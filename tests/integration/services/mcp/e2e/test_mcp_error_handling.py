# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Mock-based MCP protocol tests for error handling.

IMPORTANT: These are NOT true integration tests. They use mock JSON-RPC handlers
to test the MCP protocol error handling logic WITHOUT the real MCP SDK.

What these tests verify:
- JSON-RPC error response structure (code, message)
- Unknown tool error handling
- Malformed JSON error handling
- Protocol validation errors (missing jsonrpc field)

What these tests do NOT verify:
- Real MCP SDK error handling
- Actual network error conditions
- Real ONEX node execution failures

Why mocks are used:
The MCP SDK's streamable_http_app() requires proper task group initialization
via run() before handling requests, which is incompatible with direct ASGI
testing via httpx. These mock tests provide fast, deterministic protocol
validation without the SDK complexity.

For real MCP SDK integration tests, see:
    tests/integration/services/mcp/e2e/test_mcp_real_e2e.py

Related Tickets:
    - OMN-1408: MCP E2E Integration Tests
"""

from __future__ import annotations

import httpx
import pytest

pytestmark = [
    pytest.mark.mcp_protocol,
    pytest.mark.asyncio,
    pytest.mark.timeout(10),
]


class TestMockMCPRoutingErrors:
    """Mock-based MCP protocol tests for routing errors.

    Uses mock JSON-RPC handlers (not real MCP SDK) to verify error response format.
    """

    async def test_nonexistent_tool_returns_error(
        self,
        mcp_http_client: object,
        mcp_app_dev_mode: dict[str, object],
    ) -> None:
        """Calling nonexistent tool returns error response.

        The MCP protocol should return an error for unknown tools.
        In dev-mode, this MUST return 200 with a JSON-RPC error response.
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

        # Dev-mode MUST return 200 with JSON-RPC error (not HTTP error codes)
        assert response.status_code == 200, (
            f"Expected 200 with JSON-RPC error, got HTTP {response.status_code}"
        )

        # MANDATORY: For nonexistent tools, MCP MUST return an error in the response
        data = response.json()
        assert "error" in data, f"Expected error for nonexistent tool, got: {data}"
        # JSON-RPC error response MUST have message or code
        assert "message" in data["error"] or "code" in data["error"], (
            f"JSON-RPC error missing message/code: {data['error']}"
        )


class TestMockMCPBasicFunctionality:
    """Mock-based MCP protocol tests for basic functionality.

    Uses mock JSON-RPC handlers (not real MCP SDK) to verify protocol compliance.
    """

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
        call_history: list[dict[str, object]] = mcp_app_dev_mode["call_history"]  # type: ignore[assignment]

        initial_count = len(call_history)

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

        # Empty arguments should succeed with 200 status
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        # Verify response contains result, not error (avoid false positive on error response)
        data = response.json()
        assert "result" in data, (
            f"Expected success result, got error: {data.get('error')}"
        )

        # Verify call was recorded in history (prevents false positive if executor not called)
        assert len(call_history) > initial_count, (
            "Expected call to be recorded in history after successful execution"
        )
        latest_call = call_history[-1]
        assert latest_call["tool_name"] == "mock_compute", (
            f"Expected tool_name 'mock_compute', got '{latest_call['tool_name']}'"
        )

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

        # Tool call should succeed with 200 status
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        # Verify response contains result, not error (before asserting history)
        data = response.json()
        assert "result" in data, (
            f"Expected success result, got error: {data.get('error')}"
        )

        # Call should be recorded in history
        assert len(call_history) > initial_count, (
            "Expected call to be recorded in history"
        )
        latest_call = call_history[-1]
        assert latest_call["tool_name"] == "mock_compute"
        arguments = latest_call["arguments"]
        assert isinstance(arguments, dict), f"Expected dict, got {type(arguments)}"
        assert arguments["test_key"] == "test_value"

    async def test_tool_call_with_missing_required_argument(
        self,
        mcp_http_client: object,
        mcp_app_dev_mode: dict[str, object],
    ) -> None:
        """Tool call with missing required argument handles gracefully.

        Verifies the MCP layer handles partial arguments correctly,
        either by using defaults or returning a structured error.
        This differs from empty arguments - here we provide some fields
        but potentially omit others that may be required.
        """
        client: httpx.AsyncClient = mcp_http_client  # type: ignore[assignment]
        path = str(mcp_app_dev_mode["path"])
        call_history: list[dict[str, object]] = mcp_app_dev_mode["call_history"]  # type: ignore[assignment]

        initial_count = len(call_history)

        # Call with partial arguments - some fields present, others missing
        # The tool may expect certain fields; we intentionally provide incomplete data
        response = await client.post(
            f"{path}/",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "mock_compute",
                    "arguments": {
                        # Provide only one field, potentially missing required fields
                        "partial_field": "partial_value",
                    },
                },
                "id": 1,
            },
            headers={"Content-Type": "application/json"},
        )

        # System should handle gracefully: either succeed (lenient) or return error
        assert response.status_code == 200, (
            f"Expected 200 (graceful handling), got {response.status_code}"
        )

        data = response.json()
        # Response must be well-formed JSON-RPC: either result or error, not both
        has_result = "result" in data
        has_error = "error" in data
        assert has_result or has_error, (
            f"Expected either result or error in response, got: {data}"
        )
        assert not (has_result and has_error), (
            f"Response should not have both result and error: {data}"
        )

        # If error, verify it's a proper JSON-RPC error structure
        if has_error:
            error = data["error"]
            assert "message" in error or "code" in error, (
                f"JSON-RPC error missing message/code: {error}"
            )

        # If success, verify call was recorded in history
        if has_result:
            assert len(call_history) > initial_count, (
                "Expected call to be recorded in history on successful execution"
            )
            latest_call = call_history[-1]
            assert latest_call["tool_name"] == "mock_compute", (
                f"Expected tool_name 'mock_compute', got '{latest_call['tool_name']}'"
            )
            # Verify the partial arguments were passed through
            arguments = latest_call["arguments"]
            assert isinstance(arguments, dict), f"Expected dict, got {type(arguments)}"
            assert arguments.get("partial_field") == "partial_value", (
                f"Expected partial_field='partial_value', got {arguments}"
            )


class TestMockMCPAsyncExecution:
    """Mock-based MCP protocol tests for async execution behavior.

    Uses mock JSON-RPC handlers (not real MCP SDK) to verify async execution.
    Note: This class tests normal async execution, not timeout error scenarios.
    """

    async def test_normal_execution_completes_successfully(
        self,
        mcp_http_client: object,
        mcp_app_dev_mode: dict[str, object],
    ) -> None:
        """Tool execution completes successfully via async handler.

        This test verifies async execution works correctly and returns
        a valid response within the test timeout.
        """
        client: httpx.AsyncClient = mcp_http_client  # type: ignore[assignment]
        path = str(mcp_app_dev_mode["path"])
        call_history: list[dict[str, object]] = mcp_app_dev_mode["call_history"]  # type: ignore[assignment]

        initial_count = len(call_history)

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

        # Normal execution should succeed with 200 status
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        # Verify response contains result, not error
        data = response.json()
        assert "result" in data, (
            f"Expected success result, got error: {data.get('error')}"
        )

        # Verify call was recorded in history (ensures executor was actually invoked)
        assert len(call_history) > initial_count, (
            "Expected call to be recorded in history after successful execution"
        )
        latest_call = call_history[-1]
        assert latest_call["tool_name"] == "mock_compute", (
            f"Expected tool_name 'mock_compute', got '{latest_call['tool_name']}'"
        )
        # Verify arguments were passed through correctly
        arguments = latest_call["arguments"]
        assert isinstance(arguments, dict), f"Expected dict, got {type(arguments)}"
        assert arguments.get("input_value") == "timeout_test", (
            f"Expected input_value='timeout_test', got {arguments}"
        )


class TestMockMCPProtocolCompliance:
    """Mock-based MCP JSON-RPC protocol compliance tests.

    Uses mock JSON-RPC handlers (not real MCP SDK) to verify protocol compliance.
    """

    async def test_json_rpc_format_required(
        self,
        mcp_http_client: object,
        mcp_app_dev_mode: dict[str, object],
    ) -> None:
        """MCP endpoint requires JSON-RPC format.

        Sending invalid JSON-RPC should return an error.
        In dev-mode, the mock handler returns 200 with JSON-RPC error.
        """
        client: httpx.AsyncClient = mcp_http_client  # type: ignore[assignment]
        path = str(mcp_app_dev_mode["path"])

        # Send invalid request (missing jsonrpc field)
        response = await client.post(
            f"{path}/",
            json={"method": "tools/list", "id": 1},
            headers={"Content-Type": "application/json"},
        )

        # Dev-mode MUST return 200 with JSON-RPC error (not HTTP error codes)
        assert response.status_code == 200, (
            f"Expected 200 with JSON-RPC error in dev-mode, got HTTP {response.status_code}"
        )

        # MANDATORY: Response MUST contain JSON-RPC error for invalid request
        data = response.json()
        assert "error" in data, (
            f"Expected JSON-RPC error for missing jsonrpc field, got: {data}"
        )

        # Verify error has proper JSON-RPC structure (code and/or message)
        error = data["error"]
        assert "message" in error or "code" in error, (
            f"JSON-RPC error missing message/code: {error}"
        )

        # Verify error code is -32600 (Invalid Request) per JSON-RPC 2.0 spec
        if "code" in error:
            assert error["code"] == -32600, (
                f"Expected error code -32600 (Invalid Request), got: {error['code']}"
            )

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

        # Initialize is a required MCP method, should succeed with 200 status
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        # Verify response contains result with expected MCP initialize structure
        data = response.json()
        assert "result" in data, (
            f"Expected success result, got error: {data.get('error')}"
        )

        # MCP initialize response MUST include protocolVersion and capabilities
        result = data["result"]
        assert "protocolVersion" in result, (
            f"MCP initialize response missing protocolVersion: {result}"
        )
        assert "capabilities" in result, (
            f"MCP initialize response missing capabilities: {result}"
        )

    async def test_malformed_json_returns_parse_error(
        self,
        mcp_http_client: object,
        mcp_app_dev_mode: dict[str, object],
    ) -> None:
        """Malformed JSON body returns JSON-RPC parse error.

        The MCP protocol should return error code -32700 for invalid JSON.
        This is a standard JSON-RPC error code for parse errors.
        """
        client: httpx.AsyncClient = mcp_http_client  # type: ignore[assignment]
        path = str(mcp_app_dev_mode["path"])

        response = await client.post(
            f"{path}/",
            content="not valid json{",
            headers={"Content-Type": "application/json"},
        )

        # Malformed JSON MUST return 400 with parse error
        assert response.status_code == 400, (
            f"Expected 400 for malformed JSON, got {response.status_code}"
        )

        data = response.json()
        assert "error" in data, f"Expected error for malformed JSON, got: {data}"
        assert data["error"]["code"] == -32700, (
            f"Expected parse error code -32700, got: {data['error']['code']}"
        )

    @pytest.mark.parametrize("method", ["initialize", "tools/list", "tools/call"])
    async def test_missing_jsonrpc_field_returns_error(
        self,
        mcp_http_client: object,
        mcp_app_dev_mode: dict[str, object],
        method: str,
    ) -> None:
        """Missing jsonrpc field returns error for {method}.

        JSON-RPC 2.0 requires the jsonrpc field to be present.
        Testing across multiple MCP methods ensures consistent validation.
        In dev-mode, the mock handler returns 200 with JSON-RPC error.
        """
        client: httpx.AsyncClient = mcp_http_client  # type: ignore[assignment]
        path = str(mcp_app_dev_mode["path"])

        # Build request without jsonrpc field
        request_body: dict[str, object] = {"method": method, "id": 1}
        if method == "tools/call":
            request_body["params"] = {"name": "mock_compute", "arguments": {}}

        response = await client.post(
            f"{path}/",
            json=request_body,
            headers={"Content-Type": "application/json"},
        )

        # Dev-mode MUST return 200 with JSON-RPC error (not HTTP error codes)
        assert response.status_code == 200, (
            f"Expected 200 with JSON-RPC error in dev-mode for {method}, "
            f"got HTTP {response.status_code}"
        )

        # MANDATORY: Response MUST contain JSON-RPC error for invalid request
        data = response.json()
        assert "error" in data, (
            f"Expected JSON-RPC error for missing jsonrpc field on {method}, got: {data}"
        )

        # Verify error has proper JSON-RPC structure
        error = data["error"]
        assert "message" in error or "code" in error, (
            f"JSON-RPC error missing message/code for {method}: {error}"
        )

        # Verify error code is -32600 (Invalid Request) per JSON-RPC 2.0 spec
        if "code" in error:
            assert error["code"] == -32600, (
                f"Expected error code -32600 (Invalid Request) for {method}, "
                f"got: {error['code']}"
            )
