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
        assert "result" not in data, (
            f"Error response should not contain result field: {data}"
        )

        # JSON-RPC error response MUST have both code and message per spec
        error = data["error"]
        assert "code" in error, f"JSON-RPC error missing code field: {error}"
        assert "message" in error, f"JSON-RPC error missing message field: {error}"
        assert isinstance(error["code"], int), (
            f"JSON-RPC error code must be integer, got: {type(error['code'])}"
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
        assert "error" not in data, (
            f"Success response should not contain error field: {data}"
        )

        # MANDATORY: MCP tool call result MUST contain content array per spec
        result = data["result"]
        assert "content" in result, (
            f"MCP tool call result missing content array: {result}"
        )
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
            assert "type" in item, (
                f"Content item {i} missing required 'type' field: {item}"
            )

        # Verify call was recorded in history (prevents false positive if executor not called)
        # Use exact increment check to detect duplicate or missed recordings
        assert len(call_history) == initial_count + 1, (
            f"Expected exactly 1 new call in history (from {initial_count} to {initial_count + 1}), "
            f"got {len(call_history)}"
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
        assert "error" not in data, (
            f"Success response should not contain error field: {data}"
        )

        # MANDATORY: MCP tool call result MUST contain content array per spec
        result = data["result"]
        assert "content" in result, (
            f"MCP tool call result missing content array: {result}"
        )
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
            assert "type" in item, (
                f"Content item {i} missing required 'type' field: {item}"
            )

        # Call should be recorded in history - use exact increment check
        assert len(call_history) == initial_count + 1, (
            f"Expected exactly 1 new call in history (from {initial_count} to {initial_count + 1}), "
            f"got {len(call_history)}"
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

        # If error, verify it's a proper JSON-RPC error structure with BOTH code and message
        if has_error:
            error = data["error"]
            assert "code" in error, f"JSON-RPC error missing code field: {error}"
            assert "message" in error, f"JSON-RPC error missing message field: {error}"
            assert isinstance(error["code"], int), (
                f"JSON-RPC error code must be integer, got: {type(error['code'])}"
            )

        # If success, verify call was recorded in history and result structure
        if has_result:
            # MANDATORY: MCP tool call result MUST contain content array per spec
            result = data["result"]
            assert "content" in result, (
                f"MCP tool call result missing content array: {result}"
            )
            assert isinstance(result["content"], list), (
                f"MCP result content must be array, got: {type(result['content'])}"
            )
            # Success response MUST have non-empty content (prevents false positive)
            assert len(result["content"]) > 0, (
                f"MCP tool call result content must be non-empty: {result}"
            )
            # Each content item MUST have a 'type' field per MCP spec
            for i, item in enumerate(result["content"]):
                assert isinstance(item, dict), (
                    f"Content item {i} must be dict, got: {type(item)}"
                )
                assert "type" in item, (
                    f"Content item {i} missing required 'type' field: {item}"
                )

            # Use exact increment check for call history
            assert len(call_history) == initial_count + 1, (
                f"Expected exactly 1 new call in history (from {initial_count} to "
                f"{initial_count + 1}), got {len(call_history)}"
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

    Uses mock JSON-RPC handlers (not real MCP SDK) to verify async execution
    completes successfully. This class tests the happy path of async execution,
    not timeout or error scenarios.
    """

    async def test_async_execution_completes_successfully(
        self,
        mcp_http_client: object,
        mcp_app_dev_mode: dict[str, object],
    ) -> None:
        """Async tool execution completes successfully and returns valid response.

        This test verifies:
        1. Async execution completes within test timeout
        2. Response has correct HTTP 200 status
        3. Response contains valid MCP result structure with content array
        4. Call history is updated (executor was invoked)
        5. Arguments are passed through correctly
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
                    "arguments": {"input_value": "async_execution_test"},
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
        assert "error" not in data, (
            f"Success response should not contain error field: {data}"
        )

        # MANDATORY: MCP tool call result MUST contain content array per spec
        result = data["result"]
        assert "content" in result, (
            f"MCP tool call result missing content array: {result}"
        )
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
            assert "type" in item, (
                f"Content item {i} missing required 'type' field: {item}"
            )

        # Verify call was recorded in history (ensures executor was actually invoked)
        # Use exact increment check to detect duplicate or missed recordings
        assert len(call_history) == initial_count + 1, (
            f"Expected exactly 1 new call in history (from {initial_count} to {initial_count + 1}), "
            f"got {len(call_history)}"
        )
        latest_call = call_history[-1]
        assert latest_call["tool_name"] == "mock_compute", (
            f"Expected tool_name 'mock_compute', got '{latest_call['tool_name']}'"
        )
        # Verify arguments were passed through correctly
        arguments = latest_call["arguments"]
        assert isinstance(arguments, dict), f"Expected dict, got {type(arguments)}"
        assert arguments.get("input_value") == "async_execution_test", (
            f"Expected input_value='async_execution_test', got {arguments}"
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
        assert "result" not in data, (
            f"Error response should not contain result field: {data}"
        )

        # Verify error has proper JSON-RPC structure - BOTH code AND message required
        error = data["error"]
        assert "code" in error, f"JSON-RPC error missing code field: {error}"
        assert "message" in error, f"JSON-RPC error missing message field: {error}"
        assert isinstance(error["code"], int), (
            f"JSON-RPC error code must be integer, got: {type(error['code'])}"
        )

        # MANDATORY: error code MUST be -32600 (Invalid Request) per JSON-RPC 2.0 spec
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
        This test verifies:
        1. HTTP 200 status returned
        2. Response contains result (not error)
        3. Result contains required MCP fields (protocolVersion, capabilities)
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
        assert "error" not in data, (
            f"Success response should not contain error field: {data}"
        )

        # MCP initialize response MUST include protocolVersion and capabilities
        result = data["result"]
        assert "protocolVersion" in result, (
            f"MCP initialize response missing protocolVersion: {result}"
        )
        assert "capabilities" in result, (
            f"MCP initialize response missing capabilities: {result}"
        )
        # Verify capabilities is a dict (can be empty but must be present)
        assert isinstance(result["capabilities"], dict), (
            f"MCP capabilities must be dict, got: {type(result['capabilities'])}"
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
        assert "result" not in data, (
            f"Error response should not contain result field: {data}"
        )

        # Verify error has proper JSON-RPC structure - BOTH code AND message required
        error = data["error"]
        assert "code" in error, f"JSON-RPC error missing code field: {error}"
        assert "message" in error, f"JSON-RPC error missing message field: {error}"
        assert isinstance(error["code"], int), (
            f"JSON-RPC error code must be integer, got: {type(error['code'])}"
        )

        # MANDATORY: parse error code MUST be -32700 per JSON-RPC 2.0 spec
        assert error["code"] == -32700, (
            f"Expected parse error code -32700, got: {error['code']}"
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
        assert "result" not in data, (
            f"Error response should not contain result field for {method}: {data}"
        )

        # Verify error has proper JSON-RPC structure - BOTH code AND message required
        error = data["error"]
        assert "code" in error, (
            f"JSON-RPC error missing code field for {method}: {error}"
        )
        assert "message" in error, (
            f"JSON-RPC error missing message field for {method}: {error}"
        )
        assert isinstance(error["code"], int), (
            f"JSON-RPC error code must be integer for {method}, got: {type(error['code'])}"
        )

        # MANDATORY: error code MUST be -32600 (Invalid Request) per JSON-RPC 2.0 spec
        assert error["code"] == -32600, (
            f"Expected error code -32600 (Invalid Request) for {method}, "
            f"got: {error['code']}"
        )
