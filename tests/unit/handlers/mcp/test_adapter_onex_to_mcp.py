# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ONEXToMCPAdapter.invoke_tool() real dispatch path (OMN-2697).

Covers:
- Successful dispatch → correct CallToolResult shape
- ONEX error response → isError: True result
- Timeout → MCP error content
- Circuit-open → MCP error content
- Tool not found → InfraUnavailableError
- No executor configured → ProtocolConfigurationError

All tests use mocked AdapterONEXToolExecution (no real network).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.handlers.mcp.adapter_onex_to_mcp import (
    MCPToolParameter,
    ONEXToMCPAdapter,
)

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(
    executor: object | None = None,
) -> ONEXToMCPAdapter:
    return ONEXToMCPAdapter(node_executor=executor)  # type: ignore[arg-type]


async def _register_tool(
    adapter: ONEXToMCPAdapter,
    name: str = "my_tool",
    endpoint: str = "http://localhost:8085/execute",
    timeout_seconds: int = 10,
) -> None:
    await adapter.register_node_as_tool(
        node_name=name,
        description="A test tool",
        parameters=[
            MCPToolParameter(
                name="input_data",
                parameter_type="string",
                description="Input payload",
                required=True,
            )
        ],
        version="1.0.0",
        timeout_seconds=timeout_seconds,
    )
    # Patch execution_endpoint into the cached tool definition.
    # MCPToolDefinition is a non-frozen dataclass, so direct assignment works.
    adapter._tool_cache[name].execution_endpoint = endpoint


# ---------------------------------------------------------------------------
# R1 / R2: Successful dispatch
# ---------------------------------------------------------------------------


class TestSuccessfulDispatch:
    """Successful ONEX response maps to CallToolResult with isError: False."""

    async def test_invoke_tool_returns_content_list_on_success(self) -> None:
        """invoke_tool() returns MCP CallToolResult with content list."""
        executor = MagicMock()
        executor.execute = AsyncMock(
            return_value={"success": True, "result": {"output": "done"}}
        )

        adapter = _make_adapter(executor)
        await _register_tool(adapter)

        result = await adapter.invoke_tool("my_tool", {"input_data": "hello"})

        assert result["isError"] is False
        content = result["content"]
        assert isinstance(content, list)
        assert len(content) == 1
        assert content[0]["type"] == "text"
        # Result should be JSON-encoded payload
        text = content[0]["text"]
        assert isinstance(text, str)
        parsed = json.loads(text)
        assert parsed["output"] == "done"

    async def test_invoke_tool_threads_correlation_id_to_executor(self) -> None:
        """Correlation ID supplied by caller is forwarded to executor.execute()."""
        executor = MagicMock()
        executor.execute = AsyncMock(return_value={"success": True, "result": "ok"})

        adapter = _make_adapter(executor)
        await _register_tool(adapter)

        cid = uuid4()
        await adapter.invoke_tool("my_tool", {}, correlation_id=cid)

        call_kwargs = executor.execute.call_args.kwargs
        assert call_kwargs["correlation_id"] == cid

    async def test_invoke_tool_uses_tool_timeout_seconds(self) -> None:
        """Per-tool timeout_seconds is passed to ModelMCPToolDefinition."""
        executor = MagicMock()
        executor.execute = AsyncMock(return_value={"success": True, "result": "ok"})

        adapter = _make_adapter(executor)
        await _register_tool(adapter, timeout_seconds=42)

        await adapter.invoke_tool("my_tool", {})

        tool_arg = executor.execute.call_args[1]["tool"]
        assert tool_arg.timeout_seconds == 42

    async def test_invoke_tool_generates_correlation_id_when_absent(self) -> None:
        """A fresh UUID is generated when no correlation_id is supplied."""
        executor = MagicMock()
        executor.execute = AsyncMock(return_value={"success": True, "result": "ok"})

        adapter = _make_adapter(executor)
        await _register_tool(adapter)

        await adapter.invoke_tool("my_tool", {})

        call_kwargs = executor.execute.call_args[1]
        cid = call_kwargs["correlation_id"]
        assert isinstance(cid, UUID)

    async def test_invoke_tool_with_string_result(self) -> None:
        """Plain string result from executor is placed directly in content text."""
        executor = MagicMock()
        executor.execute = AsyncMock(
            return_value={"success": True, "result": "plain text result"}
        )

        adapter = _make_adapter(executor)
        await _register_tool(adapter)

        result = await adapter.invoke_tool("my_tool", {})

        assert result["isError"] is False
        assert result["content"][0]["text"] == "plain text result"

    async def test_protocol_fields_stripped_from_result_payload(self) -> None:
        """Envelope protocol fields are stripped from dict results before MCP serialization.

        When the orchestrator returns an envelope-shaped result the internal
        protocol fields (envelope_id, correlation_id, source, payload, metadata,
        success) must not appear in the MCP content text.  Domain fields such as
        "data" must be preserved.
        """
        executor = MagicMock()
        executor.execute = AsyncMock(
            return_value={
                "success": True,
                "result": {
                    "envelope_id": "abc",
                    "correlation_id": "xyz",
                    "source": "mcp-adapter",
                    "payload": {"arg": 1},
                    "metadata": {"k": "v"},
                    "success": True,
                    "data": "actual_value",
                },
            }
        )

        adapter = _make_adapter(executor)
        await _register_tool(adapter)

        result = await adapter.invoke_tool("my_tool", {"input_data": "hello"})

        assert result["isError"] is False
        text = result["content"][0]["text"]

        # Protocol envelope fields must be stripped
        assert "envelope_id" not in text
        assert "correlation_id" not in text
        assert '"source"' not in text
        assert "metadata" not in text
        assert '"payload"' not in text

        # Domain data must be preserved
        assert "actual_value" in text or "data" in text


# ---------------------------------------------------------------------------
# R2: ONEX error response → isError: True
# ---------------------------------------------------------------------------


class TestONEXErrorMapping:
    """ONEX error responses map to CallToolResult with isError: True."""

    async def test_onex_error_response_sets_is_error_true(self) -> None:
        """When executor returns success=False, isError is True."""
        executor = MagicMock()
        executor.execute = AsyncMock(
            return_value={"success": False, "error": "node execution failed"}
        )

        adapter = _make_adapter(executor)
        await _register_tool(adapter)

        result = await adapter.invoke_tool("my_tool", {})

        assert result["isError"] is True
        content = result["content"]
        assert isinstance(content, list)
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert "node execution failed" in content[0]["text"]

    async def test_onex_error_response_without_error_field_uses_fallback(
        self,
    ) -> None:
        """When error key is absent, a generic message appears in content."""
        executor = MagicMock()
        executor.execute = AsyncMock(return_value={"success": False})

        adapter = _make_adapter(executor)
        await _register_tool(adapter)

        result = await adapter.invoke_tool("my_tool", {})

        assert result["isError"] is True
        assert result["content"][0]["text"] == "Tool execution failed"

    async def test_timeout_maps_to_mcp_error_content(self) -> None:
        """Timeout error message from executor appears as MCP error content."""
        timeout_message = "Tool execution timed out after 10 seconds"
        executor = MagicMock()
        executor.execute = AsyncMock(
            return_value={"success": False, "error": timeout_message}
        )

        adapter = _make_adapter(executor)
        await _register_tool(adapter)

        result = await adapter.invoke_tool("my_tool", {})

        assert result["isError"] is True
        assert timeout_message in result["content"][0]["text"]

    async def test_circuit_open_maps_to_mcp_error_content(self) -> None:
        """Circuit-open message from executor appears as MCP error content."""
        circuit_message = "Service temporarily unavailable - circuit breaker open"
        executor = MagicMock()
        executor.execute = AsyncMock(
            return_value={"success": False, "error": circuit_message}
        )

        adapter = _make_adapter(executor)
        await _register_tool(adapter)

        result = await adapter.invoke_tool("my_tool", {})

        assert result["isError"] is True
        assert circuit_message in result["content"][0]["text"]

    async def test_infra_timeout_exception_maps_to_mcp_error(self) -> None:
        """InfraTimeoutError raised by execute() is caught and returned as MCP error."""
        from omnibase_infra.errors import ModelTimeoutErrorContext

        timeout_ctx = ModelTimeoutErrorContext(
            transport_type=EnumInfraTransportType.HTTP,
            operation="execute_tool",
        )
        exc = InfraTimeoutError("timed out calling orchestrator", context=timeout_ctx)

        executor = MagicMock()
        executor.execute = AsyncMock(side_effect=exc)

        adapter = _make_adapter(executor)
        await _register_tool(adapter)

        result = await adapter.invoke_tool("my_tool", {})

        assert result["isError"] is True
        text = result["content"][0]["text"]
        assert "timed out" in text.lower() or "timeout" in text.lower()

    async def test_infra_unavailable_exception_maps_to_mcp_error(self) -> None:
        """InfraUnavailableError raised by execute() is caught and returned as MCP error."""
        ctx = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.HTTP,
            operation="execute_tool",
        )
        exc = InfraUnavailableError("circuit breaker open", context=ctx)

        executor = MagicMock()
        executor.execute = AsyncMock(side_effect=exc)

        adapter = _make_adapter(executor)
        await _register_tool(adapter)

        result = await adapter.invoke_tool("my_tool", {})

        assert result["isError"] is True
        assert (
            "unavailable" in result["content"][0]["text"].lower()
            or "circuit" in result["content"][0]["text"].lower()
        )


# ---------------------------------------------------------------------------
# R3: Guard conditions
# ---------------------------------------------------------------------------


class TestGuardConditions:
    """Pre-condition checks before executor dispatch."""

    async def test_invoke_tool_raises_when_tool_not_found(self) -> None:
        """InfraUnavailableError raised when tool is not in registry."""
        adapter = _make_adapter()

        with pytest.raises(InfraUnavailableError, match="not found"):
            await adapter.invoke_tool("nonexistent_tool", {})

    async def test_invoke_tool_raises_when_no_executor_configured(self) -> None:
        """ProtocolConfigurationError raised when executor is not set."""
        adapter = _make_adapter(executor=None)
        await _register_tool(adapter)

        with pytest.raises(
            ProtocolConfigurationError, match="Node executor not configured"
        ):
            await adapter.invoke_tool("my_tool", {})

    async def test_no_mock_responses_in_invoke_tool(self) -> None:
        """Verify invoke_tool dispatches to executor, not a hardcoded stub.

        After a successful call the result must not contain the old mock fields
        ('message', 'arguments') from the pre-OMN-2697 implementation.
        """
        executor = MagicMock()
        executor.execute = AsyncMock(
            return_value={"success": True, "result": {"computed": True}}
        )

        adapter = _make_adapter(executor)
        await _register_tool(adapter)

        result = await adapter.invoke_tool("my_tool", {"k": "v"})

        # MCP shape must be present
        assert "content" in result
        assert "isError" in result
        # Old mock stub fields must be absent
        assert "message" not in result
        assert "arguments" not in result
        # Executor was actually called
        executor.execute.assert_awaited_once()

    async def test_invoke_tool_passes_arguments_to_executor(self) -> None:
        """Arguments dict is forwarded verbatim to executor.execute()."""
        executor = MagicMock()
        executor.execute = AsyncMock(return_value={"success": True, "result": "ok"})

        adapter = _make_adapter(executor)
        await _register_tool(adapter)

        args = {"input_data": "test_payload", "flag": True}
        await adapter.invoke_tool("my_tool", args)

        call_kwargs = executor.execute.call_args[1]
        assert call_kwargs["arguments"] == args

    async def test_tool_with_no_endpoint_returns_error(self) -> None:
        """When a tool has no execution_endpoint set, invoke_tool returns isError: True.

        Exercises the code path where execution_endpoint is "" (default), which
        becomes endpoint=None in ModelMCPToolDefinition. The executor raises
        InfraUnavailableError, which the adapter catches and maps to an MCP
        error result rather than propagating the exception.
        """
        ctx = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.HTTP,
            operation="execute_tool",
        )
        exc = InfraUnavailableError("no endpoint configured for tool", context=ctx)

        executor = MagicMock()
        executor.execute = AsyncMock(side_effect=exc)

        adapter = _make_adapter(executor)
        # Register the tool WITHOUT patching execution_endpoint — it stays as "".
        await adapter.register_node_as_tool(
            node_name="no_endpoint_tool",
            description="A tool with no endpoint",
            parameters=[],
            version="1.0.0",
        )

        result = await adapter.invoke_tool("no_endpoint_tool", {})

        assert result["isError"] is True
        text = result["content"][0]["text"]
        assert (
            "unavailable" in text.lower()
            or "endpoint" in text.lower()
            or "no endpoint" in text.lower()
        )
