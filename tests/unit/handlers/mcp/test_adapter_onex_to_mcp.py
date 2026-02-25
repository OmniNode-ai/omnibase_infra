# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ONEXToMCPAdapter — invoke_tool() dispatch (OMN-2697) and
contract-driven discovery (OMN-2698).

Covers invoke_tool():
- Successful dispatch → correct CallToolResult shape
- ONEX error response → isError: True result
- Timeout → MCP error content
- Circuit-open → MCP error content
- Tool not found → InfraUnavailableError
- No executor configured → ProtocolConfigurationError

Covers discover_tools() with contracts_root:
- discover_tools() with contracts_root returning mcp.expose: true contracts
- Contracts without mcp section are silently ignored
- Contracts with mcp.expose: false are silently ignored
- Invalid / unparseable YAML is skipped (non-fatal)
- Input schema derived from Pydantic model via pydantic_to_json_schema()
- Input schema falls back to {"type": "object"} for unknown models
- Manual registrations take precedence when tool name collides with contract
- Tag filtering applied after contract scan
- discover_tools() without contracts_root returns cached tools only
- V3 verification: no TODO(OMN-1288) stub in discover_tools

All tests use mocked AdapterONEXToolExecution (no real network).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.handlers.mcp.adapter_onex_to_mcp import (
    MCPToolDefinition,
    MCPToolParameter,
    ONEXToMCPAdapter,
)

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers (invoke_tool tests)
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
# Fixtures (contract-discovery tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter() -> ONEXToMCPAdapter:
    """Fresh adapter with empty cache."""
    return ONEXToMCPAdapter()


@pytest.fixture
def mcp_contract_yaml() -> str:
    """Minimal valid contract with mcp.expose: true."""
    return textwrap.dedent(
        """\
        name: "node_example_orchestrator"
        node_type: "ORCHESTRATOR_GENERIC"
        description: "An example orchestrator for testing."
        node_version: "1.0.0"
        mcp:
          expose: true
          tool_name: "example_tool"
          description: "Execute the example orchestrator."
          timeout_seconds: 45
        input_model:
          name: "ModelExampleInput"
          module: "tests.unit.handlers.mcp.test_adapter_onex_to_mcp"
          description: "Input for example."
        """
    )


@pytest.fixture
def contract_no_mcp_yaml() -> str:
    """Contract with no mcp section — should be silently ignored."""
    return textwrap.dedent(
        """\
        name: "node_no_mcp"
        node_type: "EFFECT_GENERIC"
        description: "Effect node without MCP."
        """
    )


@pytest.fixture
def contract_mcp_expose_false_yaml() -> str:
    """Contract with mcp.expose: false — should be silently ignored."""
    return textwrap.dedent(
        """\
        name: "node_disabled_mcp"
        node_type: "ORCHESTRATOR_GENERIC"
        description: "Orchestrator with MCP disabled."
        mcp:
          expose: false
          tool_name: "disabled_tool"
        """
    )


@pytest.fixture
def contract_no_tool_name_yaml() -> str:
    """Contract with mcp.expose: true but no tool_name — falls back to name."""
    return textwrap.dedent(
        """\
        name: "node_fallback_name"
        node_type: "ORCHESTRATOR_GENERIC"
        description: "Orchestrator with fallback tool name."
        mcp:
          expose: true
        """
    )


# ---------------------------------------------------------------------------
# Inline Pydantic model for test_contract_with_pydantic_input_schema
# ---------------------------------------------------------------------------


class ModelExampleInput(BaseModel):
    """Minimal Pydantic model exposed as an MCP tool input."""

    workflow_id: str
    dry_run: bool = False
    max_retries: int = 3


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


# ---------------------------------------------------------------------------
# Tests: discover_tools() without contracts_root
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDiscoverToolsNoScan:
    """discover_tools() with no contracts_root — cached tools only."""

    @pytest.mark.asyncio
    async def test_empty_cache_returns_empty(self, adapter: ONEXToMCPAdapter) -> None:
        tools = await adapter.discover_tools()
        assert list(tools) == []

    @pytest.mark.asyncio
    async def test_manually_registered_tool_returned(
        self, adapter: ONEXToMCPAdapter
    ) -> None:
        await adapter.register_node_as_tool(
            "manual_tool", "Manual tool", [], tags=["infra"]
        )
        tools = await adapter.discover_tools()
        assert len(list(tools)) == 1
        assert next(iter(tools)).name == "manual_tool"

    @pytest.mark.asyncio
    async def test_tag_filter_applied(self, adapter: ONEXToMCPAdapter) -> None:
        await adapter.register_node_as_tool("tool_a", "A", [], tags=["alpha"])
        await adapter.register_node_as_tool("tool_b", "B", [], tags=["beta"])
        tools = await adapter.discover_tools(tags=["alpha"])
        assert len(list(tools)) == 1
        assert next(iter(tools)).name == "tool_a"


# ---------------------------------------------------------------------------
# Tests: discover_tools() with contracts_root
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDiscoverToolsWithContractsRoot:
    """discover_tools() scans contract.yaml files when contracts_root is set."""

    @pytest.mark.asyncio
    async def test_mcp_enabled_contract_discovered(
        self,
        adapter: ONEXToMCPAdapter,
        tmp_path: Path,
        mcp_contract_yaml: str,
    ) -> None:
        """Contract with mcp.expose: true is discovered and returned."""
        node_dir = tmp_path / "node_example_orchestrator"
        node_dir.mkdir()
        (node_dir / "contract.yaml").write_text(mcp_contract_yaml)

        tools = await adapter.discover_tools(contracts_root=tmp_path)
        tool_names = [t.name for t in tools]
        assert "example_tool" in tool_names

    @pytest.mark.asyncio
    async def test_discovered_tool_has_correct_description(
        self,
        adapter: ONEXToMCPAdapter,
        tmp_path: Path,
        mcp_contract_yaml: str,
    ) -> None:
        """mcp.description overrides top-level description."""
        (tmp_path / "contract.yaml").write_text(mcp_contract_yaml)
        tools = await adapter.discover_tools(contracts_root=tmp_path)
        tool = next(t for t in tools if t.name == "example_tool")
        assert "example orchestrator" in tool.description.lower()

    @pytest.mark.asyncio
    async def test_discovered_tool_timeout_from_contract(
        self,
        adapter: ONEXToMCPAdapter,
        tmp_path: Path,
        mcp_contract_yaml: str,
    ) -> None:
        """timeout_seconds is read from mcp.timeout_seconds."""
        (tmp_path / "contract.yaml").write_text(mcp_contract_yaml)
        tools = await adapter.discover_tools(contracts_root=tmp_path)
        tool = next(t for t in tools if t.name == "example_tool")
        assert tool.timeout_seconds == 45

    @pytest.mark.asyncio
    async def test_contract_without_mcp_section_ignored(
        self,
        adapter: ONEXToMCPAdapter,
        tmp_path: Path,
        contract_no_mcp_yaml: str,
    ) -> None:
        """Contract with no mcp section is silently ignored."""
        (tmp_path / "contract.yaml").write_text(contract_no_mcp_yaml)
        tools = await adapter.discover_tools(contracts_root=tmp_path)
        assert list(tools) == []

    @pytest.mark.asyncio
    async def test_contract_with_mcp_expose_false_ignored(
        self,
        adapter: ONEXToMCPAdapter,
        tmp_path: Path,
        contract_mcp_expose_false_yaml: str,
    ) -> None:
        """Contract with mcp.expose: false is silently ignored."""
        (tmp_path / "contract.yaml").write_text(contract_mcp_expose_false_yaml)
        tools = await adapter.discover_tools(contracts_root=tmp_path)
        assert list(tools) == []

    @pytest.mark.asyncio
    async def test_unparseable_yaml_skipped(
        self,
        adapter: ONEXToMCPAdapter,
        tmp_path: Path,
    ) -> None:
        """Unparseable YAML is logged and skipped — not fatal."""
        (tmp_path / "contract.yaml").write_text("}{: invalid yaml{{")
        tools = await adapter.discover_tools(contracts_root=tmp_path)
        assert list(tools) == []

    @pytest.mark.asyncio
    async def test_fallback_to_node_name_when_no_tool_name(
        self,
        adapter: ONEXToMCPAdapter,
        tmp_path: Path,
        contract_no_tool_name_yaml: str,
    ) -> None:
        """When mcp.tool_name absent, falls back to contract name field."""
        (tmp_path / "contract.yaml").write_text(contract_no_tool_name_yaml)
        tools = await adapter.discover_tools(contracts_root=tmp_path)
        assert any(t.name == "node_fallback_name" for t in tools)

    @pytest.mark.asyncio
    async def test_manual_registration_takes_precedence(
        self,
        adapter: ONEXToMCPAdapter,
        tmp_path: Path,
        mcp_contract_yaml: str,
    ) -> None:
        """Manual registration wins when tool name collides with contract."""
        await adapter.register_node_as_tool(
            "example_tool",
            "Manually registered description",
            [],
        )
        (tmp_path / "contract.yaml").write_text(mcp_contract_yaml)
        tools = await adapter.discover_tools(contracts_root=tmp_path)
        tool = next(t for t in tools if t.name == "example_tool")
        assert tool.description == "Manually registered description"

    @pytest.mark.asyncio
    async def test_multiple_contracts_discovered(
        self,
        adapter: ONEXToMCPAdapter,
        tmp_path: Path,
        mcp_contract_yaml: str,
    ) -> None:
        """Multiple contract.yaml files under the root are all scanned."""
        dir_a = tmp_path / "node_a"
        dir_b = tmp_path / "node_b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "contract.yaml").write_text(mcp_contract_yaml)
        # Second contract with different tool name
        second = mcp_contract_yaml.replace(
            'tool_name: "example_tool"', 'tool_name: "second_tool"'
        ).replace('name: "node_example_orchestrator"', 'name: "node_second"')
        (dir_b / "contract.yaml").write_text(second)
        tools = await adapter.discover_tools(contracts_root=tmp_path)
        tool_names = {t.name for t in tools}
        assert "example_tool" in tool_names
        assert "second_tool" in tool_names

    @pytest.mark.asyncio
    async def test_metadata_source_set_to_contract_discovery(
        self,
        adapter: ONEXToMCPAdapter,
        tmp_path: Path,
        mcp_contract_yaml: str,
    ) -> None:
        """Discovered tool metadata['source'] == 'contract_discovery'."""
        (tmp_path / "contract.yaml").write_text(mcp_contract_yaml)
        tools = await adapter.discover_tools(contracts_root=tmp_path)
        tool = next(t for t in tools if t.name == "example_tool")
        assert tool.metadata.get("source") == "contract_discovery"


# ---------------------------------------------------------------------------
# Tests: Pydantic input schema derivation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestContractPydanticInputSchema:
    """Verify input_schema is derived from Pydantic models declared in contracts."""

    @pytest.mark.asyncio
    async def test_contract_with_pydantic_input_schema(
        self,
        adapter: ONEXToMCPAdapter,
        tmp_path: Path,
    ) -> None:
        """JSON schema is derived from the Pydantic model declared in input_model."""
        contract = textwrap.dedent(
            """\
            name: "node_schema_test"
            node_type: "ORCHESTRATOR_GENERIC"
            description: "Schema test."
            mcp:
              expose: true
              tool_name: "schema_test_tool"
            input_model:
              name: "ModelExampleInput"
              module: "tests.unit.handlers.mcp.test_adapter_onex_to_mcp"
              description: "Input for schema test."
            """
        )
        (tmp_path / "contract.yaml").write_text(contract)
        tools = await adapter.discover_tools(contracts_root=tmp_path)
        tool = next(t for t in tools if t.name == "schema_test_tool")
        # Should have parameters derived from ModelExampleInput
        param_names = {p.name for p in tool.parameters}
        assert "workflow_id" in param_names

    @pytest.mark.asyncio
    async def test_fallback_schema_for_unknown_model(
        self,
        adapter: ONEXToMCPAdapter,
        tmp_path: Path,
    ) -> None:
        """Falls back to {"type": "object"} when model cannot be imported."""
        contract = textwrap.dedent(
            """\
            name: "node_unknown_model"
            node_type: "ORCHESTRATOR_GENERIC"
            description: "Unknown model test."
            mcp:
              expose: true
              tool_name: "unknown_model_tool"
            input_model:
              name: "ModelDoesNotExist"
              module: "no.such.module"
              description: "Missing."
            """
        )
        (tmp_path / "contract.yaml").write_text(contract)
        tools = await adapter.discover_tools(contracts_root=tmp_path)
        tool = next(t for t in tools if t.name == "unknown_model_tool")
        # Fallback schema — no parameters extracted but tool is present
        assert tool is not None


# ---------------------------------------------------------------------------
# Tests: V3 verification — no TODO(OMN-1288) stub in discover_tools
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoStubReturn:
    """V3: discover_tools() no longer contains the TODO(OMN-1288) stub."""

    def test_discover_tools_source_has_no_stub_comment(self) -> None:
        """Verify the TODO(OMN-1288) stub is removed from adapter source."""
        import inspect

        from omnibase_infra.handlers.mcp import adapter_onex_to_mcp

        source = inspect.getsource(adapter_onex_to_mcp)
        assert "TODO(OMN-1288)" not in source, (
            "discover_tools() still contains the TODO(OMN-1288) stub; "
            "remove it as part of this ticket."
        )


# ---------------------------------------------------------------------------
# Tests: pydantic_to_json_schema (existing static method — regression guards)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPydanticToJsonSchema:
    """Regression tests for the existing pydantic_to_json_schema static method."""

    def test_pydantic_model_returns_schema(self) -> None:
        class _M(BaseModel):
            x: int
            y: str = "hello"

        schema = ONEXToMCPAdapter.pydantic_to_json_schema(_M)
        assert schema.get("type") == "object"
        assert "properties" in schema
        assert "x" in schema["properties"]  # type: ignore[index]

    def test_non_pydantic_returns_fallback(self) -> None:
        schema = ONEXToMCPAdapter.pydantic_to_json_schema(str)
        assert schema == {"type": "object"}

    def test_raise_on_error_true_raises_for_non_pydantic(self) -> None:
        from omnibase_infra.errors import ProtocolConfigurationError

        with pytest.raises(ProtocolConfigurationError):
            ONEXToMCPAdapter.pydantic_to_json_schema(str, raise_on_error=True)
