# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ONEXToMCPAdapter — contract-driven discovery path.

Tests cover:
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
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import BaseModel

from omnibase_infra.handlers.mcp.adapter_onex_to_mcp import (
    MCPToolDefinition,
    MCPToolParameter,
    ONEXToMCPAdapter,
)

# ---------------------------------------------------------------------------
# Fixtures
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
