# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Mock-based tests for ServiceMCPToolDiscovery via event bus registry.  # ai-slop-ok: pre-existing

These tests validate ServiceMCPToolDiscovery behavior using mock
ProjectionReaderRegistration objects — no database or other infrastructure
is required.  They are placed in the integration suite because they test
the boundary between the MCP discovery service and the registration
projection layer.

Test Categories
===============  # ai-slop-ok: pre-existing

- Mock-based tests using MagicMock/AsyncMock (no infra required)

Related Ticket: OMN-2700
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.models.contracts.model_contract_mcp_config import (
    ModelContractMcpConfig,
)
from omnibase_core.models.contracts.model_contract_mcp_tool import ModelContractMcpTool
from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.services.mcp.service_mcp_tool_discovery import (
    ServiceMCPToolDiscovery,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def _make_tool_entry(
    tool_name: str, description: str, timeout: int
) -> ModelContractMcpTool:
    """Build one valid ``mcp.tools`` entry."""
    return ModelContractMcpTool(
        tool_name=tool_name,
        description=description,
        timeout_seconds=timeout,
        input_model="pkg.models.ModelToolInput",
        output_model="pkg.models.ModelToolOutput",
        command_topic="onex.cmd.platform.tool-run.v1",
        completed_topic="onex.evt.platform.tool-run-completed.v1",
        failed_topic="onex.evt.platform.tool-run-failed.v1",
    )


def _make_projection(
    node_type: str = "orchestrator_generic",
    mcp_expose: bool = True,
    tool_name: str = "test_tool",
    description: str = "Test tool description",
    timeout: int = 30,
    state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
) -> MagicMock:
    """Build a mock ModelRegistrationProjection with MCP metadata."""

    mcp_config = (
        ModelContractMcpConfig(
            expose=True,
            tools=[_make_tool_entry(tool_name, description, timeout)],
        )
        if mcp_expose
        else None
    )
    capabilities = ModelNodeCapabilities(mcp=mcp_config)

    node_type_enum = MagicMock()
    node_type_enum.value = node_type

    proj = MagicMock()
    proj.entity_id = uuid4()
    proj.node_type = node_type_enum
    proj.node_version = MagicMock()
    proj.node_version.__str__ = lambda self: "1.0.0"
    proj.capabilities = capabilities
    proj.current_state = state

    # Make isinstance(proj, ModelRegistrationProjection) pass
    from omnibase_infra.models.projection import ModelRegistrationProjection

    proj.__class__ = ModelRegistrationProjection
    return proj


# =============================================================================
# Mock-based tests (no infrastructure required)
# =============================================================================


@pytest.mark.integration
class TestServiceMCPToolDiscoveryInit:
    """Tests for ServiceMCPToolDiscovery initialization."""

    def test_init_sets_defaults(self) -> None:
        """Should accept a ProjectionReaderRegistration and use default limit."""
        reader = MagicMock()
        svc = ServiceMCPToolDiscovery(reader)

        assert svc._reader is reader
        assert svc._query_limit == 100

    def test_init_accepts_custom_limit(self) -> None:
        """Should accept a custom query_limit."""
        reader = MagicMock()
        svc = ServiceMCPToolDiscovery(reader, query_limit=50)

        assert svc._query_limit == 50

    def test_describe_returns_metadata(self) -> None:
        """describe() should return service metadata without Consul fields."""
        reader = MagicMock()
        svc = ServiceMCPToolDiscovery(reader)

        meta = svc.describe()

        assert meta["service_name"] == "ServiceMCPToolDiscovery"
        assert meta["source"] == "event_bus_registry"
        assert meta["capability_tag"] == "mcp-enabled"
        # Must not contain consul-related keys
        assert "consul_host" not in meta
        assert "consul_port" not in meta
        assert "consul_scheme" not in meta


@pytest.mark.integration
@pytest.mark.asyncio
class TestDiscoverAll:
    """Tests for ServiceMCPToolDiscovery.discover_all() using mock reader."""

    async def test_discover_all_returns_mcp_orchestrator(self) -> None:
        """Should return a tool definition for an ACTIVE MCP orchestrator."""
        proj = _make_projection(
            node_type="orchestrator_generic",
            mcp_expose=True,
            tool_name="my_tool",
            description="My tool description",
        )
        reader = MagicMock()
        reader.get_by_capability_tag = AsyncMock(return_value=[proj])

        svc = ServiceMCPToolDiscovery(reader)
        tools = await svc.discover_all()

        assert len(tools) == 1
        assert tools[0].name == "my_tool"
        assert tools[0].description == "My tool description"
        assert tools[0].metadata["source"] == "event_bus_registry"

    async def test_discover_all_skips_non_orchestrator(self) -> None:
        """Should skip nodes that are not orchestrators."""
        proj = _make_projection(node_type="effect_generic", mcp_expose=True)
        reader = MagicMock()
        reader.get_by_capability_tag = AsyncMock(return_value=[proj])

        svc = ServiceMCPToolDiscovery(reader)
        tools = await svc.discover_all()

        assert tools == []

    async def test_discover_all_skips_mcp_expose_false(self) -> None:
        """Should skip nodes where mcp.expose=False."""
        proj = _make_projection(node_type="orchestrator_generic", mcp_expose=False)
        proj.capabilities = ModelNodeCapabilities(
            mcp=ModelContractMcpConfig(expose=False)
        )
        reader = MagicMock()
        reader.get_by_capability_tag = AsyncMock(return_value=[proj])

        svc = ServiceMCPToolDiscovery(reader)
        tools = await svc.discover_all()

        assert tools == []

    async def test_discover_all_skips_no_mcp_config(self) -> None:
        """Should skip nodes without mcp config in capabilities."""
        proj = _make_projection(node_type="orchestrator_generic")
        proj.capabilities = ModelNodeCapabilities(mcp=None)
        reader = MagicMock()
        reader.get_by_capability_tag = AsyncMock(return_value=[proj])

        svc = ServiceMCPToolDiscovery(reader)
        tools = await svc.discover_all()

        assert tools == []

    async def test_discover_all_lists_one_tool_per_entry(self) -> None:
        """Should list every ``mcp.tools`` entry of one node."""
        proj = _make_projection(node_type="orchestrator_generic", tool_name="op_a")
        proj.capabilities = ModelNodeCapabilities(
            mcp=ModelContractMcpConfig(
                expose=True,
                tools=[
                    _make_tool_entry("op_a", "First op.", 30),
                    _make_tool_entry("op_b", "Second op.", 15),
                ],
            )
        )
        reader = MagicMock()
        reader.get_by_capability_tag = AsyncMock(return_value=[proj])

        svc = ServiceMCPToolDiscovery(reader)
        tools = await svc.discover_all()

        assert {t.name: t.timeout_seconds for t in tools} == {"op_a": 30, "op_b": 15}

    async def test_discover_all_warns_on_empty_registry(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Should log WARNING when registry returns 0 eligible nodes."""
        reader = MagicMock()
        reader.get_by_capability_tag = AsyncMock(return_value=[])

        svc = ServiceMCPToolDiscovery(reader)

        with caplog.at_level(logging.WARNING):
            tools = await svc.discover_all()

        assert tools == []
        assert any("0 eligible nodes" in r.message for r in caplog.records)

    async def test_discover_all_passes_state_filter(self) -> None:
        """Should filter by ACTIVE state when querying the reader."""
        reader = MagicMock()
        reader.get_by_capability_tag = AsyncMock(return_value=[])

        svc = ServiceMCPToolDiscovery(reader)
        await svc.discover_all()

        reader.get_by_capability_tag.assert_awaited_once()
        call_kwargs = reader.get_by_capability_tag.call_args.kwargs
        assert call_kwargs.get("state") == EnumRegistrationState.ACTIVE
        assert call_kwargs.get("tag") == "mcp-enabled"

    async def test_discover_all_passes_query_limit(self) -> None:
        """Should pass configured query_limit to the reader."""
        reader = MagicMock()
        reader.get_by_capability_tag = AsyncMock(return_value=[])

        svc = ServiceMCPToolDiscovery(reader, query_limit=42)
        await svc.discover_all()

        call_kwargs = reader.get_by_capability_tag.call_args.kwargs
        assert call_kwargs.get("limit") == 42

    async def test_discover_all_multiple_tools(self) -> None:
        """Should return multiple tools when multiple eligible nodes exist."""
        projs = [
            _make_projection(
                node_type="orchestrator_generic",
                tool_name=f"tool_{i}",
                mcp_expose=True,
            )
            for i in range(3)
        ]
        reader = MagicMock()
        reader.get_by_capability_tag = AsyncMock(return_value=projs)

        svc = ServiceMCPToolDiscovery(reader)
        tools = await svc.discover_all()

        assert len(tools) == 3
        tool_names = {t.name for t in tools}
        assert tool_names == {"tool_0", "tool_1", "tool_2"}

    async def test_discover_all_uses_timeout_from_mcp_config(self) -> None:
        """Should use mcp.timeout_seconds from contract config."""
        proj = _make_projection(
            node_type="orchestrator_generic",
            mcp_expose=True,
            tool_name="timed_tool",
            timeout=120,
        )
        reader = MagicMock()
        reader.get_by_capability_tag = AsyncMock(return_value=[proj])

        svc = ServiceMCPToolDiscovery(reader)
        tools = await svc.discover_all()

        assert len(tools) == 1
        assert tools[0].timeout_seconds == 120

    async def test_discover_all_metadata_includes_entity_id(self) -> None:
        """Tool metadata should include entity_id for traceability."""
        proj = _make_projection(
            node_type="orchestrator_generic",
            mcp_expose=True,
            tool_name="traced_tool",
        )
        reader = MagicMock()
        reader.get_by_capability_tag = AsyncMock(return_value=[proj])

        svc = ServiceMCPToolDiscovery(reader)
        tools = await svc.discover_all()

        assert len(tools) == 1
        assert tools[0].metadata["entity_id"] == str(proj.entity_id)

    async def test_discover_all_no_consul_fields_in_metadata(self) -> None:
        """Tool metadata must not contain consul_host/port/scheme fields."""
        proj = _make_projection(
            node_type="orchestrator_generic",
            mcp_expose=True,
            tool_name="clean_tool",
        )
        reader = MagicMock()
        reader.get_by_capability_tag = AsyncMock(return_value=[proj])

        svc = ServiceMCPToolDiscovery(reader)
        tools = await svc.discover_all()

        assert len(tools) == 1
        meta = tools[0].metadata
        assert "consul_host" not in meta
        assert "consul_port" not in meta
        assert "service_name" not in meta or meta.get("source") != "consul_discovery"
        assert meta["source"] == "event_bus_registry"

    async def test_discover_all_propagates_registry_error(self) -> None:
        """Should propagate InfraConnectionError from the reader."""
        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext

        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="get_by_capability_tag",
            target_name="test",
            correlation_id=uuid4(),
        )
        reader = MagicMock()
        reader.get_by_capability_tag = AsyncMock(
            side_effect=InfraConnectionError("DB down", context=ctx)
        )

        svc = ServiceMCPToolDiscovery(reader)

        with pytest.raises(InfraConnectionError):
            await svc.discover_all()


@pytest.mark.integration
class TestProjectionToTools:
    """Tests for ServiceMCPToolDiscovery._projection_to_tools() edge cases."""

    def test_returns_empty_for_non_projection_object(self) -> None:
        """Should return no tools for non-ModelRegistrationProjection objects."""
        reader = MagicMock()
        svc = ServiceMCPToolDiscovery(reader)

        result = svc._projection_to_tools("not a projection", uuid4())

        assert result == []


@pytest.mark.integration
class TestMCPWrapperOrchestratorSatisfiesGate:
    """OMN-12841: the generated COMPUTE wrapper orchestrator satisfies the gate.

    Option B (contract-native) wraps a generated COMPUTE node behind a thin
    declarative ORCHESTRATOR surface. The orchestrator-only discovery gate
    (``service_mcp_tool_discovery.py:197-212``) is NOT relaxed; instead the
    wrapper-shaped projection is an ORCHESTRATOR (``node_type`` starts with
    ``orchestrator``) carrying ``mcp.expose=True`` + a ``tool_name``, so it
    passes the unchanged gate. A bare COMPUTE projection still returns ``None``,
    proving the rule is preserved (Option A -- relaxing the gate -- was rejected).
    """

    def test_wrapper_orchestrator_projection_converts_to_tool(self) -> None:
        """A wrapper-shaped orchestrator projection yields a tool definition."""
        proj = _make_projection(
            node_type="orchestrator_generic",
            mcp_expose=True,
            tool_name="node_sentiment_classifier",
            description="Classify customer review sentiment.",
        )
        reader = MagicMock()
        svc = ServiceMCPToolDiscovery(reader)

        tools = svc._projection_to_tools(proj, uuid4())

        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "node_sentiment_classifier"
        assert tool.description == "Classify customer review sentiment."

    def test_bare_compute_projection_returns_none(self) -> None:
        """A bare COMPUTE projection is NOT exposed (gate unchanged)."""
        proj = _make_projection(
            node_type="compute_generic",
            mcp_expose=True,
            tool_name="node_sentiment_classifier",
            description="Classify customer review sentiment.",
        )
        reader = MagicMock()
        svc = ServiceMCPToolDiscovery(reader)

        tools = svc._projection_to_tools(proj, uuid4())

        assert tools == []
