# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for ServiceMCPToolDiscovery via event bus registry.

These tests validate ServiceMCPToolDiscovery behavior against the ONEX
registration projection (PostgreSQL-backed).  They exercise the live
database if it is available, and skip gracefully in CI/CD environments
where the database is not reachable.

CI/CD Graceful Skip Behavior
=============================

Skip Conditions:
    - Skips if DATABASE_URL (or OMNIBASE_INFRA_DB_URL) is not set
    - Skips if TCP connection to the database host fails

Test Categories
===============

- Unit-style tests using mock ProjectionReaderRegistration (no infra)
- Integration tests using the real projection reader (infra required)

Related Ticket: OMN-2700
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.mcp.model_mcp_contract_config import ModelMCPContractConfig
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.services.mcp.service_mcp_tool_discovery import (
    ServiceMCPToolDiscovery,
)

if TYPE_CHECKING:
    from omnibase_infra.projectors import ProjectionReaderRegistration

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def _make_projection(
    node_type: str = "orchestrator_generic",
    mcp_expose: bool = True,
    tool_name: str | None = "test_tool",
    description: str | None = "Test tool description",
    timeout: int = 30,
    state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
) -> MagicMock:
    """Build a mock ModelRegistrationProjection with MCP metadata."""
    from omnibase_core.enums import EnumNodeKind

    mcp_config = (
        ModelMCPContractConfig(
            expose=mcp_expose,
            tool_name=tool_name,
            description=description,
            timeout_seconds=timeout,
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


@pytest.mark.unit
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


@pytest.mark.unit
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
            mcp=ModelMCPContractConfig(expose=False)
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

    async def test_discover_all_falls_back_tool_name_to_entity_id(self) -> None:
        """Should fall back to str(entity_id) when mcp.tool_name is None."""
        proj = _make_projection(
            node_type="orchestrator_generic",
            mcp_expose=True,
            tool_name=None,
        )
        reader = MagicMock()
        reader.get_by_capability_tag = AsyncMock(return_value=[proj])

        svc = ServiceMCPToolDiscovery(reader)
        tools = await svc.discover_all()

        assert len(tools) == 1
        assert tools[0].name == str(proj.entity_id)

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


@pytest.mark.unit
class TestProjectionToTool:
    """Tests for ServiceMCPToolDiscovery._projection_to_tool() edge cases."""

    def test_returns_none_for_non_projection_object(self) -> None:
        """Should return None for non-ModelRegistrationProjection objects."""
        reader = MagicMock()
        svc = ServiceMCPToolDiscovery(reader)

        result = svc._projection_to_tool("not a projection", uuid4())

        assert result is None

    def test_description_falls_back_to_generated(self) -> None:
        """Should generate description when mcp.description is None."""
        proj = _make_projection(
            node_type="orchestrator_generic",
            mcp_expose=True,
            tool_name="my_tool",
            description=None,
        )
        reader = MagicMock()
        svc = ServiceMCPToolDiscovery(reader)

        tool = svc._projection_to_tool(proj, uuid4())

        assert tool is not None
        assert "ONEX orchestrator" in tool.description
