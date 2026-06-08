# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ServiceMCPToolSync MCP exposure rule (OMN-12827, Plan B2).

These tests are infrastructure-free: they exercise the pure tag-classification
predicate that decides whether a node-registration event should surface as an
MCP tool. Before OMN-12827 only ORCHESTRATOR nodes were exposed; B2 relaxes the
rule so that generated COMPUTE nodes (the SEA self-extension loop output) are
also exposed, without falsely re-labeling them as orchestrators.

The exposure contract (the source of truth this test is written from):

- An ``mcp-enabled`` ORCHESTRATOR node is exposed (unchanged behavior).
- An ``mcp-enabled`` generated COMPUTE node (``node-type:compute`` + ``generated``)
  is exposed (new B2 behavior).
- A non-MCP node is never exposed.
- A NON-generated COMPUTE node is NOT exposed (only generated compute nodes
  surface as tools; arbitrary compute nodes must not leak into the registry).
- A node missing the ``mcp-enabled`` tag is never exposed regardless of type.
"""

from __future__ import annotations

import pytest

from omnibase_infra.services.mcp.service_mcp_tool_sync import ServiceMCPToolSync


@pytest.fixture
def sync() -> ServiceMCPToolSync:
    """Build a ServiceMCPToolSync with direct (mock-free) dependency injection.

    The exposure predicate is pure and does not touch the registry or bus, so a
    minimal pair of stand-ins is sufficient to construct the service.
    """

    class _StubRegistry:
        pass

    class _StubBus:
        environment = "test"

    return ServiceMCPToolSync(registry=_StubRegistry(), bus=_StubBus())  # type: ignore[arg-type]


class TestMcpExposureRule:
    """Cover the relaxed MCP exposure predicate (B2)."""

    def test_mcp_enabled_orchestrator_is_exposed(
        self, sync: ServiceMCPToolSync
    ) -> None:
        tags = [
            ServiceMCPToolSync.TAG_MCP_ENABLED,
            ServiceMCPToolSync.TAG_NODE_TYPE_ORCHESTRATOR,
            f"{ServiceMCPToolSync.TAG_PREFIX_MCP_TOOL}some_orchestrator",
        ]
        assert sync._is_mcp_exposable(tags) is True

    def test_mcp_enabled_generated_compute_is_exposed(
        self, sync: ServiceMCPToolSync
    ) -> None:
        tags = [
            ServiceMCPToolSync.TAG_MCP_ENABLED,
            ServiceMCPToolSync.TAG_NODE_TYPE_COMPUTE,
            ServiceMCPToolSync.TAG_GENERATED,
            f"{ServiceMCPToolSync.TAG_PREFIX_MCP_TOOL}generated_compute_node",
        ]
        assert sync._is_mcp_exposable(tags) is True

    def test_non_generated_compute_is_not_exposed(
        self, sync: ServiceMCPToolSync
    ) -> None:
        tags = [
            ServiceMCPToolSync.TAG_MCP_ENABLED,
            ServiceMCPToolSync.TAG_NODE_TYPE_COMPUTE,
            f"{ServiceMCPToolSync.TAG_PREFIX_MCP_TOOL}hand_authored_compute",
        ]
        assert sync._is_mcp_exposable(tags) is False

    def test_non_mcp_node_is_not_exposed(self, sync: ServiceMCPToolSync) -> None:
        tags = [
            ServiceMCPToolSync.TAG_NODE_TYPE_ORCHESTRATOR,
            f"{ServiceMCPToolSync.TAG_PREFIX_MCP_TOOL}some_orchestrator",
        ]
        assert sync._is_mcp_exposable(tags) is False

    def test_generated_compute_without_mcp_enabled_is_not_exposed(
        self, sync: ServiceMCPToolSync
    ) -> None:
        tags = [
            ServiceMCPToolSync.TAG_NODE_TYPE_COMPUTE,
            ServiceMCPToolSync.TAG_GENERATED,
            f"{ServiceMCPToolSync.TAG_PREFIX_MCP_TOOL}generated_compute_node",
        ]
        assert sync._is_mcp_exposable(tags) is False

    def test_empty_tags_is_not_exposed(self, sync: ServiceMCPToolSync) -> None:
        assert sync._is_mcp_exposable([]) is False
