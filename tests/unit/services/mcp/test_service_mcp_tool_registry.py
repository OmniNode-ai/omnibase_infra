# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ServiceMCPToolRegistry.

Comprehensive test suite covering the event-loop safe in-memory cache
for MCP tool definitions. Tests focus on:
- Version tracking and idempotent operations
- Event ordering semantics (newer event_id wins)
- Concurrent access safety via asyncio.Lock
- Full CRUD operations for tool management

The registry uses string-based event_id comparison for version tracking.
Event IDs should be monotonically increasing (e.g., Kafka offset,
timestamp-based strings, or lexicographically ordered identifiers).
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from omnibase_infra.models.mcp.model_mcp_tool_definition import ModelMCPToolDefinition
from omnibase_infra.services.mcp.service_mcp_tool_registry import ServiceMCPToolRegistry

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def registry() -> ServiceMCPToolRegistry:
    """Create a fresh ServiceMCPToolRegistry instance."""
    return ServiceMCPToolRegistry()


@pytest.fixture
def sample_tool() -> ModelMCPToolDefinition:
    """Create a sample tool definition for testing."""
    return ModelMCPToolDefinition(
        name="test_tool",
        description="A test tool for unit testing",
        version="1.0.0",
        parameters=[],
        input_schema={"type": "object", "properties": {}},
        orchestrator_node_id=str(uuid4()),
        timeout_seconds=30,
    )


@pytest.fixture
def another_tool() -> ModelMCPToolDefinition:
    """Create another sample tool definition for testing."""
    return ModelMCPToolDefinition(
        name="another_tool",
        description="Another test tool",
        version="2.0.0",
        parameters=[],
        orchestrator_node_id=str(uuid4()),
        timeout_seconds=60,
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestServiceMCPToolRegistryInitialization:
    """Test suite for ServiceMCPToolRegistry initialization."""

    def test_registry_initializes_empty(self) -> None:
        """Test registry starts with empty tools and versions."""
        registry = ServiceMCPToolRegistry()

        assert registry.tool_count == 0

    def test_tool_count_property_returns_zero_initially(
        self, registry: ServiceMCPToolRegistry
    ) -> None:
        """Test tool_count property returns 0 for empty registry."""
        assert registry.tool_count == 0


# =============================================================================
# Upsert Tool Tests
# =============================================================================


class TestServiceMCPToolRegistryUpsert:
    """Test suite for upsert_tool operation."""

    @pytest.mark.asyncio
    async def test_upsert_new_tool_succeeds(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test upserting a new tool returns True."""
        result = await registry.upsert_tool(sample_tool, event_id="event-001")

        assert result is True
        assert registry.tool_count == 1

    @pytest.mark.asyncio
    async def test_upsert_stores_tool_correctly(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test upserted tool can be retrieved."""
        await registry.upsert_tool(sample_tool, event_id="event-001")

        retrieved = await registry.get_tool(sample_tool.name)

        assert retrieved is not None
        assert retrieved.name == sample_tool.name
        assert retrieved.description == sample_tool.description
        assert retrieved.version == sample_tool.version

    @pytest.mark.asyncio
    async def test_upsert_with_same_event_id_is_idempotent(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test upserting with same event_id returns False (idempotent)."""
        event_id = "event-001"

        # First upsert succeeds
        result1 = await registry.upsert_tool(sample_tool, event_id=event_id)
        assert result1 is True

        # Second upsert with same event_id is idempotent
        result2 = await registry.upsert_tool(sample_tool, event_id=event_id)
        assert result2 is False

        # Only one tool in registry
        assert registry.tool_count == 1

    @pytest.mark.asyncio
    async def test_upsert_with_older_event_id_is_rejected(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test upserting with older event_id returns False (stale)."""
        # Insert with newer event_id first
        await registry.upsert_tool(sample_tool, event_id="event-002")

        # Try to upsert with older event_id
        sample_tool_updated = ModelMCPToolDefinition(
            name=sample_tool.name,
            description="Updated description",
            version="2.0.0",
        )
        result = await registry.upsert_tool(sample_tool_updated, event_id="event-001")

        assert result is False

        # Original tool should be unchanged
        retrieved = await registry.get_tool(sample_tool.name)
        assert retrieved is not None
        assert retrieved.description == sample_tool.description

    @pytest.mark.asyncio
    async def test_upsert_with_newer_event_id_succeeds(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test upserting with newer event_id returns True and updates tool."""
        # Insert initial tool
        await registry.upsert_tool(sample_tool, event_id="event-001")

        # Upsert with newer event_id
        updated_tool = ModelMCPToolDefinition(
            name=sample_tool.name,
            description="Updated description",
            version="2.0.0",
        )
        result = await registry.upsert_tool(updated_tool, event_id="event-002")

        assert result is True

        # Tool should be updated
        retrieved = await registry.get_tool(sample_tool.name)
        assert retrieved is not None
        assert retrieved.description == "Updated description"
        assert retrieved.version == "2.0.0"

    @pytest.mark.asyncio
    async def test_upsert_multiple_tools(
        self,
        registry: ServiceMCPToolRegistry,
        sample_tool: ModelMCPToolDefinition,
        another_tool: ModelMCPToolDefinition,
    ) -> None:
        """Test upserting multiple different tools."""
        result1 = await registry.upsert_tool(sample_tool, event_id="event-001")
        result2 = await registry.upsert_tool(another_tool, event_id="event-002")

        assert result1 is True
        assert result2 is True
        assert registry.tool_count == 2

    @pytest.mark.asyncio
    async def test_upsert_tracks_version_per_tool(
        self,
        registry: ServiceMCPToolRegistry,
        sample_tool: ModelMCPToolDefinition,
        another_tool: ModelMCPToolDefinition,
    ) -> None:
        """Test version tracking is per-tool (not global)."""
        # Insert both tools with different event_ids
        await registry.upsert_tool(sample_tool, event_id="event-003")
        await registry.upsert_tool(another_tool, event_id="event-001")

        # sample_tool has version event-003
        # another_tool has version event-001

        # Update another_tool with event-002 (older than sample_tool but newer than another_tool)
        updated_another = ModelMCPToolDefinition(
            name=another_tool.name,
            description="Updated another",
            version="3.0.0",
        )
        result = await registry.upsert_tool(updated_another, event_id="event-002")

        # Should succeed because event-002 > event-001 for another_tool
        assert result is True

        version = await registry.get_tool_version(another_tool.name)
        assert version == "event-002"


# =============================================================================
# Remove Tool Tests
# =============================================================================


class TestServiceMCPToolRegistryRemove:
    """Test suite for remove_tool operation."""

    @pytest.mark.asyncio
    async def test_remove_existing_tool_succeeds(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test removing an existing tool returns True."""
        await registry.upsert_tool(sample_tool, event_id="event-001")

        result = await registry.remove_tool(sample_tool.name, event_id="event-002")

        assert result is True
        assert registry.tool_count == 0

    @pytest.mark.asyncio
    async def test_remove_clears_tool_from_registry(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test removed tool cannot be retrieved."""
        await registry.upsert_tool(sample_tool, event_id="event-001")
        await registry.remove_tool(sample_tool.name, event_id="event-002")

        retrieved = await registry.get_tool(sample_tool.name)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_remove_with_stale_event_id_is_rejected(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test removing with stale event_id returns False."""
        await registry.upsert_tool(sample_tool, event_id="event-002")

        # Try to remove with older event_id
        result = await registry.remove_tool(sample_tool.name, event_id="event-001")

        assert result is False

        # Tool should still exist
        assert await registry.has_tool(sample_tool.name) is True

    @pytest.mark.asyncio
    async def test_remove_nonexistent_tool_returns_false(
        self, registry: ServiceMCPToolRegistry
    ) -> None:
        """Test removing non-existent tool returns False."""
        result = await registry.remove_tool("nonexistent", event_id="event-001")

        assert result is False

    @pytest.mark.asyncio
    async def test_remove_updates_version_even_for_nonexistent_tool(
        self, registry: ServiceMCPToolRegistry
    ) -> None:
        """Test remove updates version tracking even when tool doesn't exist.

        This prevents re-adding a tool with an older event after removal.
        """
        # Remove non-existent tool (sets version to event-002)
        await registry.remove_tool("future_tool", event_id="event-002")

        # Try to add the tool with older event_id
        tool = ModelMCPToolDefinition(
            name="future_tool",
            description="A tool added after removal",
            version="1.0.0",
        )
        result = await registry.upsert_tool(tool, event_id="event-001")

        # Should fail because event-001 < event-002
        assert result is False

    @pytest.mark.asyncio
    async def test_remove_then_readd_with_newer_event_succeeds(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test re-adding a tool after removal with newer event_id."""
        # Add tool
        await registry.upsert_tool(sample_tool, event_id="event-001")

        # Remove tool
        await registry.remove_tool(sample_tool.name, event_id="event-002")

        # Re-add with newer event_id
        result = await registry.upsert_tool(sample_tool, event_id="event-003")

        assert result is True
        assert registry.tool_count == 1

    @pytest.mark.asyncio
    async def test_remove_with_same_event_id_as_upsert_is_rejected(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test remove with same event_id as upsert is rejected."""
        event_id = "event-001"

        await registry.upsert_tool(sample_tool, event_id=event_id)

        # Try to remove with same event_id
        result = await registry.remove_tool(sample_tool.name, event_id=event_id)

        assert result is False
        assert await registry.has_tool(sample_tool.name) is True


# =============================================================================
# Get Tool Tests
# =============================================================================


class TestServiceMCPToolRegistryGetTool:
    """Test suite for get_tool operation."""

    @pytest.mark.asyncio
    async def test_get_existing_tool(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test retrieving an existing tool."""
        await registry.upsert_tool(sample_tool, event_id="event-001")

        retrieved = await registry.get_tool(sample_tool.name)

        assert retrieved is not None
        assert retrieved.name == sample_tool.name

    @pytest.mark.asyncio
    async def test_get_nonexistent_tool_returns_none(
        self, registry: ServiceMCPToolRegistry
    ) -> None:
        """Test retrieving non-existent tool returns None."""
        retrieved = await registry.get_tool("nonexistent")

        assert retrieved is None


# =============================================================================
# List Tools Tests
# =============================================================================


class TestServiceMCPToolRegistryListTools:
    """Test suite for list_tools operation."""

    @pytest.mark.asyncio
    async def test_list_tools_empty_registry(
        self, registry: ServiceMCPToolRegistry
    ) -> None:
        """Test list_tools returns empty list for empty registry."""
        tools = await registry.list_tools()

        assert tools == []

    @pytest.mark.asyncio
    async def test_list_tools_returns_all_tools(
        self,
        registry: ServiceMCPToolRegistry,
        sample_tool: ModelMCPToolDefinition,
        another_tool: ModelMCPToolDefinition,
    ) -> None:
        """Test list_tools returns all registered tools."""
        await registry.upsert_tool(sample_tool, event_id="event-001")
        await registry.upsert_tool(another_tool, event_id="event-002")

        tools = await registry.list_tools()

        assert len(tools) == 2
        tool_names = {t.name for t in tools}
        assert sample_tool.name in tool_names
        assert another_tool.name in tool_names

    @pytest.mark.asyncio
    async def test_list_tools_returns_snapshot(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test list_tools returns a snapshot (copy) of tools."""
        await registry.upsert_tool(sample_tool, event_id="event-001")

        tools = await registry.list_tools()

        # Modify the returned list
        tools.clear()

        # Registry should still have the tool
        assert registry.tool_count == 1


# =============================================================================
# Clear Tests
# =============================================================================


class TestServiceMCPToolRegistryClear:
    """Test suite for clear operation."""

    @pytest.mark.asyncio
    async def test_clear_removes_all_tools(
        self,
        registry: ServiceMCPToolRegistry,
        sample_tool: ModelMCPToolDefinition,
        another_tool: ModelMCPToolDefinition,
    ) -> None:
        """Test clear removes all tools from registry."""
        await registry.upsert_tool(sample_tool, event_id="event-001")
        await registry.upsert_tool(another_tool, event_id="event-002")

        await registry.clear()

        assert registry.tool_count == 0

    @pytest.mark.asyncio
    async def test_clear_removes_all_versions(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test clear removes version tracking data."""
        await registry.upsert_tool(sample_tool, event_id="event-001")

        await registry.clear()

        version = await registry.get_tool_version(sample_tool.name)
        assert version is None

    @pytest.mark.asyncio
    async def test_clear_allows_re_adding_with_any_event_id(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test after clear, tools can be added with any event_id."""
        # Add tool with high event_id
        await registry.upsert_tool(sample_tool, event_id="event-999")

        # Clear registry
        await registry.clear()

        # Should be able to add with low event_id now
        result = await registry.upsert_tool(sample_tool, event_id="event-001")
        assert result is True

    @pytest.mark.asyncio
    async def test_clear_on_empty_registry(
        self, registry: ServiceMCPToolRegistry
    ) -> None:
        """Test clear on empty registry does not raise."""
        await registry.clear()

        assert registry.tool_count == 0


# =============================================================================
# Has Tool Tests
# =============================================================================


class TestServiceMCPToolRegistryHasTool:
    """Test suite for has_tool operation."""

    @pytest.mark.asyncio
    async def test_has_tool_returns_true_for_existing_tool(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test has_tool returns True for existing tool."""
        await registry.upsert_tool(sample_tool, event_id="event-001")

        result = await registry.has_tool(sample_tool.name)

        assert result is True

    @pytest.mark.asyncio
    async def test_has_tool_returns_false_for_nonexistent_tool(
        self, registry: ServiceMCPToolRegistry
    ) -> None:
        """Test has_tool returns False for non-existent tool."""
        result = await registry.has_tool("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_has_tool_returns_false_after_removal(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test has_tool returns False after tool is removed."""
        await registry.upsert_tool(sample_tool, event_id="event-001")
        await registry.remove_tool(sample_tool.name, event_id="event-002")

        result = await registry.has_tool(sample_tool.name)

        assert result is False


# =============================================================================
# Get Tool Version Tests
# =============================================================================


class TestServiceMCPToolRegistryGetToolVersion:
    """Test suite for get_tool_version operation."""

    @pytest.mark.asyncio
    async def test_get_version_returns_event_id(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test get_tool_version returns the last event_id."""
        event_id = "event-001"
        await registry.upsert_tool(sample_tool, event_id=event_id)

        version = await registry.get_tool_version(sample_tool.name)

        assert version == event_id

    @pytest.mark.asyncio
    async def test_get_version_returns_latest_event_id(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test get_tool_version returns the most recent event_id."""
        await registry.upsert_tool(sample_tool, event_id="event-001")
        await registry.upsert_tool(sample_tool, event_id="event-002")

        version = await registry.get_tool_version(sample_tool.name)

        assert version == "event-002"

    @pytest.mark.asyncio
    async def test_get_version_returns_none_for_nonexistent_tool(
        self, registry: ServiceMCPToolRegistry
    ) -> None:
        """Test get_tool_version returns None for non-existent tool."""
        version = await registry.get_tool_version("nonexistent")

        assert version is None

    @pytest.mark.asyncio
    async def test_get_version_persists_after_removal(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test version tracking persists even after tool removal."""
        await registry.upsert_tool(sample_tool, event_id="event-001")
        await registry.remove_tool(sample_tool.name, event_id="event-002")

        version = await registry.get_tool_version(sample_tool.name)

        # Version should be event-002 from removal
        assert version == "event-002"


# =============================================================================
# Tool Count Tests
# =============================================================================


class TestServiceMCPToolRegistryToolCount:
    """Test suite for tool_count property."""

    def test_tool_count_initially_zero(self, registry: ServiceMCPToolRegistry) -> None:
        """Test tool_count is 0 initially."""
        assert registry.tool_count == 0

    @pytest.mark.asyncio
    async def test_tool_count_increments_on_upsert(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test tool_count increments when tool is added."""
        await registry.upsert_tool(sample_tool, event_id="event-001")

        assert registry.tool_count == 1

    @pytest.mark.asyncio
    async def test_tool_count_decrements_on_remove(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test tool_count decrements when tool is removed."""
        await registry.upsert_tool(sample_tool, event_id="event-001")
        await registry.remove_tool(sample_tool.name, event_id="event-002")

        assert registry.tool_count == 0

    @pytest.mark.asyncio
    async def test_tool_count_unchanged_on_failed_upsert(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test tool_count unchanged when upsert fails (stale event)."""
        await registry.upsert_tool(sample_tool, event_id="event-002")

        # Stale upsert should fail
        await registry.upsert_tool(sample_tool, event_id="event-001")

        assert registry.tool_count == 1


# =============================================================================
# Describe Tests
# =============================================================================


class TestServiceMCPToolRegistryDescribe:
    """Test suite for describe operation."""

    def test_describe_returns_metadata(self, registry: ServiceMCPToolRegistry) -> None:
        """Test describe returns observability metadata."""
        description = registry.describe()

        assert description["service_name"] == "ServiceMCPToolRegistry"
        assert description["tool_count"] == 0
        assert description["version_count"] == 0

    @pytest.mark.asyncio
    async def test_describe_reflects_tool_count(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test describe reflects current tool count."""
        await registry.upsert_tool(sample_tool, event_id="event-001")

        description = registry.describe()

        assert description["tool_count"] == 1
        assert description["version_count"] == 1

    @pytest.mark.asyncio
    async def test_describe_reflects_version_count_after_removal(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test describe reflects version_count even after tool removal.

        Version tracking persists after removal, so version_count may
        differ from tool_count.
        """
        await registry.upsert_tool(sample_tool, event_id="event-001")
        await registry.remove_tool(sample_tool.name, event_id="event-002")

        description = registry.describe()

        # Tool removed, but version tracking persists
        assert description["tool_count"] == 0
        assert description["version_count"] == 1


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestServiceMCPToolRegistryConcurrentAccess:
    """Test suite for concurrent access safety."""

    @pytest.mark.asyncio
    async def test_concurrent_upserts_are_serialized(
        self, registry: ServiceMCPToolRegistry
    ) -> None:
        """Test concurrent upserts are properly serialized via lock."""
        tools = [
            ModelMCPToolDefinition(
                name=f"tool_{i}",
                description=f"Tool {i}",
                version="1.0.0",
            )
            for i in range(10)
        ]

        # Upsert all tools concurrently
        results = await asyncio.gather(
            *[
                registry.upsert_tool(tool, event_id=f"event-{i:03d}")
                for i, tool in enumerate(tools)
            ]
        )

        # All should succeed
        assert all(results)
        assert registry.tool_count == 10

    @pytest.mark.asyncio
    async def test_concurrent_reads_and_writes(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test concurrent reads and writes don't corrupt state."""
        await registry.upsert_tool(sample_tool, event_id="event-000")

        async def update_tool(i: int) -> bool:
            updated = ModelMCPToolDefinition(
                name=sample_tool.name,
                description=f"Update {i}",
                version=f"{i}.0.0",
            )
            return await registry.upsert_tool(updated, event_id=f"event-{i:03d}")

        async def read_tool() -> ModelMCPToolDefinition | None:
            return await registry.get_tool(sample_tool.name)

        # Mix of reads and writes
        tasks = []
        for i in range(1, 11):
            tasks.append(update_tool(i))
            tasks.append(read_tool())

        await asyncio.gather(*tasks)

        # Final state should be consistent
        assert registry.tool_count == 1
        final_tool = await registry.get_tool(sample_tool.name)
        assert final_tool is not None

    @pytest.mark.asyncio
    async def test_concurrent_remove_and_upsert(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test concurrent remove and upsert operations."""
        await registry.upsert_tool(sample_tool, event_id="event-000")

        async def remove_tool() -> bool:
            return await registry.remove_tool(sample_tool.name, event_id="event-002")

        async def upsert_tool() -> bool:
            return await registry.upsert_tool(sample_tool, event_id="event-001")

        # Run concurrently - outcome depends on execution order
        results = await asyncio.gather(remove_tool(), upsert_tool())

        # Both operations should complete without error
        # Exact outcome depends on ordering, but state should be consistent
        assert isinstance(results[0], bool)
        assert isinstance(results[1], bool)


# =============================================================================
# Event ID Ordering Tests
# =============================================================================


class TestServiceMCPToolRegistryEventIdOrdering:
    """Test suite for event_id string comparison semantics."""

    @pytest.mark.asyncio
    async def test_lexicographic_ordering_of_event_ids(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test event_id uses lexicographic string comparison.

        This test verifies the comparison semantics since the registry
        uses string comparison (<=) for event_id ordering.
        """
        # Insert with "event-10"
        await registry.upsert_tool(sample_tool, event_id="event-10")

        # Try to update with "event-9" - lexicographically "event-10" < "event-9"
        # because '1' < '9' at position 6
        updated = ModelMCPToolDefinition(
            name=sample_tool.name,
            description="Updated with event-9",
            version="2.0.0",
        )
        result = await registry.upsert_tool(updated, event_id="event-9")

        # This should SUCCEED because "event-9" > "event-10" lexicographically
        assert result is True

        retrieved = await registry.get_tool(sample_tool.name)
        assert retrieved is not None
        assert retrieved.description == "Updated with event-9"

    @pytest.mark.asyncio
    async def test_numeric_padded_event_ids(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test padded numeric event_ids work as expected.

        For proper ordering, numeric event_ids should be zero-padded.
        """
        # Insert with "event-001"
        await registry.upsert_tool(sample_tool, event_id="event-001")

        # Update with "event-010" - "event-010" > "event-001" lexicographically
        updated = ModelMCPToolDefinition(
            name=sample_tool.name,
            description="Updated with event-010",
            version="2.0.0",
        )
        result = await registry.upsert_tool(updated, event_id="event-010")

        assert result is True

        # Try with "event-009" - "event-009" < "event-010"
        another_update = ModelMCPToolDefinition(
            name=sample_tool.name,
            description="Updated with event-009",
            version="3.0.0",
        )
        result2 = await registry.upsert_tool(another_update, event_id="event-009")

        assert result2 is False

    @pytest.mark.asyncio
    async def test_uuid_event_ids(
        self, registry: ServiceMCPToolRegistry, sample_tool: ModelMCPToolDefinition
    ) -> None:
        """Test UUID-based event_ids work correctly."""
        uuid1 = str(uuid4())
        uuid2 = str(uuid4())

        await registry.upsert_tool(sample_tool, event_id=uuid1)

        updated = ModelMCPToolDefinition(
            name=sample_tool.name,
            description="Updated with uuid2",
            version="2.0.0",
        )

        # UUID comparison is lexicographic - outcome depends on UUID values
        result = await registry.upsert_tool(updated, event_id=uuid2)

        # Result depends on lexicographic comparison of UUIDs
        if uuid2 > uuid1:
            assert result is True
        else:
            assert result is False
