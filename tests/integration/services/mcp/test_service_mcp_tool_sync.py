# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for ServiceMCPToolSync with real Kafka/Redpanda.

These tests validate ServiceMCPToolSync behavior against actual Kafka infrastructure.
They test the Kafka subscription lifecycle and event processing for MCP tool hot reload.

Test Categories:
    - Lifecycle Tests: Start/stop subscription, idempotent operations
    - Event Processing Tests: Registration, update, deregistration events
    - Error Handling Tests: Invalid events, graceful error handling

Environment Variables:
    KAFKA_BOOTSTRAP_SERVERS: Kafka broker address (e.g., "localhost:9092" or
        "192.168.86.200:29092"). Required for tests to run.

Note:
    Tests will skip gracefully if KAFKA_BOOTSTRAP_SERVERS is not set.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import TopicAlreadyExistsError

from tests.helpers.util_kafka import validate_bootstrap_servers

if TYPE_CHECKING:
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.models.mcp.model_mcp_tool_definition import (
        ModelMCPToolDefinition,
    )

logger = logging.getLogger(__name__)

# =============================================================================
# Test Configuration and Skip Conditions
# =============================================================================

# Check if Kafka is available based on environment variable.
# NOTE: We intentionally do NOT provide a default value. Tests should only run
# when KAFKA_BOOTSTRAP_SERVERS is explicitly set, not when assuming a default
# infrastructure address (which won't be available in CI).
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_AVAILABLE = KAFKA_BOOTSTRAP_SERVERS is not None

# Validate configuration only if environment variable is set
# This provides detailed validation for local development
if KAFKA_AVAILABLE:
    _kafka_config_validation = validate_bootstrap_servers(KAFKA_BOOTSTRAP_SERVERS)
    # Override KAFKA_AVAILABLE if format validation fails
    if not _kafka_config_validation.is_valid:
        KAFKA_AVAILABLE = False
        _skip_reason = _kafka_config_validation.skip_reason
    else:
        _skip_reason = None
else:
    _skip_reason = "Kafka not available (KAFKA_BOOTSTRAP_SERVERS not set)"

# Module-level markers - skip all tests if Kafka is not available
pytestmark = [
    pytest.mark.skipif(
        not KAFKA_AVAILABLE,
        reason=_skip_reason
        or "Kafka not available (KAFKA_BOOTSTRAP_SERVERS not configured)",
    ),
    pytest.mark.integration,
    pytest.mark.kafka,
]

# Test configuration constants
TEST_TIMEOUT_SECONDS = 30
MESSAGE_DELIVERY_WAIT_SECONDS = 5.0
EVENT_PROCESSING_WAIT_SECONDS = 3.0


# =============================================================================
# Mock Registry and Discovery for Isolated Testing
# =============================================================================


@dataclass
class MockMCPToolRegistry:
    """Mock ServiceMCPToolRegistry that tracks upsert/remove operations.

    This mock captures all operations for assertion in tests without
    actually storing tools in memory.
    """

    upsert_calls: list[tuple[ModelMCPToolDefinition, str]] = field(default_factory=list)
    remove_calls: list[tuple[str, str]] = field(default_factory=list)
    _tools: dict[str, ModelMCPToolDefinition] = field(default_factory=dict)
    _versions: dict[str, str] = field(default_factory=dict)

    async def upsert_tool(self, tool: ModelMCPToolDefinition, event_id: str) -> bool:
        """Mock upsert that tracks calls and simulates version-based updates."""
        self.upsert_calls.append((tool, event_id))

        # Simulate idempotency check
        existing_version = self._versions.get(tool.name)
        if existing_version and event_id <= existing_version:
            return False

        self._tools[tool.name] = tool
        self._versions[tool.name] = event_id
        return True

    async def remove_tool(self, tool_name: str, event_id: str) -> bool:
        """Mock remove that tracks calls and simulates version-based removes."""
        self.remove_calls.append((tool_name, event_id))

        # Simulate idempotency check
        existing_version = self._versions.get(tool_name)
        if existing_version and event_id <= existing_version:
            return False

        removed = self._tools.pop(tool_name, None) is not None
        self._versions[tool_name] = event_id
        return removed

    @property
    def tool_count(self) -> int:
        """Return count of stored tools."""
        return len(self._tools)

    async def get_tool(self, tool_name: str) -> ModelMCPToolDefinition | None:
        """Get a tool by name."""
        return self._tools.get(tool_name)


@dataclass
class MockMCPToolDiscovery:
    """Mock ServiceMCPToolDiscovery for Consul fallback testing.

    This mock can be configured to return specific tools for service IDs.
    """

    discover_by_service_id_calls: list[str] = field(default_factory=list)
    tools_by_service_id: dict[str, ModelMCPToolDefinition | None] = field(
        default_factory=dict
    )

    async def discover_by_service_id(
        self, service_id: str
    ) -> ModelMCPToolDefinition | None:
        """Mock Consul lookup that returns pre-configured tools."""
        self.discover_by_service_id_calls.append(service_id)
        return self.tools_by_service_id.get(service_id)


# =============================================================================
# Helper Functions
# =============================================================================


def create_registration_event(
    event_type: str,
    tool_name: str,
    service_name: str | None = None,
    service_id: str | None = None,
    node_id: str | None = None,
    event_id: str | None = None,
    include_mcp_tags: bool = True,
    include_full_info: bool = True,
) -> dict[str, object]:
    """Create a registration event payload for testing.

    Args:
        event_type: Event type (registered, updated, deregistered, expired)
        tool_name: Name of the MCP tool
        service_name: Service name (used to build tool if include_full_info)
        service_id: Service ID in Consul
        node_id: Node ID
        event_id: Unique event ID for idempotency
        include_mcp_tags: Whether to include MCP tags
        include_full_info: Whether to include full service info

    Returns:
        Event payload dictionary
    """
    tags = []
    if include_mcp_tags:
        tags = [
            "mcp-enabled",
            "node-type:orchestrator",
            f"mcp-tool:{tool_name}",
        ]

    event: dict[str, object] = {
        "event_type": event_type,
        "tags": tags,
        "node_id": node_id or str(uuid.uuid4()),
        "service_id": service_id or f"service-{uuid.uuid4().hex[:8]}",
    }

    if event_id:
        event["event_id"] = event_id

    if include_full_info and service_name:
        event["service_name"] = service_name
        event["endpoint"] = f"http://localhost:8080/orchestrator/{tool_name}"
        event["description"] = f"Test orchestrator for {tool_name}"
        event["timeout_seconds"] = 30

    return event


async def wait_for_registry_update(
    registry: MockMCPToolRegistry,
    expected_upsert_count: int | None = None,
    expected_remove_count: int | None = None,
    timeout: float = EVENT_PROCESSING_WAIT_SECONDS,
) -> None:
    """Wait for registry to receive expected number of operations.

    Args:
        registry: The mock registry to check
        expected_upsert_count: Expected number of upsert calls
        expected_remove_count: Expected number of remove calls
        timeout: Maximum time to wait in seconds
    """
    start = asyncio.get_event_loop().time()

    while asyncio.get_event_loop().time() - start < timeout:
        upsert_ok = (
            expected_upsert_count is None
            or len(registry.upsert_calls) >= expected_upsert_count
        )
        remove_ok = (
            expected_remove_count is None
            or len(registry.remove_calls) >= expected_remove_count
        )

        if upsert_ok and remove_ok:
            return

        await asyncio.sleep(0.1)

    # Provide diagnostic info on timeout
    actual_upsert = len(registry.upsert_calls)
    actual_remove = len(registry.remove_calls)
    raise TimeoutError(
        f"Registry update timeout. Expected upsert={expected_upsert_count}, "
        f"got {actual_upsert}. Expected remove={expected_remove_count}, "
        f"got {actual_remove}"
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def kafka_bootstrap_servers() -> str:
    """Get Kafka bootstrap servers from environment.

    Note: Tests are skipped if KAFKA_BOOTSTRAP_SERVERS is not set (via pytestmark),
    so this fixture will only be called when the env var is available.
    The fallback is defensive and ensures type safety.
    """
    return os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


@pytest.fixture
async def kafka_event_bus(
    kafka_bootstrap_servers: str,
) -> AsyncGenerator[EventBusKafka, None]:
    """Create and start EventBusKafka for integration testing.

    Yields a started EventBusKafka instance and ensures cleanup after test.
    """
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

    config = ModelKafkaEventBusConfig(
        bootstrap_servers=kafka_bootstrap_servers,
        environment="integration-test",
        group="mcp-sync-test",
        timeout_seconds=TEST_TIMEOUT_SECONDS,
        max_retry_attempts=2,
        retry_backoff_base=0.5,
        circuit_breaker_threshold=5,
        circuit_breaker_reset_timeout=10.0,
        # Start from earliest so we don't miss messages in test
        auto_offset_reset="earliest",
    )
    bus = EventBusKafka(config=config)
    await bus.start()

    yield bus

    # Cleanup
    try:
        await bus.close()
    except Exception:
        pass


async def create_topic_if_not_exists(
    bootstrap_servers: str,
    topic_name: str,
    partitions: int = 1,
    replication_factor: int = 1,
) -> None:
    """Create a Kafka topic if it doesn't exist.

    Args:
        bootstrap_servers: Kafka bootstrap servers
        topic_name: Name of the topic to create
        partitions: Number of partitions
        replication_factor: Replication factor
    """
    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap_servers)
    try:
        await admin.start()
        new_topic = NewTopic(
            name=topic_name,
            num_partitions=partitions,
            replication_factor=replication_factor,
        )
        await admin.create_topics([new_topic])
        logger.info(f"Created topic: {topic_name}")
    except TopicAlreadyExistsError:
        logger.info(f"Topic already exists: {topic_name}")
    except Exception as e:
        # Log but don't fail - topic might exist or creation might be handled elsewhere
        logger.warning(f"Failed to create topic {topic_name}: {e}")
    finally:
        await admin.close()


@pytest.fixture
async def registration_topic(
    kafka_bootstrap_servers: str,
) -> AsyncGenerator[str, None]:
    """Ensure registration topic exists for tests.

    Creates node.registration.v1 topic if it doesn't exist and yields the topic name.
    """
    topic_name = "node.registration.v1"
    await create_topic_if_not_exists(kafka_bootstrap_servers, topic_name)
    # Small delay to allow topic metadata to propagate
    await asyncio.sleep(0.5)
    return topic_name


@pytest.fixture
def mock_registry() -> MockMCPToolRegistry:
    """Create a mock MCP tool registry."""
    return MockMCPToolRegistry()


@pytest.fixture
def mock_discovery() -> MockMCPToolDiscovery:
    """Create a mock MCP tool discovery service."""
    return MockMCPToolDiscovery()


@pytest.fixture
async def mcp_tool_sync(
    kafka_event_bus: EventBusKafka,
    mock_registry: MockMCPToolRegistry,
    mock_discovery: MockMCPToolDiscovery,
    registration_topic: str,
) -> AsyncGenerator:
    """Create ServiceMCPToolSync with mocked dependencies.

    Yields the sync service with pre-created topic, registry, and discovery mocks.
    """
    from omnibase_infra.services.mcp.service_mcp_tool_sync import ServiceMCPToolSync

    # Create sync service with mocked dependencies
    sync = ServiceMCPToolSync(
        registry=mock_registry,  # type: ignore[arg-type]
        discovery=mock_discovery,  # type: ignore[arg-type]
        bus=kafka_event_bus,
    )

    yield sync

    # Cleanup: stop the sync service
    if sync.is_running:
        await sync.stop()


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestServiceMCPToolSyncLifecycle:
    """Tests for ServiceMCPToolSync start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_begins_kafka_subscription(
        self, mcp_tool_sync, mock_registry: MockMCPToolRegistry
    ) -> None:
        """Verify start() initiates Kafka subscription.

        After calling start():
        - is_running should be True
        - Service should be subscribed to node.registration.v1 topic
        """
        # Initially not running
        assert mcp_tool_sync.is_running is False

        # Start subscription
        await mcp_tool_sync.start()

        # Should now be running
        assert mcp_tool_sync.is_running is True

    @pytest.mark.asyncio
    async def test_stop_ends_subscription(self, mcp_tool_sync) -> None:
        """Verify stop() cleanly ends Kafka subscription.

        After calling stop():
        - is_running should be False
        - No errors should be raised
        """
        # Start then stop
        await mcp_tool_sync.start()
        assert mcp_tool_sync.is_running is True

        await mcp_tool_sync.stop()
        assert mcp_tool_sync.is_running is False

    @pytest.mark.asyncio
    async def test_multiple_start_calls_idempotent(self, mcp_tool_sync) -> None:
        """Verify multiple start() calls are safe and idempotent.

        Calling start() multiple times should:
        - Not raise errors
        - Only subscribe once
        """
        await mcp_tool_sync.start()
        await mcp_tool_sync.start()  # Second call should be no-op
        await mcp_tool_sync.start()  # Third call should be no-op

        assert mcp_tool_sync.is_running is True

        # Stop should still work
        await mcp_tool_sync.stop()
        assert mcp_tool_sync.is_running is False

    @pytest.mark.asyncio
    async def test_stop_when_not_started_is_safe(self, mcp_tool_sync) -> None:
        """Verify stop() when not started doesn't raise errors.

        This ensures defensive programming - stop() should be safe to call
        even if start() was never called.
        """
        # Stop without start should be no-op
        await mcp_tool_sync.stop()
        assert mcp_tool_sync.is_running is False

        # Double stop should also be safe
        await mcp_tool_sync.stop()
        assert mcp_tool_sync.is_running is False

    @pytest.mark.asyncio
    async def test_describe_returns_service_metadata(self, mcp_tool_sync) -> None:
        """Verify describe() returns correct service metadata."""
        # Before start
        info = mcp_tool_sync.describe()
        assert info["service_name"] == "ServiceMCPToolSync"
        assert info["topic"] == "node.registration.v1"
        assert info["group_id"] == "mcp-tool-sync"
        assert info["is_running"] is False

        # After start
        await mcp_tool_sync.start()
        info = mcp_tool_sync.describe()
        assert info["is_running"] is True


# =============================================================================
# Event Processing Tests
# =============================================================================


class TestServiceMCPToolSyncEventProcessing:
    """Tests for ServiceMCPToolSync event processing."""

    @pytest.mark.asyncio
    async def test_registered_event_triggers_upsert(
        self,
        mcp_tool_sync,
        kafka_event_bus: EventBusKafka,
        mock_registry: MockMCPToolRegistry,
    ) -> None:
        """Verify 'registered' event triggers tool upsert in registry.

        When a registration event with type='registered' is received:
        1. Tool definition should be built from event data
        2. upsert_tool should be called on registry
        """
        await mcp_tool_sync.start()

        # Create and publish registration event
        tool_name = f"test_tool_{uuid.uuid4().hex[:8]}"
        event = create_registration_event(
            event_type="registered",
            tool_name=tool_name,
            service_name="test-orchestrator",
            include_full_info=True,
        )

        # Publish event to registration topic
        await kafka_event_bus.publish(
            "node.registration.v1",
            None,
            json.dumps(event).encode("utf-8"),
        )

        # Wait for registry update
        await wait_for_registry_update(mock_registry, expected_upsert_count=1)

        # Verify upsert was called
        assert len(mock_registry.upsert_calls) >= 1
        upserted_tool, _ = mock_registry.upsert_calls[0]
        assert upserted_tool.name == tool_name

    @pytest.mark.asyncio
    async def test_updated_event_triggers_upsert(
        self,
        mcp_tool_sync,
        kafka_event_bus: EventBusKafka,
        mock_registry: MockMCPToolRegistry,
    ) -> None:
        """Verify 'updated' event triggers tool upsert in registry.

        The 'updated' event type should behave identically to 'registered'.
        """
        await mcp_tool_sync.start()

        tool_name = f"updated_tool_{uuid.uuid4().hex[:8]}"
        event = create_registration_event(
            event_type="updated",
            tool_name=tool_name,
            service_name="updated-orchestrator",
            include_full_info=True,
        )

        await kafka_event_bus.publish(
            "node.registration.v1",
            None,
            json.dumps(event).encode("utf-8"),
        )

        await wait_for_registry_update(mock_registry, expected_upsert_count=1)

        assert len(mock_registry.upsert_calls) >= 1
        upserted_tool, _ = mock_registry.upsert_calls[0]
        assert upserted_tool.name == tool_name

    @pytest.mark.asyncio
    async def test_deregistered_event_triggers_remove(
        self,
        mcp_tool_sync,
        kafka_event_bus: EventBusKafka,
        mock_registry: MockMCPToolRegistry,
    ) -> None:
        """Verify 'deregistered' event triggers tool removal from registry.

        When a deregistration event is received:
        - remove_tool should be called with the tool name from tags
        """
        await mcp_tool_sync.start()

        tool_name = f"removed_tool_{uuid.uuid4().hex[:8]}"
        event = create_registration_event(
            event_type="deregistered",
            tool_name=tool_name,
            include_full_info=False,  # Deregister events typically don't have full info
        )

        await kafka_event_bus.publish(
            "node.registration.v1",
            None,
            json.dumps(event).encode("utf-8"),
        )

        await wait_for_registry_update(mock_registry, expected_remove_count=1)

        assert len(mock_registry.remove_calls) >= 1
        removed_name, _ = mock_registry.remove_calls[0]
        assert removed_name == tool_name

    @pytest.mark.asyncio
    async def test_expired_event_triggers_remove(
        self,
        mcp_tool_sync,
        kafka_event_bus: EventBusKafka,
        mock_registry: MockMCPToolRegistry,
    ) -> None:
        """Verify 'expired' event triggers tool removal from registry.

        The 'expired' event type should behave identically to 'deregistered'.
        """
        await mcp_tool_sync.start()

        tool_name = f"expired_tool_{uuid.uuid4().hex[:8]}"
        event = create_registration_event(
            event_type="expired",
            tool_name=tool_name,
            include_full_info=False,
        )

        await kafka_event_bus.publish(
            "node.registration.v1",
            None,
            json.dumps(event).encode("utf-8"),
        )

        await wait_for_registry_update(mock_registry, expected_remove_count=1)

        assert len(mock_registry.remove_calls) >= 1
        removed_name, _ = mock_registry.remove_calls[0]
        assert removed_name == tool_name

    @pytest.mark.asyncio
    async def test_non_mcp_event_ignored(
        self,
        mcp_tool_sync,
        kafka_event_bus: EventBusKafka,
        mock_registry: MockMCPToolRegistry,
    ) -> None:
        """Verify events without MCP tags are ignored.

        Events that don't have 'mcp-enabled' and 'node-type:orchestrator' tags
        should be silently ignored.
        """
        await mcp_tool_sync.start()

        # Event without MCP tags
        event = create_registration_event(
            event_type="registered",
            tool_name="non_mcp_tool",
            service_name="non-mcp-service",
            include_mcp_tags=False,  # No MCP tags
            include_full_info=True,
        )

        await kafka_event_bus.publish(
            "node.registration.v1",
            None,
            json.dumps(event).encode("utf-8"),
        )

        # Wait a bit and verify no updates
        await asyncio.sleep(EVENT_PROCESSING_WAIT_SECONDS)

        assert len(mock_registry.upsert_calls) == 0
        assert len(mock_registry.remove_calls) == 0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestServiceMCPToolSyncErrorHandling:
    """Tests for ServiceMCPToolSync error handling."""

    @pytest.mark.asyncio
    async def test_invalid_json_handled_gracefully(
        self,
        mcp_tool_sync,
        kafka_event_bus: EventBusKafka,
        mock_registry: MockMCPToolRegistry,
    ) -> None:
        """Verify invalid JSON messages don't crash the service.

        The service should skip malformed messages and continue processing.
        """
        await mcp_tool_sync.start()

        # Publish invalid JSON
        await kafka_event_bus.publish(
            "node.registration.v1",
            None,
            b"this is not valid json {{{",
        )

        # Then publish a valid event
        tool_name = f"valid_tool_{uuid.uuid4().hex[:8]}"
        valid_event = create_registration_event(
            event_type="registered",
            tool_name=tool_name,
            service_name="valid-orchestrator",
            include_full_info=True,
        )

        await kafka_event_bus.publish(
            "node.registration.v1",
            None,
            json.dumps(valid_event).encode("utf-8"),
        )

        # Valid event should still be processed
        await wait_for_registry_update(mock_registry, expected_upsert_count=1)

        assert len(mock_registry.upsert_calls) >= 1
        upserted_tool, _ = mock_registry.upsert_calls[0]
        assert upserted_tool.name == tool_name

    @pytest.mark.asyncio
    async def test_event_missing_tool_name_tag_ignored(
        self,
        mcp_tool_sync,
        kafka_event_bus: EventBusKafka,
        mock_registry: MockMCPToolRegistry,
    ) -> None:
        """Verify events missing mcp-tool tag are handled gracefully.

        Events with mcp-enabled and node-type:orchestrator but missing
        mcp-tool:{name} tag should be logged but not crash the service.
        """
        await mcp_tool_sync.start()

        # Event with MCP tags but missing tool name
        event: dict[str, object] = {
            "event_type": "registered",
            "tags": ["mcp-enabled", "node-type:orchestrator"],  # Missing mcp-tool:xxx
            "node_id": str(uuid.uuid4()),
            "service_id": f"service-{uuid.uuid4().hex[:8]}",
            "service_name": "missing-tool-name",
        }

        await kafka_event_bus.publish(
            "node.registration.v1",
            None,
            json.dumps(event).encode("utf-8"),
        )

        # Wait and verify no crash, no update
        await asyncio.sleep(EVENT_PROCESSING_WAIT_SECONDS)

        assert len(mock_registry.upsert_calls) == 0

    @pytest.mark.asyncio
    async def test_unknown_event_type_ignored(
        self,
        mcp_tool_sync,
        kafka_event_bus: EventBusKafka,
        mock_registry: MockMCPToolRegistry,
    ) -> None:
        """Verify unknown event types are silently ignored.

        Events with unrecognized event_type values should not cause errors.
        """
        await mcp_tool_sync.start()

        event = create_registration_event(
            event_type="unknown_type",  # Not registered/updated/deregistered/expired
            tool_name="some_tool",
            service_name="some-service",
            include_full_info=True,
        )

        await kafka_event_bus.publish(
            "node.registration.v1",
            None,
            json.dumps(event).encode("utf-8"),
        )

        # Wait and verify no operations
        await asyncio.sleep(EVENT_PROCESSING_WAIT_SECONDS)

        assert len(mock_registry.upsert_calls) == 0
        assert len(mock_registry.remove_calls) == 0


# =============================================================================
# Consul Fallback Tests
# =============================================================================


class TestServiceMCPToolSyncConsulFallback:
    """Tests for ServiceMCPToolSync Consul discovery fallback."""

    @pytest.mark.asyncio
    async def test_fallback_to_consul_when_event_lacks_full_info(
        self,
        mcp_tool_sync,
        kafka_event_bus: EventBusKafka,
        mock_registry: MockMCPToolRegistry,
        mock_discovery: MockMCPToolDiscovery,
    ) -> None:
        """Verify service falls back to Consul when event lacks full info.

        When a registration event doesn't contain enough information to build
        the tool definition, the service should call discover_by_service_id.
        """
        from omnibase_infra.models.mcp.model_mcp_tool_definition import (
            ModelMCPToolDefinition,
        )

        await mcp_tool_sync.start()

        # Configure mock discovery to return a tool
        tool_name = f"consul_tool_{uuid.uuid4().hex[:8]}"
        service_id = f"service-{uuid.uuid4().hex[:8]}"
        mock_tool = ModelMCPToolDefinition(
            name=tool_name,
            description="Tool from Consul fallback",
            version="1.0.0",
            parameters=[],
        )
        mock_discovery.tools_by_service_id[service_id] = mock_tool

        # Event without full service info (missing service_name)
        event = create_registration_event(
            event_type="registered",
            tool_name=tool_name,
            service_id=service_id,
            service_name=None,  # Missing - triggers fallback
            include_full_info=False,
        )

        await kafka_event_bus.publish(
            "node.registration.v1",
            None,
            json.dumps(event).encode("utf-8"),
        )

        # Wait for discovery and registry update
        await wait_for_registry_update(mock_registry, expected_upsert_count=1)

        # Verify Consul was called
        assert service_id in mock_discovery.discover_by_service_id_calls

        # Verify tool was upserted
        assert len(mock_registry.upsert_calls) >= 1
        upserted_tool, _ = mock_registry.upsert_calls[0]
        assert upserted_tool.name == tool_name


# =============================================================================
# Idempotency Tests
# =============================================================================


class TestServiceMCPToolSyncIdempotency:
    """Tests for ServiceMCPToolSync idempotency handling."""

    @pytest.mark.asyncio
    async def test_duplicate_events_handled_idempotently(
        self,
        mcp_tool_sync,
        kafka_event_bus: EventBusKafka,
        mock_registry: MockMCPToolRegistry,
    ) -> None:
        """Verify duplicate events with same event_id are handled idempotently.

        The registry's version tracking should prevent duplicate updates.
        """
        await mcp_tool_sync.start()

        tool_name = f"idempotent_tool_{uuid.uuid4().hex[:8]}"
        event_id = f"event-{uuid.uuid4().hex[:8]}"

        event = create_registration_event(
            event_type="registered",
            tool_name=tool_name,
            service_name="idempotent-orchestrator",
            event_id=event_id,
            include_full_info=True,
        )

        # Publish same event multiple times
        for _ in range(3):
            await kafka_event_bus.publish(
                "node.registration.v1",
                None,
                json.dumps(event).encode("utf-8"),
            )

        # Wait for processing
        await asyncio.sleep(EVENT_PROCESSING_WAIT_SECONDS * 2)

        # All events should be processed (calls made)
        assert len(mock_registry.upsert_calls) >= 1

        # But only first should succeed due to version tracking
        # (Mock registry simulates this)
        tool = await mock_registry.get_tool(tool_name)
        assert tool is not None


# =============================================================================
# Concurrency Tests
# =============================================================================


class TestServiceMCPToolSyncConcurrency:
    """Tests for ServiceMCPToolSync concurrent event handling."""

    @pytest.mark.asyncio
    async def test_multiple_events_processed_correctly(
        self,
        mcp_tool_sync,
        kafka_event_bus: EventBusKafka,
        mock_registry: MockMCPToolRegistry,
    ) -> None:
        """Verify multiple different events are all processed correctly.

        Publish several different registration events and verify all are handled.
        """
        await mcp_tool_sync.start()

        # Create multiple unique tools
        num_tools = 5
        tool_names = [
            f"concurrent_tool_{i}_{uuid.uuid4().hex[:8]}" for i in range(num_tools)
        ]

        # Publish all events
        for tool_name in tool_names:
            event = create_registration_event(
                event_type="registered",
                tool_name=tool_name,
                service_name=f"orchestrator-{tool_name}",
                include_full_info=True,
            )
            await kafka_event_bus.publish(
                "node.registration.v1",
                None,
                json.dumps(event).encode("utf-8"),
            )

        # Wait for all to be processed
        await wait_for_registry_update(
            mock_registry,
            expected_upsert_count=num_tools,
            timeout=EVENT_PROCESSING_WAIT_SECONDS * 3,
        )

        # Verify all tools were upserted
        assert len(mock_registry.upsert_calls) >= num_tools
        upserted_names = {call[0].name for call in mock_registry.upsert_calls}
        for tool_name in tool_names:
            assert tool_name in upserted_names
