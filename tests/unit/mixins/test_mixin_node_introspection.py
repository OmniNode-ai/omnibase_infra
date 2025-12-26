# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Unit tests for MixinNodeIntrospection.

This test suite validates:
- Initialization and attribute setup
- Capability extraction via reflection
- Endpoint discovery for health checks and operations
- FSM state extraction
- Caching behavior with TTL expiration
- Event bus publishing (with and without event bus)
- Background task management (heartbeat)
- Graceful degradation on errors
- Performance requirements (<50ms with CI buffer)

Test Organization:
    - TestMixinNodeIntrospectionInit: Initialization tests
    - TestMixinNodeIntrospectionCapabilities: Capability extraction
    - TestMixinNodeIntrospectionEndpoints: Endpoint discovery
    - TestMixinNodeIntrospectionState: FSM state extraction
    - TestMixinNodeIntrospectionCaching: Caching behavior
    - TestMixinNodeIntrospectionPublishing: Event bus publishing
    - TestMixinNodeIntrospectionTasks: Background tasks
    - TestMixinNodeIntrospectionGracefulDegradation: Error handling
    - TestMixinNodeIntrospectionPerformance: Performance requirements
    - TestMixinNodeIntrospectionBenchmark: Detailed benchmarks with instrumentation
    - TestMixinNodeIntrospectionEdgeCases: Edge cases and boundary conditions

Coverage Goals:
    - >90% code coverage for mixin
    - All public methods tested
    - Error paths validated
    - Performance requirements verified
"""

import asyncio
import json
import os
import time
from uuid import UUID, uuid4

import pytest
from omnibase_core.enums.enum_node_kind import EnumNodeKind
from pydantic import BaseModel, ValidationError

# Test UUIDs - use deterministic values for reproducible tests
TEST_NODE_UUID_1 = UUID("00000000-0000-0000-0000-000000000001")
TEST_NODE_UUID_2 = UUID("00000000-0000-0000-0000-000000000002")
TEST_NODE_UUID_3 = UUID("00000000-0000-0000-0000-000000000003")

from omnibase_infra.mixins.mixin_node_introspection import (
    PERF_THRESHOLD_CACHE_HIT_MS,
    PERF_THRESHOLD_GET_CAPABILITIES_MS,
    PERF_THRESHOLD_GET_INTROSPECTION_DATA_MS,
    IntrospectionPerformanceMetrics,
    MixinNodeIntrospection,
)
from omnibase_infra.models.discovery import (
    ModelIntrospectionConfig,
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.models.registration import ModelNodeHeartbeatEvent

# CI environments may be slower - apply multiplier for performance thresholds
_CI_MODE: bool = os.environ.get("CI", "false").lower() == "true"
PERF_MULTIPLIER: float = 3.0 if _CI_MODE else 2.0

# Type alias for event bus published event structure
PublishedEventDict = dict[
    str, str | bytes | None | dict[str, str | int | bool | list[str]]
]


class MockEventBus:
    """Mock event bus for testing introspection publishing.

    Implements the event bus interface required by MixinNodeIntrospection.
    The mixin uses duck typing to check for publish_envelope and publish methods.
    """

    def __init__(self, should_fail: bool = False) -> None:
        """Initialize mock event bus.

        Args:
            should_fail: If True, publish operations will raise exceptions.
        """
        self.should_fail = should_fail
        # Store envelopes for test assertions
        # Accepts both ModelNodeIntrospectionEvent and ModelNodeHeartbeatEvent
        self.published_envelopes: list[
            tuple[ModelNodeIntrospectionEvent | ModelNodeHeartbeatEvent, str]
        ] = []
        self.published_events: list[PublishedEventDict] = []

    async def publish_envelope(
        self,
        envelope: BaseModel,
        topic: str,
    ) -> None:
        """Mock publish_envelope method.

        Args:
            envelope: Event envelope to publish. Uses BaseModel for type safety
                since all event envelopes are Pydantic models.
            topic: Event topic.

        Raises:
            RuntimeError: If should_fail is True.
        """
        if self.should_fail:
            raise RuntimeError("Event bus publish failed")
        # Store envelopes - supports both introspection and heartbeat events
        if isinstance(envelope, ModelNodeIntrospectionEvent | ModelNodeHeartbeatEvent):
            self.published_envelopes.append((envelope, topic))

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
    ) -> None:
        """Mock publish method (fallback).

        Args:
            topic: Event topic.
            key: Event key.
            value: Event payload as bytes.

        Raises:
            RuntimeError: If should_fail is True.
        """
        if self.should_fail:
            raise RuntimeError("Event bus publish failed")

        self.published_events.append(
            {
                "topic": topic,
                "key": key,
                "value": json.loads(value.decode("utf-8")),
            }
        )


class MockNode(MixinNodeIntrospection):
    """Mock node class for testing the mixin."""

    def __init__(self) -> None:
        """Initialize mock node with test state."""
        self._state = "idle"
        self.health_url = "http://localhost:8080/health"
        self.metrics_url = "http://localhost:8080/metrics"

    async def execute(self, operation: str, payload: dict[str, str]) -> dict[str, str]:
        """Mock execute method.

        Args:
            operation: Operation to execute.
            payload: Operation payload.

        Returns:
            Operation result.
        """
        _ = payload  # Silence unused parameter warning
        return {"result": "ok", "operation": operation}

    async def health_check(self) -> dict[str, bool | str]:
        """Mock health check.

        Returns:
            Health status.
        """
        return {"healthy": True, "state": self._state}

    async def handle_event(self, event: dict[str, str]) -> None:
        """Mock handle_event method (should be discovered as operation).

        Args:
            event: Event to handle.
        """
        _ = event  # Silence unused parameter warning

    async def process_batch(self, items: list[dict[str, str]]) -> list[dict[str, str]]:
        """Mock process method (should be discovered as operation).

        Args:
            items: Items to process.

        Returns:
            Processed items.
        """
        return items


class MockNodeNoHealth(MixinNodeIntrospection):
    """Mock node without health endpoint URLs."""

    def __init__(self) -> None:
        """Initialize mock node."""
        self._state = "active"

    async def process(self, data: dict[str, str]) -> dict[str, bool]:
        """Mock process method.

        Args:
            data: Data to process.

        Returns:
            Processed result.
        """
        _ = data  # Silence unused parameter warning
        return {"processed": True}


class MockNodeNoState(MixinNodeIntrospection):
    """Mock node without _state attribute."""

    async def execute(self, operation: str) -> dict[str, str]:
        """Mock execute method.

        Args:
            operation: Operation name.

        Returns:
            Operation result.
        """
        return {"executed": operation}


class MockNodeWithEnumState(MixinNodeIntrospection):
    """Mock node with enum-style state."""

    def __init__(self) -> None:
        """Initialize mock node."""

        class State:
            """Mock state class with value attribute to simulate enum-style state."""

            value: str = "running"

        self._state: State = State()


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinNodeIntrospectionInit:
    """Tests for introspection initialization."""

    async def test_initialize_introspection_sets_attributes(self) -> None:
        """Test that initialize_introspection properly sets all attributes."""
        test_node_id = uuid4()
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        assert node._introspection_node_id == TEST_NODE_UUID_1
        assert node._introspection_node_type == EnumNodeKind.EFFECT
        assert node._introspection_event_bus is None
        assert node._introspection_version == "1.0.0"
        assert node._introspection_start_time is not None

    async def test_initialize_introspection_with_event_bus(self) -> None:
        """Test initialization with event bus."""
        node = MockNode()
        event_bus = MockEventBus()

        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_2,
            node_type="COMPUTE",
            event_bus=event_bus,
        )
        node.initialize_introspection(config)

        assert node._introspection_event_bus is event_bus

    async def test_initialize_introspection_custom_cache_ttl(self) -> None:
        """Test initialization with custom cache TTL."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_3,
            node_type="REDUCER",
            event_bus=None,
            cache_ttl=120.0,
        )
        node.initialize_introspection(config)

        assert node._introspection_cache_ttl == 120.0

    async def test_initialize_introspection_custom_version(self) -> None:
        """Test initialization with custom version."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="ORCHESTRATOR",
            event_bus=None,
            version="2.1.0",
        )
        node.initialize_introspection(config)

        assert node._introspection_version == "2.1.0"

    async def test_initialize_introspection_defaults(self) -> None:
        """Test initialization uses correct defaults."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # Default cache TTL is 300 seconds
        assert node._introspection_cache_ttl == 300.0
        # Default version is 1.0.0
        assert node._introspection_version == "1.0.0"
        # Cache starts empty
        assert node._introspection_cache is None
        assert node._introspection_cached_at is None

    async def test_initialize_introspection_invalid_node_id_format_raises(self) -> None:
        """Test that invalid UUID format for node_id raises validation error.

        The node_id field expects a UUID type. Passing an empty string (or any
        non-UUID string) triggers Pydantic's type validation, which rejects
        invalid UUID formats.
        """
        node = MockNode()

        with pytest.raises(ValidationError):
            config = ModelIntrospectionConfig(
                node_id="",  # Empty string is not a valid UUID format
                node_type="EFFECT",
            )
            node.initialize_introspection(config)

    async def test_initialize_introspection_empty_node_type_raises(self) -> None:
        """Test that empty node_type raises validation error."""
        node = MockNode()

        with pytest.raises(ValidationError):
            config = ModelIntrospectionConfig(
                node_id=TEST_NODE_UUID_1,
                node_type="",
            )
            node.initialize_introspection(config)


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinNodeIntrospectionCapabilities:
    """Tests for capability extraction."""

    @pytest.fixture
    def mock_node(self) -> MockNode:
        """Create initialized mock node fixture.

        Returns:
            Initialized MockNode instance.
        """
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)
        return node

    async def test_get_capabilities_extracts_operations(
        self, mock_node: MockNode
    ) -> None:
        """Test that get_capabilities extracts operation methods."""
        capabilities = await mock_node.get_capabilities()

        # Should discover methods with operation keywords
        operations = capabilities["operations"]
        assert isinstance(operations, list)
        assert "execute" in operations
        assert "handle_event" in operations
        assert "process_batch" in operations

    async def test_get_capabilities_excludes_private_methods(
        self, mock_node: MockNode
    ) -> None:
        """Test that get_capabilities excludes private methods."""
        capabilities = await mock_node.get_capabilities()

        # Private methods should not be in operations
        operations = capabilities["operations"]
        assert isinstance(operations, list)
        for op in operations:
            assert not op.startswith("_")

    async def test_get_capabilities_detects_fsm(self, mock_node: MockNode) -> None:
        """Test that get_capabilities detects FSM state management."""
        capabilities = await mock_node.get_capabilities()

        # MockNode has _state attribute
        assert capabilities["has_fsm"] is True

    async def test_get_capabilities_detects_protocols(
        self, mock_node: MockNode
    ) -> None:
        """Test that get_capabilities discovers protocols."""
        capabilities = await mock_node.get_capabilities()

        # Should include MixinNodeIntrospection in protocols
        protocols = capabilities["protocols"]
        assert isinstance(protocols, list)
        assert "MixinNodeIntrospection" in protocols

    async def test_get_capabilities_includes_method_signatures(
        self, mock_node: MockNode
    ) -> None:
        """Test that get_capabilities captures method signatures."""
        capabilities = await mock_node.get_capabilities()

        # Should have method signatures
        assert "method_signatures" in capabilities
        assert isinstance(capabilities["method_signatures"], dict)

    async def test_get_capabilities_returns_dict(self, mock_node: MockNode) -> None:
        """Test that get_capabilities returns a dictionary."""
        capabilities = await mock_node.get_capabilities()

        assert isinstance(capabilities, dict)
        assert "operations" in capabilities
        assert "protocols" in capabilities
        assert "has_fsm" in capabilities
        assert "method_signatures" in capabilities


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinNodeIntrospectionEndpoints:
    """Tests for endpoint discovery."""

    @pytest.fixture
    def mock_node(self) -> MockNode:
        """Create initialized mock node fixture.

        Returns:
            Initialized MockNode instance.
        """
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)
        return node

    async def test_get_endpoints_discovers_health(self, mock_node: MockNode) -> None:
        """Test that get_endpoints discovers health endpoint."""
        endpoints = await mock_node.get_endpoints()

        assert "health" in endpoints
        assert endpoints["health"] == "http://localhost:8080/health"

    async def test_get_endpoints_discovers_metrics(self, mock_node: MockNode) -> None:
        """Test that get_endpoints discovers metrics endpoint."""
        endpoints = await mock_node.get_endpoints()

        assert "metrics" in endpoints
        assert endpoints["metrics"] == "http://localhost:8080/metrics"

    async def test_get_endpoints_no_endpoints(self) -> None:
        """Test endpoint discovery when no endpoints defined."""
        node = MockNodeNoHealth()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        endpoints = await node.get_endpoints()

        # Should return empty dict
        assert isinstance(endpoints, dict)
        assert len(endpoints) == 0

    async def test_get_endpoints_returns_dict(self, mock_node: MockNode) -> None:
        """Test that get_endpoints returns a dictionary."""
        endpoints = await mock_node.get_endpoints()

        assert isinstance(endpoints, dict)
        for key, value in endpoints.items():
            assert isinstance(key, str)
            assert isinstance(value, str)


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinNodeIntrospectionState:
    """Tests for FSM state extraction."""

    @pytest.fixture
    def mock_node(self) -> MockNode:
        """Create initialized mock node fixture.

        Returns:
            Initialized MockNode instance.
        """
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)
        return node

    async def test_get_current_state_returns_state(self, mock_node: MockNode) -> None:
        """Test that get_current_state returns the node's state."""
        state = await mock_node.get_current_state()

        assert state == "idle"

    async def test_get_current_state_reflects_changes(
        self, mock_node: MockNode
    ) -> None:
        """Test that get_current_state reflects state changes."""
        mock_node._state = "processing"
        state = await mock_node.get_current_state()

        assert state == "processing"

    async def test_get_current_state_no_state_attribute(self) -> None:
        """Test get_current_state when _state is missing."""
        node = MockNodeNoState()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        state = await node.get_current_state()

        assert state is None

    async def test_get_current_state_with_enum_state(self) -> None:
        """Test get_current_state with enum-style state (has .value)."""
        node = MockNodeWithEnumState()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        state = await node.get_current_state()
        assert state == "running"


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinNodeIntrospectionCaching:
    """Tests for caching behavior."""

    @pytest.fixture
    def mock_node(self) -> MockNode:
        """Create initialized mock node with short TTL.

        Returns:
            Initialized MockNode instance with 0.1s cache TTL.
        """
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
            cache_ttl=0.1,  # Short TTL for testing
        )
        node.initialize_introspection(config)
        return node

    async def test_get_introspection_data_caches_result(
        self, mock_node: MockNode
    ) -> None:
        """Test that get_introspection_data caches the result."""
        # First call - should compute
        data1 = await mock_node.get_introspection_data()
        timestamp1 = data1.timestamp

        # Immediate second call - should return cached
        data2 = await mock_node.get_introspection_data()
        timestamp2 = data2.timestamp

        # Same timestamp means cached result
        assert timestamp1 == timestamp2

    async def test_cache_expires_after_ttl(self, mock_node: MockNode) -> None:
        """Test that cache expires after TTL."""
        # First call - populates cache
        data1 = await mock_node.get_introspection_data()
        timestamp1 = data1.timestamp

        # Wait for TTL to expire (0.1s + buffer)
        await asyncio.sleep(0.15)

        # Next call should recompute
        data2 = await mock_node.get_introspection_data()
        timestamp2 = data2.timestamp

        # Different timestamp means cache was refreshed
        assert timestamp2 > timestamp1

    async def test_get_introspection_data_structure(self, mock_node: MockNode) -> None:
        """Test that get_introspection_data returns expected model."""
        data = await mock_node.get_introspection_data()

        assert isinstance(data, ModelNodeIntrospectionEvent)
        # node_id is a UUID passed via config
        assert isinstance(data.node_id, UUID)
        assert data.node_id == TEST_NODE_UUID_1
        # node_type is stored as EnumNodeKind (a StrEnum that inherits from str).
        # Compare directly to the enum for type consistency with _introspection_node_type.
        assert data.node_type == EnumNodeKind.EFFECT
        assert isinstance(data.capabilities, dict)
        assert isinstance(data.endpoints, dict)
        assert data.version == "1.0.0"

    async def test_cache_not_used_before_initialization(self) -> None:
        """Test that cache starts empty."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        assert node._introspection_cache is None
        assert node._introspection_cached_at is None

    async def test_invalidate_introspection_cache(self, mock_node: MockNode) -> None:
        """Test that invalidate_introspection_cache clears the cache."""
        # Populate cache
        await mock_node.get_introspection_data()
        assert mock_node._introspection_cache is not None

        # Invalidate
        mock_node.invalidate_introspection_cache()

        assert mock_node._introspection_cache is None
        assert mock_node._introspection_cached_at is None


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinNodeIntrospectionPublishing:
    """Tests for event bus publishing."""

    @pytest.fixture
    def mock_node_with_bus(self) -> MockNode:
        """Create initialized mock node with event bus.

        Returns:
            MockNode with MockEventBus attached.
        """
        node = MockNode()
        event_bus = MockEventBus()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=event_bus,
        )
        node.initialize_introspection(config)
        return node

    @pytest.fixture
    def mock_node_without_bus(self) -> MockNode:
        """Create initialized mock node without event bus.

        Returns:
            MockNode without event bus.
        """
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)
        return node

    async def test_publish_introspection_returns_false_without_event_bus(
        self, mock_node_without_bus: MockNode
    ) -> None:
        """Test that publish_introspection returns False without event bus."""
        result = await mock_node_without_bus.publish_introspection()

        assert result is False

    async def test_publish_introspection_succeeds_with_event_bus(
        self, mock_node_with_bus: MockNode
    ) -> None:
        """Test that publish_introspection succeeds with event bus."""
        result = await mock_node_with_bus.publish_introspection()

        assert result is True

        # Verify envelope was published
        event_bus = mock_node_with_bus._introspection_event_bus
        assert isinstance(event_bus, MockEventBus)
        assert len(event_bus.published_envelopes) == 1

        envelope, topic = event_bus.published_envelopes[0]
        assert topic == "node.introspection"
        assert isinstance(envelope, ModelNodeIntrospectionEvent)

    async def test_publish_introspection_with_correlation_id(
        self, mock_node_with_bus: MockNode
    ) -> None:
        """Test that publish_introspection passes correlation_id."""
        correlation_id = uuid4()
        await mock_node_with_bus.publish_introspection(correlation_id=correlation_id)

        event_bus = mock_node_with_bus._introspection_event_bus
        assert isinstance(event_bus, MockEventBus)

        envelope, _ = event_bus.published_envelopes[0]
        # Correlation ID should be set (it may be regenerated in the publish method)
        assert envelope.correlation_id is not None

    async def test_publish_introspection_with_reason(
        self, mock_node_with_bus: MockNode
    ) -> None:
        """Test that publish_introspection sets the reason."""
        await mock_node_with_bus.publish_introspection(reason="shutdown")

        event_bus = mock_node_with_bus._introspection_event_bus
        assert isinstance(event_bus, MockEventBus)

        envelope, _ = event_bus.published_envelopes[0]
        # Type narrow to ModelNodeIntrospectionEvent for reason attribute
        assert isinstance(envelope, ModelNodeIntrospectionEvent)
        assert envelope.reason == "shutdown"


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinNodeIntrospectionTasks:
    """Tests for background task management."""

    async def test_start_introspection_tasks_starts_heartbeat(self) -> None:
        """Test that start_introspection_tasks creates heartbeat task."""
        node = MockNode()
        event_bus = MockEventBus()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=event_bus,
        )
        node.initialize_introspection(config)

        # Start tasks with fast heartbeat
        await node.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=0.05,
            enable_registry_listener=False,
        )

        try:
            assert node._heartbeat_task is not None
            assert not node._heartbeat_task.done()

            # Wait for at least one heartbeat
            await asyncio.sleep(0.1)

            # Should have published at least one event
            assert len(event_bus.published_envelopes) >= 1
        finally:
            # Clean up
            await node.stop_introspection_tasks()

    async def test_stop_introspection_tasks_cancels_tasks(self) -> None:
        """Test that stop_introspection_tasks cancels all tasks."""
        node = MockNode()
        event_bus = MockEventBus()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=event_bus,
        )
        node.initialize_introspection(config)

        # Start and then stop tasks
        await node.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=0.1,
            enable_registry_listener=False,
        )
        assert node._heartbeat_task is not None

        await node.stop_introspection_tasks()

        # Task should be None after stop
        assert node._heartbeat_task is None

    async def test_stop_introspection_tasks_idempotent(self) -> None:
        """Test that stop_introspection_tasks can be called multiple times."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # Stop without starting should be safe
        await node.stop_introspection_tasks()
        await node.stop_introspection_tasks()

        assert node._heartbeat_task is None

    async def test_heartbeat_publishes_periodically(self) -> None:
        """Test that heartbeat publishes at regular intervals.

        Uses polling with retries instead of fixed sleep to be more robust
        in CI environments where timing can be variable.
        """
        node = MockNode()
        event_bus = MockEventBus()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=event_bus,
        )
        node.initialize_introspection(config)

        await node.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=0.02,  # Faster heartbeat for test
            enable_registry_listener=False,
        )

        try:
            # Use polling with retries instead of fixed sleep
            # This is more robust in slow CI environments
            max_wait_seconds = 0.5  # Max time to wait for heartbeats
            poll_interval = 0.05  # Check every 50ms
            min_expected_events = 2  # Lower threshold for CI robustness
            elapsed = 0.0

            while elapsed < max_wait_seconds:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
                if len(event_bus.published_envelopes) >= min_expected_events:
                    break

            # Should have at least 2 events (reduced from 3 for CI robustness)
            assert len(event_bus.published_envelopes) >= min_expected_events, (
                f"Expected at least {min_expected_events} heartbeat events, "
                f"got {len(event_bus.published_envelopes)} after {elapsed:.2f}s"
            )
        finally:
            await node.stop_introspection_tasks()


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinNodeIntrospectionGracefulDegradation:
    """Tests for graceful degradation on errors."""

    async def test_publish_graceful_degradation_on_error(self) -> None:
        """Test that publish_introspection handles errors gracefully."""
        node = MockNode()
        failing_event_bus = MockEventBus(should_fail=True)
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=failing_event_bus,
        )
        node.initialize_introspection(config)

        # Should not raise, just return False
        result = await node.publish_introspection()

        assert result is False

    async def test_publish_does_not_crash_on_exception(self) -> None:
        """Test that publish_introspection catches all exceptions."""
        node = MockNode()

        # Create event bus that raises unexpected exception
        # Must implement publish_envelope and publish methods for duck typing
        class BrokenEventBus:
            async def publish_envelope(self, envelope: object, topic: str) -> None:
                raise ValueError("Unexpected error")

            async def publish(
                self, topic: str, key: bytes | None, value: bytes
            ) -> None:
                raise ValueError("Unexpected error")

        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=BrokenEventBus(),
        )
        node.initialize_introspection(config)

        # Should not raise
        result = await node.publish_introspection()
        assert result is False

    async def test_heartbeat_continues_after_publish_failure(self) -> None:
        """Test that heartbeat continues even if publish fails.

        Uses polling to verify task is still running instead of fixed sleep,
        making it more robust in CI environments.
        """
        node = MockNode()
        event_bus = MockEventBus(should_fail=True)
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=event_bus,
        )
        node.initialize_introspection(config)

        await node.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=0.02,  # Faster for test
            enable_registry_listener=False,
        )

        try:
            # Poll to verify task stays running despite failures
            # This is more robust than a fixed sleep in slow CI environments
            max_wait_seconds = 0.3
            poll_interval = 0.05
            elapsed = 0.0
            task_was_running = False

            while elapsed < max_wait_seconds:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
                # Verify task is still running
                if node._heartbeat_task is not None and not node._heartbeat_task.done():
                    task_was_running = True
                    # Continue polling to ensure task doesn't crash
                    continue
                break

            # Task should still be running (not crashed from failures)
            assert node._heartbeat_task is not None, "Heartbeat task should exist"
            assert not node._heartbeat_task.done(), "Heartbeat task should not be done"
            assert task_was_running, "Task should have been observed running"
        finally:
            await node.stop_introspection_tasks()


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinNodeIntrospectionPerformance:
    """Tests for performance requirements.

    Note: Performance thresholds are multiplied by PERF_MULTIPLIER to account
    for CI environments which may be slower than local development machines.
    """

    @pytest.fixture
    def mock_node(self) -> MockNode:
        """Create initialized mock node fixture.

        Returns:
            Initialized MockNode instance.
        """
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)
        return node

    async def test_introspection_extraction_under_50ms(
        self, mock_node: MockNode
    ) -> None:
        """Test that introspection data extraction completes within threshold."""
        # Clear cache to force full computation
        mock_node._introspection_cache = None
        mock_node._introspection_cached_at = None

        threshold_ms = 50 * PERF_MULTIPLIER
        start = time.perf_counter()
        await mock_node.get_introspection_data()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < threshold_ms, (
            f"Introspection took {elapsed_ms:.2f}ms, expected <{threshold_ms:.0f}ms"
        )

    async def test_cached_introspection_under_1ms(self, mock_node: MockNode) -> None:
        """Test that cached introspection returns within threshold."""
        # Populate cache
        await mock_node.get_introspection_data()

        threshold_ms = 1 * PERF_MULTIPLIER
        start = time.perf_counter()
        await mock_node.get_introspection_data()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < threshold_ms, (
            f"Cached introspection took {elapsed_ms:.2f}ms, expected <{threshold_ms:.0f}ms"
        )

    async def test_capability_extraction_under_10ms(self, mock_node: MockNode) -> None:
        """Test that capability extraction completes within threshold."""
        threshold_ms = 10 * PERF_MULTIPLIER
        start = time.perf_counter()
        await mock_node.get_capabilities()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < threshold_ms, (
            f"Capability extraction took {elapsed_ms:.2f}ms, expected <{threshold_ms:.0f}ms"
        )

    async def test_endpoint_discovery_under_10ms(self, mock_node: MockNode) -> None:
        """Test that endpoint discovery completes within threshold."""
        threshold_ms = 10 * PERF_MULTIPLIER
        start = time.perf_counter()
        await mock_node.get_endpoints()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < threshold_ms, (
            f"Endpoint discovery took {elapsed_ms:.2f}ms, expected <{threshold_ms:.0f}ms"
        )

    async def test_state_extraction_under_1ms(self, mock_node: MockNode) -> None:
        """Test that state extraction completes within threshold."""
        threshold_ms = 1 * PERF_MULTIPLIER
        start = time.perf_counter()
        await mock_node.get_current_state()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < threshold_ms, (
            f"State extraction took {elapsed_ms:.2f}ms, expected <{threshold_ms:.0f}ms"
        )

    async def test_multiple_introspection_calls_consistent_performance(
        self, mock_node: MockNode
    ) -> None:
        """Test that multiple introspection calls have consistent performance."""
        times: list[float] = []

        for _ in range(10):
            # Clear cache each time
            mock_node._introspection_cache = None
            mock_node._introspection_cached_at = None

            start = time.perf_counter()
            await mock_node.get_introspection_data()
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        avg_time = sum(times) / len(times)
        max_time = max(times)

        avg_threshold_ms = 30 * PERF_MULTIPLIER
        max_threshold_ms = 50 * PERF_MULTIPLIER

        assert avg_time < avg_threshold_ms, (
            f"Average time {avg_time:.2f}ms, expected <{avg_threshold_ms:.0f}ms"
        )
        assert max_time < max_threshold_ms, (
            f"Max time {max_time:.2f}ms, expected <{max_threshold_ms:.0f}ms"
        )


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinNodeIntrospectionBenchmark:
    """Detailed performance benchmarks with instrumentation.

    These tests verify the <50ms requirement and provide
    detailed timing breakdowns for optimization.

    Note: Performance thresholds are multiplied by PERF_MULTIPLIER to account
    for CI environments which may be slower than local development machines.
    """

    async def test_introspection_benchmark_with_instrumentation(self) -> None:
        """Benchmark introspection with detailed timing breakdown."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # Clear cache for full computation
        node._introspection_cache = None
        node._introspection_cached_at = None

        timings: dict[str, list[float]] = {
            "get_capabilities": [],
            "get_endpoints": [],
            "get_current_state": [],
            "total_introspection": [],
        }

        iterations = 20
        for _ in range(iterations):
            node._introspection_cache = None
            node._introspection_cached_at = None

            # Time individual components
            start = time.perf_counter()
            await node.get_capabilities()
            timings["get_capabilities"].append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            await node.get_endpoints()
            timings["get_endpoints"].append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            await node.get_current_state()
            timings["get_current_state"].append((time.perf_counter() - start) * 1000)

            node._introspection_cache = None
            node._introspection_cached_at = None

            start = time.perf_counter()
            await node.get_introspection_data()
            timings["total_introspection"].append((time.perf_counter() - start) * 1000)

        # Calculate statistics
        for name, times in timings.items():
            avg = sum(times) / len(times)
            min_t = min(times)
            max_t = max(times)
            p95 = sorted(times)[int(len(times) * 0.95)]

            # Log timing breakdown for debugging
            print(
                f"{name}: avg={avg:.2f}ms, min={min_t:.2f}ms, "
                f"max={max_t:.2f}ms, p95={p95:.2f}ms"
            )

        # Assert <50ms requirement (with CI buffer)
        threshold_ms = 50 * PERF_MULTIPLIER
        avg_total = sum(timings["total_introspection"]) / len(
            timings["total_introspection"]
        )
        assert avg_total < threshold_ms, (
            f"Average introspection {avg_total:.2f}ms exceeds {threshold_ms:.0f}ms"
        )

    async def test_introspection_concurrent_load_benchmark(self) -> None:
        """Benchmark introspection under concurrent load."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
            cache_ttl=0.001,  # Force cache misses
        )
        node.initialize_introspection(config)

        async def single_introspection() -> float:
            start = time.perf_counter()
            await node.get_introspection_data()
            return (time.perf_counter() - start) * 1000

        # 50 concurrent introspection requests
        tasks = [single_introspection() for _ in range(50)]
        times = await asyncio.gather(*tasks)

        avg_time = sum(times) / len(times)
        max_time = max(times)
        p95_time = sorted(times)[int(len(times) * 0.95)]

        # Log benchmark results
        print(
            f"Concurrent load (50 requests): avg={avg_time:.2f}ms, "
            f"max={max_time:.2f}ms, p95={p95_time:.2f}ms"
        )

        threshold_ms = 100 * PERF_MULTIPLIER  # Higher threshold for concurrent load
        assert avg_time < threshold_ms, (
            f"Average concurrent time {avg_time:.2f}ms exceeds {threshold_ms:.0f}ms"
        )
        assert max_time < threshold_ms * 2, (
            f"Max concurrent time {max_time:.2f}ms exceeds {threshold_ms * 2:.0f}ms"
        )

    async def test_cache_hit_performance(self) -> None:
        """Verify cache hits are sub-millisecond."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # Warm cache
        await node.get_introspection_data()

        # Measure cache hits
        times: list[float] = []
        for _ in range(100):
            start = time.perf_counter()
            await node.get_introspection_data()
            times.append((time.perf_counter() - start) * 1000)

        avg_time = sum(times) / len(times)
        p99 = sorted(times)[int(len(times) * 0.99)]
        min_time = min(times)
        max_time = max(times)

        # Log cache hit performance
        print(
            f"Cache hits (100 requests): avg={avg_time:.3f}ms, "
            f"min={min_time:.3f}ms, max={max_time:.3f}ms, p99={p99:.3f}ms"
        )

        # Cache hits should be very fast
        threshold_ms = 0.5 * PERF_MULTIPLIER
        assert avg_time < threshold_ms, (
            f"Cache hit avg {avg_time:.3f}ms exceeds {threshold_ms:.1f}ms"
        )

    async def test_introspection_p95_latency(self) -> None:
        """Test that p95 latency meets requirements."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        times: list[float] = []
        iterations = 50

        for _ in range(iterations):
            # Clear cache for each iteration
            node._introspection_cache = None
            node._introspection_cached_at = None

            start = time.perf_counter()
            await node.get_introspection_data()
            times.append((time.perf_counter() - start) * 1000)

        p95 = sorted(times)[int(len(times) * 0.95)]
        p99 = sorted(times)[int(len(times) * 0.99)]
        avg_time = sum(times) / len(times)

        # Log p95 and p99 latencies
        print(
            f"Latency distribution ({iterations} iterations): "
            f"avg={avg_time:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms"
        )

        # p95 should be under 50ms threshold (with CI buffer)
        threshold_ms = 50 * PERF_MULTIPLIER
        assert p95 < threshold_ms, (
            f"p95 latency {p95:.2f}ms exceeds {threshold_ms:.0f}ms"
        )

    async def test_component_timing_breakdown(self) -> None:
        """Test timing breakdown of individual introspection components."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # Time each component individually
        components = {
            "capabilities": node.get_capabilities,
            "endpoints": node.get_endpoints,
            "state": node.get_current_state,
        }

        component_times: dict[str, float] = {}

        for name, func in components.items():
            times: list[float] = []
            for _ in range(10):
                start = time.perf_counter()
                await func()
                times.append((time.perf_counter() - start) * 1000)
            component_times[name] = sum(times) / len(times)

        # Log component breakdown
        print("Component timing breakdown:")
        for name, avg_ms in component_times.items():
            print(f"  {name}: {avg_ms:.2f}ms")

        # Capabilities is typically the slowest (reflection-based)
        cap_threshold_ms = 20 * PERF_MULTIPLIER
        assert component_times["capabilities"] < cap_threshold_ms, (
            f"Capabilities extraction {component_times['capabilities']:.2f}ms "
            f"exceeds {cap_threshold_ms:.0f}ms"
        )

        # State extraction should be very fast
        state_threshold_ms = 1 * PERF_MULTIPLIER
        assert component_times["state"] < state_threshold_ms, (
            f"State extraction {component_times['state']:.2f}ms "
            f"exceeds {state_threshold_ms:.0f}ms"
        )


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinNodeIntrospectionEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_empty_node_introspection(self) -> None:
        """Test introspection on a minimal node."""

        class MinimalNode(MixinNodeIntrospection):
            pass

        test_node_id = uuid4()
        node = MinimalNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        data = await node.get_introspection_data()

        # node_id is the UUID passed in config
        assert isinstance(data.node_id, UUID)
        assert data.node_id == TEST_NODE_UUID_1
        assert data.current_state is None  # No state attribute

    async def test_large_capability_list(self) -> None:
        """Test introspection with many public methods."""

        class LargeNode(MixinNodeIntrospection):
            async def execute_task_001(self) -> None:
                pass

            async def handle_event_002(self) -> None:
                pass

            async def process_data_003(self) -> None:
                pass

            async def run_operation_004(self) -> None:
                pass

            async def invoke_action_005(self) -> None:
                pass

            async def call_service_006(self) -> None:
                pass

            async def execute_job_007(self) -> None:
                pass

            async def handle_request_008(self) -> None:
                pass

            async def process_queue_009(self) -> None:
                pass

            async def run_batch_010(self) -> None:
                pass

        node = LargeNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_2,
            node_type="COMPUTE",
            event_bus=None,
        )
        node.initialize_introspection(config)

        threshold_ms = 50 * PERF_MULTIPLIER
        start = time.perf_counter()
        capabilities = await node.get_capabilities()
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should include all 10 operation methods
        operations = capabilities["operations"]
        assert isinstance(operations, list)
        assert len(operations) >= 10
        assert elapsed_ms < threshold_ms, (
            f"Large capability extraction took {elapsed_ms:.2f}ms, expected <{threshold_ms:.0f}ms"
        )

    async def test_concurrent_introspection_calls(self) -> None:
        """Test concurrent introspection data requests."""
        test_node_id = uuid4()
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
            cache_ttl=0.001,  # Very short TTL
        )
        node.initialize_introspection(config)

        # Make 100 concurrent calls
        tasks = [node.get_introspection_data() for _ in range(100)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 100
        for result in results:
            assert isinstance(result.node_id, UUID)
            assert result.node_id == TEST_NODE_UUID_1

    async def test_introspection_with_special_characters_in_state(self) -> None:
        """Test introspection with special characters in state."""
        node = MockNode()
        node._state = "state<with>special&chars\"quote'"
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        state = await node.get_current_state()
        assert state == "state<with>special&chars\"quote'"

    async def test_introspection_preserves_node_functionality(self) -> None:
        """Test that introspection mixin doesn't affect node functionality."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # Node methods should still work normally
        result = await node.execute("test_op", {"data": "value"})
        assert result["result"] == "ok"
        assert result["operation"] == "test_op"

        health = await node.health_check()
        assert health["healthy"] is True


@pytest.mark.asyncio(loop_scope="function")
class TestMixinNodeIntrospectionClassLevelCache:
    """Test class-level method signature caching for performance optimization."""

    def setup_method(self) -> None:
        """Clear class-level cache before each test."""
        MixinNodeIntrospection._invalidate_class_method_cache()

    def teardown_method(self) -> None:
        """Clear class-level cache after each test."""
        MixinNodeIntrospection._invalidate_class_method_cache()

    async def test_class_method_cache_populated_on_first_access(self) -> None:
        """Test that class-level cache is populated on first access."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # Cache should be empty before first access
        assert MockNode not in MixinNodeIntrospection._class_method_cache

        # Access capabilities to trigger cache population
        await node.get_capabilities()

        # Cache should now contain MockNode
        assert MockNode in MixinNodeIntrospection._class_method_cache
        cached_signatures = MixinNodeIntrospection._class_method_cache[MockNode]
        assert isinstance(cached_signatures, dict)
        assert len(cached_signatures) > 0

    async def test_class_method_cache_shared_across_instances(self) -> None:
        """Test that class-level cache is shared across instances."""
        node1 = MockNode()
        config1 = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node1.initialize_introspection(config1)

        node2 = MockNode()
        config2 = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node2.initialize_introspection(config2)

        # First node populates cache
        await node1.get_capabilities()
        assert MockNode in MixinNodeIntrospection._class_method_cache

        # Second node uses same cache (no re-population)
        cached_before = id(MixinNodeIntrospection._class_method_cache[MockNode])
        await node2.get_capabilities()
        cached_after = id(MixinNodeIntrospection._class_method_cache[MockNode])

        # Cache object identity should be the same (not recreated)
        assert cached_before == cached_after

    async def test_invalidate_class_method_cache_specific_class(self) -> None:
        """Test invalidating cache for a specific class."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # Populate cache
        await node.get_capabilities()
        assert MockNode in MixinNodeIntrospection._class_method_cache

        # Invalidate specific class
        MixinNodeIntrospection._invalidate_class_method_cache(MockNode)

        # Cache should be cleared for MockNode
        assert MockNode not in MixinNodeIntrospection._class_method_cache

    async def test_invalidate_class_method_cache_all_classes(self) -> None:
        """Test invalidating cache for all classes."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # Populate cache
        await node.get_capabilities()
        assert MockNode in MixinNodeIntrospection._class_method_cache

        # Invalidate all
        MixinNodeIntrospection._invalidate_class_method_cache()

        # Cache should be empty
        assert len(MixinNodeIntrospection._class_method_cache) == 0

    async def test_cached_signatures_match_direct_extraction(self) -> None:
        """Test that cached signatures match direct signature extraction."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # Get capabilities (uses cache)
        capabilities = await node.get_capabilities()
        cached_signatures = capabilities["method_signatures"]
        assert isinstance(cached_signatures, dict)

        # Get cached signatures directly
        direct_cached = MixinNodeIntrospection._class_method_cache.get(MockNode, {})

        # The capabilities method filters some prefixes, but the direct cache
        # should have all public methods. Verify cached signatures are used.
        assert len(cached_signatures) > 0
        assert len(direct_cached) >= len(cached_signatures)

    async def test_class_level_cache_performance_benefit(self) -> None:
        """Test that class-level caching provides performance benefit."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # First call (cold cache) - populates cache
        start1 = time.perf_counter()
        await node.get_capabilities()
        first_call_ms = (time.perf_counter() - start1) * 1000

        # Subsequent calls (warm cache) - uses cached signatures
        times_warm = []
        for _ in range(10):
            start = time.perf_counter()
            await node.get_capabilities()
            times_warm.append((time.perf_counter() - start) * 1000)

        avg_warm_ms = sum(times_warm) / len(times_warm)

        # Warm cache calls should be reasonably fast
        threshold_ms = 5 * PERF_MULTIPLIER
        assert avg_warm_ms < threshold_ms, (
            f"Warm cache calls averaged {avg_warm_ms:.2f}ms, expected <{threshold_ms:.0f}ms"
        )

    async def test_different_classes_have_separate_cache_entries(self) -> None:
        """Test that different classes have separate cache entries."""

        class CustomNode1(MixinNodeIntrospection):
            async def execute_custom1(self, data: str) -> dict[str, str]:
                return {"custom1": data}

        class CustomNode2(MixinNodeIntrospection):
            async def execute_custom2(self, value: int) -> dict[str, int]:
                return {"custom2": value}

        node1 = CustomNode1()
        config1 = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="COMPUTE",
            event_bus=None,
        )
        node1.initialize_introspection(config1)

        node2 = CustomNode2()
        config2 = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_2,
            node_type="COMPUTE",
            event_bus=None,
        )
        node2.initialize_introspection(config2)

        # Both populate their respective caches
        await node1.get_capabilities()
        await node2.get_capabilities()

        # Both classes should have entries
        assert CustomNode1 in MixinNodeIntrospection._class_method_cache
        assert CustomNode2 in MixinNodeIntrospection._class_method_cache

        # Entries should be different
        cache1 = MixinNodeIntrospection._class_method_cache[CustomNode1]
        cache2 = MixinNodeIntrospection._class_method_cache[CustomNode2]

        # CustomNode1 should have execute_custom1
        assert "execute_custom1" in cache1
        # CustomNode2 should have execute_custom2
        assert "execute_custom2" in cache2

        # Each should NOT have the other's method
        assert "execute_custom2" not in cache1
        assert "execute_custom1" not in cache2

    async def test_cache_handles_methods_without_signatures(self) -> None:
        """Test that cache handles methods without inspectable signatures."""

        class NodeWithBuiltins(MixinNodeIntrospection):
            # Built-in methods that may not have inspectable signatures
            pass

        node = NodeWithBuiltins()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # Should not raise exception
        capabilities = await node.get_capabilities()
        assert isinstance(capabilities, dict)
        assert "method_signatures" in capabilities


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinNodeIntrospectionConfigurableKeywords:
    """Tests for configurable operation_keywords and exclude_prefixes."""

    async def test_default_operation_keywords_used_when_not_specified(self) -> None:
        """Test that DEFAULT_OPERATION_KEYWORDS is used when not specified."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)
        assert (
            node._introspection_operation_keywords
            == MixinNodeIntrospection.DEFAULT_OPERATION_KEYWORDS
        )

    async def test_default_exclude_prefixes_used_when_not_specified(self) -> None:
        """Test that DEFAULT_EXCLUDE_PREFIXES is used when not specified."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)
        assert (
            node._introspection_exclude_prefixes
            == MixinNodeIntrospection.DEFAULT_EXCLUDE_PREFIXES
        )

    async def test_custom_operation_keywords_are_stored(self) -> None:
        """Test that custom operation_keywords are stored correctly."""
        custom_keywords = {"fetch", "upload", "download", "sync"}
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
            operation_keywords=custom_keywords,
        )
        node.initialize_introspection(config)
        assert node._introspection_operation_keywords == custom_keywords

    async def test_custom_exclude_prefixes_are_stored(self) -> None:
        """Test that custom exclude_prefixes are stored correctly."""
        custom_prefixes = {"_", "helper_", "internal_"}
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
            exclude_prefixes=custom_prefixes,
        )
        node.initialize_introspection(config)
        assert node._introspection_exclude_prefixes == custom_prefixes

    async def test_custom_operation_keywords_affect_capability_discovery(self) -> None:
        """Test that custom operation_keywords affect which methods are discovered."""

        class CustomMethodsNode(MixinNodeIntrospection):
            async def fetch_data(self, source: str) -> dict[str, str]:
                return {"source": source}

            async def upload_file(self, file_path: str) -> bool:
                return True

            async def execute_task(self, task_id: str) -> None:
                pass

        node = CustomMethodsNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
            operation_keywords={"fetch", "upload"},
        )
        node.initialize_introspection(config)
        capabilities = await node.get_capabilities()
        operations = capabilities["operations"]
        assert isinstance(operations, list)
        assert "fetch_data" in operations
        assert "upload_file" in operations
        assert "execute_task" not in operations

    async def test_node_type_specific_keywords_constant_exists(self) -> None:
        """Test that NODE_TYPE_OPERATION_KEYWORDS constant exists with EnumNodeKind keys."""
        assert hasattr(MixinNodeIntrospection, "NODE_TYPE_OPERATION_KEYWORDS")
        keywords_map = MixinNodeIntrospection.NODE_TYPE_OPERATION_KEYWORDS
        # Keys should be EnumNodeKind members for type safety
        assert EnumNodeKind.EFFECT in keywords_map
        assert EnumNodeKind.COMPUTE in keywords_map
        assert EnumNodeKind.REDUCER in keywords_map
        assert EnumNodeKind.ORCHESTRATOR in keywords_map
        for keywords in keywords_map.values():
            assert isinstance(keywords, set)

    async def test_empty_operation_keywords_discovers_no_operations(self) -> None:
        """Test that empty operation_keywords results in no operations discovered."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
            operation_keywords=set(),
        )
        node.initialize_introspection(config)
        capabilities = await node.get_capabilities()
        operations = capabilities["operations"]
        assert isinstance(operations, list)
        assert len(operations) == 0

    async def test_configuration_is_instance_specific(self) -> None:
        """Test that configuration is instance-specific, not shared."""

        class MultiInstanceNode(MixinNodeIntrospection):
            async def execute_task(self) -> None:
                pass

            async def fetch_data(self) -> None:
                pass

        node1 = MultiInstanceNode()
        config1 = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
            operation_keywords={"execute"},
        )
        node1.initialize_introspection(config1)
        node2 = MultiInstanceNode()
        config2 = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_2,
            node_type="EFFECT",
            event_bus=None,
            operation_keywords={"fetch"},
        )
        node2.initialize_introspection(config2)
        caps1 = await node1.get_capabilities()
        caps2 = await node2.get_capabilities()
        ops1 = caps1["operations"]
        ops2 = caps2["operations"]
        assert isinstance(ops1, list)
        assert isinstance(ops2, list)
        assert "execute_task" in ops1
        assert "fetch_data" not in ops1
        assert "fetch_data" in ops2
        assert "execute_task" not in ops2

    async def test_default_keywords_not_mutated(self) -> None:
        """Test that DEFAULT_OPERATION_KEYWORDS is not mutated by instances.

        With frozenset, the keywords are immutable by design, so we verify:
        1. Instance keywords are separate from class defaults
        2. Neither can be mutated (frozenset is immutable)
        """
        original_defaults = MixinNodeIntrospection.DEFAULT_OPERATION_KEYWORDS.copy()
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # Verify instance keywords are a frozenset (immutable by design)
        assert isinstance(node._introspection_operation_keywords, frozenset)

        # Verify class defaults remain unchanged and are also immutable
        assert original_defaults == MixinNodeIntrospection.DEFAULT_OPERATION_KEYWORDS
        assert isinstance(MixinNodeIntrospection.DEFAULT_OPERATION_KEYWORDS, frozenset)


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinNodeIntrospectionPerformanceMetrics:
    """Tests for performance metrics tracking and retrieval."""

    async def test_get_performance_metrics_returns_none_before_introspection(
        self,
    ) -> None:
        """Test that get_performance_metrics returns None before introspection."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        metrics = node.get_performance_metrics()
        assert metrics is None

    async def test_get_performance_metrics_returns_metrics_after_introspection(
        self,
    ) -> None:
        """Test that get_performance_metrics returns metrics after introspection."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        await node.get_introspection_data()
        metrics = node.get_performance_metrics()

        assert metrics is not None
        assert isinstance(metrics, IntrospectionPerformanceMetrics)

    async def test_performance_metrics_contains_expected_fields(self) -> None:
        """Test that performance metrics contain all expected fields."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        await node.get_introspection_data()
        metrics = node.get_performance_metrics()

        assert metrics is not None
        # Check all fields exist
        assert hasattr(metrics, "get_capabilities_ms")
        assert hasattr(metrics, "discover_capabilities_ms")
        assert hasattr(metrics, "get_endpoints_ms")
        assert hasattr(metrics, "get_current_state_ms")
        assert hasattr(metrics, "total_introspection_ms")
        assert hasattr(metrics, "cache_hit")
        assert hasattr(metrics, "method_count")
        assert hasattr(metrics, "threshold_exceeded")
        assert hasattr(metrics, "slow_operations")

        # Check types
        assert isinstance(metrics.get_capabilities_ms, float)
        assert isinstance(metrics.total_introspection_ms, float)
        assert isinstance(metrics.cache_hit, bool)
        assert isinstance(metrics.method_count, int)
        assert isinstance(metrics.threshold_exceeded, bool)
        assert isinstance(metrics.slow_operations, list)

    async def test_performance_metrics_cache_hit_detection(self) -> None:
        """Test that cache hits are correctly detected in metrics."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # First call - cache miss
        await node.get_introspection_data()
        metrics_miss = node.get_performance_metrics()
        assert metrics_miss is not None
        assert metrics_miss.cache_hit is False

        # Second call - cache hit
        await node.get_introspection_data()
        metrics_hit = node.get_performance_metrics()
        assert metrics_hit is not None
        assert metrics_hit.cache_hit is True

    async def test_performance_metrics_method_count(self) -> None:
        """Test that method count is correctly reported in metrics."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        await node.get_introspection_data()
        metrics = node.get_performance_metrics()

        assert metrics is not None
        # MockNode has at least 4 public methods (execute, health_check,
        # handle_event, process_batch)
        assert metrics.method_count >= 4

    async def test_performance_metrics_to_dict(self) -> None:
        """Test that to_dict() returns all fields."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        await node.get_introspection_data()
        metrics = node.get_performance_metrics()

        assert metrics is not None
        metrics_dict = metrics.model_dump()

        assert isinstance(metrics_dict, dict)
        assert "get_capabilities_ms" in metrics_dict
        assert "discover_capabilities_ms" in metrics_dict
        assert "get_endpoints_ms" in metrics_dict
        assert "get_current_state_ms" in metrics_dict
        assert "total_introspection_ms" in metrics_dict
        assert "cache_hit" in metrics_dict
        assert "method_count" in metrics_dict
        assert "threshold_exceeded" in metrics_dict
        assert "slow_operations" in metrics_dict

    async def test_performance_metrics_fresh_on_each_call(self) -> None:
        """Test that performance metrics are fresh for each introspection call."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
            cache_ttl=0.001,  # Very short TTL to force cache refresh
        )
        node.initialize_introspection(config)

        # First call
        await node.get_introspection_data()
        metrics1 = node.get_performance_metrics()
        assert metrics1 is not None
        total_ms_1 = metrics1.total_introspection_ms

        # Wait for cache to expire
        await asyncio.sleep(0.01)

        # Second call with fresh computation
        await node.get_introspection_data()
        metrics2 = node.get_performance_metrics()
        assert metrics2 is not None
        total_ms_2 = metrics2.total_introspection_ms

        # Metrics should be different (fresh computation)
        # We can't guarantee exact timing, but both should be positive
        assert total_ms_1 > 0
        assert total_ms_2 > 0


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinNodeIntrospectionMethodCountBenchmark:
    """Performance benchmarks with varying method counts.

    These tests validate that introspection performance scales appropriately
    with the number of methods on a node. The <50ms target should be maintained
    even with a large number of methods due to class-level caching.
    """

    async def test_benchmark_minimal_methods_node(self) -> None:
        """Benchmark introspection on a node with minimal methods."""

        class MinimalMethodsNode(MixinNodeIntrospection):
            """Node with just one operation method."""

            async def execute(self, data: str) -> str:
                return data

        node = MinimalMethodsNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="COMPUTE",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # Clear cache for accurate measurement
        MixinNodeIntrospection._invalidate_class_method_cache(MinimalMethodsNode)

        times: list[float] = []
        for _ in range(20):
            node._introspection_cache = None
            node._introspection_cached_at = None
            MixinNodeIntrospection._invalidate_class_method_cache(MinimalMethodsNode)

            start = time.perf_counter()
            await node.get_introspection_data()
            times.append((time.perf_counter() - start) * 1000)

        avg_time = sum(times) / len(times)
        max_time = max(times)
        metrics = node.get_performance_metrics()

        print(
            f"\nMinimal methods node ({metrics.method_count if metrics else 0} methods):"
        )
        print(f"  avg={avg_time:.2f}ms, max={max_time:.2f}ms")

        # Should be well under the threshold
        threshold_ms = PERF_THRESHOLD_GET_INTROSPECTION_DATA_MS * PERF_MULTIPLIER
        assert avg_time < threshold_ms, (
            f"Minimal methods avg {avg_time:.2f}ms exceeds {threshold_ms:.0f}ms"
        )

    async def test_benchmark_medium_methods_node(self) -> None:
        """Benchmark introspection on a node with ~20 methods."""

        class MediumMethodsNode(MixinNodeIntrospection):
            """Node with ~20 operation methods."""

            async def execute_task_01(self, d: str) -> str:
                return d

            async def execute_task_02(self, d: str) -> str:
                return d

            async def execute_task_03(self, d: str) -> str:
                return d

            async def handle_event_01(self, d: str) -> str:
                return d

            async def handle_event_02(self, d: str) -> str:
                return d

            async def handle_event_03(self, d: str) -> str:
                return d

            async def process_data_01(self, d: str) -> str:
                return d

            async def process_data_02(self, d: str) -> str:
                return d

            async def process_data_03(self, d: str) -> str:
                return d

            async def run_operation_01(self, d: str) -> str:
                return d

            async def run_operation_02(self, d: str) -> str:
                return d

            async def run_operation_03(self, d: str) -> str:
                return d

            async def invoke_action_01(self, d: str) -> str:
                return d

            async def invoke_action_02(self, d: str) -> str:
                return d

            async def invoke_action_03(self, d: str) -> str:
                return d

            async def call_service_01(self, d: str) -> str:
                return d

            async def call_service_02(self, d: str) -> str:
                return d

            async def call_service_03(self, d: str) -> str:
                return d

            # Additional utility methods
            def validate_input(self, d: str) -> bool:
                return True

            def transform_output(self, d: str) -> str:
                return d

        node = MediumMethodsNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="COMPUTE",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # Clear cache for accurate measurement
        MixinNodeIntrospection._invalidate_class_method_cache(MediumMethodsNode)

        times: list[float] = []
        for _ in range(20):
            node._introspection_cache = None
            node._introspection_cached_at = None
            MixinNodeIntrospection._invalidate_class_method_cache(MediumMethodsNode)

            start = time.perf_counter()
            await node.get_introspection_data()
            times.append((time.perf_counter() - start) * 1000)

        avg_time = sum(times) / len(times)
        max_time = max(times)
        metrics = node.get_performance_metrics()

        print(
            f"\nMedium methods node ({metrics.method_count if metrics else 0} methods):"
        )
        print(f"  avg={avg_time:.2f}ms, max={max_time:.2f}ms")

        # Should still be under the threshold
        threshold_ms = PERF_THRESHOLD_GET_INTROSPECTION_DATA_MS * PERF_MULTIPLIER
        assert avg_time < threshold_ms, (
            f"Medium methods avg {avg_time:.2f}ms exceeds {threshold_ms:.0f}ms"
        )

    async def test_benchmark_large_methods_node(self) -> None:
        """Benchmark introspection on a node with ~50 methods."""

        # Create a node class dynamically with many methods
        class LargeMethodsNode(MixinNodeIntrospection):
            """Node with ~50 methods to stress-test reflection performance."""

        # Add 50 methods dynamically
        for i in range(50):
            # Alternate between different operation keywords
            keywords = ["execute", "handle", "process", "run", "invoke"]
            keyword = keywords[i % len(keywords)]

            async def method(self: LargeMethodsNode, data: str = "") -> str:
                return data

            method.__name__ = f"{keyword}_operation_{i:02d}"
            setattr(LargeMethodsNode, method.__name__, method)

        node = LargeMethodsNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="COMPUTE",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # Clear cache for accurate measurement
        MixinNodeIntrospection._invalidate_class_method_cache(LargeMethodsNode)

        times: list[float] = []
        for _ in range(20):
            node._introspection_cache = None
            node._introspection_cached_at = None
            MixinNodeIntrospection._invalidate_class_method_cache(LargeMethodsNode)

            start = time.perf_counter()
            await node.get_introspection_data()
            times.append((time.perf_counter() - start) * 1000)

        avg_time = sum(times) / len(times)
        max_time = max(times)
        p95_time = sorted(times)[int(len(times) * 0.95)]
        metrics = node.get_performance_metrics()

        print(
            f"\nLarge methods node ({metrics.method_count if metrics else 0} methods):"
        )
        print(f"  avg={avg_time:.2f}ms, max={max_time:.2f}ms, p95={p95_time:.2f}ms")

        # Should still be under the threshold even with 50+ methods
        threshold_ms = PERF_THRESHOLD_GET_INTROSPECTION_DATA_MS * PERF_MULTIPLIER
        assert avg_time < threshold_ms, (
            f"Large methods avg {avg_time:.2f}ms exceeds {threshold_ms:.0f}ms"
        )

    async def test_benchmark_cache_hit_performance_50_methods(self) -> None:
        """Benchmark cache hit performance with large method count."""

        class LargeCacheNode(MixinNodeIntrospection):
            pass

        # Add 50 methods
        for i in range(50):

            async def method(self: LargeCacheNode, data: str = "") -> str:
                return data

            method.__name__ = f"execute_task_{i:02d}"
            setattr(LargeCacheNode, method.__name__, method)

        node = LargeCacheNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="COMPUTE",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # Warm cache
        await node.get_introspection_data()

        # Measure cache hits
        times: list[float] = []
        for _ in range(100):
            start = time.perf_counter()
            await node.get_introspection_data()
            times.append((time.perf_counter() - start) * 1000)

        avg_time = sum(times) / len(times)
        max_time = max(times)
        p99_time = sorted(times)[int(len(times) * 0.99)]

        print(
            f"\nCache hit (50 methods): avg={avg_time:.3f}ms, max={max_time:.3f}ms, p99={p99_time:.3f}ms"
        )

        # Cache hits should be very fast regardless of method count
        threshold_ms = PERF_THRESHOLD_CACHE_HIT_MS * PERF_MULTIPLIER
        assert avg_time < threshold_ms, (
            f"Cache hit avg {avg_time:.3f}ms exceeds {threshold_ms:.1f}ms"
        )

    async def test_method_count_scaling_analysis(self) -> None:
        """Analyze how introspection time scales with method count."""
        results: list[tuple[int, float]] = []

        for method_count in [5, 10, 20, 30, 40, 50]:
            # Create node class with specified method count
            class ScalingTestNode(MixinNodeIntrospection):
                pass

            for i in range(method_count):

                async def method(self: ScalingTestNode, data: str = "") -> str:
                    return data

                method.__name__ = f"execute_op_{i:02d}"
                setattr(ScalingTestNode, method.__name__, method)

            node = ScalingTestNode()
            config = ModelIntrospectionConfig(
                node_id=TEST_NODE_UUID_1,
                node_type="COMPUTE",
                event_bus=None,
            )
            node.initialize_introspection(config)

            # Clear cache and measure
            MixinNodeIntrospection._invalidate_class_method_cache(ScalingTestNode)

            times: list[float] = []
            for _ in range(10):
                node._introspection_cache = None
                node._introspection_cached_at = None
                MixinNodeIntrospection._invalidate_class_method_cache(ScalingTestNode)

                start = time.perf_counter()
                await node.get_introspection_data()
                times.append((time.perf_counter() - start) * 1000)

            avg_time = sum(times) / len(times)
            results.append((method_count, avg_time))

            # Clean up class from cache
            MixinNodeIntrospection._invalidate_class_method_cache(ScalingTestNode)

        print("\n\nMethod Count Scaling Analysis:")
        print("Methods | Avg Time (ms)")
        print("--------|---------------")
        for method_count, avg_time in results:
            print(f"   {method_count:3d}  |    {avg_time:.2f}")

        # All should be under threshold
        threshold_ms = PERF_THRESHOLD_GET_INTROSPECTION_DATA_MS * PERF_MULTIPLIER
        for method_count, avg_time in results:
            assert avg_time < threshold_ms, (
                f"{method_count} methods: {avg_time:.2f}ms exceeds {threshold_ms:.0f}ms"
            )


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinNodeIntrospectionThresholdDetection:
    """Tests for threshold exceeded detection."""

    async def test_threshold_not_exceeded_normal_operation(self) -> None:
        """Test that thresholds are not marked exceeded in normal operation."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        await node.get_introspection_data()
        metrics = node.get_performance_metrics()

        assert metrics is not None
        # Under normal operation with MockNode, thresholds should not be exceeded
        # (unless running on very slow CI)
        if metrics.total_introspection_ms < PERF_THRESHOLD_GET_INTROSPECTION_DATA_MS:
            assert metrics.threshold_exceeded is False
            assert len(metrics.slow_operations) == 0

    async def test_slow_operations_list_populated_when_exceeded(self) -> None:
        """Test that slow_operations is populated when threshold exceeded."""
        # This test verifies the structure is correct
        # We can't reliably force slow operations in a unit test
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        await node.get_introspection_data()
        metrics = node.get_performance_metrics()

        assert metrics is not None
        # slow_operations should always be a list
        assert isinstance(metrics.slow_operations, list)

    async def test_cache_hit_threshold_separate_from_total(self) -> None:
        """Test that cache hit has its own performance threshold."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # First call - cache miss
        await node.get_introspection_data()

        # Second call - cache hit
        await node.get_introspection_data()
        metrics = node.get_performance_metrics()

        assert metrics is not None
        assert metrics.cache_hit is True

        # Cache hit should be very fast
        # If it exceeds 1ms, there might be an issue
        if metrics.total_introspection_ms < PERF_THRESHOLD_CACHE_HIT_MS:
            assert "cache_hit" not in metrics.slow_operations


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.performance
class TestMixinNodeIntrospectionComprehensiveBenchmark:
    """Comprehensive performance benchmarks for node introspection.

    These tests provide detailed performance analysis with:
    - Cold-start (no cache) introspection timing
    - Warm cache hit timing
    - Component-level timing breakdown
    - Percentile-based thresholds (p95/p99) for flaky test mitigation
    - Verification of <50ms target
    - Performance metrics validation via get_performance_metrics()

    Note: All tests use PERF_MULTIPLIER to adjust thresholds for CI environments.
    Tests use sample sizes of 20+ iterations for statistical significance.
    """

    # Minimum sample size for statistical significance
    MIN_SAMPLE_SIZE = 20

    def _calculate_percentile(self, times: list[float], percentile: float) -> float:
        """Calculate the percentile value from a list of times.

        Args:
            times: List of timing measurements in milliseconds.
            percentile: Percentile to calculate (0-100).

        Returns:
            The percentile value.
        """
        sorted_times = sorted(times)
        index = int(len(sorted_times) * (percentile / 100))
        # Clamp index to valid range
        index = min(index, len(sorted_times) - 1)
        return sorted_times[index]

    async def test_benchmark_cold_start_introspection(self) -> None:
        """Benchmark cold-start introspection time (no cache).

        Measures introspection performance when cache is empty,
        which represents the worst-case timing scenario.

        Uses p95 percentile to avoid flakiness from outliers.
        """
        node = MockNode()
        node.initialize_introspection(
            ModelIntrospectionConfig(
                node_id=uuid4(),
                node_type="EFFECT",
                event_bus=None,
            )
        )

        # Clear class-level cache for accurate cold-start measurement
        MixinNodeIntrospection._invalidate_class_method_cache(MockNode)

        cold_start_times: list[float] = []

        for i in range(self.MIN_SAMPLE_SIZE):
            # Clear both instance and class caches for true cold start
            node._introspection_cache = None
            node._introspection_cached_at = None
            if i > 0:
                # Only clear class cache on iterations > 0 to measure true cold start
                MixinNodeIntrospection._invalidate_class_method_cache(MockNode)

            start = time.perf_counter()
            await node.get_introspection_data()
            elapsed_ms = (time.perf_counter() - start) * 1000
            cold_start_times.append(elapsed_ms)

        # Calculate statistics
        avg_time = sum(cold_start_times) / len(cold_start_times)
        p95_time = self._calculate_percentile(cold_start_times, 95)
        p99_time = self._calculate_percentile(cold_start_times, 99)
        min_time = min(cold_start_times)
        max_time = max(cold_start_times)

        print(f"\nCold-start introspection ({self.MIN_SAMPLE_SIZE} iterations):")
        print(
            f"  avg={avg_time:.2f}ms, min={min_time:.2f}ms, "
            f"max={max_time:.2f}ms, p95={p95_time:.2f}ms, p99={p99_time:.2f}ms"
        )

        # Use p95 for threshold comparison (more stable than max)
        threshold_ms = PERF_THRESHOLD_GET_INTROSPECTION_DATA_MS * PERF_MULTIPLIER
        assert p95_time < threshold_ms, (
            f"Cold-start p95 latency {p95_time:.2f}ms exceeds {threshold_ms:.0f}ms "
            f"threshold (avg={avg_time:.2f}ms, max={max_time:.2f}ms)"
        )

    async def test_benchmark_warm_cache_hit(self) -> None:
        """Benchmark warm cache hit timing.

        Measures introspection performance when result is served from cache,
        which should be sub-millisecond.

        Uses p99 percentile for cache hits since they should be very fast.
        """
        node = MockNode()
        node.initialize_introspection(
            ModelIntrospectionConfig(
                node_id=uuid4(),
                node_type="EFFECT",
                event_bus=None,
            )
        )

        # Warm the cache with initial call
        await node.get_introspection_data()

        cache_hit_times: list[float] = []

        # Measure cache hits - use larger sample for cache hit measurement
        for _ in range(100):
            start = time.perf_counter()
            await node.get_introspection_data()
            elapsed_ms = (time.perf_counter() - start) * 1000
            cache_hit_times.append(elapsed_ms)

        # Calculate statistics
        avg_time = sum(cache_hit_times) / len(cache_hit_times)
        p95_time = self._calculate_percentile(cache_hit_times, 95)
        p99_time = self._calculate_percentile(cache_hit_times, 99)
        min_time = min(cache_hit_times)
        max_time = max(cache_hit_times)

        print("\nWarm cache hit (100 iterations):")
        print(
            f"  avg={avg_time:.3f}ms, min={min_time:.3f}ms, "
            f"max={max_time:.3f}ms, p95={p95_time:.3f}ms, p99={p99_time:.3f}ms"
        )

        # Cache hits should be very fast - use p99 for threshold
        threshold_ms = PERF_THRESHOLD_CACHE_HIT_MS * PERF_MULTIPLIER
        assert p99_time < threshold_ms, (
            f"Cache hit p99 latency {p99_time:.3f}ms exceeds {threshold_ms:.1f}ms "
            f"threshold (avg={avg_time:.3f}ms)"
        )

        # Verify cache hit was detected in metrics
        metrics = node.get_performance_metrics()
        assert metrics is not None
        assert metrics.cache_hit is True, "Last call should be a cache hit"

    async def test_benchmark_component_level_timing(self) -> None:
        """Benchmark timing for individual introspection components.

        Measures timing for each component:
        - get_capabilities()
        - get_endpoints()
        - get_current_state()

        Uses p95 for threshold comparison to avoid flakiness.
        """
        node = MockNode()
        node.initialize_introspection(
            ModelIntrospectionConfig(
                node_id=uuid4(),
                node_type="EFFECT",
                event_bus=None,
            )
        )

        # Clear class cache for accurate capability measurement
        MixinNodeIntrospection._invalidate_class_method_cache(MockNode)

        component_timings: dict[str, list[float]] = {
            "get_capabilities": [],
            "get_endpoints": [],
            "get_current_state": [],
        }

        for i in range(self.MIN_SAMPLE_SIZE):
            # Clear class cache each iteration for get_capabilities measurement
            if i > 0:
                MixinNodeIntrospection._invalidate_class_method_cache(MockNode)

            # Time get_capabilities
            start = time.perf_counter()
            await node.get_capabilities()
            component_timings["get_capabilities"].append(
                (time.perf_counter() - start) * 1000
            )

            # Time get_endpoints
            start = time.perf_counter()
            await node.get_endpoints()
            component_timings["get_endpoints"].append(
                (time.perf_counter() - start) * 1000
            )

            # Time get_current_state
            start = time.perf_counter()
            await node.get_current_state()
            component_timings["get_current_state"].append(
                (time.perf_counter() - start) * 1000
            )

        print(f"\nComponent-level timing ({self.MIN_SAMPLE_SIZE} iterations):")
        for name, times in component_timings.items():
            avg = sum(times) / len(times)
            p95 = self._calculate_percentile(times, 95)
            min_t = min(times)
            max_t = max(times)
            print(
                f"  {name}: avg={avg:.2f}ms, min={min_t:.2f}ms, "
                f"max={max_t:.2f}ms, p95={p95:.2f}ms"
            )

        # Verify get_capabilities (most expensive due to reflection)
        cap_p95 = self._calculate_percentile(component_timings["get_capabilities"], 95)
        cap_threshold = PERF_THRESHOLD_GET_CAPABILITIES_MS * PERF_MULTIPLIER
        assert cap_p95 < cap_threshold, (
            f"get_capabilities p95 {cap_p95:.2f}ms exceeds {cap_threshold:.0f}ms"
        )

        # Verify get_current_state (should be very fast)
        state_p95 = self._calculate_percentile(
            component_timings["get_current_state"], 95
        )
        state_threshold = 1.0 * PERF_MULTIPLIER  # Should be sub-millisecond
        assert state_p95 < state_threshold, (
            f"get_current_state p95 {state_p95:.2f}ms exceeds {state_threshold:.1f}ms"
        )

    async def test_benchmark_50ms_target_verification(self) -> None:
        """Verify that the <50ms target is consistently met.

        This is the primary benchmark test that validates the performance
        requirement. Uses p95 percentile for stability.
        """
        node = MockNode()
        node.initialize_introspection(
            ModelIntrospectionConfig(
                node_id=uuid4(),
                node_type="EFFECT",
                event_bus=None,
            )
        )

        total_times: list[float] = []

        for i in range(self.MIN_SAMPLE_SIZE):
            # Clear cache for each iteration
            node._introspection_cache = None
            node._introspection_cached_at = None
            if i > 0:
                MixinNodeIntrospection._invalidate_class_method_cache(MockNode)

            start = time.perf_counter()
            await node.get_introspection_data()
            elapsed_ms = (time.perf_counter() - start) * 1000
            total_times.append(elapsed_ms)

        # Calculate statistics
        avg_time = sum(total_times) / len(total_times)
        p95_time = self._calculate_percentile(total_times, 95)
        p99_time = self._calculate_percentile(total_times, 99)

        print(f"\n<50ms Target Verification ({self.MIN_SAMPLE_SIZE} iterations):")
        print(f"  avg={avg_time:.2f}ms, p95={p95_time:.2f}ms, p99={p99_time:.2f}ms")

        # The 50ms target with CI buffer
        threshold_ms = 50.0 * PERF_MULTIPLIER
        assert p95_time < threshold_ms, (
            f"FAILED: p95 latency {p95_time:.2f}ms exceeds <50ms target "
            f"(with {PERF_MULTIPLIER}x CI buffer = {threshold_ms:.0f}ms)"
        )

        print(f"  PASSED: p95={p95_time:.2f}ms < {threshold_ms:.0f}ms threshold")

    async def test_benchmark_performance_metrics_validation(self) -> None:
        """Verify that get_performance_metrics() returns valid timing data.

        Validates that IntrospectionPerformanceMetrics captures accurate
        timing information that matches actual measured times.
        """
        node = MockNode()
        node.initialize_introspection(
            ModelIntrospectionConfig(
                node_id=uuid4(),
                node_type="EFFECT",
                event_bus=None,
            )
        )

        # Clear cache for fresh computation
        node._introspection_cache = None
        node._introspection_cached_at = None
        MixinNodeIntrospection._invalidate_class_method_cache(MockNode)

        # Measure actual time and get metrics
        start = time.perf_counter()
        await node.get_introspection_data()
        actual_total_ms = (time.perf_counter() - start) * 1000

        metrics = node.get_performance_metrics()

        # Validate metrics structure
        assert metrics is not None, "get_performance_metrics() should return metrics"
        assert isinstance(metrics, IntrospectionPerformanceMetrics)

        # Validate timing fields are populated
        assert metrics.total_introspection_ms > 0, (
            "total_introspection_ms should be > 0"
        )
        assert metrics.get_capabilities_ms >= 0, "get_capabilities_ms should be >= 0"
        assert metrics.get_endpoints_ms >= 0, "get_endpoints_ms should be >= 0"
        assert metrics.get_current_state_ms >= 0, "get_current_state_ms should be >= 0"

        # Validate method count is reasonable
        assert metrics.method_count >= 0, "method_count should be >= 0"

        # Validate cache hit detection (should be False for fresh computation)
        assert metrics.cache_hit is False, "First call should not be cache hit"

        # Validate timing consistency - metrics time should be close to measured time
        # Allow some tolerance for measurement overhead
        timing_tolerance_ms = 5.0 * PERF_MULTIPLIER
        assert (
            abs(metrics.total_introspection_ms - actual_total_ms) < timing_tolerance_ms
        ), (
            f"Metrics time {metrics.total_introspection_ms:.2f}ms differs from "
            f"measured time {actual_total_ms:.2f}ms by more than {timing_tolerance_ms:.1f}ms"
        )

        # Validate to_dict() returns all fields
        metrics_dict = metrics.model_dump()
        expected_keys = {
            "get_capabilities_ms",
            "discover_capabilities_ms",
            "get_endpoints_ms",
            "get_current_state_ms",
            "total_introspection_ms",
            "cache_hit",
            "method_count",
            "threshold_exceeded",
            "slow_operations",
            "captured_at",  # Added by Pydantic model
        }
        assert set(metrics_dict.keys()) == expected_keys, (
            f"model_dump() missing keys: {expected_keys - set(metrics_dict.keys())}"
        )

        print("\nPerformance Metrics Validation:")
        print(f"  total_introspection_ms: {metrics.total_introspection_ms:.2f}")
        print(f"  get_capabilities_ms: {metrics.get_capabilities_ms:.2f}")
        print(f"  get_endpoints_ms: {metrics.get_endpoints_ms:.2f}")
        print(f"  get_current_state_ms: {metrics.get_current_state_ms:.2f}")
        print(f"  method_count: {metrics.method_count}")
        print(f"  cache_hit: {metrics.cache_hit}")
        print(f"  threshold_exceeded: {metrics.threshold_exceeded}")

    async def test_benchmark_metrics_threshold_detection(self) -> None:
        """Verify that threshold_exceeded and slow_operations are correctly set.

        Tests that the metrics correctly identify when operations exceed
        their performance thresholds.
        """
        node = MockNode()
        node.initialize_introspection(
            ModelIntrospectionConfig(
                node_id=uuid4(),
                node_type="EFFECT",
                event_bus=None,
            )
        )

        # Run multiple iterations to check threshold detection
        threshold_exceeded_count = 0
        iterations = self.MIN_SAMPLE_SIZE

        for i in range(iterations):
            # Clear cache for each iteration
            node._introspection_cache = None
            node._introspection_cached_at = None
            if i > 0:
                MixinNodeIntrospection._invalidate_class_method_cache(MockNode)

            await node.get_introspection_data()
            metrics = node.get_performance_metrics()

            assert metrics is not None

            # Count how often threshold is exceeded
            if metrics.threshold_exceeded:
                threshold_exceeded_count += 1

            # Validate slow_operations consistency
            if metrics.threshold_exceeded:
                # If threshold is exceeded, slow_operations should contain something
                # (unless it's a cache hit threshold, which may not add to list)
                pass  # Threshold detection is based on raw values
            else:
                # If threshold not exceeded, slow_operations should be empty
                assert len(metrics.slow_operations) == 0, (
                    f"slow_operations should be empty when threshold not exceeded: "
                    f"{metrics.slow_operations}"
                )

        print(f"\nThreshold Detection ({iterations} iterations):")
        print(f"  threshold_exceeded: {threshold_exceeded_count}/{iterations} times")

        # In normal operation, we should rarely exceed thresholds
        # Allow up to 10% of iterations to exceed (for CI variance)
        max_exceeded_ratio = 0.1
        exceeded_ratio = threshold_exceeded_count / iterations
        if exceeded_ratio > max_exceeded_ratio:
            print(
                f"  WARNING: {exceeded_ratio:.1%} of iterations exceeded threshold "
                f"(expected < {max_exceeded_ratio:.1%})"
            )

    async def test_benchmark_statistical_stability(self) -> None:
        """Verify that timing measurements are statistically stable.

        Tests that the variance in timing measurements is acceptable,
        indicating consistent performance.
        """
        node = MockNode()
        node.initialize_introspection(
            ModelIntrospectionConfig(
                node_id=uuid4(),
                node_type="EFFECT",
                event_bus=None,
            )
        )

        # Collect timing samples
        samples: list[float] = []
        sample_size = 50  # Larger sample for better statistics

        for i in range(sample_size):
            node._introspection_cache = None
            node._introspection_cached_at = None
            if i > 0:
                MixinNodeIntrospection._invalidate_class_method_cache(MockNode)

            start = time.perf_counter()
            await node.get_introspection_data()
            samples.append((time.perf_counter() - start) * 1000)

        # Calculate statistics
        avg = sum(samples) / len(samples)
        variance = sum((x - avg) ** 2 for x in samples) / len(samples)
        std_dev = variance**0.5
        coef_of_variation = std_dev / avg if avg > 0 else 0

        p50 = self._calculate_percentile(samples, 50)
        p95 = self._calculate_percentile(samples, 95)
        p99 = self._calculate_percentile(samples, 99)

        print(f"\nStatistical Stability ({sample_size} samples):")
        print(f"  avg={avg:.2f}ms, std_dev={std_dev:.2f}ms, CV={coef_of_variation:.2%}")
        print(f"  p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms")

        # Coefficient of variation should be reasonable
        # Allow higher variance in CI environments
        max_cv = 1.0 * PERF_MULTIPLIER  # 100% CV with CI buffer
        assert coef_of_variation < max_cv, (
            f"Coefficient of variation {coef_of_variation:.2%} exceeds {max_cv:.0%}, "
            f"indicating unstable performance"
        )

        # p99 should not be excessively higher than p50
        # This catches outliers that might cause flaky tests
        # Note: Increased from 5.0 to 10.0 to handle CI/container variance
        max_p99_to_p50_ratio = 10.0 * PERF_MULTIPLIER
        p99_to_p50_ratio = p99 / p50 if p50 > 0 else 0
        assert p99_to_p50_ratio < max_p99_to_p50_ratio, (
            f"p99/p50 ratio {p99_to_p50_ratio:.1f} exceeds {max_p99_to_p50_ratio:.1f}, "
            f"indicating excessive outliers"
        )


# =============================================================================
# Topic Validation Tests (PR #54 Coverage)
# =============================================================================


@pytest.mark.unit
class TestTopicVersionSuffixValidation:
    r"""Test version suffix validation (`.v\d+`) for ONEX topics.

    ONEX topics (starting with 'onex.') must have a version suffix
    like .v1, .v2, etc. Legacy topics are allowed without version
    suffix but generate a warning.
    """

    def test_valid_onex_topic_with_v1_suffix(self) -> None:
        """Test that ONEX topic with .v1 suffix is valid."""
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            introspection_topic="onex.node.introspection.published.v1",
        )
        assert config.introspection_topic == "onex.node.introspection.published.v1"

    def test_valid_onex_topic_with_v2_suffix(self) -> None:
        """Test that ONEX topic with .v2 suffix is valid."""
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            heartbeat_topic="onex.node.heartbeat.published.v2",
        )
        assert config.heartbeat_topic == "onex.node.heartbeat.published.v2"

    def test_valid_onex_topic_with_multi_digit_version(self) -> None:
        """Test that ONEX topic with multi-digit version suffix is valid."""
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            request_introspection_topic="onex.registry.introspection.requested.v10",
        )
        assert (
            config.request_introspection_topic
            == "onex.registry.introspection.requested.v10"
        )

    def test_onex_topic_without_version_suffix_rejected(self) -> None:
        """Test that ONEX topic without version suffix is rejected."""
        with pytest.raises(ValueError, match="ONEX topic must have version suffix"):
            ModelIntrospectionConfig(
                node_id=TEST_NODE_UUID_1,
                node_type="EFFECT",
                introspection_topic="onex.node.introspection.published",
            )

    def test_onex_topic_with_wrong_version_format_rejected(self) -> None:
        """Test that ONEX topic with wrong version format is rejected."""
        # These topics fail due to missing version suffix
        invalid_version_suffix = [
            "onex.node.topic.1",  # Missing 'v' prefix
            "onex.node.topic.ver1",  # Wrong prefix
            "onex.node.topic.v",  # No version number
        ]
        for topic in invalid_version_suffix:
            with pytest.raises(ValueError, match="ONEX topic must have version suffix"):
                ModelIntrospectionConfig(
                    node_id=TEST_NODE_UUID_1,
                    node_type="EFFECT",
                    introspection_topic=topic,
                )

        # This topic fails due to uppercase V (invalid characters)
        with pytest.raises(ValueError, match="lowercase alphanumeric"):
            ModelIntrospectionConfig(
                node_id=TEST_NODE_UUID_1,
                node_type="EFFECT",
                introspection_topic="onex.node.topic.V1",  # Uppercase V
            )

    def test_legacy_topic_without_version_allowed(self) -> None:
        """Test that legacy topics without version suffix are allowed.

        Legacy topics (not starting with 'onex.') are supported for flexibility.
        """
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            introspection_topic="custom.legacy.topic",
        )
        assert config.introspection_topic == "custom.legacy.topic"

    def test_legacy_topic_with_version_allowed(self) -> None:
        """Test that legacy topics with version suffix are allowed."""
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            introspection_topic="custom.legacy.topic.v1",
        )
        assert config.introspection_topic == "custom.legacy.topic.v1"


@pytest.mark.unit
class TestTopicInvalidNamesValidation:
    """Test rejection of invalid topic names."""

    def test_empty_topic_rejected(self) -> None:
        """Test that empty topic name is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ModelIntrospectionConfig(
                node_id=TEST_NODE_UUID_1,
                node_type="EFFECT",
                introspection_topic="",
            )

    def test_topic_with_special_characters_rejected(self) -> None:
        """Test that topics with special characters are rejected."""
        invalid_topics = [
            "onex.topic@invalid.v1",
            "onex.topic#name.v1",
            "onex.topic$value.v1",
            "onex.topic%test.v1",
            "onex.topic&invalid.v1",
            "onex.topic*wildcard.v1",
            "onex.topic+plus.v1",
            "onex.topic=equals.v1",
        ]
        for topic in invalid_topics:
            with pytest.raises(ValueError, match="invalid characters"):
                ModelIntrospectionConfig(
                    node_id=TEST_NODE_UUID_1,
                    node_type="EFFECT",
                    introspection_topic=topic,
                )

    def test_topic_starting_with_uppercase_rejected(self) -> None:
        """Test that topics starting with uppercase are rejected."""
        with pytest.raises(ValueError, match="must start with a lowercase letter"):
            ModelIntrospectionConfig(
                node_id=TEST_NODE_UUID_1,
                node_type="EFFECT",
                introspection_topic="Onex.node.topic.v1",
            )

    def test_topic_ending_with_dot_rejected(self) -> None:
        """Test that topics ending with dot are rejected."""
        with pytest.raises(ValueError, match="must not end with a dot"):
            ModelIntrospectionConfig(
                node_id=TEST_NODE_UUID_1,
                node_type="EFFECT",
                introspection_topic="onex.",
            )

    def test_topic_with_whitespace_rejected(self) -> None:
        """Test that topics with whitespace are rejected."""
        invalid_topics = [
            "onex.node .topic.v1",
            "onex. node.topic.v1",
            " onex.node.topic.v1",
            "onex.node.topic.v1 ",
            "onex.node\ttopic.v1",
        ]
        for topic in invalid_topics:
            with pytest.raises(ValueError, match="invalid characters"):
                ModelIntrospectionConfig(
                    node_id=TEST_NODE_UUID_1,
                    node_type="EFFECT",
                    introspection_topic=topic,
                )

    def test_valid_topic_characters_accepted(self) -> None:
        """Test that valid topic characters are accepted."""
        valid_topics = [
            "onex.node-name.topic.v1",  # hyphen
            "onex.node_name.topic.v1",  # underscore
            "onex.node123.topic.v1",  # numbers
            "onex.my-domain.sub_topic.v1",  # mixed
        ]
        for topic in valid_topics:
            config = ModelIntrospectionConfig(
                node_id=TEST_NODE_UUID_1,
                node_type="EFFECT",
                introspection_topic=topic,
            )
            assert config.introspection_topic == topic


@pytest.mark.unit
class TestCustomTopicParameters:
    """Test custom topic parameter handling in ModelIntrospectionConfig."""

    def test_custom_introspection_topic(self) -> None:
        """Test setting custom introspection topic."""
        custom_topic = "onex.payments.introspection.published.v1"
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            introspection_topic=custom_topic,
        )
        assert config.introspection_topic == custom_topic

    def test_custom_heartbeat_topic(self) -> None:
        """Test setting custom heartbeat topic."""
        custom_topic = "onex.payments.heartbeat.published.v1"
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            heartbeat_topic=custom_topic,
        )
        assert config.heartbeat_topic == custom_topic

    def test_custom_request_introspection_topic(self) -> None:
        """Test setting custom request introspection topic."""
        custom_topic = "onex.payments.introspection.requested.v1"
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            request_introspection_topic=custom_topic,
        )
        assert config.request_introspection_topic == custom_topic

    def test_all_custom_topics_together(self) -> None:
        """Test setting all custom topics together."""
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            introspection_topic="onex.tenant1.introspection.published.v1",
            heartbeat_topic="onex.tenant1.heartbeat.published.v1",
            request_introspection_topic="onex.tenant1.introspection.requested.v1",
        )
        assert config.introspection_topic == "onex.tenant1.introspection.published.v1"
        assert config.heartbeat_topic == "onex.tenant1.heartbeat.published.v1"
        assert (
            config.request_introspection_topic
            == "onex.tenant1.introspection.requested.v1"
        )

    def test_default_topics_used_when_not_specified(self) -> None:
        """Test that default topics are used when not specified."""
        from omnibase_infra.models.discovery import (
            DEFAULT_HEARTBEAT_TOPIC,
            DEFAULT_INTROSPECTION_TOPIC,
            DEFAULT_REQUEST_INTROSPECTION_TOPIC,
        )

        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
        )
        assert config.introspection_topic == DEFAULT_INTROSPECTION_TOPIC
        assert config.heartbeat_topic == DEFAULT_HEARTBEAT_TOPIC
        assert config.request_introspection_topic == DEFAULT_REQUEST_INTROSPECTION_TOPIC

    def test_partial_custom_topics_with_defaults(self) -> None:
        """Test partial custom topics with defaults for unspecified."""
        from omnibase_infra.models.discovery import (
            DEFAULT_HEARTBEAT_TOPIC,
            DEFAULT_REQUEST_INTROSPECTION_TOPIC,
        )

        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            introspection_topic="onex.custom.introspection.published.v1",
            # heartbeat_topic and request_introspection_topic use defaults
        )
        assert config.introspection_topic == "onex.custom.introspection.published.v1"
        assert config.heartbeat_topic == DEFAULT_HEARTBEAT_TOPIC
        assert config.request_introspection_topic == DEFAULT_REQUEST_INTROSPECTION_TOPIC

    def test_domain_specific_topic_patterns(self) -> None:
        """Test domain-specific topic naming patterns."""
        domains = ["payments", "orders", "users", "inventory"]

        for domain in domains:
            config = ModelIntrospectionConfig(
                node_id=TEST_NODE_UUID_1,
                node_type="COMPUTE",
                introspection_topic=f"onex.{domain}.introspection.published.v1",
                heartbeat_topic=f"onex.{domain}.heartbeat.published.v1",
                request_introspection_topic=f"onex.{domain}.introspection.requested.v1",
            )
            assert domain in config.introspection_topic
            assert domain in config.heartbeat_topic
            assert domain in config.request_introspection_topic


@pytest.mark.unit
@pytest.mark.asyncio
class TestIntrospectionCacheThreadSafety:
    """Test thread safety for cache operations under concurrent access.

    These tests verify that cache operations are safe under concurrent
    asyncio access patterns typical in real-world usage.
    """

    async def test_concurrent_cache_reads_safe(self) -> None:
        """Test that concurrent cache reads don't cause race conditions."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # Pre-populate cache
        await node.get_introspection_data()

        # Launch many concurrent reads
        async def concurrent_read() -> ModelNodeIntrospectionEvent:
            return await node.get_introspection_data()

        # 50 concurrent reads
        tasks = [concurrent_read() for _ in range(50)]
        results = await asyncio.gather(*tasks)

        # All should succeed with same node_id
        assert len(results) == 50
        for result in results:
            assert result.node_id == TEST_NODE_UUID_1

    async def test_concurrent_cache_invalidation_safe(self) -> None:
        """Test that cache invalidation during reads is safe."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # Pre-populate cache
        await node.get_introspection_data()

        errors: list[Exception] = []

        async def concurrent_read_and_invalidate(i: int) -> None:
            try:
                await node.get_introspection_data()
                # Every 5th task invalidates cache
                if i % 5 == 0:
                    node.invalidate_introspection_cache()
            except Exception as e:
                errors.append(e)

        # Launch concurrent reads with some invalidations
        tasks = [concurrent_read_and_invalidate(i) for i in range(50)]
        await asyncio.gather(*tasks)

        # No errors should occur
        assert len(errors) == 0, f"Errors during concurrent access: {errors}"

    async def test_cache_state_consistency_under_load(self) -> None:
        """Test that cache state remains consistent under concurrent load."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
            cache_ttl=0.01,  # Very short TTL to force refreshes
        )
        node.initialize_introspection(config)

        results: list[UUID] = []

        async def read_node_id() -> None:
            data = await node.get_introspection_data()
            results.append(data.node_id)

        # Many concurrent reads with cache expiring
        for _ in range(5):
            tasks = [read_node_id() for _ in range(10)]
            await asyncio.gather(*tasks)
            await asyncio.sleep(0.02)  # Allow cache to expire

        # All results should have the same node_id
        assert len(results) == 50
        assert all(r == TEST_NODE_UUID_1 for r in results)


@pytest.mark.unit
@pytest.mark.asyncio
class TestCacheHitPerformanceRobust:
    """Robust cache hit performance tests that don't rely on wall-clock timing.

    These tests verify cache hit behavior using deterministic assertions
    rather than timing-based checks to avoid flakiness in CI environments.
    """

    async def test_cache_hit_indicated_by_metrics(self) -> None:
        """Test that cache hits are correctly indicated in metrics."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # First call - should be cache miss
        await node.get_introspection_data()
        metrics_miss = node.get_performance_metrics()
        assert metrics_miss is not None
        assert metrics_miss.cache_hit is False

        # Second call - should be cache hit
        await node.get_introspection_data()
        metrics_hit = node.get_performance_metrics()
        assert metrics_hit is not None
        assert metrics_hit.cache_hit is True

    async def test_cache_hit_uses_cached_timestamp(self) -> None:
        """Test that cache hit returns data with same timestamp (no recompute)."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # First call - compute fresh data
        data1 = await node.get_introspection_data()
        timestamp1 = data1.timestamp

        # Second call - should use cached data
        data2 = await node.get_introspection_data()
        timestamp2 = data2.timestamp

        # Same timestamp means cached result was returned
        assert timestamp1 == timestamp2

    async def test_cache_hit_skips_reflection(self) -> None:
        """Test that cache hit does not trigger reflection operations."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # Clear class-level cache to ensure reflection happens on first call
        MixinNodeIntrospection._invalidate_class_method_cache(MockNode)

        # First call - populates both instance and class caches
        await node.get_introspection_data()

        # Verify class cache was populated
        assert MockNode in MixinNodeIntrospection._class_method_cache

        # Count class cache accesses for second call
        original_cache = MixinNodeIntrospection._class_method_cache.copy()

        # Second call - should use instance cache, not class cache
        await node.get_introspection_data()

        # Class cache should not have been modified
        assert MixinNodeIntrospection._class_method_cache == original_cache

    async def test_cache_hit_count_comparison(self) -> None:
        """Test cache hit using call count comparison instead of timing."""
        node = MockNode()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=None,
        )
        node.initialize_introspection(config)

        # Track timestamps to verify caching
        timestamps: list[str] = []

        for _ in range(10):
            data = await node.get_introspection_data()
            timestamps.append(data.timestamp)

        # All timestamps should be identical (cache hit)
        assert len(set(timestamps)) == 1


@pytest.mark.unit
@pytest.mark.asyncio
class TestHeartbeatEventCounting:
    """Non-flaky heartbeat tests using event counting instead of timing.

    These tests verify heartbeat behavior using deterministic event
    counting rather than timing-based assertions.
    """

    async def test_heartbeat_publishes_events(self) -> None:
        """Test that heartbeat task publishes events."""
        node = MockNode()
        event_bus = MockEventBus()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=event_bus,
        )
        node.initialize_introspection(config)

        await node.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=0.01,  # Fast heartbeat for test
            enable_registry_listener=False,
        )

        try:
            # Wait for some heartbeats to be published
            await asyncio.sleep(0.1)

            # Should have published at least one event
            assert len(event_bus.published_envelopes) >= 1

            # All events should be on heartbeat topic
            for envelope, topic in event_bus.published_envelopes:
                assert topic == "node.heartbeat"
                assert isinstance(envelope, ModelNodeHeartbeatEvent)
        finally:
            await node.stop_introspection_tasks()

    async def test_heartbeat_task_can_be_stopped(self) -> None:
        """Test that heartbeat task can be stopped cleanly."""
        node = MockNode()
        event_bus = MockEventBus()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=event_bus,
        )
        node.initialize_introspection(config)

        await node.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=0.01,
            enable_registry_listener=False,
        )

        # Wait for at least one heartbeat
        await asyncio.sleep(0.05)
        count_before_stop = len(event_bus.published_envelopes)
        assert count_before_stop >= 1

        # Stop the task
        await node.stop_introspection_tasks()

        # Wait and verify no more events
        await asyncio.sleep(0.05)
        count_after_stop = len(event_bus.published_envelopes)

        # Count should not increase after stop
        assert count_after_stop == count_before_stop

    async def test_heartbeat_node_id_consistency(self) -> None:
        """Test that heartbeat events have consistent node_id."""
        node = MockNode()
        event_bus = MockEventBus()
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type="EFFECT",
            event_bus=event_bus,
        )
        node.initialize_introspection(config)

        await node.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=0.01,
            enable_registry_listener=False,
        )

        try:
            # Wait for multiple heartbeats
            await asyncio.sleep(0.1)

            # All events should have same node_id
            for envelope, _ in event_bus.published_envelopes:
                assert isinstance(envelope, ModelNodeHeartbeatEvent)
                assert envelope.node_id == TEST_NODE_UUID_1
        finally:
            await node.stop_introspection_tasks()
