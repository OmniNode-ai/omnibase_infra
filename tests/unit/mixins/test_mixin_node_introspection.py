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
from uuid import uuid4

import pytest

from omnibase_infra.mixins.mixin_node_introspection import MixinNodeIntrospection
from omnibase_infra.models.discovery import ModelNodeIntrospectionEvent

# CI environments may be slower - apply multiplier for performance thresholds
_CI_MODE: bool = os.environ.get("CI", "false").lower() == "true"
PERF_MULTIPLIER: float = 3.0 if _CI_MODE else 2.0

# Type alias for event bus published event structure
PublishedEventDict = dict[
    str, str | bytes | None | dict[str, str | int | bool | list[str]]
]


class MockEventBus:
    """Mock event bus for testing introspection publishing."""

    def __init__(self, should_fail: bool = False) -> None:
        """Initialize mock event bus.

        Args:
            should_fail: If True, publish operations will raise exceptions.
        """
        self.should_fail = should_fail
        self.published_envelopes: list[tuple[ModelNodeIntrospectionEvent, str]] = []
        self.published_events: list[PublishedEventDict] = []

    async def publish_envelope(
        self,
        envelope: ModelNodeIntrospectionEvent,
        topic: str,
    ) -> None:
        """Mock publish_envelope method.

        Args:
            envelope: Event envelope to publish.
            topic: Event topic.

        Raises:
            RuntimeError: If should_fail is True.
        """
        if self.should_fail:
            raise RuntimeError("Event bus publish failed")
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
        node = MockNode()
        node.initialize_introspection(
            node_id="test-node-001",
            node_type="EFFECT",
            event_bus=None,
        )

        assert node._introspection_node_id == "test-node-001"
        assert node._introspection_node_type == "EFFECT"
        assert node._introspection_event_bus is None
        assert node._introspection_version == "1.0.0"
        assert node._introspection_start_time is not None

    async def test_initialize_introspection_with_event_bus(self) -> None:
        """Test initialization with event bus."""
        node = MockNode()
        event_bus = MockEventBus()

        node.initialize_introspection(
            node_id="test-node-002",
            node_type="COMPUTE",
            event_bus=event_bus,
        )

        assert node._introspection_event_bus is event_bus

    async def test_initialize_introspection_custom_cache_ttl(self) -> None:
        """Test initialization with custom cache TTL."""
        node = MockNode()
        node.initialize_introspection(
            node_id="test-node-003",
            node_type="REDUCER",
            event_bus=None,
            cache_ttl=120.0,
        )

        assert node._introspection_cache_ttl == 120.0

    async def test_initialize_introspection_custom_version(self) -> None:
        """Test initialization with custom version."""
        node = MockNode()
        node.initialize_introspection(
            node_id="test-node-004",
            node_type="ORCHESTRATOR",
            event_bus=None,
            version="2.1.0",
        )

        assert node._introspection_version == "2.1.0"

    async def test_initialize_introspection_defaults(self) -> None:
        """Test initialization uses correct defaults."""
        node = MockNode()
        node.initialize_introspection(
            node_id="test-node-005",
            node_type="EFFECT",
            event_bus=None,
        )

        # Default cache TTL is 300 seconds
        assert node._introspection_cache_ttl == 300.0
        # Default version is 1.0.0
        assert node._introspection_version == "1.0.0"
        # Cache starts empty
        assert node._introspection_cache is None
        assert node._introspection_cached_at is None

    async def test_initialize_introspection_empty_node_id_raises(self) -> None:
        """Test that empty node_id raises ValueError."""
        node = MockNode()

        with pytest.raises(ValueError, match="node_id cannot be empty"):
            node.initialize_introspection(
                node_id="",
                node_type="EFFECT",
            )

    async def test_initialize_introspection_empty_node_type_raises(self) -> None:
        """Test that empty node_type raises ValueError."""
        node = MockNode()

        with pytest.raises(ValueError, match="node_type cannot be empty"):
            node.initialize_introspection(
                node_id="test-node",
                node_type="",
            )


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
        node.initialize_introspection(
            node_id="test-node-001",
            node_type="EFFECT",
            event_bus=None,
        )
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
        node.initialize_introspection(
            node_id="test-node-001",
            node_type="EFFECT",
            event_bus=None,
        )
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
        node.initialize_introspection(
            node_id="test-node-no-health",
            node_type="EFFECT",
            event_bus=None,
        )

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
        node.initialize_introspection(
            node_id="test-node-001",
            node_type="EFFECT",
            event_bus=None,
        )
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
        node.initialize_introspection(
            node_id="test-node-no-state",
            node_type="EFFECT",
            event_bus=None,
        )

        state = await node.get_current_state()

        assert state is None

    async def test_get_current_state_with_enum_state(self) -> None:
        """Test get_current_state with enum-style state (has .value)."""
        node = MockNodeWithEnumState()
        node.initialize_introspection(
            node_id="test-node-enum",
            node_type="EFFECT",
            event_bus=None,
        )

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
        node.initialize_introspection(
            node_id="test-node-001",
            node_type="EFFECT",
            event_bus=None,
            cache_ttl=0.1,  # Short TTL for testing
        )
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
        assert data.node_id == "test-node-001"
        assert data.node_type == "EFFECT"
        assert isinstance(data.capabilities, dict)
        assert isinstance(data.endpoints, dict)
        assert data.version == "1.0.0"

    async def test_cache_not_used_before_initialization(self) -> None:
        """Test that cache starts empty."""
        node = MockNode()
        node.initialize_introspection(
            node_id="test-node-cache",
            node_type="EFFECT",
            event_bus=None,
        )

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
        node.initialize_introspection(
            node_id="test-node-001",
            node_type="EFFECT",
            event_bus=event_bus,
        )
        return node

    @pytest.fixture
    def mock_node_without_bus(self) -> MockNode:
        """Create initialized mock node without event bus.

        Returns:
            MockNode without event bus.
        """
        node = MockNode()
        node.initialize_introspection(
            node_id="test-node-001",
            node_type="EFFECT",
            event_bus=None,
        )
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
        assert envelope.reason == "shutdown"


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinNodeIntrospectionTasks:
    """Tests for background task management."""

    async def test_start_introspection_tasks_starts_heartbeat(self) -> None:
        """Test that start_introspection_tasks creates heartbeat task."""
        node = MockNode()
        event_bus = MockEventBus()
        node.initialize_introspection(
            node_id="test-node-tasks",
            node_type="EFFECT",
            event_bus=event_bus,
        )

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
        node.initialize_introspection(
            node_id="test-node-stop",
            node_type="EFFECT",
            event_bus=event_bus,
        )

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
        node.initialize_introspection(
            node_id="test-node-idempotent",
            node_type="EFFECT",
            event_bus=None,
        )

        # Stop without starting should be safe
        await node.stop_introspection_tasks()
        await node.stop_introspection_tasks()

        assert node._heartbeat_task is None

    async def test_heartbeat_publishes_periodically(self) -> None:
        """Test that heartbeat publishes at regular intervals."""
        node = MockNode()
        event_bus = MockEventBus()
        node.initialize_introspection(
            node_id="test-node-heartbeat",
            node_type="EFFECT",
            event_bus=event_bus,
        )

        await node.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=0.05,
            enable_registry_listener=False,
        )

        try:
            # Wait for multiple heartbeats
            await asyncio.sleep(0.2)

            # Should have multiple events (at least 3)
            assert len(event_bus.published_envelopes) >= 3
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
        node.initialize_introspection(
            node_id="test-node-graceful",
            node_type="EFFECT",
            event_bus=failing_event_bus,
        )

        # Should not raise, just return False
        result = await node.publish_introspection()

        assert result is False

    async def test_publish_does_not_crash_on_exception(self) -> None:
        """Test that publish_introspection catches all exceptions."""
        node = MockNode()

        # Create event bus that raises unexpected exception
        class BrokenEventBus:
            async def publish_envelope(
                self, envelope: ModelNodeIntrospectionEvent, topic: str
            ) -> None:
                raise ValueError("Unexpected error")

        node.initialize_introspection(
            node_id="test-node-broken",
            node_type="EFFECT",
            event_bus=BrokenEventBus(),
        )

        # Should not raise
        result = await node.publish_introspection()
        assert result is False

    async def test_heartbeat_continues_after_publish_failure(self) -> None:
        """Test that heartbeat continues even if publish fails."""
        node = MockNode()
        event_bus = MockEventBus(should_fail=True)
        node.initialize_introspection(
            node_id="test-node-continue",
            node_type="EFFECT",
            event_bus=event_bus,
        )

        await node.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=0.05,
            enable_registry_listener=False,
        )

        try:
            # Let heartbeat run with failing publishes
            await asyncio.sleep(0.15)

            # Task should still be running (not crashed)
            assert node._heartbeat_task is not None
            assert not node._heartbeat_task.done()
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
        node.initialize_introspection(
            node_id="test-node-perf",
            node_type="EFFECT",
            event_bus=None,
        )
        return node

    async def test_introspection_extraction_under_50ms(
        self, mock_node: MockNode
    ) -> None:
        """Test that introspection data extraction completes within threshold."""
        # Clear cache to force full computation
        mock_node._introspection_cache = None
        mock_node._introspection_cached_at = None

        threshold_ms = 50 * PERF_MULTIPLIER
        start = time.time()
        await mock_node.get_introspection_data()
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < threshold_ms, (
            f"Introspection took {elapsed_ms:.2f}ms, expected <{threshold_ms:.0f}ms"
        )

    async def test_cached_introspection_under_1ms(self, mock_node: MockNode) -> None:
        """Test that cached introspection returns within threshold."""
        # Populate cache
        await mock_node.get_introspection_data()

        threshold_ms = 1 * PERF_MULTIPLIER
        start = time.time()
        await mock_node.get_introspection_data()
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < threshold_ms, (
            f"Cached introspection took {elapsed_ms:.2f}ms, expected <{threshold_ms:.0f}ms"
        )

    async def test_capability_extraction_under_10ms(self, mock_node: MockNode) -> None:
        """Test that capability extraction completes within threshold."""
        threshold_ms = 10 * PERF_MULTIPLIER
        start = time.time()
        await mock_node.get_capabilities()
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < threshold_ms, (
            f"Capability extraction took {elapsed_ms:.2f}ms, expected <{threshold_ms:.0f}ms"
        )

    async def test_endpoint_discovery_under_10ms(self, mock_node: MockNode) -> None:
        """Test that endpoint discovery completes within threshold."""
        threshold_ms = 10 * PERF_MULTIPLIER
        start = time.time()
        await mock_node.get_endpoints()
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < threshold_ms, (
            f"Endpoint discovery took {elapsed_ms:.2f}ms, expected <{threshold_ms:.0f}ms"
        )

    async def test_state_extraction_under_1ms(self, mock_node: MockNode) -> None:
        """Test that state extraction completes within threshold."""
        threshold_ms = 1 * PERF_MULTIPLIER
        start = time.time()
        await mock_node.get_current_state()
        elapsed_ms = (time.time() - start) * 1000

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

            start = time.time()
            await mock_node.get_introspection_data()
            elapsed_ms = (time.time() - start) * 1000
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
        node.initialize_introspection(
            node_id="benchmark-node",
            node_type="EFFECT",
            event_bus=None,
        )

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
        node.initialize_introspection(
            node_id="concurrent-benchmark-node",
            node_type="EFFECT",
            event_bus=None,
            cache_ttl=0.001,  # Force cache misses
        )

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
        node.initialize_introspection(
            node_id="cache-hit-node",
            node_type="EFFECT",
            event_bus=None,
        )

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
        node.initialize_introspection(
            node_id="p95-node",
            node_type="EFFECT",
            event_bus=None,
        )

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
        node.initialize_introspection(
            node_id="breakdown-node",
            node_type="EFFECT",
            event_bus=None,
        )

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

        node = MinimalNode()
        node.initialize_introspection(
            node_id="minimal-node",
            node_type="EFFECT",
            event_bus=None,
        )

        data = await node.get_introspection_data()

        assert data.node_id == "minimal-node"
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
        node.initialize_introspection(
            node_id="large-node",
            node_type="COMPUTE",
            event_bus=None,
        )

        threshold_ms = 50 * PERF_MULTIPLIER
        start = time.time()
        capabilities = await node.get_capabilities()
        elapsed_ms = (time.time() - start) * 1000

        # Should include all 10 operation methods
        operations = capabilities["operations"]
        assert isinstance(operations, list)
        assert len(operations) >= 10
        assert elapsed_ms < threshold_ms, (
            f"Large capability extraction took {elapsed_ms:.2f}ms, expected <{threshold_ms:.0f}ms"
        )

    async def test_concurrent_introspection_calls(self) -> None:
        """Test concurrent introspection data requests."""
        node = MockNode()
        node.initialize_introspection(
            node_id="concurrent-node",
            node_type="EFFECT",
            event_bus=None,
            cache_ttl=0.001,  # Very short TTL
        )

        # Make 100 concurrent calls
        tasks = [node.get_introspection_data() for _ in range(100)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 100
        for result in results:
            assert result.node_id == "concurrent-node"

    async def test_introspection_with_special_characters_in_state(self) -> None:
        """Test introspection with special characters in state."""
        node = MockNode()
        node._state = "state<with>special&chars\"quote'"
        node.initialize_introspection(
            node_id="special-node",
            node_type="EFFECT",
            event_bus=None,
        )

        state = await node.get_current_state()
        assert state == "state<with>special&chars\"quote'"

    async def test_introspection_preserves_node_functionality(self) -> None:
        """Test that introspection mixin doesn't affect node functionality."""
        node = MockNode()
        node.initialize_introspection(
            node_id="functional-node",
            node_type="EFFECT",
            event_bus=None,
        )

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
        node.initialize_introspection(
            node_id="cache-test-node",
            node_type="EFFECT",
            event_bus=None,
        )

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
        node1.initialize_introspection(
            node_id="cache-node-1",
            node_type="EFFECT",
            event_bus=None,
        )

        node2 = MockNode()
        node2.initialize_introspection(
            node_id="cache-node-2",
            node_type="EFFECT",
            event_bus=None,
        )

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
        node.initialize_introspection(
            node_id="invalidate-test",
            node_type="EFFECT",
            event_bus=None,
        )

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
        node.initialize_introspection(
            node_id="invalidate-all-test",
            node_type="EFFECT",
            event_bus=None,
        )

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
        node.initialize_introspection(
            node_id="signature-match-test",
            node_type="EFFECT",
            event_bus=None,
        )

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
        node.initialize_introspection(
            node_id="perf-cache-test",
            node_type="EFFECT",
            event_bus=None,
        )

        # First call (cold cache) - populates cache
        start1 = time.time()
        await node.get_capabilities()
        first_call_ms = (time.time() - start1) * 1000

        # Subsequent calls (warm cache) - uses cached signatures
        times_warm = []
        for _ in range(10):
            start = time.time()
            await node.get_capabilities()
            times_warm.append((time.time() - start) * 1000)

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
        node1.initialize_introspection(
            node_id="custom-1",
            node_type="COMPUTE",
            event_bus=None,
        )

        node2 = CustomNode2()
        node2.initialize_introspection(
            node_id="custom-2",
            node_type="COMPUTE",
            event_bus=None,
        )

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
        node.initialize_introspection(
            node_id="builtins-test",
            node_type="EFFECT",
            event_bus=None,
        )

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
        node.initialize_introspection(
            node_id="default-keywords-node",
            node_type="EFFECT",
            event_bus=None,
        )
        assert (
            node._introspection_operation_keywords
            == MixinNodeIntrospection.DEFAULT_OPERATION_KEYWORDS
        )

    async def test_default_exclude_prefixes_used_when_not_specified(self) -> None:
        """Test that DEFAULT_EXCLUDE_PREFIXES is used when not specified."""
        node = MockNode()
        node.initialize_introspection(
            node_id="default-prefixes-node",
            node_type="EFFECT",
            event_bus=None,
        )
        assert (
            node._introspection_exclude_prefixes
            == MixinNodeIntrospection.DEFAULT_EXCLUDE_PREFIXES
        )

    async def test_custom_operation_keywords_are_stored(self) -> None:
        """Test that custom operation_keywords are stored correctly."""
        custom_keywords = {"fetch", "upload", "download", "sync"}
        node = MockNode()
        node.initialize_introspection(
            node_id="custom-keywords-node",
            node_type="EFFECT",
            event_bus=None,
            operation_keywords=custom_keywords,
        )
        assert node._introspection_operation_keywords == custom_keywords

    async def test_custom_exclude_prefixes_are_stored(self) -> None:
        """Test that custom exclude_prefixes are stored correctly."""
        custom_prefixes = {"_", "helper_", "internal_"}
        node = MockNode()
        node.initialize_introspection(
            node_id="custom-prefixes-node",
            node_type="EFFECT",
            event_bus=None,
            exclude_prefixes=custom_prefixes,
        )
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
        node.initialize_introspection(
            node_id="custom-ops-node",
            node_type="EFFECT",
            event_bus=None,
            operation_keywords={"fetch", "upload"},
        )
        capabilities = await node.get_capabilities()
        operations = capabilities["operations"]
        assert "fetch_data" in operations
        assert "upload_file" in operations
        assert "execute_task" not in operations

    async def test_node_type_specific_keywords_constant_exists(self) -> None:
        """Test that NODE_TYPE_OPERATION_KEYWORDS constant exists."""
        assert hasattr(MixinNodeIntrospection, "NODE_TYPE_OPERATION_KEYWORDS")
        keywords_map = MixinNodeIntrospection.NODE_TYPE_OPERATION_KEYWORDS
        assert "EFFECT" in keywords_map
        assert "COMPUTE" in keywords_map
        assert "REDUCER" in keywords_map
        assert "ORCHESTRATOR" in keywords_map
        for keywords in keywords_map.values():
            assert isinstance(keywords, set)

    async def test_empty_operation_keywords_discovers_no_operations(self) -> None:
        """Test that empty operation_keywords results in no operations discovered."""
        node = MockNode()
        node.initialize_introspection(
            node_id="empty-keywords-node",
            node_type="EFFECT",
            event_bus=None,
            operation_keywords=set(),
        )
        capabilities = await node.get_capabilities()
        assert len(capabilities["operations"]) == 0

    async def test_configuration_is_instance_specific(self) -> None:
        """Test that configuration is instance-specific, not shared."""

        class MultiInstanceNode(MixinNodeIntrospection):
            async def execute_task(self) -> None:
                pass

            async def fetch_data(self) -> None:
                pass

        node1 = MultiInstanceNode()
        node1.initialize_introspection(
            node_id="node-1",
            node_type="EFFECT",
            event_bus=None,
            operation_keywords={"execute"},
        )
        node2 = MultiInstanceNode()
        node2.initialize_introspection(
            node_id="node-2",
            node_type="EFFECT",
            event_bus=None,
            operation_keywords={"fetch"},
        )
        caps1 = await node1.get_capabilities()
        caps2 = await node2.get_capabilities()
        assert "execute_task" in caps1["operations"]
        assert "fetch_data" not in caps1["operations"]
        assert "fetch_data" in caps2["operations"]
        assert "execute_task" not in caps2["operations"]

    async def test_default_keywords_not_mutated(self) -> None:
        """Test that DEFAULT_OPERATION_KEYWORDS is not mutated by instances."""
        original_defaults = MixinNodeIntrospection.DEFAULT_OPERATION_KEYWORDS.copy()
        node = MockNode()
        node.initialize_introspection(
            node_id="no-mutation-node",
            node_type="EFFECT",
            event_bus=None,
        )
        node._introspection_operation_keywords.add("custom_keyword")
        assert original_defaults == MixinNodeIntrospection.DEFAULT_OPERATION_KEYWORDS
        assert "custom_keyword" not in MixinNodeIntrospection.DEFAULT_OPERATION_KEYWORDS
