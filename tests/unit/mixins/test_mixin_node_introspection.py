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
- Thread safety and race condition handling

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
    - TestMixinNodeIntrospectionThreadSafety: Race condition and concurrency tests

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
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

import pytest

from omnibase_infra.mixins.mixin_node_introspection import (
    PERF_THRESHOLD_CACHE_HIT_MS,
    PERF_THRESHOLD_GET_CAPABILITIES_MS,
    PERF_THRESHOLD_GET_INTROSPECTION_DATA_MS,
    IntrospectionPerformanceMetrics,
    MixinNodeIntrospection,
)
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

    async def subscribe(
        self,
        topic: str,
        group_id: str,
        on_message: Callable[[Any], Awaitable[None]],
    ) -> Callable[[], Awaitable[None]]:
        """Mock subscribe method for protocol compliance.

        Args:
            topic: Topic to subscribe to.
            group_id: Consumer group ID.
            on_message: Callback function for messages.

        Returns:
            An async unsubscribe function.
        """

        async def unsubscribe() -> None:
            pass

        return unsubscribe


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

        with pytest.raises(ValueError, match="node_id cannot be None or empty"):
            node.initialize_introspection(
                node_id="",
                node_type="EFFECT",
            )

    async def test_initialize_introspection_empty_node_type_raises(self) -> None:
        """Test that empty node_type raises ValueError."""
        node = MockNode()

        with pytest.raises(ValueError, match="node_type cannot be None or empty"):
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

        # Invalidate (async method for thread-safe cache access)
        await mock_node.invalidate_introspection_cache()

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
        assert topic == MixinNodeIntrospection.DEFAULT_INTROSPECTION_TOPIC
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

    async def test_cache_lock_thread_safety(self) -> None:
        """Test that cache operations are thread-safe with async lock.

        This test verifies that concurrent cache reads, writes, and invalidations
        do not cause race conditions when using the async lock.
        """
        node = MockNode()
        node.initialize_introspection(
            node_id="lock-test-node",
            node_type="EFFECT",
            event_bus=None,
            cache_ttl=0.001,  # Very short TTL to force frequent cache misses
        )

        # Verify the lock is initialized
        assert hasattr(node, "_introspection_cache_lock")
        assert isinstance(node._introspection_cache_lock, asyncio.Lock)

        # Create mixed operations: reads, writes (via get_introspection_data),
        # and invalidations
        async def mixed_operations(idx: int) -> str:
            """Perform a mix of cache operations."""
            if idx % 3 == 0:
                await node.invalidate_introspection_cache()
                return "invalidate"
            else:
                await node.get_introspection_data()
                return "read"

        # Run 50 concurrent mixed operations
        tasks = [mixed_operations(i) for i in range(50)]
        results = await asyncio.gather(*tasks)

        # All operations should complete without deadlock or error
        assert len(results) == 50
        assert "invalidate" in results
        assert "read" in results

        # Final state should be consistent
        data = await node.get_introspection_data()
        assert data.node_id == "lock-test-node"

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


@pytest.mark.unit
@pytest.mark.asyncio(loop_scope="function")
class TestMixinNodeIntrospectionThreadSafety:
    """Comprehensive thread safety and race condition tests.

    This test class validates that the MixinNodeIntrospection cache operations
    are thread-safe under various concurrent access patterns:

    1. Multiple concurrent reads (get_introspection_data)
    2. Multiple concurrent invalidations (invalidate_introspection_cache)
    3. Mixed concurrent reads and invalidations
    4. Rapid alternating read/invalidate patterns
    5. Cache state consistency after concurrent operations
    6. Lock contention under high load
    7. No data corruption or deadlocks

    Thread Safety Implementation:
        The mixin uses asyncio.Lock to protect:
        - _introspection_cache (dict or None)
        - _introspection_cached_at (float or None)

        All tests verify that these invariants hold under concurrent access.
    """

    async def test_concurrent_reads_return_consistent_data(self) -> None:
        """Test that multiple concurrent get_introspection_data calls return consistent data.

        This test verifies that when many coroutines simultaneously request
        introspection data, all receive identical, valid results without
        data corruption.
        """
        node = MockNode()
        node.initialize_introspection(
            node_id="concurrent-read-node",
            node_type="EFFECT",
            event_bus=None,
            cache_ttl=60.0,  # Long TTL to ensure cache hits
        )

        # Prime the cache first
        initial_data = await node.get_introspection_data()

        # Run many concurrent reads
        num_concurrent = 100
        results = await asyncio.gather(
            *[node.get_introspection_data() for _ in range(num_concurrent)]
        )

        # All results should be valid and consistent
        assert len(results) == num_concurrent
        for i, result in enumerate(results):
            assert result.node_id == "concurrent-read-node", (
                f"Result {i} has wrong node_id: {result.node_id}"
            )
            assert result.node_type == "EFFECT", (
                f"Result {i} has wrong node_type: {result.node_type}"
            )
            # Verify structural consistency with initial data
            assert result.capabilities == initial_data.capabilities, (
                f"Result {i} has different capabilities than initial data"
            )

    async def test_concurrent_invalidations_are_safe(self) -> None:
        """Test that multiple concurrent invalidate_introspection_cache calls are safe.

        This test verifies that rapidly invalidating the cache from multiple
        coroutines does not cause errors or leave the cache in an inconsistent state.
        """
        node = MockNode()
        node.initialize_introspection(
            node_id="concurrent-invalidate-node",
            node_type="EFFECT",
            event_bus=None,
            cache_ttl=60.0,
        )

        # Prime the cache
        await node.get_introspection_data()
        assert node._introspection_cache is not None

        # Run many concurrent invalidations
        num_concurrent = 50
        await asyncio.gather(
            *[node.invalidate_introspection_cache() for _ in range(num_concurrent)]
        )

        # Cache should be invalidated (None)
        assert node._introspection_cache is None
        assert node._introspection_cached_at is None

        # Should be able to repopulate cache normally
        data = await node.get_introspection_data()
        assert data.node_id == "concurrent-invalidate-node"
        assert node._introspection_cache is not None

    async def test_invalidation_during_concurrent_reads(self) -> None:
        """Test that invalidation during multiple concurrent reads is handled safely.

        This is the key race condition scenario: invalidation happens while
        other coroutines are reading. All operations should complete without
        errors or data corruption.
        """
        node = MockNode()
        node.initialize_introspection(
            node_id="mixed-ops-node",
            node_type="EFFECT",
            event_bus=None,
            cache_ttl=0.001,  # Very short TTL to force cache misses
        )

        errors: list[Exception] = []
        results: list[str] = []

        async def read_operation(idx: int) -> None:
            """Perform a read operation."""
            try:
                data = await node.get_introspection_data()
                assert data.node_id == "mixed-ops-node"
                results.append(f"read-{idx}")
            except Exception as e:
                errors.append(e)

        async def invalidate_operation(idx: int) -> None:
            """Perform an invalidation operation."""
            try:
                await node.invalidate_introspection_cache()
                results.append(f"invalidate-{idx}")
            except Exception as e:
                errors.append(e)

        # Create mixed tasks: 70% reads, 30% invalidations
        tasks: list[asyncio.Task[None]] = []
        for i in range(100):
            if i % 3 == 0:
                tasks.append(asyncio.create_task(invalidate_operation(i)))
            else:
                tasks.append(asyncio.create_task(read_operation(i)))

        await asyncio.gather(*tasks)

        # No errors should have occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 100

        # Final state should be consistent
        final_data = await node.get_introspection_data()
        assert final_data.node_id == "mixed-ops-node"

    async def test_rapid_alternating_read_invalidate_cycles(self) -> None:
        """Test rapid alternating read/invalidate cycles.

        This tests a worst-case contention pattern where reads and invalidations
        rapidly alternate, maximizing lock contention.
        """
        node = MockNode()
        node.initialize_introspection(
            node_id="alternating-node",
            node_type="COMPUTE",
            event_bus=None,
            cache_ttl=0.0001,  # Extremely short TTL
        )

        cycle_count = 50
        errors: list[Exception] = []

        async def alternating_cycle(cycle_id: int) -> None:
            """Perform alternating read-invalidate-read cycle."""
            try:
                # Read
                data1 = await node.get_introspection_data()
                assert data1.node_id == "alternating-node"

                # Invalidate
                await node.invalidate_introspection_cache()

                # Read again (forces cache rebuild)
                data2 = await node.get_introspection_data()
                assert data2.node_id == "alternating-node"
            except Exception as e:
                errors.append(e)

        # Run many alternating cycles concurrently
        await asyncio.gather(*[alternating_cycle(i) for i in range(cycle_count)])

        assert len(errors) == 0, f"Errors during alternating cycles: {errors}"

    async def test_cache_state_consistency_invariants(self) -> None:
        """Test that cache state invariants hold under concurrent access.

        Invariants:
        1. If _introspection_cache is not None, _introspection_cached_at must be set
        2. If _introspection_cached_at is None, _introspection_cache must be None
        3. Both must transition atomically together
        """
        node = MockNode()
        node.initialize_introspection(
            node_id="invariant-node",
            node_type="EFFECT",
            event_bus=None,
            cache_ttl=0.001,
        )

        invariant_violations: list[str] = []

        async def check_and_operate(op_type: str) -> None:
            """Check invariants and perform operation."""
            # Check invariant before operation
            cache = node._introspection_cache
            cached_at = node._introspection_cached_at

            if cache is not None and cached_at is None:
                invariant_violations.append(
                    f"cache set but cached_at is None before {op_type}"
                )
            if cache is None and cached_at is not None:
                invariant_violations.append(
                    f"cache is None but cached_at is set before {op_type}"
                )

            # Perform operation
            if op_type == "read":
                await node.get_introspection_data()
            else:
                await node.invalidate_introspection_cache()

            # Small yield to allow other coroutines to run
            await asyncio.sleep(0)

            # Check invariant after operation
            cache = node._introspection_cache
            cached_at = node._introspection_cached_at

            if cache is not None and cached_at is None:
                invariant_violations.append(
                    f"cache set but cached_at is None after {op_type}"
                )
            if cache is None and cached_at is not None:
                invariant_violations.append(
                    f"cache is None but cached_at is set after {op_type}"
                )

        # Run mixed operations
        tasks: list[asyncio.Task[None]] = []
        for i in range(100):
            op = "invalidate" if i % 4 == 0 else "read"
            tasks.append(asyncio.create_task(check_and_operate(op)))

        await asyncio.gather(*tasks)

        assert len(invariant_violations) == 0, (
            f"Invariant violations: {invariant_violations}"
        )

    async def test_no_deadlock_under_high_contention(self) -> None:
        """Test that high contention does not cause deadlocks.

        This test uses a timeout to detect potential deadlocks. If the test
        completes within the timeout, no deadlock occurred.
        """
        node = MockNode()
        node.initialize_introspection(
            node_id="deadlock-test-node",
            node_type="EFFECT",
            event_bus=None,
            cache_ttl=0.0001,
        )

        completed = 0

        async def contention_operation(idx: int) -> None:
            """Perform operations under high contention."""
            nonlocal completed
            for _ in range(10):
                if idx % 2 == 0:
                    await node.get_introspection_data()
                else:
                    await node.invalidate_introspection_cache()
                # Yield to maximize contention
                await asyncio.sleep(0)
            completed += 1

        # Run with timeout to detect deadlock
        num_tasks = 50
        try:
            await asyncio.wait_for(
                asyncio.gather(*[contention_operation(i) for i in range(num_tasks)]),
                timeout=10.0,  # 10 second timeout
            )
        except TimeoutError:
            pytest.fail(
                f"Deadlock detected: only {completed}/{num_tasks} tasks completed"
            )

        assert completed == num_tasks

    async def test_stale_read_prevention(self) -> None:
        """Test that invalidation properly prevents stale reads.

        When a cache is invalidated, subsequent reads should return fresh data,
        not stale cached data.
        """
        node = MockNode()
        node.initialize_introspection(
            node_id="stale-prevention-node",
            node_type="EFFECT",
            event_bus=None,
            cache_ttl=60.0,  # Long TTL
        )

        # Populate cache
        initial_data = await node.get_introspection_data()
        initial_timestamp = initial_data.timestamp

        # Wait a small amount to ensure timestamp difference
        await asyncio.sleep(0.01)

        # Invalidate cache
        await node.invalidate_introspection_cache()

        # Read again - should get fresh data with new timestamp
        fresh_data = await node.get_introspection_data()

        assert fresh_data.timestamp >= initial_timestamp, (
            "Fresh data should have timestamp >= initial timestamp"
        )
        assert fresh_data.node_id == "stale-prevention-node"

    async def test_concurrent_reads_with_expiring_cache(self) -> None:
        """Test concurrent reads when cache TTL expires mid-operation.

        This tests the race condition where multiple readers find an expired
        cache simultaneously and all try to rebuild it.
        """
        node = MockNode()
        node.initialize_introspection(
            node_id="expiring-cache-node",
            node_type="EFFECT",
            event_bus=None,
            cache_ttl=0.001,  # 1ms TTL - will expire quickly
        )

        # Prime the cache
        await node.get_introspection_data()

        # Wait for cache to expire
        await asyncio.sleep(0.005)

        # Now run many concurrent reads - all will find expired cache
        num_concurrent = 50
        results = await asyncio.gather(
            *[node.get_introspection_data() for _ in range(num_concurrent)]
        )

        # All should succeed with valid data
        assert len(results) == num_concurrent
        for result in results:
            assert result.node_id == "expiring-cache-node"
            assert result.node_type == "EFFECT"

    async def test_lock_acquisition_fairness(self) -> None:
        """Test that lock acquisition is reasonably fair under contention.

        This test verifies that no coroutine is starved of lock access
        by checking that all operations complete in reasonable order.
        """
        node = MockNode()
        node.initialize_introspection(
            node_id="fairness-node",
            node_type="EFFECT",
            event_bus=None,
            cache_ttl=0.001,
        )

        completion_order: list[int] = []
        start_order: list[int] = []
        order_lock = asyncio.Lock()

        async def tracked_operation(idx: int) -> None:
            """Track operation start and completion order."""
            async with order_lock:
                start_order.append(idx)

            # Perform actual operation
            await node.get_introspection_data()

            async with order_lock:
                completion_order.append(idx)

        num_tasks = 30
        await asyncio.gather(*[tracked_operation(i) for i in range(num_tasks)])

        # All tasks should complete
        assert len(completion_order) == num_tasks
        assert set(completion_order) == set(range(num_tasks))

        # Check for reasonable fairness: first starter shouldn't be last completer
        # (allowing some flexibility due to async scheduling)
        first_starters = set(start_order[:10])
        last_completers = set(completion_order[-5:])

        # At least half of early starters should have completed before last 5
        early_completed = first_starters - last_completers
        assert len(early_completed) >= 5, (
            f"Potential starvation: {len(early_completed)}/10 early starters "
            f"completed before last 5 completers"
        )

    async def test_exception_in_concurrent_operation_does_not_corrupt_cache(
        self,
    ) -> None:
        """Test that an exception in one coroutine doesn't corrupt cache for others.

        If one operation fails, the cache should remain in a consistent state
        for other concurrent operations.
        """
        node = MockNode()
        node.initialize_introspection(
            node_id="exception-safety-node",
            node_type="EFFECT",
            event_bus=None,
            cache_ttl=60.0,
        )

        # Prime cache
        await node.get_introspection_data()

        errors: list[Exception] = []
        successes: list[int] = []

        async def potentially_failing_read(idx: int, should_fail: bool) -> None:
            """Read that may raise an exception."""
            try:
                if should_fail:
                    # Simulate an error after getting data
                    await node.get_introspection_data()
                    raise ValueError(f"Simulated error in task {idx}")

                data = await node.get_introspection_data()
                assert data.node_id == "exception-safety-node"
                successes.append(idx)
            except ValueError as e:
                errors.append(e)

        # Run mix of failing and succeeding tasks
        tasks: list[asyncio.Task[None]] = []
        for i in range(50):
            should_fail = i % 10 == 0  # 10% failure rate
            tasks.append(asyncio.create_task(potentially_failing_read(i, should_fail)))

        await asyncio.gather(*tasks, return_exceptions=True)

        # Should have expected number of successes and errors
        assert len(errors) == 5, f"Expected 5 errors, got {len(errors)}"
        assert len(successes) == 45, f"Expected 45 successes, got {len(successes)}"

        # Cache should still be valid for normal operations
        final_data = await node.get_introspection_data()
        assert final_data.node_id == "exception-safety-node"

    async def test_concurrent_initialization_safety(self) -> None:
        """Test that concurrent access during initialization is handled safely.

        This is an edge case where operations might be called before
        initialization is complete.
        """

        class SlowInitNode(MixinNodeIntrospection):
            """Node with delayed state setup."""

            def __init__(self) -> None:
                self._state = "initializing"

            async def slow_init(self) -> None:
                """Simulate slow initialization."""
                await asyncio.sleep(0.01)
                self._state = "ready"

        nodes: list[SlowInitNode] = []
        errors: list[Exception] = []

        async def init_and_introspect(idx: int) -> None:
            """Initialize and immediately introspect."""
            try:
                node = SlowInitNode()
                node.initialize_introspection(
                    node_id=f"slow-init-{idx}",
                    node_type="EFFECT",
                    event_bus=None,
                )
                nodes.append(node)

                # Start introspection before slow_init completes
                init_task = asyncio.create_task(node.slow_init())
                data = await node.get_introspection_data()

                assert data.node_id == f"slow-init-{idx}"
                await init_task

            except Exception as e:
                errors.append(e)

        await asyncio.gather(*[init_and_introspect(i) for i in range(20)])

        assert len(errors) == 0, f"Initialization errors: {errors}"
        assert len(nodes) == 20

    async def test_multiple_nodes_independent_caches(self) -> None:
        """Test that multiple node instances have independent caches.

        Concurrent operations on different nodes should not interfere
        with each other's caches.
        """
        nodes: list[MockNode] = []
        for i in range(10):
            node = MockNode()
            node.initialize_introspection(
                node_id=f"independent-node-{i}",
                node_type="EFFECT",
                event_bus=None,
                cache_ttl=60.0,
            )
            nodes.append(node)

        async def operate_on_node(node: MockNode, op_count: int) -> list[str]:
            """Perform operations on a single node."""
            node_ids: list[str] = []
            for _ in range(op_count):
                data = await node.get_introspection_data()
                node_ids.append(data.node_id)
                if _ % 3 == 0:
                    await node.invalidate_introspection_cache()
            return node_ids

        # Run concurrent operations on all nodes
        results = await asyncio.gather(*[operate_on_node(node, 20) for node in nodes])

        # Each node's operations should only see its own node_id
        for i, node_ids in enumerate(results):
            expected_id = f"independent-node-{i}"
            for node_id in node_ids:
                assert node_id == expected_id, (
                    f"Node {i} saw foreign node_id: {node_id}"
                )


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
        assert isinstance(operations, list)
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
        ops1 = caps1["operations"]
        ops2 = caps2["operations"]
        assert isinstance(ops1, list)
        assert isinstance(ops2, list)
        assert "execute_task" in ops1
        assert "fetch_data" not in ops1
        assert "fetch_data" in ops2
        assert "execute_task" not in ops2

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


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinNodeIntrospectionPerformanceMetrics:
    """Tests for performance metrics tracking and retrieval."""

    async def test_get_performance_metrics_returns_none_before_introspection(
        self,
    ) -> None:
        """Test that get_performance_metrics returns None before introspection."""
        node = MockNode()
        node.initialize_introspection(
            node_id="metrics-test-node",
            node_type="EFFECT",
            event_bus=None,
        )

        metrics = node.get_performance_metrics()
        assert metrics is None

    async def test_get_performance_metrics_returns_metrics_after_introspection(
        self,
    ) -> None:
        """Test that get_performance_metrics returns metrics after introspection."""
        node = MockNode()
        node.initialize_introspection(
            node_id="metrics-test-node",
            node_type="EFFECT",
            event_bus=None,
        )

        await node.get_introspection_data()
        metrics = node.get_performance_metrics()

        assert metrics is not None
        assert isinstance(metrics, IntrospectionPerformanceMetrics)

    async def test_performance_metrics_contains_expected_fields(self) -> None:
        """Test that performance metrics contain all expected fields."""
        node = MockNode()
        node.initialize_introspection(
            node_id="metrics-fields-node",
            node_type="EFFECT",
            event_bus=None,
        )

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
        node.initialize_introspection(
            node_id="cache-hit-metrics-node",
            node_type="EFFECT",
            event_bus=None,
        )

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
        node.initialize_introspection(
            node_id="method-count-metrics-node",
            node_type="EFFECT",
            event_bus=None,
        )

        await node.get_introspection_data()
        metrics = node.get_performance_metrics()

        assert metrics is not None
        # MockNode has at least 4 public methods (execute, health_check,
        # handle_event, process_batch)
        assert metrics.method_count >= 4

    async def test_performance_metrics_to_dict(self) -> None:
        """Test that to_dict() returns all fields."""
        node = MockNode()
        node.initialize_introspection(
            node_id="to-dict-node",
            node_type="EFFECT",
            event_bus=None,
        )

        await node.get_introspection_data()
        metrics = node.get_performance_metrics()

        assert metrics is not None
        metrics_dict = metrics.to_dict()

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
        node.initialize_introspection(
            node_id="fresh-metrics-node",
            node_type="EFFECT",
            event_bus=None,
            cache_ttl=0.001,  # Very short TTL to force cache refresh
        )

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
        node.initialize_introspection(
            node_id="minimal-methods-node",
            node_type="COMPUTE",
            event_bus=None,
        )

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
        node.initialize_introspection(
            node_id="medium-methods-node",
            node_type="COMPUTE",
            event_bus=None,
        )

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
        node.initialize_introspection(
            node_id="large-methods-node",
            node_type="COMPUTE",
            event_bus=None,
        )

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
        node.initialize_introspection(
            node_id="large-cache-node",
            node_type="COMPUTE",
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
            node.initialize_introspection(
                node_id=f"scaling-test-{method_count}",
                node_type="COMPUTE",
                event_bus=None,
            )

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
        node.initialize_introspection(
            node_id="threshold-normal-node",
            node_type="EFFECT",
            event_bus=None,
        )

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
        node.initialize_introspection(
            node_id="slow-ops-structure-node",
            node_type="EFFECT",
            event_bus=None,
        )

        await node.get_introspection_data()
        metrics = node.get_performance_metrics()

        assert metrics is not None
        # slow_operations should always be a list
        assert isinstance(metrics.slow_operations, list)

    async def test_cache_hit_threshold_separate_from_total(self) -> None:
        """Test that cache hit has its own performance threshold."""
        node = MockNode()
        node.initialize_introspection(
            node_id="cache-threshold-node",
            node_type="EFFECT",
            event_bus=None,
        )

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


class TestMixinNodeIntrospectionConfigModel:
    """Test ModelIntrospectionConfig usage with initialize_introspection."""

    async def test_initialize_with_config_model(self) -> None:
        """Test initialization using ModelIntrospectionConfig."""
        from omnibase_infra.mixins import ModelIntrospectionConfig

        config = ModelIntrospectionConfig(
            node_id="config-model-node",
            node_type="EFFECT",
            version="2.0.0",
            cache_ttl=600.0,
        )

        node = MockNode()
        node.initialize_introspection(config=config)

        assert node._introspection_node_id == "config-model-node"
        assert node._introspection_node_type == "EFFECT"
        assert node._introspection_version == "2.0.0"
        assert node._introspection_cache_ttl == 600.0

    async def test_config_model_with_event_bus(self) -> None:
        """Test config model with event bus."""
        from omnibase_infra.mixins import ModelIntrospectionConfig

        event_bus = MockEventBus()
        config = ModelIntrospectionConfig(
            node_id="config-eventbus-node",
            node_type="COMPUTE",
            event_bus=event_bus,
        )

        node = MockNode()
        node.initialize_introspection(config=config)

        assert node._introspection_event_bus is event_bus

        # Verify publishing works
        success = await node.publish_introspection(reason="test")
        assert success is True

    async def test_config_model_with_custom_keywords(self) -> None:
        """Test config model with custom operation keywords."""
        from omnibase_infra.mixins import ModelIntrospectionConfig

        custom_keywords = {"fetch", "upload", "download"}
        config = ModelIntrospectionConfig(
            node_id="config-keywords-node",
            node_type="EFFECT",
            operation_keywords=custom_keywords,
        )

        node = MockNode()
        node.initialize_introspection(config=config)

        assert node._introspection_operation_keywords == custom_keywords

    async def test_config_model_with_custom_topics(self) -> None:
        """Test config model with custom topic configuration."""
        from omnibase_infra.mixins import ModelIntrospectionConfig

        config = ModelIntrospectionConfig(
            node_id="config-topics-node",
            node_type="EFFECT",
            introspection_topic="onex.custom.introspection.topic",
            heartbeat_topic="onex.custom.heartbeat.topic",
            request_introspection_topic="onex.custom.request.topic",
        )

        node = MockNode()
        node.initialize_introspection(config=config)

        assert node._introspection_topic == "onex.custom.introspection.topic"
        assert node._heartbeat_topic == "onex.custom.heartbeat.topic"
        assert node._request_introspection_topic == "onex.custom.request.topic"

    async def test_legacy_params_still_work(self) -> None:
        """Test that legacy individual parameters still work."""
        node = MockNode()
        node.initialize_introspection(
            node_id="legacy-node",
            node_type="REDUCER",
            version="1.5.0",
        )

        assert node._introspection_node_id == "legacy-node"
        assert node._introspection_node_type == "REDUCER"
        assert node._introspection_version == "1.5.0"

    async def test_config_model_overrides_legacy_params(self) -> None:
        """Test that config model takes precedence over individual params."""
        from omnibase_infra.mixins import ModelIntrospectionConfig

        config = ModelIntrospectionConfig(
            node_id="config-priority-node",
            node_type="ORCHESTRATOR",
            version="3.0.0",
        )

        node = MockNode()
        # Pass both config and legacy params - config should win
        node.initialize_introspection(
            config=config,
            node_id="should-be-ignored",
            node_type="SHOULD-BE-IGNORED",
            version="0.0.1",
        )

        assert node._introspection_node_id == "config-priority-node"
        assert node._introspection_node_type == "ORCHESTRATOR"
        assert node._introspection_version == "3.0.0"

    async def test_error_when_neither_config_nor_required_params(self) -> None:
        """Test error raised when neither config nor required params provided."""
        node = MockNode()

        with pytest.raises(ValueError, match="Either config or both node_id"):
            node.initialize_introspection()

        with pytest.raises(ValueError, match="Either config or both node_id"):
            node.initialize_introspection(node_id="only-id")

        with pytest.raises(ValueError, match="Either config or both node_id"):
            node.initialize_introspection(node_type="only-type")

    async def test_config_model_validation(self) -> None:
        """Test that config model validates inputs."""
        from pydantic import ValidationError

        from omnibase_infra.mixins import ModelIntrospectionConfig

        # Empty node_id should fail validation
        with pytest.raises(ValidationError):
            ModelIntrospectionConfig(
                node_id="",
                node_type="EFFECT",
            )

        # Empty node_type should fail validation
        with pytest.raises(ValidationError):
            ModelIntrospectionConfig(
                node_id="valid-id",
                node_type="",
            )

        # Negative cache_ttl should fail validation
        with pytest.raises(ValidationError):
            ModelIntrospectionConfig(
                node_id="valid-id",
                node_type="EFFECT",
                cache_ttl=-1.0,
            )

    async def test_config_model_exports(self) -> None:
        """Test that ModelIntrospectionConfig is properly exported."""
        # Test module-level export
        from omnibase_infra.mixins.mixin_node_introspection import (
            ModelIntrospectionConfig,
        )

        assert ModelIntrospectionConfig is not None

        # Test package-level export
        from omnibase_infra.mixins import ModelIntrospectionConfig as PkgConfig

        assert PkgConfig is not None
        assert PkgConfig is ModelIntrospectionConfig


@pytest.mark.unit
@pytest.mark.asyncio
class TestModelIntrospectionConfigTopicValidation:
    """Tests for topic name validation in ModelIntrospectionConfig.

    These tests verify that the Pydantic field_validator for topic names
    correctly validates and rejects invalid topic name formats.

    Topic names must:
    - Start with "onex." prefix (ONEX naming convention)
    - Contain only alphanumeric characters, dots, hyphens, and underscores
    - Be non-empty after the prefix
    """

    async def test_valid_topic_names_accepted(self) -> None:
        """Test that valid topic names following ONEX convention are accepted."""
        from omnibase_infra.mixins import ModelIntrospectionConfig

        # Valid topic names with various patterns
        config = ModelIntrospectionConfig(
            node_id="test-node",
            node_type="EFFECT",
            introspection_topic="onex.node.introspection.published.v1",
            heartbeat_topic="onex.node.heartbeat.published.v1",
            request_introspection_topic="onex.registry.introspection.requested.v1",
        )

        assert config.introspection_topic == "onex.node.introspection.published.v1"
        assert config.heartbeat_topic == "onex.node.heartbeat.published.v1"
        assert (
            config.request_introspection_topic
            == "onex.registry.introspection.requested.v1"
        )

    async def test_none_values_allowed(self) -> None:
        """Test that None values pass validation (use defaults)."""
        from omnibase_infra.mixins import ModelIntrospectionConfig

        config = ModelIntrospectionConfig(
            node_id="test-node",
            node_type="EFFECT",
            introspection_topic=None,
            heartbeat_topic=None,
            request_introspection_topic=None,
        )

        assert config.introspection_topic is None
        assert config.heartbeat_topic is None
        assert config.request_introspection_topic is None

    async def test_missing_onex_prefix_rejected(self) -> None:
        """Test that topic names without 'onex.' prefix are rejected."""
        from pydantic import ValidationError

        from omnibase_infra.mixins import ModelIntrospectionConfig

        # Test introspection_topic without prefix
        with pytest.raises(ValidationError) as exc_info:
            ModelIntrospectionConfig(
                node_id="test-node",
                node_type="EFFECT",
                introspection_topic="node.introspection.published.v1",
            )

        error_message = str(exc_info.value)
        assert "onex." in error_message
        assert "prefix" in error_message.lower()

    async def test_invalid_characters_rejected(self) -> None:
        """Test that topic names with invalid characters are rejected."""
        from pydantic import ValidationError

        from omnibase_infra.mixins import ModelIntrospectionConfig

        # Test with spaces
        with pytest.raises(ValidationError) as exc_info:
            ModelIntrospectionConfig(
                node_id="test-node",
                node_type="EFFECT",
                introspection_topic="onex.node introspection.published.v1",
            )

        error_message = str(exc_info.value)
        assert "invalid characters" in error_message.lower()

        # Test with special characters
        with pytest.raises(ValidationError):
            ModelIntrospectionConfig(
                node_id="test-node",
                node_type="EFFECT",
                heartbeat_topic="onex.node@heartbeat#published!v1",
            )

        # Test with slashes
        with pytest.raises(ValidationError):
            ModelIntrospectionConfig(
                node_id="test-node",
                node_type="EFFECT",
                request_introspection_topic="onex/registry/introspection",
            )

    async def test_empty_after_prefix_rejected(self) -> None:
        """Test that topic names with only 'onex.' prefix are rejected."""
        from pydantic import ValidationError

        from omnibase_infra.mixins import ModelIntrospectionConfig

        with pytest.raises(ValidationError) as exc_info:
            ModelIntrospectionConfig(
                node_id="test-node",
                node_type="EFFECT",
                introspection_topic="onex.",
            )

        error_message = str(exc_info.value)
        assert "content after" in error_message.lower()

    async def test_valid_characters_in_topic_names(self) -> None:
        """Test that valid characters (alphanumeric, dots, hyphens, underscores) are allowed."""
        from omnibase_infra.mixins import ModelIntrospectionConfig

        # Test with hyphens and underscores
        config = ModelIntrospectionConfig(
            node_id="test-node",
            node_type="EFFECT",
            introspection_topic="onex.my-node_introspection.published.v1",
            heartbeat_topic="onex.node-1_heartbeat.published.v2",
            request_introspection_topic="onex.registry_v2.introspection-request.v1",
        )

        assert config.introspection_topic == "onex.my-node_introspection.published.v1"
        assert config.heartbeat_topic == "onex.node-1_heartbeat.published.v2"
        assert (
            config.request_introspection_topic
            == "onex.registry_v2.introspection-request.v1"
        )

    async def test_multiple_topic_validation_errors(self) -> None:
        """Test that multiple invalid topics produce multiple validation errors."""
        from pydantic import ValidationError

        from omnibase_infra.mixins import ModelIntrospectionConfig

        with pytest.raises(ValidationError) as exc_info:
            ModelIntrospectionConfig(
                node_id="test-node",
                node_type="EFFECT",
                introspection_topic="bad.introspection.topic",
                heartbeat_topic="bad.heartbeat.topic",
                request_introspection_topic="bad.request.topic",
            )

        # Should have 3 validation errors
        errors = exc_info.value.errors()
        assert len(errors) == 3

        # All should be about missing prefix
        for error in errors:
            assert "onex." in str(error["msg"])

    async def test_case_sensitive_prefix(self) -> None:
        """Test that the 'onex.' prefix is case-sensitive."""
        from pydantic import ValidationError

        from omnibase_infra.mixins import ModelIntrospectionConfig

        # ONEX. (uppercase) should fail
        with pytest.raises(ValidationError):
            ModelIntrospectionConfig(
                node_id="test-node",
                node_type="EFFECT",
                introspection_topic="ONEX.node.introspection.published.v1",
            )

        # Onex. (mixed case) should fail
        with pytest.raises(ValidationError):
            ModelIntrospectionConfig(
                node_id="test-node",
                node_type="EFFECT",
                introspection_topic="Onex.node.introspection.published.v1",
            )

    async def test_validation_error_messages_are_descriptive(self) -> None:
        """Test that validation error messages provide clear guidance."""
        from pydantic import ValidationError

        from omnibase_infra.mixins import ModelIntrospectionConfig

        # Test missing prefix error message
        with pytest.raises(ValidationError) as exc_info:
            ModelIntrospectionConfig(
                node_id="test-node",
                node_type="EFFECT",
                introspection_topic="custom.topic.name",
            )

        error_msg = str(exc_info.value)
        assert "onex." in error_msg
        assert "custom.topic.name" in error_msg  # Shows the actual invalid value

        # Test invalid characters error message
        with pytest.raises(ValidationError) as exc_info:
            ModelIntrospectionConfig(
                node_id="test-node",
                node_type="EFFECT",
                heartbeat_topic="onex.topic with spaces",
            )

        error_msg = str(exc_info.value)
        assert "invalid characters" in error_msg.lower()
        assert "onex.topic with spaces" in error_msg


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinNodeIntrospectionCustomTopics:
    """Tests for custom topic configuration via ModelIntrospectionConfig.

    These tests verify that custom topics are correctly used when publishing
    introspection events, heartbeats, and when setting up registry listeners.
    """

    async def test_custom_introspection_topic_used_in_publishing(self) -> None:
        """Test that custom introspection topic is used when publishing."""
        from omnibase_infra.mixins import ModelIntrospectionConfig

        event_bus = MockEventBus()
        custom_topic = "onex.my.custom.introspection.topic.v1"

        config = ModelIntrospectionConfig(
            node_id="custom-topic-node",
            node_type="EFFECT",
            event_bus=event_bus,
            introspection_topic=custom_topic,
        )

        node = MockNode()
        node.initialize_introspection(config=config)

        # Publish introspection
        success = await node.publish_introspection(reason="test")
        assert success is True

        # Verify the custom topic was used
        assert len(event_bus.published_envelopes) == 1
        _, topic = event_bus.published_envelopes[0]
        assert topic == custom_topic

    async def test_custom_topics_default_to_class_defaults(self) -> None:
        """Test that topics default to class-level defaults when not specified."""
        node = MockNode()
        node.initialize_introspection(
            node_id="default-topics-node",
            node_type="EFFECT",
            event_bus=None,
        )

        # Verify class-level defaults are used
        assert (
            node._introspection_topic
            == MixinNodeIntrospection.DEFAULT_INTROSPECTION_TOPIC
        )
        assert node._heartbeat_topic == MixinNodeIntrospection.DEFAULT_HEARTBEAT_TOPIC
        assert (
            node._request_introspection_topic
            == MixinNodeIntrospection.DEFAULT_REQUEST_INTROSPECTION_TOPIC
        )

    async def test_custom_heartbeat_topic_used_in_heartbeat_publishing(self) -> None:
        """Test that custom heartbeat topic is used in heartbeat publishing."""
        from omnibase_infra.mixins import ModelIntrospectionConfig

        event_bus = MockEventBus()
        custom_heartbeat_topic = "onex.my.custom.heartbeat.topic.v1"

        config = ModelIntrospectionConfig(
            node_id="custom-heartbeat-node",
            node_type="COMPUTE",
            event_bus=event_bus,
            heartbeat_topic=custom_heartbeat_topic,
        )

        node = MockNode()
        node.initialize_introspection(config=config)

        # Start heartbeat task with fast interval
        await node.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=0.05,
            enable_registry_listener=False,
        )

        try:
            # Wait for at least one heartbeat
            await asyncio.sleep(0.1)

            # Verify the custom heartbeat topic was used
            assert len(event_bus.published_envelopes) >= 1

            # Check that heartbeat events use the custom topic
            for _, topic in event_bus.published_envelopes:
                assert topic == custom_heartbeat_topic
        finally:
            await node.stop_introspection_tasks()

    async def test_custom_request_topic_stored_for_registry_listener(self) -> None:
        """Test that custom request introspection topic is stored correctly.

        The registry listener uses the request_introspection_topic when
        subscribing to introspection requests. This test verifies the topic
        is correctly stored for use by the registry listener.
        """
        from omnibase_infra.mixins import ModelIntrospectionConfig

        custom_request_topic = "onex.my.custom.request.introspection.topic.v1"

        config = ModelIntrospectionConfig(
            node_id="custom-request-topic-node",
            node_type="REDUCER",
            request_introspection_topic=custom_request_topic,
        )

        node = MockNode()
        node.initialize_introspection(config=config)

        # Verify the custom request topic is stored
        assert node._request_introspection_topic == custom_request_topic

    async def test_all_custom_topics_used_together(self) -> None:
        """Test that all three custom topics work when configured together."""
        from omnibase_infra.mixins import ModelIntrospectionConfig

        event_bus = MockEventBus()
        custom_introspection = "onex.custom.intro.topic"
        custom_heartbeat = "onex.custom.heartbeat.topic"
        custom_request = "onex.custom.request.topic"

        config = ModelIntrospectionConfig(
            node_id="all-custom-topics-node",
            node_type="ORCHESTRATOR",
            event_bus=event_bus,
            introspection_topic=custom_introspection,
            heartbeat_topic=custom_heartbeat,
            request_introspection_topic=custom_request,
        )

        node = MockNode()
        node.initialize_introspection(config=config)

        # Verify all topics are stored correctly
        assert node._introspection_topic == custom_introspection
        assert node._heartbeat_topic == custom_heartbeat
        assert node._request_introspection_topic == custom_request

        # Test introspection publishing uses custom topic
        await node.publish_introspection(reason="test")
        assert len(event_bus.published_envelopes) == 1
        _, intro_topic = event_bus.published_envelopes[0]
        assert intro_topic == custom_introspection

        # Clear and test heartbeat uses custom topic
        event_bus.published_envelopes.clear()
        await node.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=0.05,
            enable_registry_listener=False,
        )

        try:
            await asyncio.sleep(0.1)
            assert len(event_bus.published_envelopes) >= 1
            for _, hb_topic in event_bus.published_envelopes:
                assert hb_topic == custom_heartbeat
        finally:
            await node.stop_introspection_tasks()

    async def test_partial_custom_topics_with_defaults(self) -> None:
        """Test that unspecified topics fall back to defaults."""
        from omnibase_infra.mixins import ModelIntrospectionConfig

        custom_introspection = "onex.only.introspection.custom"

        config = ModelIntrospectionConfig(
            node_id="partial-topics-node",
            node_type="EFFECT",
            introspection_topic=custom_introspection,
            # heartbeat_topic and request_introspection_topic not specified
        )

        node = MockNode()
        node.initialize_introspection(config=config)

        # Custom topic should be set
        assert node._introspection_topic == custom_introspection
        # Others should use class-level defaults
        assert node._heartbeat_topic == MixinNodeIntrospection.DEFAULT_HEARTBEAT_TOPIC
        assert (
            node._request_introspection_topic
            == MixinNodeIntrospection.DEFAULT_REQUEST_INTROSPECTION_TOPIC
        )

    async def test_introspection_topic_with_fallback_publish_method(self) -> None:
        """Test custom topic used with fallback publish method (no publish_envelope).

        This test validates the fallback path when an event bus doesn't have
        publish_envelope method. Uses legacy parameter path to bypass Pydantic
        protocol validation.
        """

        class NoEnvelopeEventBus:
            """Mock event bus without publish_envelope method (only publish)."""

            def __init__(self) -> None:
                self.published_events: list[dict[str, str | bytes | None]] = []

            async def publish(
                self,
                topic: str,
                key: bytes | None,
                value: bytes,
            ) -> None:
                self.published_events.append(
                    {
                        "topic": topic,
                        "key": key,
                        "value": value,
                    }
                )

        # Use legacy parameter path to bypass Pydantic protocol validation
        # This allows testing the fallback path with a minimal event bus
        event_bus = NoEnvelopeEventBus()
        custom_topic = "onex.fallback.publish.topic.v1"

        node = MockNode()
        # Cast to Any to bypass type checking for testing fallback behavior
        node.initialize_introspection(
            node_id="fallback-publish-node",
            node_type="EFFECT",
            event_bus=event_bus,  # type: ignore[arg-type]
            introspection_topic=custom_topic,
        )

        # Publish using fallback method
        success = await node.publish_introspection(reason="fallback-test")
        assert success is True

        # Verify custom topic was used in fallback publish
        assert len(event_bus.published_events) == 1
        assert event_bus.published_events[0]["topic"] == custom_topic

    async def test_heartbeat_topic_with_fallback_publish_method(self) -> None:
        """Test custom heartbeat topic used with fallback publish method.

        This test validates the fallback path when an event bus doesn't have
        publish_envelope method. Uses legacy parameter path to bypass Pydantic
        protocol validation.
        """

        class NoEnvelopeEventBus:
            """Mock event bus without publish_envelope method (only publish)."""

            def __init__(self) -> None:
                self.published_events: list[dict[str, str | bytes | None]] = []

            async def publish(
                self,
                topic: str,
                key: bytes | None,
                value: bytes,
            ) -> None:
                self.published_events.append(
                    {
                        "topic": topic,
                        "key": key,
                        "value": value,
                    }
                )

        # Use legacy parameter path to bypass Pydantic protocol validation
        # This allows testing the fallback path with a minimal event bus
        event_bus = NoEnvelopeEventBus()
        custom_heartbeat_topic = "onex.fallback.heartbeat.topic.v1"

        node = MockNode()
        # Cast to Any to bypass type checking for testing fallback behavior
        node.initialize_introspection(
            node_id="fallback-heartbeat-node",
            node_type="COMPUTE",
            event_bus=event_bus,  # type: ignore[arg-type]
            heartbeat_topic=custom_heartbeat_topic,
        )

        # Start heartbeat
        await node.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=0.05,
            enable_registry_listener=False,
        )

        try:
            await asyncio.sleep(0.1)

            # Verify custom heartbeat topic was used in fallback publish
            assert len(event_bus.published_events) >= 1
            for event in event_bus.published_events:
                assert event["topic"] == custom_heartbeat_topic
        finally:
            await node.stop_introspection_tasks()

    async def test_empty_topic_strings_use_defaults(self) -> None:
        """Test that None topic values correctly fall back to defaults."""
        from omnibase_infra.mixins import ModelIntrospectionConfig

        # Create config with explicit None values
        config = ModelIntrospectionConfig(
            node_id="none-topics-node",
            node_type="EFFECT",
            introspection_topic=None,
            heartbeat_topic=None,
            request_introspection_topic=None,
        )

        node = MockNode()
        node.initialize_introspection(config=config)

        # All should fall back to class-level defaults
        assert (
            node._introspection_topic
            == MixinNodeIntrospection.DEFAULT_INTROSPECTION_TOPIC
        )
        assert node._heartbeat_topic == MixinNodeIntrospection.DEFAULT_HEARTBEAT_TOPIC
        assert (
            node._request_introspection_topic
            == MixinNodeIntrospection.DEFAULT_REQUEST_INTROSPECTION_TOPIC
        )

    async def test_get_introspection_data_includes_topic_info(self) -> None:
        """Test that introspection data includes configured topic information.

        The mixin stores topic configuration for debugging and observability.
        This test verifies topics are properly reported in introspection output.
        """
        from omnibase_infra.mixins import ModelIntrospectionConfig

        custom_introspection = "onex.debug.introspection.topic"
        custom_heartbeat = "onex.debug.heartbeat.topic"
        custom_request = "onex.debug.request.topic"

        config = ModelIntrospectionConfig(
            node_id="topic-info-node",
            node_type="EFFECT",
            introspection_topic=custom_introspection,
            heartbeat_topic=custom_heartbeat,
            request_introspection_topic=custom_request,
        )

        node = MockNode()
        node.initialize_introspection(config=config)

        # Verify internal attributes are set (for debugging/logging purposes)
        assert hasattr(node, "_introspection_topic")
        assert hasattr(node, "_heartbeat_topic")
        assert hasattr(node, "_request_introspection_topic")

        assert node._introspection_topic == custom_introspection
        assert node._heartbeat_topic == custom_heartbeat
        assert node._request_introspection_topic == custom_request

    async def test_legacy_params_custom_introspection_topic(self) -> None:
        """Test custom introspection topic via legacy parameter path."""
        event_bus = MockEventBus()
        custom_topic = "onex.legacy.introspection.topic.v1"

        node = MockNode()
        node.initialize_introspection(
            node_id="legacy-intro-topic-node",
            node_type="EFFECT",
            event_bus=event_bus,
            introspection_topic=custom_topic,
        )

        # Verify topic is stored correctly
        assert node._introspection_topic == custom_topic

        # Verify topic is used when publishing
        success = await node.publish_introspection(reason="legacy-test")
        assert success is True

        assert len(event_bus.published_envelopes) == 1
        _, topic = event_bus.published_envelopes[0]
        assert topic == custom_topic

    async def test_legacy_params_custom_heartbeat_topic(self) -> None:
        """Test custom heartbeat topic via legacy parameter path."""
        event_bus = MockEventBus()
        custom_heartbeat_topic = "onex.legacy.heartbeat.topic.v1"

        node = MockNode()
        node.initialize_introspection(
            node_id="legacy-heartbeat-topic-node",
            node_type="COMPUTE",
            event_bus=event_bus,
            heartbeat_topic=custom_heartbeat_topic,
        )

        # Verify topic is stored correctly
        assert node._heartbeat_topic == custom_heartbeat_topic

        # Start heartbeat task with fast interval
        await node.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=0.05,
            enable_registry_listener=False,
        )

        try:
            # Wait for at least one heartbeat
            await asyncio.sleep(0.1)

            # Verify the custom heartbeat topic was used
            assert len(event_bus.published_envelopes) >= 1

            # Check that heartbeat events use the custom topic
            for _, topic in event_bus.published_envelopes:
                assert topic == custom_heartbeat_topic
        finally:
            await node.stop_introspection_tasks()

    async def test_legacy_params_custom_request_topic(self) -> None:
        """Test custom request introspection topic via legacy parameter path."""
        custom_request_topic = "onex.legacy.request.introspection.topic.v1"

        node = MockNode()
        node.initialize_introspection(
            node_id="legacy-request-topic-node",
            node_type="REDUCER",
            request_introspection_topic=custom_request_topic,
        )

        # Verify topic is stored correctly
        assert node._request_introspection_topic == custom_request_topic

    async def test_legacy_params_all_custom_topics(self) -> None:
        """Test all three custom topics via legacy parameter path."""
        event_bus = MockEventBus()
        custom_introspection = "onex.legacy.intro.topic"
        custom_heartbeat = "onex.legacy.heartbeat.topic"
        custom_request = "onex.legacy.request.topic"

        node = MockNode()
        node.initialize_introspection(
            node_id="legacy-all-topics-node",
            node_type="ORCHESTRATOR",
            event_bus=event_bus,
            introspection_topic=custom_introspection,
            heartbeat_topic=custom_heartbeat,
            request_introspection_topic=custom_request,
        )

        # Verify all topics are stored correctly
        assert node._introspection_topic == custom_introspection
        assert node._heartbeat_topic == custom_heartbeat
        assert node._request_introspection_topic == custom_request

        # Test introspection publishing uses custom topic
        await node.publish_introspection(reason="legacy-test")
        assert len(event_bus.published_envelopes) == 1
        _, intro_topic = event_bus.published_envelopes[0]
        assert intro_topic == custom_introspection

        # Clear and test heartbeat uses custom topic
        event_bus.published_envelopes.clear()
        await node.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=0.05,
            enable_registry_listener=False,
        )

        try:
            await asyncio.sleep(0.1)
            assert len(event_bus.published_envelopes) >= 1
            for _, hb_topic in event_bus.published_envelopes:
                assert hb_topic == custom_heartbeat
        finally:
            await node.stop_introspection_tasks()

    async def test_registry_listener_uses_custom_request_topic(self) -> None:
        """Test that registry listener subscribes to the custom request topic.

        This test verifies that the registry listener loop uses the
        custom request_introspection_topic when subscribing to the event bus.
        """

        class CapturingEventBus:
            """Event bus that captures subscription topic."""

            def __init__(self) -> None:
                self.subscribed_topic: str | None = None
                self.subscribed_group_id: str | None = None
                self.published_envelopes: list[tuple[Any, str]] = []

            async def publish_envelope(
                self,
                envelope: Any,
                topic: str,
            ) -> None:
                self.published_envelopes.append((envelope, topic))

            async def publish(
                self,
                topic: str,
                key: bytes | None,
                value: bytes,
            ) -> None:
                pass

            async def subscribe(
                self,
                topic: str,
                group_id: str,
                on_message: Callable[[Any], Awaitable[None]],
            ) -> Callable[[], Awaitable[None]]:
                self.subscribed_topic = topic
                self.subscribed_group_id = group_id

                async def unsubscribe() -> None:
                    pass

                return unsubscribe

        from omnibase_infra.mixins import ModelIntrospectionConfig

        event_bus = CapturingEventBus()
        custom_request_topic = "onex.custom.registry.request.topic.v1"

        config = ModelIntrospectionConfig(
            node_id="registry-custom-topic-node",
            node_type="EFFECT",
            event_bus=event_bus,  # type: ignore[arg-type]
            request_introspection_topic=custom_request_topic,
        )

        node = MockNode()
        node.initialize_introspection(config=config)

        # Start registry listener
        await node.start_introspection_tasks(
            enable_heartbeat=False,
            enable_registry_listener=True,
        )

        try:
            # Give time for subscription to complete
            await asyncio.sleep(0.1)

            # Verify the custom request topic was used for subscription
            assert event_bus.subscribed_topic == custom_request_topic
            assert (
                event_bus.subscribed_group_id
                == "introspection-registry-custom-topic-node"
            )
        finally:
            await node.stop_introspection_tasks()

    async def test_registry_listener_uses_custom_request_topic_legacy_params(
        self,
    ) -> None:
        """Test registry listener with custom request topic via legacy params."""

        class CapturingEventBus:
            """Event bus that captures subscription topic."""

            def __init__(self) -> None:
                self.subscribed_topic: str | None = None
                self.subscribed_group_id: str | None = None

            async def publish_envelope(
                self,
                envelope: Any,
                topic: str,
            ) -> None:
                pass

            async def publish(
                self,
                topic: str,
                key: bytes | None,
                value: bytes,
            ) -> None:
                pass

            async def subscribe(
                self,
                topic: str,
                group_id: str,
                on_message: Callable[[Any], Awaitable[None]],
            ) -> Callable[[], Awaitable[None]]:
                self.subscribed_topic = topic
                self.subscribed_group_id = group_id

                async def unsubscribe() -> None:
                    pass

                return unsubscribe

        event_bus = CapturingEventBus()
        custom_request_topic = "onex.legacy.registry.request.topic.v1"

        node = MockNode()
        node.initialize_introspection(
            node_id="legacy-registry-topic-node",
            node_type="COMPUTE",
            event_bus=event_bus,  # type: ignore[arg-type]
            request_introspection_topic=custom_request_topic,
        )

        # Start registry listener
        await node.start_introspection_tasks(
            enable_heartbeat=False,
            enable_registry_listener=True,
        )

        try:
            # Give time for subscription to complete
            await asyncio.sleep(0.1)

            # Verify the custom request topic was used for subscription
            assert event_bus.subscribed_topic == custom_request_topic
            assert (
                event_bus.subscribed_group_id
                == "introspection-legacy-registry-topic-node"
            )
        finally:
            await node.stop_introspection_tasks()

    async def test_default_topics_used_without_custom_config(self) -> None:
        """Test that class-level default topics are used when no custom topics provided."""
        event_bus = MockEventBus()

        node = MockNode()
        node.initialize_introspection(
            node_id="default-topics-verify-node",
            node_type="EFFECT",
            event_bus=event_bus,
        )

        # Verify class-level defaults are used
        assert (
            node._introspection_topic
            == MixinNodeIntrospection.DEFAULT_INTROSPECTION_TOPIC
        )
        assert node._heartbeat_topic == MixinNodeIntrospection.DEFAULT_HEARTBEAT_TOPIC
        assert (
            node._request_introspection_topic
            == MixinNodeIntrospection.DEFAULT_REQUEST_INTROSPECTION_TOPIC
        )

        # Verify introspection uses default topic
        await node.publish_introspection(reason="default-test")
        assert len(event_bus.published_envelopes) == 1
        _, topic = event_bus.published_envelopes[0]
        assert topic == MixinNodeIntrospection.DEFAULT_INTROSPECTION_TOPIC

    async def test_config_model_default_topics_used(self) -> None:
        """Test that default topics are used when config model has None topics."""
        from omnibase_infra.mixins import ModelIntrospectionConfig

        event_bus = MockEventBus()

        # Create config without specifying any topics
        config = ModelIntrospectionConfig(
            node_id="config-default-topics-node",
            node_type="EFFECT",
            event_bus=event_bus,
        )

        node = MockNode()
        node.initialize_introspection(config=config)

        # Verify class-level defaults are used
        assert (
            node._introspection_topic
            == MixinNodeIntrospection.DEFAULT_INTROSPECTION_TOPIC
        )
        assert node._heartbeat_topic == MixinNodeIntrospection.DEFAULT_HEARTBEAT_TOPIC
        assert (
            node._request_introspection_topic
            == MixinNodeIntrospection.DEFAULT_REQUEST_INTROSPECTION_TOPIC
        )

        # Verify introspection uses default topic
        await node.publish_introspection(reason="config-default-test")
        assert len(event_bus.published_envelopes) == 1
        _, topic = event_bus.published_envelopes[0]
        assert topic == MixinNodeIntrospection.DEFAULT_INTROSPECTION_TOPIC


@pytest.mark.asyncio(loop_scope="function")
class TestMixinNodeIntrospectionClassLevelDefaults:
    """Test class-level default topic constants for subclass overrides."""

    async def test_class_level_default_topics_exist(self) -> None:
        """Test that class-level default topic constants are defined."""
        assert hasattr(MixinNodeIntrospection, "DEFAULT_INTROSPECTION_TOPIC")
        assert hasattr(MixinNodeIntrospection, "DEFAULT_HEARTBEAT_TOPIC")
        assert hasattr(MixinNodeIntrospection, "DEFAULT_REQUEST_INTROSPECTION_TOPIC")

        # Verify they have the expected values
        assert (
            MixinNodeIntrospection.DEFAULT_INTROSPECTION_TOPIC
            == "onex.node.introspection.published.v1"
        )
        assert (
            MixinNodeIntrospection.DEFAULT_HEARTBEAT_TOPIC
            == "onex.node.heartbeat.published.v1"
        )
        assert (
            MixinNodeIntrospection.DEFAULT_REQUEST_INTROSPECTION_TOPIC
            == "onex.registry.introspection.requested.v1"
        )

    async def test_subclass_can_override_default_topics(self) -> None:
        """Test that subclasses can override default topic class variables."""

        class TenantNode(MixinNodeIntrospection):
            """A tenant-specific node with custom topic defaults."""

            DEFAULT_INTROSPECTION_TOPIC = "onex.tenant1.introspection.published.v1"
            DEFAULT_HEARTBEAT_TOPIC = "onex.tenant1.heartbeat.published.v1"
            DEFAULT_REQUEST_INTROSPECTION_TOPIC = (
                "onex.tenant1.introspection.requested.v1"
            )

        node = TenantNode()
        node.initialize_introspection(
            node_id="tenant-node",
            node_type="EFFECT",
            event_bus=None,
        )

        # Verify that the subclass defaults are used
        assert node._introspection_topic == "onex.tenant1.introspection.published.v1"
        assert node._heartbeat_topic == "onex.tenant1.heartbeat.published.v1"
        assert (
            node._request_introspection_topic
            == "onex.tenant1.introspection.requested.v1"
        )

    async def test_subclass_defaults_used_in_publishing(self) -> None:
        """Test that subclass default topics are used in publish operations."""

        class CustomTopicNode(MixinNodeIntrospection):
            """A node with custom topic defaults."""

            DEFAULT_INTROSPECTION_TOPIC = "onex.custom.introspection.published.v1"

        event_bus = MockEventBus()
        node = CustomTopicNode()
        node.initialize_introspection(
            node_id="custom-topic-node",
            node_type="EFFECT",
            event_bus=event_bus,
        )

        await node.publish_introspection(reason="test")

        # Verify the custom topic was used
        assert len(event_bus.published_envelopes) == 1
        _, topic = event_bus.published_envelopes[0]
        assert topic == "onex.custom.introspection.published.v1"

    async def test_config_model_overrides_subclass_defaults(self) -> None:
        """Test that config model topics override subclass defaults."""
        from omnibase_infra.mixins import ModelIntrospectionConfig

        class SubclassNode(MixinNodeIntrospection):
            """A subclass with custom defaults."""

            DEFAULT_INTROSPECTION_TOPIC = "onex.subclass.introspection.published.v1"

        event_bus = MockEventBus()

        # Config specifies a different topic that should override subclass default
        config = ModelIntrospectionConfig(
            node_id="override-node",
            node_type="EFFECT",
            event_bus=event_bus,
            introspection_topic="onex.config.introspection.published.v1",
        )

        node = SubclassNode()
        node.initialize_introspection(config=config)

        # Verify config topic overrides subclass default
        assert node._introspection_topic == "onex.config.introspection.published.v1"

        await node.publish_introspection(reason="override-test")

        _, topic = event_bus.published_envelopes[0]
        assert topic == "onex.config.introspection.published.v1"

    async def test_multiple_subclasses_have_independent_defaults(self) -> None:
        """Test that different subclasses can have independent default topics."""

        class TenantANode(MixinNodeIntrospection):
            """Node for Tenant A."""

            DEFAULT_INTROSPECTION_TOPIC = "onex.tenantA.introspection.published.v1"

        class TenantBNode(MixinNodeIntrospection):
            """Node for Tenant B."""

            DEFAULT_INTROSPECTION_TOPIC = "onex.tenantB.introspection.published.v1"

        event_bus_a = MockEventBus()
        event_bus_b = MockEventBus()

        node_a = TenantANode()
        node_a.initialize_introspection(
            node_id="tenant-a-node",
            node_type="EFFECT",
            event_bus=event_bus_a,
        )

        node_b = TenantBNode()
        node_b.initialize_introspection(
            node_id="tenant-b-node",
            node_type="EFFECT",
            event_bus=event_bus_b,
        )

        # Verify independent defaults
        assert node_a._introspection_topic == "onex.tenantA.introspection.published.v1"
        assert node_b._introspection_topic == "onex.tenantB.introspection.published.v1"

        # Verify publishing uses correct topics
        await node_a.publish_introspection(reason="tenant-a")
        await node_b.publish_introspection(reason="tenant-b")

        _, topic_a = event_bus_a.published_envelopes[0]
        _, topic_b = event_bus_b.published_envelopes[0]

        assert topic_a == "onex.tenantA.introspection.published.v1"
        assert topic_b == "onex.tenantB.introspection.published.v1"

    async def test_class_defaults_have_expected_values(self) -> None:
        """Test that class defaults have the expected ONEX topic values."""
        # Verify class defaults have the expected values following ONEX naming convention
        assert (
            MixinNodeIntrospection.DEFAULT_INTROSPECTION_TOPIC
            == "onex.node.introspection.published.v1"
        )
        assert (
            MixinNodeIntrospection.DEFAULT_HEARTBEAT_TOPIC
            == "onex.node.heartbeat.published.v1"
        )
        assert (
            MixinNodeIntrospection.DEFAULT_REQUEST_INTROSPECTION_TOPIC
            == "onex.registry.introspection.requested.v1"
        )
