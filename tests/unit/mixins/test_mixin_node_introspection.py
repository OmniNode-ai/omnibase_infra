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
- Performance requirements (<50ms)

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

Coverage Goals:
    - >90% code coverage for mixin
    - All public methods tested
    - Error paths validated
    - Performance requirements verified
"""

import asyncio
import time
from typing import Any
from uuid import uuid4

import pytest

from omnibase_infra.mixins.mixin_node_introspection import MixinNodeIntrospection
from omnibase_infra.models.discovery import ModelNodeIntrospectionEvent


class MockEventBus:
    """Mock event bus for testing introspection publishing."""

    def __init__(self, should_fail: bool = False) -> None:
        """Initialize mock event bus.

        Args:
            should_fail: If True, publish operations will raise exceptions.
        """
        self.should_fail = should_fail
        self.published_envelopes: list[tuple[Any, str]] = []
        self.published_events: list[dict[str, Any]] = []

    async def publish_envelope(
        self,
        envelope: Any,
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
        import json

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

    async def execute(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Mock execute method.

        Args:
            operation: Operation to execute.
            payload: Operation payload.

        Returns:
            Operation result.
        """
        return {"result": "ok", "operation": operation}

    async def health_check(self) -> dict[str, Any]:
        """Mock health check.

        Returns:
            Health status.
        """
        return {"healthy": True, "state": self._state}

    async def handle_event(self, event: dict[str, Any]) -> None:
        """Mock handle_event method (should be discovered as operation).

        Args:
            event: Event to handle.
        """
        _ = event  # Silence unused parameter warning

    async def process_batch(self, items: list[Any]) -> list[Any]:
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

    async def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Mock process method.

        Args:
            data: Data to process.

        Returns:
            Processed result.
        """
        return {"processed": True}


class MockNodeNoState(MixinNodeIntrospection):
    """Mock node without _state attribute."""

    async def execute(self, operation: str) -> dict[str, Any]:
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
            value = "running"

        self._state = State()


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
        assert "MixinNodeIntrospection" in capabilities["protocols"]

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
            async def publish_envelope(self, envelope: Any, topic: str) -> None:
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
    """Tests for performance requirements."""

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
        """Test that introspection data extraction completes in under 50ms."""
        # Clear cache to force full computation
        mock_node._introspection_cache = None
        mock_node._introspection_cached_at = None

        start = time.time()
        await mock_node.get_introspection_data()
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < 50, f"Introspection took {elapsed_ms:.2f}ms, expected <50ms"

    async def test_cached_introspection_under_1ms(self, mock_node: MockNode) -> None:
        """Test that cached introspection returns in under 1ms."""
        # Populate cache
        await mock_node.get_introspection_data()

        start = time.time()
        await mock_node.get_introspection_data()
        elapsed_ms = (time.time() - start) * 1000

        assert (
            elapsed_ms < 1
        ), f"Cached introspection took {elapsed_ms:.2f}ms, expected <1ms"

    async def test_capability_extraction_under_10ms(self, mock_node: MockNode) -> None:
        """Test that capability extraction completes in under 10ms."""
        start = time.time()
        await mock_node.get_capabilities()
        elapsed_ms = (time.time() - start) * 1000

        assert (
            elapsed_ms < 10
        ), f"Capability extraction took {elapsed_ms:.2f}ms, expected <10ms"

    async def test_endpoint_discovery_under_10ms(self, mock_node: MockNode) -> None:
        """Test that endpoint discovery completes in under 10ms."""
        start = time.time()
        await mock_node.get_endpoints()
        elapsed_ms = (time.time() - start) * 1000

        assert (
            elapsed_ms < 10
        ), f"Endpoint discovery took {elapsed_ms:.2f}ms, expected <10ms"

    async def test_state_extraction_under_1ms(self, mock_node: MockNode) -> None:
        """Test that state extraction completes in under 1ms."""
        start = time.time()
        await mock_node.get_current_state()
        elapsed_ms = (time.time() - start) * 1000

        assert (
            elapsed_ms < 1
        ), f"State extraction took {elapsed_ms:.2f}ms, expected <1ms"

    async def test_multiple_introspection_calls_consistent_performance(
        self, mock_node: MockNode
    ) -> None:
        """Test that multiple introspection calls have consistent performance."""
        times = []

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

        assert avg_time < 30, f"Average time {avg_time:.2f}ms, expected <30ms"
        assert max_time < 50, f"Max time {max_time:.2f}ms, expected <50ms"


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

        start = time.time()
        capabilities = await node.get_capabilities()
        elapsed_ms = (time.time() - start) * 1000

        # Should include all 10 operation methods
        assert len(capabilities["operations"]) >= 10
        assert elapsed_ms < 50

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
