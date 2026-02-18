# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for topic-based routing via registry (OMN-1613).

Tests the registry-driven topic routing:
1. Nodes register with topic subscriptions
2. Topic index built in Consul KV
3. Runtime queries for subscribers
4. Correct node IDs returned for routing

Test Categories:
    - Topic Subscriber Index: Testing topic -> node_id index operations
    - Topic Index Idempotency: Testing idempotent index updates
    - Topic Query Routing: Testing subscriber queries for routing
    - Registration to Routing: End-to-end flow tests

These tests use an in-memory mock of Consul KV to enable testing without
requiring a live Consul instance. For live Consul tests, see
tests/integration/handlers/test_consul_handler_integration.py.

Related Ticket: OMN-1613
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TypeVar
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.handlers.mixins.mixin_consul_topic_index import (
    MixinConsulTopicIndex,
)
from omnibase_infra.models.registration.model_node_event_bus_config import (
    ModelEventBusTopicEntry,
    ModelNodeEventBusConfig,
)

# Type variable for retry function
T = TypeVar("T")

# Module-level markers
pytestmark = [
    pytest.mark.asyncio,
]


# =============================================================================
# Mock Consul KV Store
# =============================================================================


class MockConsulKV:
    """In-memory mock of Consul KV store for testing.

    This mock provides a simple dict-based KV store that mimics the
    Consul python-consul client's KV interface.
    """

    def __init__(self) -> None:
        """Initialize empty KV store."""
        self._store: dict[str, str] = {}
        self._index = 0

    def get(
        self, key: str, recurse: bool = False
    ) -> tuple[int, dict[str, object] | None]:
        """Get value from KV store.

        Args:
            key: KV key path.
            recurse: Whether to recurse (ignored in mock).

        Returns:
            Tuple of (index, data dict or None if not found).
        """
        if key not in self._store:
            return self._index, None

        value = self._store[key]
        return self._index, {
            "Key": key,
            "Value": value.encode("utf-8"),
            "Flags": 0,
            "CreateIndex": self._index,
            "ModifyIndex": self._index,
        }

    def put(self, key: str, value: str) -> bool:
        """Put value to KV store.

        Args:
            key: KV key path.
            value: Value to store.

        Returns:
            True on success.
        """
        self._store[key] = value
        self._index += 1
        return True

    def delete(self, key: str) -> bool:
        """Delete key from KV store.

        Args:
            key: KV key path.

        Returns:
            True on success.
        """
        if key in self._store:
            del self._store[key]
        self._index += 1
        return True


class MockConsulClient:
    """Mock Consul client for testing."""

    def __init__(self) -> None:
        """Initialize mock client with KV store."""
        self.kv = MockConsulKV()


# =============================================================================
# Test Implementation of MixinConsulTopicIndex
# =============================================================================


class _TopicIndexMixinImpl(MixinConsulTopicIndex):
    """Testable implementation of MixinConsulTopicIndex.

    This class provides the required dependencies for the mixin:
    - _client: Mock Consul client
    - _config: Mock configuration
    - _execute_with_retry: Simple passthrough (no retry in tests)
    """

    def __init__(self) -> None:
        """Initialize with mock Consul client."""
        self._client: MockConsulClient | None = MockConsulClient()
        self._config: object | None = {}

    async def _execute_with_retry(
        self,
        operation: str,
        func: Callable[[], T],
        correlation_id: UUID,
    ) -> T:
        """Execute without retry for testing.

        Args:
            operation: Operation name (for logging).
            func: Function to execute.
            correlation_id: Correlation ID for tracing.

        Returns:
            Result of the function call.
        """
        return func()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def topic_index_mixin() -> _TopicIndexMixinImpl:
    """Create a testable topic index mixin instance.

    Returns:
        _TopicIndexMixinImpl with mock Consul client.
    """
    return _TopicIndexMixinImpl()


@pytest.fixture
def correlation_id() -> UUID:
    """Generate a unique correlation ID for test tracing.

    Returns:
        UUID for correlation tracking.
    """
    return uuid4()


@pytest.fixture
def sample_node_ids() -> list[str]:
    """Return sample node ID strings for testing.

    Returns:
        List of node ID strings (UUID format).
    """
    return [
        str(uuid4()),
        str(uuid4()),
        str(uuid4()),
    ]


@pytest.fixture
def sample_topics() -> list[str]:
    """Return sample topic strings for testing.

    Returns:
        List of topic strings.
    """
    return [
        "onex.evt.intent-classified.v1",
        "onex.evt.registration-requested.v1",
        "onex.cmd.register-node.v1",
    ]


# =============================================================================
# Topic Subscriber Index Tests
# =============================================================================


class TestTopicSubscriberIndex:
    """Tests for topic -> subscriber index operations."""

    async def test_single_node_single_topic(
        self,
        topic_index_mixin: _TopicIndexMixinImpl,
        correlation_id: UUID,
    ) -> None:
        """One node subscribing to one topic.

        Validates:
        1. Node can be added to a topic's subscriber list
        2. Subscriber list correctly contains the node ID
        3. Subscriber list is stored as sorted JSON array
        """
        node_id = str(uuid4())
        topic = "onex.evt.test-event.v1"

        # Create event bus config with single topic
        event_bus = ModelNodeEventBusConfig(
            subscribe_topics=[
                ModelEventBusTopicEntry(topic=topic),
            ],
        )

        # Update topic index FIRST (reads old state, updates index),
        # then store event bus config (stores new state for next time)
        await topic_index_mixin._update_topic_index(node_id, event_bus, correlation_id)
        await topic_index_mixin._store_node_event_bus(
            node_id, event_bus, correlation_id
        )

        # Verify subscriber list
        subscribers = await topic_index_mixin._get_topic_subscribers(
            topic, correlation_id
        )
        assert node_id in subscribers
        assert len(subscribers) == 1

    async def test_single_node_multiple_topics(
        self,
        topic_index_mixin: _TopicIndexMixinImpl,
        correlation_id: UUID,
        sample_topics: list[str],
    ) -> None:
        """One node subscribing to multiple topics.

        Validates:
        1. Node can subscribe to multiple topics
        2. Node appears in each topic's subscriber list
        3. Each topic's subscriber list is independent
        """
        node_id = str(uuid4())

        # Create event bus config with multiple topics
        event_bus = ModelNodeEventBusConfig(
            subscribe_topics=[
                ModelEventBusTopicEntry(topic=topic) for topic in sample_topics
            ],
        )

        # Update topic index FIRST, then store event bus config
        await topic_index_mixin._update_topic_index(node_id, event_bus, correlation_id)
        await topic_index_mixin._store_node_event_bus(
            node_id, event_bus, correlation_id
        )

        # Verify node is in all topic subscriber lists
        for topic in sample_topics:
            subscribers = await topic_index_mixin._get_topic_subscribers(
                topic, correlation_id
            )
            assert node_id in subscribers
            assert len(subscribers) == 1

    async def test_multiple_nodes_same_topic(
        self,
        topic_index_mixin: _TopicIndexMixinImpl,
        correlation_id: UUID,
        sample_node_ids: list[str],
    ) -> None:
        """Multiple nodes subscribing to same topic.

        Validates:
        1. Multiple nodes can subscribe to the same topic
        2. All node IDs appear in the subscriber list
        3. Subscriber list is properly aggregated
        """
        topic = "onex.evt.shared-event.v1"

        # Register each node with the same topic
        for node_id in sample_node_ids:
            event_bus = ModelNodeEventBusConfig(
                subscribe_topics=[
                    ModelEventBusTopicEntry(topic=topic),
                ],
            )
            # Update topic index FIRST, then store event bus config
            await topic_index_mixin._update_topic_index(
                node_id, event_bus, correlation_id
            )
            await topic_index_mixin._store_node_event_bus(
                node_id, event_bus, correlation_id
            )

        # Verify all nodes are in subscriber list
        subscribers = await topic_index_mixin._get_topic_subscribers(
            topic, correlation_id
        )
        assert len(subscribers) == len(sample_node_ids)
        for node_id in sample_node_ids:
            assert node_id in subscribers

    async def test_multiple_nodes_different_topics(
        self,
        topic_index_mixin: _TopicIndexMixinImpl,
        correlation_id: UUID,
    ) -> None:
        """Multiple nodes subscribing to different topics.

        Validates:
        1. Different nodes can have different topic subscriptions
        2. Each topic only contains its subscribers
        3. No cross-contamination between topic subscriber lists
        """
        # Node A subscribes to topic A
        node_a = str(uuid4())
        topic_a = "onex.evt.topic-a.v1"
        event_bus_a = ModelNodeEventBusConfig(
            subscribe_topics=[ModelEventBusTopicEntry(topic=topic_a)],
        )
        # Update topic index FIRST, then store event bus config
        await topic_index_mixin._update_topic_index(node_a, event_bus_a, correlation_id)
        await topic_index_mixin._store_node_event_bus(
            node_a, event_bus_a, correlation_id
        )

        # Node B subscribes to topic B
        node_b = str(uuid4())
        topic_b = "onex.evt.topic-b.v1"
        event_bus_b = ModelNodeEventBusConfig(
            subscribe_topics=[ModelEventBusTopicEntry(topic=topic_b)],
        )
        # Update topic index FIRST, then store event bus config
        await topic_index_mixin._update_topic_index(node_b, event_bus_b, correlation_id)
        await topic_index_mixin._store_node_event_bus(
            node_b, event_bus_b, correlation_id
        )

        # Verify topic A only has node A
        subscribers_a = await topic_index_mixin._get_topic_subscribers(
            topic_a, correlation_id
        )
        assert subscribers_a == [node_a]

        # Verify topic B only has node B
        subscribers_b = await topic_index_mixin._get_topic_subscribers(
            topic_b, correlation_id
        )
        assert subscribers_b == [node_b]

    async def test_node_with_no_subscriptions(
        self,
        topic_index_mixin: _TopicIndexMixinImpl,
        correlation_id: UUID,
    ) -> None:
        """Node without event_bus is not in any topic index.

        Validates:
        1. Empty subscribe_topics list doesn't create index entries
        2. Topic subscriber lists remain unaffected
        """
        node_id = str(uuid4())

        # Create event bus config with no subscriptions
        event_bus = ModelNodeEventBusConfig(
            subscribe_topics=[],
            publish_topics=[
                ModelEventBusTopicEntry(topic="onex.evt.output.v1"),
            ],
        )

        # Update topic index FIRST, then store event bus config
        await topic_index_mixin._update_topic_index(node_id, event_bus, correlation_id)
        await topic_index_mixin._store_node_event_bus(
            node_id, event_bus, correlation_id
        )

        # Verify node is not in any subscriber lists (check a non-existent topic)
        subscribers = await topic_index_mixin._get_topic_subscribers(
            "onex.evt.output.v1", correlation_id
        )
        assert node_id not in subscribers
        assert subscribers == []


# =============================================================================
# Topic Index Idempotency Tests
# =============================================================================


class TestTopicIndexIdempotency:
    """Tests for idempotent index updates."""

    async def test_add_subscriber_idempotent(
        self,
        topic_index_mixin: _TopicIndexMixinImpl,
        correlation_id: UUID,
    ) -> None:
        """Adding same subscriber twice doesn't duplicate.

        Validates:
        1. Calling _add_subscriber_to_topic multiple times is safe
        2. Node ID only appears once in subscriber list
        3. Order is maintained (sorted)
        """
        node_id = str(uuid4())
        topic = "onex.evt.idempotent-test.v1"

        # Add subscriber multiple times
        await topic_index_mixin._add_subscriber_to_topic(topic, node_id, correlation_id)
        await topic_index_mixin._add_subscriber_to_topic(topic, node_id, correlation_id)
        await topic_index_mixin._add_subscriber_to_topic(topic, node_id, correlation_id)

        # Verify only one entry
        subscribers = await topic_index_mixin._get_topic_subscribers(
            topic, correlation_id
        )
        assert subscribers == [node_id]
        assert len(subscribers) == 1

    async def test_remove_subscriber_idempotent(
        self,
        topic_index_mixin: _TopicIndexMixinImpl,
        correlation_id: UUID,
    ) -> None:
        """Removing non-existent subscriber is safe.

        Validates:
        1. Removing subscriber that doesn't exist doesn't raise
        2. Removing subscriber multiple times is safe
        3. Subscriber list remains consistent
        """
        node_id = str(uuid4())
        topic = "onex.evt.remove-test.v1"

        # Remove subscriber that was never added (should not raise)
        await topic_index_mixin._remove_subscriber_from_topic(
            topic, node_id, correlation_id
        )

        # Add then remove multiple times
        await topic_index_mixin._add_subscriber_to_topic(topic, node_id, correlation_id)
        await topic_index_mixin._remove_subscriber_from_topic(
            topic, node_id, correlation_id
        )
        await topic_index_mixin._remove_subscriber_from_topic(
            topic, node_id, correlation_id
        )

        # Verify subscriber list is empty
        subscribers = await topic_index_mixin._get_topic_subscribers(
            topic, correlation_id
        )
        assert subscribers == []

    async def test_index_delta_computation(
        self,
        topic_index_mixin: _TopicIndexMixinImpl,
        correlation_id: UUID,
    ) -> None:
        """Delta correctly identifies topics to add/remove.

        Validates:
        1. Re-registering with same topics doesn't change index
        2. Re-registering with new topics adds to index
        3. Re-registering with removed topics removes from index
        """
        node_id = str(uuid4())
        topic_a = "onex.evt.delta-a.v1"
        topic_b = "onex.evt.delta-b.v1"
        topic_c = "onex.evt.delta-c.v1"

        # Initial registration with topics A and B
        # Update topic index FIRST, then store event bus config
        event_bus_1 = ModelNodeEventBusConfig(
            subscribe_topics=[
                ModelEventBusTopicEntry(topic=topic_a),
                ModelEventBusTopicEntry(topic=topic_b),
            ],
        )
        await topic_index_mixin._update_topic_index(
            node_id, event_bus_1, correlation_id
        )
        await topic_index_mixin._store_node_event_bus(
            node_id, event_bus_1, correlation_id
        )

        # Verify initial state
        assert node_id in await topic_index_mixin._get_topic_subscribers(
            topic_a, correlation_id
        )
        assert node_id in await topic_index_mixin._get_topic_subscribers(
            topic_b, correlation_id
        )
        assert node_id not in await topic_index_mixin._get_topic_subscribers(
            topic_c, correlation_id
        )

        # Re-register with topics B and C (remove A, keep B, add C)
        # Update topic index FIRST (reads old state), then store new state
        event_bus_2 = ModelNodeEventBusConfig(
            subscribe_topics=[
                ModelEventBusTopicEntry(topic=topic_b),
                ModelEventBusTopicEntry(topic=topic_c),
            ],
        )
        await topic_index_mixin._update_topic_index(
            node_id, event_bus_2, correlation_id
        )
        await topic_index_mixin._store_node_event_bus(
            node_id, event_bus_2, correlation_id
        )

        # Verify delta was applied correctly
        subscribers_a = await topic_index_mixin._get_topic_subscribers(
            topic_a, correlation_id
        )
        subscribers_b = await topic_index_mixin._get_topic_subscribers(
            topic_b, correlation_id
        )
        subscribers_c = await topic_index_mixin._get_topic_subscribers(
            topic_c, correlation_id
        )

        assert node_id not in subscribers_a  # Removed
        assert node_id in subscribers_b  # Kept
        assert node_id in subscribers_c  # Added

    async def test_full_reregistration_idempotent(
        self,
        topic_index_mixin: _TopicIndexMixinImpl,
        correlation_id: UUID,
    ) -> None:
        """Full re-registration with same config is idempotent.

        Validates:
        1. Re-registering with identical config produces same result
        2. No duplicates in subscriber lists
        3. Index remains consistent after multiple updates
        """
        node_id = str(uuid4())
        topic = "onex.evt.reregister.v1"

        event_bus = ModelNodeEventBusConfig(
            subscribe_topics=[ModelEventBusTopicEntry(topic=topic)],
        )

        # Register multiple times
        # Update topic index FIRST, then store event bus config
        for _ in range(3):
            await topic_index_mixin._update_topic_index(
                node_id, event_bus, correlation_id
            )
            await topic_index_mixin._store_node_event_bus(
                node_id, event_bus, correlation_id
            )

        # Verify only one entry
        subscribers = await topic_index_mixin._get_topic_subscribers(
            topic, correlation_id
        )
        assert subscribers == [node_id]


# =============================================================================
# Topic Query Routing Tests
# =============================================================================


class TestTopicQueryRouting:
    """Tests for querying topic subscribers for routing."""

    async def test_query_returns_all_subscribers(
        self,
        topic_index_mixin: _TopicIndexMixinImpl,
        correlation_id: UUID,
        sample_node_ids: list[str],
    ) -> None:
        """Query returns all node IDs for topic.

        Validates:
        1. All registered subscribers are returned
        2. List is complete (no missing entries)
        3. List is sorted for consistency
        """
        topic = "onex.evt.all-subscribers.v1"

        # Register all nodes
        for node_id in sample_node_ids:
            await topic_index_mixin._add_subscriber_to_topic(
                topic, node_id, correlation_id
            )

        # Query subscribers
        subscribers = await topic_index_mixin._get_topic_subscribers(
            topic, correlation_id
        )

        # Verify all nodes present
        assert len(subscribers) == len(sample_node_ids)
        for node_id in sample_node_ids:
            assert node_id in subscribers

        # Verify sorted
        assert subscribers == sorted(subscribers)

    async def test_query_unknown_topic_returns_empty(
        self,
        topic_index_mixin: _TopicIndexMixinImpl,
        correlation_id: UUID,
    ) -> None:
        """Query for non-existent topic returns empty list.

        Validates:
        1. Unknown topic doesn't raise error
        2. Returns empty list (not None)
        3. Graceful handling of missing KV key
        """
        unknown_topic = "onex.evt.nonexistent-topic.v1"

        subscribers = await topic_index_mixin._get_topic_subscribers(
            unknown_topic, correlation_id
        )

        assert subscribers == []
        assert isinstance(subscribers, list)

    async def test_query_returns_strings(
        self,
        topic_index_mixin: _TopicIndexMixinImpl,
        correlation_id: UUID,
    ) -> None:
        """Query returns string node IDs.

        Validates:
        1. Node IDs are returned as strings
        2. Can be converted to UUIDs by caller
        3. Format matches UUID string representation
        """
        node_id = str(uuid4())
        topic = "onex.evt.string-ids.v1"

        await topic_index_mixin._add_subscriber_to_topic(topic, node_id, correlation_id)

        subscribers = await topic_index_mixin._get_topic_subscribers(
            topic, correlation_id
        )

        assert len(subscribers) == 1
        assert isinstance(subscribers[0], str)
        # Verify it's a valid UUID format
        UUID(subscribers[0])

    async def test_query_after_subscriber_removal(
        self,
        topic_index_mixin: _TopicIndexMixinImpl,
        correlation_id: UUID,
    ) -> None:
        """Query reflects subscriber removals correctly.

        Validates:
        1. Removed subscribers don't appear in query results
        2. Remaining subscribers still appear
        3. Query is consistent with index state
        """
        node_a = str(uuid4())
        node_b = str(uuid4())
        topic = "onex.evt.removal-query.v1"

        # Add both nodes
        await topic_index_mixin._add_subscriber_to_topic(topic, node_a, correlation_id)
        await topic_index_mixin._add_subscriber_to_topic(topic, node_b, correlation_id)

        # Remove node A
        await topic_index_mixin._remove_subscriber_from_topic(
            topic, node_a, correlation_id
        )

        # Query should only return node B
        subscribers = await topic_index_mixin._get_topic_subscribers(
            topic, correlation_id
        )
        assert subscribers == [node_b]


# =============================================================================
# Registration to Routing End-to-End Tests
# =============================================================================


class TestRegistrationToRouting:
    """End-to-end: registration -> index -> query."""

    async def test_full_registration_to_query_flow(
        self,
        topic_index_mixin: _TopicIndexMixinImpl,
        correlation_id: UUID,
    ) -> None:
        """Register node -> query topic -> get node ID.

        Validates complete flow:
        1. Node registers with event bus config
        2. Topic index is updated
        3. Query returns correct node ID
        4. Multiple queries return consistent results
        """
        node_id = str(uuid4())
        topic = "onex.evt.full-flow.v1"

        # Step 1: Create event bus configuration
        event_bus = ModelNodeEventBusConfig(
            subscribe_topics=[
                ModelEventBusTopicEntry(
                    topic=topic,
                    event_type="ModelTestEvent",
                    message_category="EVENT",
                    description="Test event for full flow",
                ),
            ],
            publish_topics=[
                ModelEventBusTopicEntry(
                    topic="onex.evt.output.v1",
                    event_type="ModelOutputEvent",
                ),
            ],
        )

        # Step 2: Update topic index FIRST (reads old state, computes delta)
        await topic_index_mixin._update_topic_index(node_id, event_bus, correlation_id)

        # Step 3: Store event bus config (saves new state for next delta computation)
        await topic_index_mixin._store_node_event_bus(
            node_id, event_bus, correlation_id
        )

        # Step 4: Query for subscribers
        subscribers = await topic_index_mixin._get_topic_subscribers(
            topic, correlation_id
        )

        # Step 5: Verify node ID returned
        assert node_id in subscribers
        assert len(subscribers) == 1

        # Step 6: Verify consistent on multiple queries
        subscribers2 = await topic_index_mixin._get_topic_subscribers(
            topic, correlation_id
        )
        assert subscribers == subscribers2

    async def test_deregistration_removes_from_index(
        self,
        topic_index_mixin: _TopicIndexMixinImpl,
        correlation_id: UUID,
    ) -> None:
        """Deregistering node removes from topic index.

        Validates:
        1. Node is initially in subscriber list
        2. Re-registering with empty subscriptions removes from index
        3. Query no longer returns the node
        """
        node_id = str(uuid4())
        topic = "onex.evt.deregister.v1"

        # Initial registration
        # Update topic index FIRST, then store event bus config
        event_bus_1 = ModelNodeEventBusConfig(
            subscribe_topics=[ModelEventBusTopicEntry(topic=topic)],
        )
        await topic_index_mixin._update_topic_index(
            node_id, event_bus_1, correlation_id
        )
        await topic_index_mixin._store_node_event_bus(
            node_id, event_bus_1, correlation_id
        )

        # Verify node is in index
        subscribers = await topic_index_mixin._get_topic_subscribers(
            topic, correlation_id
        )
        assert node_id in subscribers

        # Deregister by updating with empty subscriptions
        # Update topic index FIRST (reads stored state to compute delta), then store
        event_bus_2 = ModelNodeEventBusConfig(
            subscribe_topics=[],
        )
        await topic_index_mixin._update_topic_index(
            node_id, event_bus_2, correlation_id
        )
        await topic_index_mixin._store_node_event_bus(
            node_id, event_bus_2, correlation_id
        )

        # Verify node is removed from index
        subscribers = await topic_index_mixin._get_topic_subscribers(
            topic, correlation_id
        )
        assert node_id not in subscribers
        assert subscribers == []

    async def test_multi_node_routing_scenario(
        self,
        topic_index_mixin: _TopicIndexMixinImpl,
        correlation_id: UUID,
    ) -> None:
        """Multi-node scenario simulating real routing.

        Simulates:
        - Node A: classifier that subscribes to input, publishes classifications
        - Node B: processor that subscribes to classifications
        - Node C: logger that subscribes to both classifications and processed
        """
        node_a = str(uuid4())  # Classifier
        node_b = str(uuid4())  # Processor
        node_c = str(uuid4())  # Logger

        topic_input = "onex.evt.input.v1"
        topic_classified = "onex.evt.classified.v1"
        topic_processed = "onex.evt.processed.v1"

        # Node A: Classifier
        # Update topic index FIRST, then store event bus config
        event_bus_a = ModelNodeEventBusConfig(
            subscribe_topics=[ModelEventBusTopicEntry(topic=topic_input)],
            publish_topics=[ModelEventBusTopicEntry(topic=topic_classified)],
        )
        await topic_index_mixin._update_topic_index(node_a, event_bus_a, correlation_id)
        await topic_index_mixin._store_node_event_bus(
            node_a, event_bus_a, correlation_id
        )

        # Node B: Processor
        event_bus_b = ModelNodeEventBusConfig(
            subscribe_topics=[ModelEventBusTopicEntry(topic=topic_classified)],
            publish_topics=[ModelEventBusTopicEntry(topic=topic_processed)],
        )
        await topic_index_mixin._update_topic_index(node_b, event_bus_b, correlation_id)
        await topic_index_mixin._store_node_event_bus(
            node_b, event_bus_b, correlation_id
        )

        # Node C: Logger (subscribes to multiple)
        event_bus_c = ModelNodeEventBusConfig(
            subscribe_topics=[
                ModelEventBusTopicEntry(topic=topic_classified),
                ModelEventBusTopicEntry(topic=topic_processed),
            ],
        )
        await topic_index_mixin._update_topic_index(node_c, event_bus_c, correlation_id)
        await topic_index_mixin._store_node_event_bus(
            node_c, event_bus_c, correlation_id
        )

        # Verify routing topology
        # Input topic: only classifier
        input_subscribers = await topic_index_mixin._get_topic_subscribers(
            topic_input, correlation_id
        )
        assert input_subscribers == [node_a]

        # Classified topic: processor and logger
        classified_subscribers = await topic_index_mixin._get_topic_subscribers(
            topic_classified, correlation_id
        )
        assert len(classified_subscribers) == 2
        assert node_b in classified_subscribers
        assert node_c in classified_subscribers

        # Processed topic: only logger
        processed_subscribers = await topic_index_mixin._get_topic_subscribers(
            topic_processed, correlation_id
        )
        assert processed_subscribers == [node_c]


# =============================================================================
# KV Storage Format Tests
# =============================================================================


class TestKVStorageFormat:
    """Tests for verifying the Consul KV storage format."""

    async def test_subscribe_topics_stored_as_json_array(
        self,
        topic_index_mixin: _TopicIndexMixinImpl,
        correlation_id: UUID,
    ) -> None:
        """Verify subscribe_topics stored as JSON array of strings.

        Validates:
        1. Topics stored at correct KV path
        2. Format is JSON array
        3. Contains topic strings only (not full entry objects)
        """
        node_id = str(uuid4())
        topics = ["onex.evt.a.v1", "onex.evt.b.v1"]

        event_bus = ModelNodeEventBusConfig(
            subscribe_topics=[ModelEventBusTopicEntry(topic=t) for t in topics],
        )
        await topic_index_mixin._store_node_event_bus(
            node_id, event_bus, correlation_id
        )

        # Read raw value from KV
        raw_value = await topic_index_mixin.kv_get_raw(
            f"onex/nodes/{node_id}/event_bus/subscribe_topics",
            correlation_id,
        )

        assert raw_value is not None
        parsed = json.loads(raw_value)
        assert isinstance(parsed, list)
        assert set(parsed) == set(topics)

    async def test_topic_subscribers_stored_as_sorted_json_array(
        self,
        topic_index_mixin: _TopicIndexMixinImpl,
        correlation_id: UUID,
    ) -> None:
        """Verify topic subscribers stored as sorted JSON array.

        Validates:
        1. Subscribers stored at correct KV path
        2. Format is JSON array
        3. Array is sorted for consistent ordering
        """
        topic = "onex.evt.sorted-test.v1"
        node_ids = [str(uuid4()) for _ in range(3)]

        # Add in random order
        for node_id in node_ids:
            await topic_index_mixin._add_subscriber_to_topic(
                topic, node_id, correlation_id
            )

        # Read raw value from KV
        raw_value = await topic_index_mixin.kv_get_raw(
            f"onex/topics/{topic}/subscribers",
            correlation_id,
        )

        assert raw_value is not None
        parsed = json.loads(raw_value)
        assert isinstance(parsed, list)
        assert parsed == sorted(parsed)  # Verify sorted
        assert set(parsed) == set(node_ids)


__all__: list[str] = [
    "TestTopicSubscriberIndex",
    "TestTopicIndexIdempotency",
    "TestTopicQueryRouting",
    "TestRegistrationToRouting",
    "TestKVStorageFormat",
]
