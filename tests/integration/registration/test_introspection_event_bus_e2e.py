# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for event bus topic storage in registry (OMN-1613).

Tests the full flow:
1. Contract with event_bus topics -> Introspection
2. Introspection event contains resolved topics
3. Consul KV stores topic data
4. Runtime can query subscribers for topic

Test Categories:
    - Introspection Extraction: Contract -> introspection event with event_bus
    - Consul Storage: Event bus config stored in Consul KV
    - Topic Reverse Index: Topic -> node_id lookup
    - Idempotency: Re-registration and index updates

Infrastructure Requirements:
    Tests requiring Consul use real Consul infrastructure.
    Tests will skip gracefully if infrastructure is unavailable.

Related Tickets:
    - OMN-1613: Add event_bus topic storage to registry for dynamic topic routing
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumIntrospectionReason
from omnibase_infra.mixins import MixinNodeIntrospection
from omnibase_infra.models.discovery import ModelIntrospectionConfig
from omnibase_infra.models.registration.model_node_event_bus_config import (
    ModelEventBusTopicEntry,
    ModelNodeEventBusConfig,
)
from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)
from tests.integration.handlers.conftest import CONSUL_AVAILABLE

if TYPE_CHECKING:
    from omnibase_core.types import JsonType
    from omnibase_infra.handlers import HandlerConsul

# Note: pytest.mark.asyncio is applied per test class, not module-level,
# because TestModelNodeEventBusConfig contains synchronous tests.


# =============================================================================
# Mock Fixtures
# =============================================================================


class MockEventBus:
    """Mock event bus for testing introspection without Kafka."""

    def __init__(self) -> None:
        """Initialize mock event bus."""
        self.published_envelopes: list[tuple[object, str]] = []

    async def publish_envelope(self, envelope: object, topic: str) -> None:
        """Mock publish_envelope method."""
        self.published_envelopes.append((envelope, topic))


class MockContract:
    """Mock contract with event_bus subcontract for testing.

    Note: The MixinNodeIntrospection._extract_event_bus_config method looks for
    `publish_topics` and `subscribe_topics` attributes on the event_bus subcontract.

    The ContractCapabilityExtractor requires:
    - `node_type` field for contract type extraction
    - `contract_version` field (ModelSemVer) for version extraction
    """

    def __init__(
        self,
        publish_suffixes: list[str] | None = None,
        subscribe_suffixes: list[str] | None = None,
        node_type: str = "EFFECT",
    ) -> None:
        """Initialize mock contract.

        Args:
            publish_suffixes: Topic suffixes the node publishes to.
            subscribe_suffixes: Topic suffixes the node subscribes to.
            node_type: Node type for ContractCapabilityExtractor (default: EFFECT).
        """
        self.event_bus = self._EventBusSubcontract(
            publish_topics=publish_suffixes or [],
            subscribe_topics=subscribe_suffixes or [],
        )
        # Required by ContractCapabilityExtractor._extract_contract_type
        self.node_type = node_type
        # Required by ContractCapabilityExtractor._extract_version
        self.contract_version = ModelSemVer(major=1, minor=0, patch=0)

    class _EventBusSubcontract:
        """Mock event_bus subcontract.

        Uses `publish_topics` and `subscribe_topics` to match the expected
        interface used by MixinNodeIntrospection._extract_event_bus_config.
        """

        def __init__(
            self, publish_topics: list[str], subscribe_topics: list[str]
        ) -> None:
            self.publish_topics = publish_topics
            self.subscribe_topics = subscribe_topics


class IntrospectionTestNode(MixinNodeIntrospection):
    """Test node implementing MixinNodeIntrospection for E2E testing."""

    def __init__(
        self,
        node_id: UUID,
        event_bus: MockEventBus | None = None,
        contract: MockContract | None = None,
    ) -> None:
        """Initialize test node.

        Args:
            node_id: Unique node identifier.
            event_bus: Optional event bus for publishing.
            contract: Optional contract with event_bus subcontract.
        """
        self._state = "initialized"
        self.health_url = f"http://localhost:8080/{node_id}/health"

        config = ModelIntrospectionConfig(
            node_id=node_id,
            node_type=EnumNodeKind.EFFECT,
            event_bus=event_bus,
            version="1.0.0",
            introspection_topic="node.introspection",
        )
        self.initialize_introspection(config)

        # Manually set contract after initialization since it's not in ModelIntrospectionConfig
        # This simulates nodes that provide contract for event_bus extraction
        if contract is not None:
            self._introspection_contract = contract

    async def execute_effect(
        self, operation: str, payload: dict[str, object]
    ) -> dict[str, object]:
        """Mock execute method for EFFECT node."""
        return {"result": "ok", "operation": operation}


# =============================================================================
# Introspection Event Bus Extraction Tests
# =============================================================================


@pytest.mark.asyncio
class TestIntrospectionEventBusExtraction:
    """Tests for extracting event_bus from contract into introspection event."""

    async def test_introspection_extracts_event_bus_from_contract(self) -> None:
        """Full flow: contract -> introspection -> event contains resolved topics."""
        # Arrange: Create node with contract that has event_bus topics
        node_id = uuid4()
        contract = MockContract(
            publish_suffixes=["onex.evt.node-registered.v1"],
            subscribe_suffixes=["onex.evt.intent-classified.v1"],
        )
        node = IntrospectionTestNode(node_id=node_id, contract=contract)

        # Act: Get introspection data
        with patch.dict(os.environ, {"ONEX_ENV": "dev"}):
            data = await node.get_introspection_data()

        # Assert: Event bus config is populated with resolved topics
        assert data.event_bus is not None
        assert len(data.event_bus.publish_topics) == 1
        assert len(data.event_bus.subscribe_topics) == 1

        # Topics should be resolved with environment prefix
        assert data.event_bus.publish_topic_strings == [
            "dev.onex.evt.node-registered.v1"
        ]
        assert data.event_bus.subscribe_topic_strings == [
            "dev.onex.evt.intent-classified.v1"
        ]

    async def test_introspection_event_with_multiple_topics(self) -> None:
        """Introspection event contains all resolved topics from contract."""
        # Arrange: Contract with multiple publish and subscribe topics
        node_id = uuid4()
        contract = MockContract(
            publish_suffixes=[
                "onex.evt.node-registered.v1",
                "onex.evt.node-heartbeat.v1",
            ],
            subscribe_suffixes=[
                "onex.evt.intent-classified.v1",
                "onex.cmd.introspect-node.v1",
            ],
        )
        node = IntrospectionTestNode(node_id=node_id, contract=contract)

        # Act
        with patch.dict(os.environ, {"ONEX_ENV": "prod"}):
            data = await node.get_introspection_data()

        # Assert
        assert data.event_bus is not None
        assert len(data.event_bus.publish_topics) == 2
        assert len(data.event_bus.subscribe_topics) == 2

        # Verify prod environment prefix
        assert (
            "prod.onex.evt.node-registered.v1" in data.event_bus.publish_topic_strings
        )
        assert "prod.onex.evt.node-heartbeat.v1" in data.event_bus.publish_topic_strings
        assert (
            "prod.onex.evt.intent-classified.v1"
            in data.event_bus.subscribe_topic_strings
        )
        assert (
            "prod.onex.cmd.introspect-node.v1" in data.event_bus.subscribe_topic_strings
        )

    async def test_introspection_without_contract_has_no_event_bus(self) -> None:
        """Node without contract produces introspection event with event_bus=None."""
        # Arrange: Node without contract
        node_id = uuid4()
        node = IntrospectionTestNode(node_id=node_id)

        # Act
        data = await node.get_introspection_data()

        # Assert: event_bus is None when no contract is provided
        assert data.event_bus is None

    async def test_introspection_with_empty_event_bus_subcontract(self) -> None:
        """Contract with empty event_bus returns None (no topics to register).

        When both publish_topics and subscribe_topics are empty, the
        _extract_event_bus_config method returns None because there are
        no topics to register in the index.
        """
        # Arrange: Contract with empty topic lists
        node_id = uuid4()
        contract = MockContract(publish_suffixes=[], subscribe_suffixes=[])
        node = IntrospectionTestNode(node_id=node_id, contract=contract)

        # Act
        data = await node.get_introspection_data()

        # Assert: event_bus is None when both topic lists are empty
        # This is intentional - no point in registering empty config
        assert data.event_bus is None

    async def test_different_env_prefixes(self) -> None:
        """Different ONEX_ENV values produce different topics."""
        node_id = uuid4()
        contract = MockContract(
            subscribe_suffixes=["onex.evt.intent-classified.v1"],
        )

        # Test dev environment
        node_dev = IntrospectionTestNode(node_id=node_id, contract=contract)
        with patch.dict(os.environ, {"ONEX_ENV": "dev"}):
            data_dev = await node_dev.get_introspection_data()

        # Test staging environment
        node_staging = IntrospectionTestNode(node_id=uuid4(), contract=contract)
        with patch.dict(os.environ, {"ONEX_ENV": "staging"}):
            data_staging = await node_staging.get_introspection_data()

        # Assert: Different environments produce different prefixes
        assert data_dev.event_bus is not None
        assert data_staging.event_bus is not None
        assert data_dev.event_bus.subscribe_topic_strings == [
            "dev.onex.evt.intent-classified.v1"
        ]
        assert data_staging.event_bus.subscribe_topic_strings == [
            "staging.onex.evt.intent-classified.v1"
        ]


# =============================================================================
# Model Tests
# =============================================================================


class TestModelNodeEventBusConfig:
    """Tests for ModelNodeEventBusConfig and related models."""

    def test_event_bus_topic_entry_creation(self) -> None:
        """ModelEventBusTopicEntry can be created with required and optional fields."""
        # With only required field
        entry = ModelEventBusTopicEntry(topic="dev.onex.evt.test.v1")
        assert entry.topic == "dev.onex.evt.test.v1"
        assert entry.event_type is None
        assert entry.message_category == "EVENT"  # default
        assert entry.description is None

        # With all fields
        full_entry = ModelEventBusTopicEntry(
            topic="dev.onex.cmd.test.v1",
            event_type="ModelTestCommand",
            message_category="COMMAND",
            description="Test command topic",
        )
        assert full_entry.topic == "dev.onex.cmd.test.v1"
        assert full_entry.event_type == "ModelTestCommand"
        assert full_entry.message_category == "COMMAND"
        assert full_entry.description == "Test command topic"

    def test_event_bus_config_topic_string_extraction(self) -> None:
        """ModelNodeEventBusConfig extracts topic strings correctly."""
        config = ModelNodeEventBusConfig(
            subscribe_topics=[
                ModelEventBusTopicEntry(topic="dev.onex.evt.input.v1"),
                ModelEventBusTopicEntry(topic="dev.onex.evt.input2.v1"),
            ],
            publish_topics=[
                ModelEventBusTopicEntry(topic="dev.onex.evt.output.v1"),
            ],
        )

        # Property methods extract only topic strings
        assert config.subscribe_topic_strings == [
            "dev.onex.evt.input.v1",
            "dev.onex.evt.input2.v1",
        ]
        assert config.publish_topic_strings == ["dev.onex.evt.output.v1"]

    def test_event_bus_config_immutability(self) -> None:
        """ModelNodeEventBusConfig is frozen (immutable)."""
        config = ModelNodeEventBusConfig(
            subscribe_topics=[ModelEventBusTopicEntry(topic="dev.test.v1")],
        )

        # Attempting to modify should raise
        with pytest.raises(Exception):  # ValidationError for frozen model
            config.subscribe_topics = []  # type: ignore[misc]


# =============================================================================
# Consul Integration Tests
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.skipif(
    not CONSUL_AVAILABLE,
    reason="Consul not available (cannot connect to remote infrastructure)",
)
class TestConsulEventBusStorage:
    """Tests for storing event_bus config in Consul KV.

    These tests require real Consul infrastructure.
    """

    @pytest.fixture
    def unique_node_id(self) -> str:
        """Generate unique node ID for test isolation."""
        return f"test-node-{uuid4().hex[:12]}"

    @pytest.fixture
    def sample_event_bus_config(self) -> ModelNodeEventBusConfig:
        """Create sample event_bus config for testing."""
        return ModelNodeEventBusConfig(
            subscribe_topics=[
                ModelEventBusTopicEntry(
                    topic="dev.onex.evt.intent-classified.v1",
                    event_type="ModelIntentClassified",
                ),
                ModelEventBusTopicEntry(
                    topic="dev.onex.evt.node-heartbeat.v1",
                ),
            ],
            publish_topics=[
                ModelEventBusTopicEntry(
                    topic="dev.onex.evt.node-registered.v1",
                    event_type="ModelNodeRegistered",
                ),
            ],
        )

    async def test_store_node_event_bus(
        self,
        initialized_consul_handler: HandlerConsul,
        unique_node_id: str,
        sample_event_bus_config: ModelNodeEventBusConfig,
    ) -> None:
        """Event bus config stored in Consul KV during node registration."""
        handler = initialized_consul_handler
        correlation_id = uuid4()

        # Act: Store event_bus config using the mixin method
        await handler._store_node_event_bus(
            node_id=unique_node_id,
            event_bus=sample_event_bus_config,
            correlation_id=correlation_id,
        )

        # Assert: Verify subscribe_topics stored
        get_envelope = {
            "operation": "consul.kv_get",
            "payload": {
                "key": f"onex/nodes/{unique_node_id}/event_bus/subscribe_topics",
            },
            "correlation_id": str(uuid4()),
        }
        result = await handler.execute(get_envelope)
        assert result.result.status == "success"

        stored_topics = json.loads(result.result.payload.data.value)
        assert stored_topics == sample_event_bus_config.subscribe_topic_strings

        # Cleanup
        await self._cleanup_node_keys(handler, unique_node_id)

    async def test_topic_reverse_index_created(
        self,
        initialized_consul_handler: HandlerConsul,
        unique_node_id: str,
        sample_event_bus_config: ModelNodeEventBusConfig,
    ) -> None:
        """Topic -> node_id reverse index created in Consul."""
        handler = initialized_consul_handler
        correlation_id = uuid4()

        # Act: Update topic index
        await handler._update_topic_index(
            node_id=unique_node_id,
            event_bus=sample_event_bus_config,
            correlation_id=correlation_id,
        )

        # Also store the node config (required for idempotent updates)
        await handler._store_node_event_bus(
            node_id=unique_node_id,
            event_bus=sample_event_bus_config,
            correlation_id=correlation_id,
        )

        # Assert: Check reverse index for first subscribe topic
        topic = sample_event_bus_config.subscribe_topic_strings[0]
        get_envelope = {
            "operation": "consul.kv_get",
            "payload": {
                "key": f"onex/topics/{topic}/subscribers",
            },
            "correlation_id": str(uuid4()),
        }
        result = await handler.execute(get_envelope)
        assert result.result.status == "success"

        subscribers = json.loads(result.result.payload.data.value)
        assert unique_node_id in subscribers

        # Cleanup
        await self._cleanup_node_keys(handler, unique_node_id)
        await self._cleanup_topic_keys(
            handler, sample_event_bus_config.subscribe_topic_strings
        )

    async def test_get_topic_subscribers(
        self,
        initialized_consul_handler: HandlerConsul,
        unique_node_id: str,
        sample_event_bus_config: ModelNodeEventBusConfig,
    ) -> None:
        """RuntimeHostProcess.get_subscribers_for_topic() returns node IDs."""
        handler = initialized_consul_handler
        correlation_id = uuid4()

        # Arrange: Update index FIRST (reads previous state), then store SECOND
        # The update method reads previous state to compute delta, so order matters:
        # 1. _update_topic_index reads previous (empty) state, adds new topics to reverse index
        # 2. _store_node_event_bus writes new state for next delta computation
        await handler._update_topic_index(
            node_id=unique_node_id,
            event_bus=sample_event_bus_config,
            correlation_id=correlation_id,
        )
        await handler._store_node_event_bus(
            node_id=unique_node_id,
            event_bus=sample_event_bus_config,
            correlation_id=correlation_id,
        )

        # Act: Query subscribers for topic
        topic = sample_event_bus_config.subscribe_topic_strings[0]
        subscribers = await handler._get_topic_subscribers(
            topic=topic,
            correlation_id=correlation_id,
        )

        # Assert
        assert unique_node_id in subscribers

        # Cleanup
        await self._cleanup_node_keys(handler, unique_node_id)
        await self._cleanup_topic_keys(
            handler, sample_event_bus_config.subscribe_topic_strings
        )

    async def test_multiple_nodes_same_topic(
        self,
        initialized_consul_handler: HandlerConsul,
    ) -> None:
        """Multiple nodes subscribing to same topic all appear in index."""
        handler = initialized_consul_handler
        correlation_id = uuid4()
        topic = "dev.onex.evt.shared-topic.v1"

        node_ids = [f"test-node-{uuid4().hex[:8]}" for _ in range(3)]
        config = ModelNodeEventBusConfig(
            subscribe_topics=[ModelEventBusTopicEntry(topic=topic)],
        )

        try:
            # Register all nodes - update index FIRST, then store
            for node_id in node_ids:
                await handler._update_topic_index(
                    node_id=node_id,
                    event_bus=config,
                    correlation_id=correlation_id,
                )
                await handler._store_node_event_bus(
                    node_id=node_id,
                    event_bus=config,
                    correlation_id=correlation_id,
                )

            # Query subscribers
            subscribers = await handler._get_topic_subscribers(
                topic=topic,
                correlation_id=correlation_id,
            )

            # All nodes should be in the subscriber list
            for node_id in node_ids:
                assert node_id in subscribers

        finally:
            # Cleanup
            for node_id in node_ids:
                await self._cleanup_node_keys(handler, node_id)
            await self._cleanup_topic_keys(handler, [topic])

    async def _cleanup_node_keys(self, handler: HandlerConsul, node_id: str) -> None:
        """Helper to cleanup node-related Consul keys."""
        keys_to_delete = [
            f"onex/nodes/{node_id}/event_bus/subscribe_topics",
            f"onex/nodes/{node_id}/event_bus/publish_topics",
            f"onex/nodes/{node_id}/event_bus/subscribe_entries",
            f"onex/nodes/{node_id}/event_bus/publish_entries",
        ]
        for key in keys_to_delete:
            try:
                # Use KV delete if available, otherwise just ignore
                if hasattr(handler, "_client") and handler._client is not None:
                    handler._client.kv.delete(key)
            except Exception:
                pass  # Ignore cleanup errors

    async def _cleanup_topic_keys(
        self, handler: HandlerConsul, topics: list[str]
    ) -> None:
        """Helper to cleanup topic-related Consul keys."""
        for topic in topics:
            try:
                if hasattr(handler, "_client") and handler._client is not None:
                    handler._client.kv.delete(f"onex/topics/{topic}/subscribers")
            except Exception:
                pass  # Ignore cleanup errors


@pytest.mark.skipif(
    not CONSUL_AVAILABLE,
    reason="Consul not available (cannot connect to remote infrastructure)",
)
@pytest.mark.asyncio
class TestIdempotencyE2E:
    """Tests for idempotent registration behavior.

    These tests require real Consul infrastructure.
    """

    @pytest.fixture
    def unique_node_id(self) -> str:
        """Generate unique node ID for test isolation."""
        return f"test-node-{uuid4().hex[:12]}"

    async def test_reregistration_no_duplicate_subscribers(
        self,
        initialized_consul_handler: HandlerConsul,
        unique_node_id: str,
    ) -> None:
        """Re-registering same node doesn't create duplicate index entries."""
        handler = initialized_consul_handler
        correlation_id = uuid4()
        topic = "dev.onex.evt.idempotent-test.v1"

        config = ModelNodeEventBusConfig(
            subscribe_topics=[ModelEventBusTopicEntry(topic=topic)],
        )

        try:
            # Register twice - update index FIRST, then store
            for _ in range(2):
                await handler._update_topic_index(
                    node_id=unique_node_id,
                    event_bus=config,
                    correlation_id=correlation_id,
                )
                await handler._store_node_event_bus(
                    node_id=unique_node_id,
                    event_bus=config,
                    correlation_id=correlation_id,
                )

            # Query subscribers
            subscribers = await handler._get_topic_subscribers(
                topic=topic,
                correlation_id=correlation_id,
            )

            # Should only appear once (idempotent)
            assert subscribers.count(unique_node_id) == 1

        finally:
            # Cleanup
            try:
                if hasattr(handler, "_client") and handler._client is not None:
                    handler._client.kv.delete(
                        f"onex/nodes/{unique_node_id}/event_bus/subscribe_topics"
                    )
                    handler._client.kv.delete(
                        f"onex/nodes/{unique_node_id}/event_bus/publish_topics"
                    )
                    handler._client.kv.delete(
                        f"onex/nodes/{unique_node_id}/event_bus/subscribe_entries"
                    )
                    handler._client.kv.delete(
                        f"onex/nodes/{unique_node_id}/event_bus/publish_entries"
                    )
                    handler._client.kv.delete(f"onex/topics/{topic}/subscribers")
            except Exception:
                pass

    async def test_contract_change_updates_index(
        self,
        initialized_consul_handler: HandlerConsul,
        unique_node_id: str,
    ) -> None:
        """Changing contract topics updates index correctly."""
        handler = initialized_consul_handler
        correlation_id = uuid4()
        old_topic = "dev.onex.evt.old-topic.v1"
        new_topic = "dev.onex.evt.new-topic.v1"

        # Initial config
        config_v1 = ModelNodeEventBusConfig(
            subscribe_topics=[ModelEventBusTopicEntry(topic=old_topic)],
        )

        # Updated config
        config_v2 = ModelNodeEventBusConfig(
            subscribe_topics=[ModelEventBusTopicEntry(topic=new_topic)],
        )

        try:
            # Register with v1 config
            await handler._store_node_event_bus(
                node_id=unique_node_id,
                event_bus=config_v1,
                correlation_id=correlation_id,
            )
            await handler._update_topic_index(
                node_id=unique_node_id,
                event_bus=config_v1,
                correlation_id=correlation_id,
            )

            # Update to v2 config
            await handler._update_topic_index(
                node_id=unique_node_id,
                event_bus=config_v2,
                correlation_id=correlation_id,
            )
            await handler._store_node_event_bus(
                node_id=unique_node_id,
                event_bus=config_v2,
                correlation_id=correlation_id,
            )

            # Check old topic - node should be removed
            old_subscribers = await handler._get_topic_subscribers(
                topic=old_topic,
                correlation_id=correlation_id,
            )
            assert unique_node_id not in old_subscribers

            # Check new topic - node should be present
            new_subscribers = await handler._get_topic_subscribers(
                topic=new_topic,
                correlation_id=correlation_id,
            )
            assert unique_node_id in new_subscribers

        finally:
            # Cleanup
            try:
                if hasattr(handler, "_client") and handler._client is not None:
                    handler._client.kv.delete(
                        f"onex/nodes/{unique_node_id}/event_bus/subscribe_topics"
                    )
                    handler._client.kv.delete(
                        f"onex/nodes/{unique_node_id}/event_bus/publish_topics"
                    )
                    handler._client.kv.delete(
                        f"onex/nodes/{unique_node_id}/event_bus/subscribe_entries"
                    )
                    handler._client.kv.delete(
                        f"onex/nodes/{unique_node_id}/event_bus/publish_entries"
                    )
                    handler._client.kv.delete(f"onex/topics/{old_topic}/subscribers")
                    handler._client.kv.delete(f"onex/topics/{new_topic}/subscribers")
            except Exception:
                pass


# =============================================================================
# Introspection Event Publishing Tests (Mock-Based)
# =============================================================================


@pytest.mark.asyncio
class TestIntrospectionEventPublishing:
    """Tests for publishing introspection events with event_bus data."""

    async def test_introspection_event_published_with_event_bus(self) -> None:
        """Introspection event published to event bus contains event_bus config."""
        # Arrange
        node_id = uuid4()
        event_bus = MockEventBus()
        contract = MockContract(
            subscribe_suffixes=["onex.evt.test-input.v1"],
            publish_suffixes=["onex.evt.test-output.v1"],
        )
        node = IntrospectionTestNode(
            node_id=node_id, event_bus=event_bus, contract=contract
        )

        # Act: Publish introspection
        with patch.dict(os.environ, {"ONEX_ENV": "dev"}):
            success = await node.publish_introspection(reason="startup")

        # Assert
        assert success is True
        assert len(event_bus.published_envelopes) == 1

        # The published envelope contains the introspection event
        envelope, _topic = event_bus.published_envelopes[0]
        # Extract payload from envelope (ModelEventEnvelope wraps the event)
        if hasattr(envelope, "payload"):
            event = envelope.payload
        else:
            event = envelope

        assert isinstance(event, ModelNodeIntrospectionEvent)
        assert event.event_bus is not None
        assert "dev.onex.evt.test-input.v1" in event.event_bus.subscribe_topic_strings
        assert "dev.onex.evt.test-output.v1" in event.event_bus.publish_topic_strings


__all__ = [
    "TestIntrospectionEventBusExtraction",
    "TestModelNodeEventBusConfig",
    "TestConsulEventBusStorage",
    "TestIdempotencyE2E",
    "TestIntrospectionEventPublishing",
]
